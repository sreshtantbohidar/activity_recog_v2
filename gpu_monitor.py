"""Who is holding the GPU, and can this service still run?

The activity service shares one card with whatever else the host runs, and the
engine needs ``gpu_memory_utilization`` of that card before it will start. When
another process grows, the first symptom used to be a job that failed or an
engine that stayed down; this module answers the question directly so the
Monitor screen can show it - including in red, before an operator notices.

Dependency-light by design (the readers and the process-owner lookup are
injected) so the parsing, the percentages and the alert rule are unit-testable
without a GPU.

Percentages are of the device total, which is what "how much of the card is
this" means to an operator; the share of *used* memory is derived from them.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

# The engine's start-up check, in the same terms the entrypoint gate uses.
DEFAULT_UTILIZATION = 0.80

# Measured cost of the audio stage the engine is parked for: Qwen2.5-Omni-3B
# 4-bit plus faster-whisper large-v3 fp16 is about 5 GB, and it runs on the card
# while the engine is asleep.
AUDIO_STAGE_MIB = 5 * 1024

# Below this margin over the engine's requirement the service is fragile: the
# next process to grow takes the card below what a restart needs.
THIN_MARGIN_MIB = 512

DEVICE_COMMAND = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                  "--format=csv,noheader,nounits"]
PROCESS_COMMAND = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                   "--format=csv,noheader,nounits"]

# Command lines that mean "this process is ours": the in-container engine, the
# API worker, and the helper processes we exec (audio stage, VRAM gate).
APP_MARKERS = ("vllm", "gunicorn", "app:app", "/app/")

# The two processes that make up the vision engine: the server the entrypoint
# supervises (its command line starts with "vllm serve") and the EngineCore child
# it spawns, which is the one holding the weights and the several GB that
# outlive a /sleep. Both count towards the engine's footprint.
ENGINE_MARKERS = ("vllm",)
SERVER_MARKERS = ("vllm serve", "vllm.entrypoints")

OK, WARNING, CRITICAL = "ok", "warning", "critical"


def _run(command: list[str], runner: Callable[..., object] = subprocess.run) -> Optional[str]:
    """Run an nvidia-smi query, returning its output or None when unavailable."""
    try:
        completed = runner(command, capture_output=True, text=True, timeout=20, check=False)
        if getattr(completed, "returncode", 1) != 0:
            return None
        return getattr(completed, "stdout", "") or ""
    except Exception:
        return None


def _split(output: Optional[str]) -> list[list[str]]:
    """CSV rows as trimmed field lists, skipping blanks."""
    if not output:
        return []
    rows = []
    for line in output.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0]:
            rows.append(fields)
    return rows


def _as_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_device(runner: Callable[..., object] = subprocess.run) -> Optional[dict]:
    """Device name and MiB totals, or None when the card cannot be read."""
    rows = _split(_run(DEVICE_COMMAND, runner))
    if not rows or len(rows[0]) < 4:
        return None
    name, total, used, free = rows[0][0], *(_as_int(field) for field in rows[0][1:4])
    if total is None or used is None:
        return None
    return {"name": name, "total_mib": total, "used_mib": used,
            "free_mib": free if free is not None else max(0, total - used)}


def read_processes(runner: Callable[..., object] = subprocess.run) -> list[dict]:
    """Every compute process on the card, as pid/name/MiB."""
    processes = []
    for row in _split(_run(PROCESS_COMMAND, runner)):
        if len(row) < 3:
            continue
        pid, used = _as_int(row[0]), _as_int(row[2])
        if pid is None or used is None:
            continue
        processes.append({"pid": pid, "name": row[1], "used_mib": used})
    return processes


def process_command_line(pid: int) -> str:
    """A process's command line, or "" when it is not visible to us."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except Exception:
        return ""


def is_app_process(pid: int) -> bool:
    """True when the process belongs to this service (engine, worker, helpers)."""
    command_line = process_command_line(pid).lower()
    return bool(command_line) and any(marker in command_line for marker in APP_MARKERS)


def engine_processes(processes: Iterable[dict],
                     command_reader: Optional[Callable[[int], str]] = None) -> list[dict]:
    """The compute processes on the card that belong to the vision engine."""
    reader = command_reader or process_command_line
    return [process for process in processes
            if any(marker in reader(process["pid"]).lower() for marker in ENGINE_MARKERS)]


def engine_memory_mib(runner: Callable[..., object] = subprocess.run,
                      command_reader: Optional[Callable[[int], str]] = None) -> Optional[int]:
    """VRAM in MiB still held by the engine, or None when it cannot be read.

    None means "unknown" - nvidia-smi failed, or no engine process was listed -
    and callers must read it as "say nothing", never as "the engine holds
    nothing": the idle recycle in engine_power.py restarts the engine on this
    number, and a failed read must not stop a running engine.
    """
    processes = read_processes(runner)
    if not processes:
        return None
    engine = engine_processes(processes, command_reader)
    if not engine:
        return None
    return sum(process["used_mib"] for process in engine)


def find_engine_server_pid(proc_root: str = "/proc",
                           command_reader: Optional[Callable[[int], str]] = None) -> Optional[int]:
    """PID of the vLLM server the entrypoint supervises, or None.

    Stopping that process is how an idle engine is recycled: the entrypoint
    waits on it and starts it again, and the EngineCore child dies with it.
    """
    reader = command_reader or process_command_line
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        command = reader(int(entry)).lower()
        if any(marker in command for marker in SERVER_MARKERS):
            return int(entry)
    return None


def percentage(part: int, whole: Optional[int]) -> float:
    """One decimal place, so the table and the chart agree."""
    if not whole:
        return 0.0
    return round(part * 100.0 / whole, 1)


def evaluate(device: dict, processes: Iterable[dict], utilization: float,
             engine: Optional[dict] = None) -> dict:
    """Decide whether the service can still run on this card.

    The engine is the constraint: it needs ``utilization`` of the device total
    *in addition to* whatever other processes hold, so what matters is how much
    the card would have if our own processes released it.

    ``engine`` is the engine's own state (``{"reachable": bool, "sleeping":
    bool|None}``) when the caller can ask it, and None when it cannot. It
    changes what the alert means, not the arithmetic: an engine that is awake
    and serving is not a failure, so the same shortfall is reported as a
    restart risk instead of "not enough VRAM to run". Only a parked or absent
    engine makes the budget check a live blocker.
    """
    processes = list(processes)
    total = device["total_mib"]
    compute_used = sum(process["used_mib"] for process in processes)
    # Desktop, driver and CUDA contexts: memory the card reports as used that no
    # compute process owns. It never comes back, so it counts against the engine.
    unattributed = max(0, device["used_mib"] - compute_used)
    others_used = sum(process["used_mib"] for process in processes if not process["app"]) + unattributed
    available = max(0, total - others_used)
    required = int(total * float(utilization))
    margin = available - required

    state = engine or {}
    known = engine is not None
    reachable = bool(state.get("reachable"))
    sleeping = state.get("sleeping")
    running = reachable and sleeping is False
    parked = reachable and sleeping is True

    if running:
        # The engine is up and holding its budget, so the card is in a valid
        # state whatever the numbers say: what a start would need is a risk to
        # plan around, not an outage to report in red.
        status = OK
        detail = (f"Engine is running: {device['used_mib']} MiB of {total} MiB is in use "
                  f"({percentage(device['used_mib'], total)}%) and other processes hold "
                  f"{others_used} MiB. A restart would need {required} MiB and {available} MiB "
                  f"would be free ({margin} MiB spare).")
        if available < required:
            detail = (f"Engine is running: {device['used_mib']} MiB of {total} MiB is in use "
                      f"({percentage(device['used_mib'], total)}%) and other processes hold "
                      f"{others_used} MiB. It could not start again until the card is freed: a "
                      f"restart needs {required} MiB against {available} MiB free.")
    elif available < required:
        status = CRITICAL
        detail = (f"Not enough VRAM to {'wake the parked engine' if parked else 'run the activity app'}: "
                  f"the engine needs {required} MiB and only {available} MiB is available - other "
                  f"processes hold {others_used} MiB.")
    elif margin < THIN_MARGIN_MIB:
        status = WARNING
        detail = (f"Tight VRAM: {available} MiB available against the engine's {required} MiB - only "
                  f"{margin} MiB of margin. Another process growing would stop a restart.")
    else:
        status = OK
        detail = (f"Enough VRAM: {available} MiB available for the engine's {required} MiB budget "
                  f"({margin} MiB spare).")
    return {"ok": status == OK, "status": status, "detail": detail, "required_mib": required,
            "available_mib": available, "shortfall_mib": max(0, required - available),
            "margin_mib": margin, "others_used_mib": others_used, "unattributed_mib": unattributed,
            "engine_reachable": reachable if known else None, "engine_sleeping": sleeping,
            "engine_running": bool(running) if known else None,
            "restart_ok": available >= required,
            "audio_stage_mib": AUDIO_STAGE_MIB, "audio_ok": available >= AUDIO_STAGE_MIB}


def build_snapshot(utilization: float = DEFAULT_UTILIZATION, *,
                   device_reader: Callable[[], Optional[dict]] = read_device,
                   process_reader: Callable[[], list[dict]] = read_processes,
                   owner_check: Callable[[int], bool] = is_app_process,
                   engine_reader: Optional[Callable[[], Optional[dict]]] = None,
                   clock: Callable[[], float] = time.time) -> dict:
    """One monitor frame: device, per-process usage, and the resource verdict.

    ``engine_reader`` reports whether the engine is up and awake; without it the
    frame can only judge the budget, not whether the service is actually down.
    """
    device = device_reader()
    if not device:
        return {"timestamp": datetime.fromtimestamp(clock(), timezone.utc).isoformat(timespec="seconds"),
                "available": False, "utilization": float(utilization),
                "device": None, "processes": [],
                "resource": {"ok": False, "status": WARNING, "detail": "GPU memory usage is unavailable",
                             "required_mib": None, "available_mib": None, "shortfall_mib": None,
                             "engine_running": None, "engine_reachable": None, "restart_ok": None,
                             "audio_ok": None}}
    total = device["total_mib"]
    processes = []
    for process in process_reader():
        item = dict(process)
        item["app"] = bool(owner_check(process["pid"]))
        item["used_percent"] = percentage(process["used_mib"], total)
        processes.append(item)
    processes.sort(key=lambda item: item["used_mib"], reverse=True)
    device = dict(device)
    device["used_percent"] = percentage(device["used_mib"], total)
    device["free_percent"] = percentage(max(0, total - device["used_mib"]), total)
    engine = None
    if engine_reader is not None:
        try:
            engine = engine_reader()
        except Exception:
            # The verdict must not fail because the engine could not be asked.
            engine = None
    resource = evaluate(device, processes, utilization, engine)
    resource["utilization"] = float(utilization)
    resource["required_percent"] = percentage(resource["required_mib"] or 0, total)
    return {"timestamp": datetime.fromtimestamp(clock(), timezone.utc).isoformat(timespec="seconds"),
            "available": True, "utilization": float(utilization),
            "device": device, "processes": processes, "resource": resource}
