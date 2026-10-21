"""Wait until the card can hold the vLLM engine before starting it.

vLLM refuses to start when the free memory is below
``gpu_memory_utilization`` of the device total, so an engine started while
another process holds the card does not fail on a bug: it fails on arithmetic,
after loading 7.4 GiB of weights. On a shared host that is expensive and noisy,
and the entrypoint supervisor used to retry it every five seconds forever.

This gate is what the supervisor asks first. It blocks until the required
memory is back and then lets exactly one engine start, so a restart always
happens when it can succeed. When the card cannot be measured at all (no
``nvidia-smi``), the gate opens and vLLM's own check stays the authority.

The wait is dependency-light by design (the probes and the clock are injected)
so its behaviour is unit-testable without a GPU, and ``python3 gpu_vram_gate.py``
is the same code the container entrypoint runs.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from typing import Callable, Optional

logger = logging.getLogger("vram_gate")

# How often the free memory is re-read while waiting, and how often the wait is
# reported. Long enough to be cheap, short enough to react to a freeing card.
POLL_SECONDS = 5.0
REPORT_SECONDS = 60.0


def query_mib(field: str, runner: Callable[..., object] = subprocess.run) -> Optional[int]:
    """One integer MiB reading from ``nvidia-smi``, or None when unavailable.

    The container only sees the GPU its reservation attaches, so the first line
    is the device the engine will use.
    """
    try:
        completed = runner(
            ["nvidia-smi", f"--query-gpu=memory.{field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False)
        if getattr(completed, "returncode", 1) != 0:
            return None
        first = (getattr(completed, "stdout", "") or "").strip().splitlines()
        return int(first[0].strip()) if first else None
    except Exception:  # no driver, no nvidia-smi, unparsable output
        logger.debug("Could not read memory.%s from nvidia-smi", field, exc_info=True)
        return None


def required_mib(utilization: float, total_mib: int) -> int:
    """The free memory the engine needs before it will start.

    The same ratio vLLM checks, computed from the same total it will read.
    """
    return int(total_mib * float(utilization))


class VramGate:
    """Blocks until the engine's memory budget is available."""

    def __init__(self, utilization: float,
                 total_fn: Callable[[], Optional[int]],
                 free_fn: Callable[[], Optional[int]],
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 interval: float = POLL_SECONDS,
                 max_wait: float = 0.0,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.utilization = float(utilization)
        self.total_fn = total_fn
        self.free_fn = free_fn
        self.sleep = sleep
        self.clock = clock
        self.interval = max(0.1, float(interval))
        self.max_wait = float(max_wait or 0)
        self.should_stop = should_stop or (lambda: False)

    def wait(self) -> bool:
        """True once the engine may start; False when the wait was abandoned.

        Unmeasurable memory opens the gate: refusing to start an engine because
        ``nvidia-smi`` is missing would turn a diagnostic gap into an outage.
        """
        try:
            total = self.total_fn()
        except Exception:
            logger.warning("Could not read the GPU memory total; starting the engine without waiting",
                           exc_info=True)
            return True
        needed = required_mib(self.utilization, total) if total else None
        if not needed:
            logger.warning("GPU memory total is unavailable; starting the engine without waiting")
            return True

        started = self.clock()
        reported = started
        while True:
            if self.should_stop():
                logger.warning("Shutdown requested while waiting for VRAM; not starting the engine")
                return False
            try:
                free = self.free_fn()
            except Exception:  # a failing probe must never block the engine for good
                logger.warning("Could not read the free GPU memory; starting the engine without waiting",
                               exc_info=True)
                return True
            if free is None:
                logger.warning("Free GPU memory is unavailable; starting the engine without waiting")
                return True
            if free >= needed:
                waited = self.clock() - started
                if waited >= self.interval:
                    logger.info("GPU memory is free again (%s MiB of %s MiB needed); starting the engine",
                                free, needed)
                return True
            now = self.clock()
            if now - reported >= REPORT_SECONDS or now == started:
                logger.warning("Waiting for GPU memory: %s MiB free, %s MiB needed for "
                               "gpu_memory_utilization=%s - another process is holding the card",
                               free, needed, self.utilization)
                reported = now
            if self.max_wait and now - started >= self.max_wait:
                logger.error("Gave up waiting for GPU memory after %.0fs (%s MiB free, %s MiB needed)",
                             now - started, free, needed)
                return False
            self.sleep(self.interval)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI used by the entrypoint: exit 0 when an engine start can succeed."""
    parser = argparse.ArgumentParser(description="Wait for the vLLM engine's GPU memory budget")
    parser.add_argument("--util", type=float, default=0.80, help="gpu_memory_utilization")
    parser.add_argument("--interval", type=float, default=POLL_SECONDS)
    parser.add_argument("--max-wait", type=float, default=0.0, help="seconds to wait; 0 waits forever")
    arguments = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[entrypoint] %(message)s")
    gate = VramGate(arguments.util,
                    total_fn=lambda: query_mib("total"),
                    free_fn=lambda: query_mib("free"),
                    interval=arguments.interval,
                    max_wait=arguments.max_wait)
    return 0 if gate.wait() else 1


if __name__ == "__main__":
    raise SystemExit(main())
