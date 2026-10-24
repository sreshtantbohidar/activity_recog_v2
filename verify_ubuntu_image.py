"""Fast, network-free verification for the Ubuntu offline image."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")

checks: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


for module in ("torch", "transformers", "vllm", "faster_whisper", "flask", "ultralytics"):
    try:
        loaded = importlib.import_module(module)
        record(f"import {module}", True, getattr(loaded, "__version__", "ok"))
    except Exception as exc:
        record(f"import {module}", False, str(exc))

required = {
    "application": Path("/app/app.py"),
    "entrypoint": Path("/app/entrypoint.sh"),
    "vision config": Path("/app/hf_models/Qwen3-VL-8B-Instruct-AWQ-4bit/config.json"),
    "audio config": Path("/app/hf_models/Qwen2.5-Omni-3B/config.json"),
    "whisper weights": Path("/app/whisper_models/faster-whisper-large-v3/model.bin"),
    "YOLO weights": Path("/app/object_detect_model_256_class.pt"),
}
for name, path in required.items():
    size = path.stat().st_size if path.is_file() else 0
    record(name, size > 0, f"{path} ({size} bytes)")

try:
    from transformers import AutoConfig

    for model in (
        "/app/hf_models/Qwen3-VL-8B-Instruct-AWQ-4bit",
        "/app/hf_models/Qwen2.5-Omni-3B",
    ):
        config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        record(f"offline config {Path(model).name}", True, config.model_type)
except Exception as exc:
    record("offline Transformers resolution", False, str(exc))

failed = [name for name, ok, _ in checks if not ok]
print(f"RESULT: {len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)

