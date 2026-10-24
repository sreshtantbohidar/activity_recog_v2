"""Verify the offline image actually contains every model and library.

Run inside the built image (no GPU needed):
    docker run --rm --entrypoint python activity-recognition:2.0.0 /app/deploy/verify_offline_image.py

Checks, in order:
  1. the heavy libraries import (torch, transformers, vllm, faster_whisper)
  2. the Qwen2.5-Omni classes exist in the installed transformers
  3. the faster-whisper large-v3 weights are baked at /opt/whisper
  4. every HuggingFace checkpoint needed at runtime exists in the local cache
  5. HF_HUB_OFFLINE mode resolves every checkpoint with no network access
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

failures = []
checks = []


def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    if not ok:
        failures.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


# 1. Libraries ----------------------------------------------------------------
try:
    import torch
    check("torch", True, torch.__version__)
except Exception as exc:
    check("torch", False, str(exc))
    torch = None

try:
    import transformers
    check("transformers", True, transformers.__version__)
except Exception as exc:
    check("transformers", False, str(exc))
    transformers = None

try:
    import vllm
    check("vllm", True, vllm.__version__)
except Exception as exc:
    check("vllm", False, str(exc))

try:
    import faster_whisper
    check("faster_whisper", True, faster_whisper.__version__)
except Exception as exc:
    check("faster_whisper", False, str(exc))

# 2. Omni support --------------------------------------------------------------
if transformers is not None:
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # noqa: F401
        check("qwen2.5-omni classes", True)
    except Exception as exc:
        check("qwen2.5-omni classes", False, str(exc))

# 3. Whisper weights -----------------------------------------------------------
whisper_root = Path("/opt/whisper")
model_dirs = list(whisper_root.glob("models--Systran--faster-whisper-large-v3/snapshots/*")) if whisper_root.exists() else []
if model_dirs:
    files = sorted(p.name for p in model_dirs[0].iterdir())
    needed = {"model.bin", "config.json", "tokenizer.json", "vocabulary"}
    missing = {n for n in needed if not any(n in f for f in files)}
    check("whisper large-v3 weights baked", not missing,
          f"{len(files)} files; missing={missing or 'none'}")
else:
    check("whisper large-v3 weights baked", False, f"{whisper_root} not found")

# 4. HF checkpoints ------------------------------------------------------------
import huggingface_hub.constants as hf_const

cache = Path(hf_const.HF_HUB_CACHE)
print(f"  (cache: {cache})")
REQUIRED = {
    "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit": ["config.json", "*.safetensors"],
    "Qwen/Qwen2.5-Omni-3B": ["config.json", "*.safetensors"],
}
for repo, patterns in REQUIRED.items():
    repo_dir = cache / f"models--{repo.replace('/', '--')}"
    snaps = list(repo_dir.glob("snapshots/*")) if repo_dir.exists() else []
    if not snaps:
        check(f"checkpoint {repo}", False, "not in cache")
        continue
    snapshot = snaps[0]
    files = [p.name for p in snapshot.iterdir()]
    missing = []
    for pattern in patterns:
        if "*" in pattern:
            if not list(snapshot.glob(pattern)):
                missing.append(pattern)
        elif pattern not in files:
            missing.append(pattern)
    size_gb = sum(f.stat().st_size for f in Path(repo_dir).rglob("*") if f.is_file()) / 1e9
    check(f"checkpoint {repo}", not missing, f"{size_gb:.1f} GB; missing={missing or 'none'}")

# 5. Offline resolution ----------------------------------------------------------
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
try:
    from huggingface_hub import snapshot_download
    for repo in REQUIRED:
        path = snapshot_download(repo, local_files_only=True)
        check(f"offline resolve {repo}", bool(path), str(path))
except Exception as exc:
    check("offline resolve", False, str(exc))

print()
if failures:
    print(f"RESULT: {len(checks) - len(failures)}/{len(checks)} passed -- FAILURES: {failures}")
    sys.exit(1)
print(f"RESULT: all {len(checks)} checks passed -- image is fully offline-capable")
