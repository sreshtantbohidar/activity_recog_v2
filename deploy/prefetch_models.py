# Resumable model prefetch on the build host (Windows python).
#
# Downloads the three runtime checkpoints into <project>/hf_models with the
# standard HuggingFace cache layout so `COPY hf_models /models/huggingface/hub`
# bakes them into the image without any network access at build time.
#
#   python deploy\prefetch_models.py
#
# Re-running it resumes incomplete downloads; it skips complete ones.
from __future__ import annotations

import os
import sys
from pathlib import Path

# hf_transfer's rust downloader survives flaky links (chunked, multi-connection,
# automatic retries) where the default single-stream download stalls.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "hf_models"
REPOS = (
    "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
    "Qwen/Qwen2.5-Omni-3B",
)
ATTEMPTS = 5


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    failures = []
    for repo in REPOS:
        print(f"== {repo} -> {CACHE}", flush=True)
        done = False
        for attempt in range(1, ATTEMPTS + 1):
            try:
                path = snapshot_download(repo_id=repo, cache_dir=str(CACHE))
                print(f"   done: {path}", flush=True)
                done = True
                break
            except Exception as exc:  # resume from .incomplete on the next try
                print(f"   attempt {attempt}/{ATTEMPTS} failed ({exc}); resuming", flush=True)
        if not done:
            failures.append(repo)
    if failures:
        print(f"INCOMPLETE: {failures}")
        return 1
    print("PREFETCH-OK")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
