#!/usr/bin/env python3
"""Run one audio-analysis stage in an isolated CUDA process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from audio_analysis import analyze_video_audio


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: audio_worker.py VIDEO OUTPUT_JSON", file=sys.stderr)
        return 2
    result = analyze_video_audio(sys.argv[1], device="cuda")
    Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
