"""Temporary validation probe: production audio stage over test clips.

Parks the vLLM engine, runs analyze_video_audio (Whisper transcript + Qwen2.5-Omni
sound description) and prints the parsed result, then restores the engine.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from audio_analysis import AUDIO_STAGE_LOCK, analyze_video_audio, release_qwen_audio
from engine_power import EnginePowerManager
from vision_backend import sleep_vllm, wake_vllm

TEST_DIR = Path("/testvideos")
engine_power = EnginePowerManager(sleep_vllm, wake_vllm, release_qwen_audio,
    host=os.environ.get("VLLM_VISION_HOST"), idle_seconds=0)

names = sys.argv[1:] or ["operation.mp4", "training.mp4", "deployment.mp4"]
videos = [TEST_DIR / n for n in names if (TEST_DIR / n).exists()]
print(f"probing {len(videos)} clip(s): {[v.name for v in videos]}", flush=True)

parked = engine_power.park()
print(f"vLLM parked: {parked}", flush=True)
try:
    with AUDIO_STAGE_LOCK:
        for path in videos:
            started = time.time()
            result = analyze_video_audio(str(path), device="cuda")
            print("=" * 70, flush=True)
            print(f"{path.name} ({time.time() - started:.1f}s)", flush=True)
            print("transcript:", json.dumps(result.get("transcript", []), ensure_ascii=False), flush=True)
            print("sounds:", [e.get("label") for e in result.get("sound_events", [])], flush=True)
            if result.get("warnings"):
                print("warnings:", result["warnings"], flush=True)
finally:
    release_qwen_audio()
    try:
        engine_power.restore()
    except Exception as exc:  # noqa: BLE001
        print("restore failed:", exc, flush=True)
print("PROBE-DONE", flush=True)
