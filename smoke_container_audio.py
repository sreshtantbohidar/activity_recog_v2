import os
import sys

sys.path.insert(0, "/app")
os.chdir("/app")
import vision_backend as vb
from audio_analysis import AUDIO_STAGE_LOCK, analyze_video_audio, release_qwen_audio
from response_schema import (
    AUDIO_ANALYSIS_SCHEMA,
    DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA,
    VIDEO_ANALYSIS_ENVELOPE_SCHEMA,
)

HOST = os.environ.get("VLLM_VISION_HOST", "http://127.0.0.1:8001")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run_audio(video):
    with AUDIO_STAGE_LOCK:
        vb.sleep_vllm(host=HOST)
        try:
            return analyze_video_audio(video, device="cuda")
        finally:
            release_qwen_audio()
            vb.wake_vllm(host=HOST)


check(set(VIDEO_ANALYSIS_ENVELOPE_SCHEMA["properties"]) == {"filename", "media_type", "analysis"},
      "Video-analysis endpoint envelope changed")
check(set(DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA["properties"]) == {"filename", "media_type", "analysis"},
      "Detailed-video endpoint envelope changed")
check({"has_audio", "transcript", "sound_events", "warnings"} <= set(AUDIO_ANALYSIS_SCHEMA["properties"]),
      "Audio-analysis public fields changed")

for filename, expects_speech in (("operation.mp4", True), ("training.mp4", True),
                                 ("joint_exercise.mp4", False), ("patrolling.mp4", False)):
    result = run_audio("/testvideos/" + filename)
    transcript = result["transcript"]
    sounds = result["sound_events"]
    print(filename, "spoken:", [item["text"] for item in transcript], flush=True)
    print(filename, "background:", [item["label"] for item in sounds], flush=True)
    check(isinstance(transcript, list) and isinstance(sounds, list), filename + " changed audio field types")
    check(all(set(item) == {"start_sec", "end_sec", "text"} for item in transcript),
          filename + " changed transcript segment fields")
    check(all({"start_sec", "end_sec", "label", "confidence"} <= set(item) for item in sounds),
          filename + " changed sound-event fields")
    check(bool(transcript) is expects_speech, filename + " speech expectation mismatch")
    if filename == "operation.mp4":
        combined = " ".join(item["text"].lower() for item in transcript)
        check("command to bravo" in combined and "perimeter" in combined,
              "operation transcript did not recover the spoken commands")
        check("radio chatter" not in combined and "heavy vehicle engine" not in combined,
              "sound-description example was copied into the transcript")
        check(bool(sounds) and any("radio" in item["label"].lower() for item in sounds),
              "operation background radio description was not retained")
    if filename == "training.mp4":
        check("hold left flank" in " ".join(item["text"].lower() for item in transcript),
              "training speech transcript was not recovered")

print("Both unchanged API envelopes and all supplied sample audio checks passed", flush=True)
