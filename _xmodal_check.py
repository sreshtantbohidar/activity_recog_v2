"""Verify the bounded cross-modal summary prompt against the live vLLM engine.

Reproduces the long-video condition (many frames + verbose sound labels) that
previously overflowed the 8192-token context and produced
"Cross-modal summary unavailable".
"""
import sys

sys.path.insert(0, "/app")

import app as appmod
from media_utils import build_multimodal_summary_prompt

FRAMES = [
    {
        "timestamp_sec": float(i),
        "detected_activity": ["operation"],
        "scene_summary": "A military vehicle with a mounted machine gun in a desert "
        "landscape with dust rising in the distance, frame %d" % i,
        "reasoning": "The presence of a mounted weapon on a military vehicle and a "
        "visible dust cloud suggest active engagement or combat operation in a "
        "rural desert setting. " * 2,
    }
    for i in range(296)
]
SPEECH = [
    {"start_sec": i * 30, "end_sec": i * 30 + 30,
     "text": "I can't see anything. Oh, keep doing what you're doing."}
    for i in range(10)
]
SOUNDS = [
    {
        "start_sec": i * 30, "end_sec": i * 30 + 30, "confidence": 1.0,
        "label": "I can hear gunfire, artillery fire at intervals (0.38 to 5.72, "
        "6.49 to 8.77, and 9.71 to 10.00), general noise throughout the duration "
        "(0.00 to 10.00), automatic gunfire, and a man speaking (6.07 to 6.53).",
    }
    for i in range(10)
]


def old_prompt_length():
    """Reconstruct the pre-fix (unbounded) prompt size for comparison."""
    visual = [
        "[{:.1f}s] Activities: {} | Scene: {} | Evidence: {}".format(
            f["timestamp_sec"], ", ".join(f["detected_activity"]),
            f["scene_summary"], f["reasoning"])
        for f in FRAMES[:300]
    ]
    speech = [
        "[{:.1f}-{:.1f}s] {}".format(s["start_sec"], s["end_sec"], s["text"])
        for s in SPEECH[:500]
    ]
    sounds = [
        "[{:.1f}-{:.1f}s] {} (confidence {:.2f})".format(
            s["start_sec"], s["end_sec"], s["label"], s["confidence"])
        for s in SOUNDS[:500]
    ]
    body = "\n".join(visual) + "\n".join(speech) + "\n".join(sounds)
    return 900 + len(body)


old_chars = old_prompt_length()
print("OLD prompt chars ~", old_chars, "~tokens", old_chars // 4)

prompt = build_multimodal_summary_prompt(FRAMES, SPEECH, SOUNDS)
print("NEW prompt chars", len(prompt), "~tokens", len(prompt) // 4)

appmod.engine_power.ensure_awake()
out = appmod._generate_cross_modal_summary(prompt)
print("RESULT_OK", bool(out and out.strip()))
print("RESULT_HEAD", (out or "")[:200].replace("\n", " "))
