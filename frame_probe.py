"""Diagnostic: send sampled frames to the vLLM engine and show raw output.

Usage:  python /tmp/frame_probe.py /tmp/deployment.mp4 [max_tokens] [prompt_mode]
prompt_mode: new (default, verbose definitions) | old (list only) | min
"""
import base64
import io
import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, "/app")

from decord import VideoReader, cpu  # noqa: E402
from PIL import Image  # noqa: E402

import media_utils as m  # noqa: E402

path = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("VLLM_MAX_NEW_TOKENS", "256"))
mode = sys.argv[3] if len(sys.argv) > 3 else "new"

PROMPTS = {
    "new": m.ACTIVITY_PROMPT,
    "old": ("Analyze only directly visible evidence. Classify using exact labels: deployment, "
            "mobilization, movement, disposition, training, infra, patrolling, visit, internal security, "
            "miscellaneous, elint, operation, procurement, technology, political diplomacy, joint exercise, foreign "
            "relation. Do not infer intent, identity, nationality, or unseen events. Return JSON only with "
            "scene_summary, key_objects, location_type, detected_activity, confidence, reasoning."),
    "min": "Return JSON only with scene_summary, key_objects, location_type, detected_activity, confidence, reasoning.",
}
prompt = PROMPTS[mode]
model = os.environ.get("VLLM_VISION_MODEL", "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit")
print("prompt_mode=%s chars=%d max_tokens=%d" % (mode, len(prompt), max_tokens))

vr = VideoReader(path, ctx=cpu(0))
n = len(vr)
for i in sorted({0, n // 2, n - 1}):
    img = Image.fromarray(vr[i].asnumpy())
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = Request("http://127.0.0.1:8001/v1/chat/completions",
                  data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=300) as resp:
            data = json.load(resp)
    except HTTPError as exc:
        print("=== frame %d  HTTP %s BODY: %s" % (i, exc.code, exc.read().decode("utf-8", "replace")[:500]))
        continue
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    parsed = m.parse_json_text(text)
    print("=== frame %d  finish_reason=%s  chars=%d  json=%s"
          % (i, choice.get("finish_reason"), len(text), "OK" if "error" not in parsed else parsed.get("error")))
    if "error" in parsed:
        print("TAIL:", repr(text[-300:]))
    else:
        print("ACTS:", parsed.get("detected_activity"), "CONF:", parsed.get("confidence"))
