"""Isolate the vision crash: send one frame at original vs downscaled size.

Usage: python /tmp/img_probe.py <video> <resize_long_side or 0> [prompt_mode]
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
resize = int(sys.argv[2]) if len(sys.argv) > 2 else 0
frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
prompt = m.ACTIVITY_PROMPT
model = os.environ.get("VLLM_VISION_MODEL", "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit")

vr = VideoReader(path, ctx=cpu(0))
img = Image.fromarray(vr[frame_idx].asnumpy())
if resize:
    img = img.resize((resize, int(round(img.height * resize / img.width))), Image.LANCZOS)
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=88)
print("video=%s size=%dx%d bytes=%d" % (path, img.width, img.height, buf.tell()))

b64 = base64.b64encode(buf.getvalue()).decode("ascii")
payload = {
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        {"type": "text", "text": prompt},
    ]}],
    "temperature": 0.1, "max_tokens": 256, "stream": False,
}
req = Request("http://127.0.0.1:8001/v1/chat/completions",
              data=json.dumps(payload).encode("utf-8"),
              headers={"Content-Type": "application/json"}, method="POST")
try:
    with urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    parsed = m.parse_json_text(text)
    print("RESULT: finish=%s json=%s acts=%s"
          % (choice.get("finish_reason"), "OK" if "error" not in parsed else parsed.get("error"),
             parsed.get("detected_activity")))
except HTTPError as exc:
    print("RESULT: HTTP %s %s" % (exc.code, exc.read().decode("utf-8", "replace")[:300]))
