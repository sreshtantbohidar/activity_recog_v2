"""End-to-end exercise of the internal request queue against the live container."""
import json
import ssl
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://192.168.1.16:5009"
LONG_CLIP = "https://192.168.1.125:9000/media_analytics/input/video/__api_test_long.mp4"


def maybe_json(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw[:200]


def multipart(field, filename, data, content_type="application/octet-stream"):
    boundary = "----queuee2e" + uuid.uuid4().hex
    body = b""
    body += ("--%s\r\n" % boundary).encode()
    body += ('Content-Disposition: form-data; name="%s"; filename="%s"\r\n' % (field, filename)).encode()
    body += ("Content-Type: %s\r\n\r\n" % content_type).encode()
    body += data + b"\r\n"
    body += ("--%s--\r\n" % boundary).encode()
    return body, "multipart/form-data; boundary=%s" % boundary


def call(path, data=None, ctype=None):
    request = urllib.request.Request(BASE + path, data=data, method="POST" if data is not None else "GET")
    if ctype:
        request.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, maybe_json(response.read()), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, maybe_json(error.read()), dict(error.headers)


def queue():
    return call("/queue")[1]


def upload(path, filename, data, content_type):
    body, ctype = multipart("file", filename, data, content_type)
    return call(path, body, ctype)


def wait(job_id, timeout=900):
    deadline = time.time() + timeout
    seen = set()
    while time.time() < deadline:
        _, job, _ = call("/queue/" + job_id)
        if job["status"] not in seen:
            seen.add(job["status"])
            print("      %-9s after %5.1fs" % (job["status"], time.time() - started))
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(2)
    raise AssertionError("job %s never finished" % job_id)


def show(job, label):
    print("   %s: status=%s http=%s duration=%s queued=%s error=%s"
          % (label, job["status"], job["http_status"], job["duration_seconds"],
             job["queued_seconds"], job["error"]))
    return job


image = open("test_image.png", "rb").read()
clip = open("test_clip.mp4", "rb").read()

print("=== 1. an image upload is accepted as a job, not run inline ===")
status, accepted, headers = upload("/analyze-image", "test_image.png", image, "image/png")
assert status == 202, (status, accepted)
print("   HTTP", status, "queued position", accepted["queue_position"], "job", accepted["job_id"][:8])
assert headers.get("Location") == accepted["status_url"], headers
print("   Location header matches status_url")
print("   queue right after accept:", json.dumps(queue()["queue"]))
started = time.time()
show(wait(accepted["job_id"]), "image job")

print()
print("=== 2. later requests wait for the running job (serialisation) ===")
_, second, _ = upload("/analyze-image", "test_image.png", image, "image/png")
status, third, _ = call("/detect-objects", json.dumps({"id": 999902}).encode(), "application/json")
assert status == 202, (status, third)
snapshot = queue()
print("   snapshot:", json.dumps(snapshot["queue"]))
for job in snapshot["jobs"]:
    print("     #%-2s %-20s %-9s position=%s" % (job["seq"], job["endpoint"], job["status"], job["queue_position"]))
assert snapshot["queue"]["pending"] >= 1, snapshot
assert snapshot["queue"]["worker_alive"]
show(wait(second["job_id"]), "second image job")
over_long = show(wait(third["job_id"]), "over-long video job")
# Regression: the 413 must be recorded as a failure, not as a 200 "completion".
assert over_long["status"] == "failed", over_long
assert over_long["http_status"] == 413, over_long
assert "maximum allowed" in (over_long["error"] or ""), over_long
print("   over-long video correctly recorded as failed/413")

print()
print("=== 3. bad input is still rejected synchronously, not queued ===")
status, rejected, _ = upload("/analyze-image", "notes.txt", b"not media", "text/plain")
print("   unsupported extension -> HTTP", status, rejected)
assert status == 400 and "Unsupported image format" in rejected["error"], (status, rejected)
context = ssl._create_unverified_context()
long_clip = urllib.request.urlopen(LONG_CLIP, context=context, timeout=120).read()
print("   fetched %d bytes of the over-long clip" % len(long_clip))
status, rejected, _ = upload("/analyze-video", "__api_test_long.mp4", long_clip, "video/mp4")
print("   over-long upload -> HTTP", status, rejected)
assert status == 413 and "maximum allowed" in rejected["error"], (status, rejected)
print("   queue did not grow:", json.dumps(queue()["queue"]))

print()
print("=== 4. a detailed video runs in the queue and writes its PDF report ===")
status, accepted, _ = upload("/analyze-video-detailed", "test_clip.mp4", clip, "video/mp4")
assert status == 202, (status, accepted)
started = time.time()
detailed = show(wait(accepted["job_id"]), "detailed video job")
payload = detailed["response"]
print("   response keys:", sorted(payload.keys()))
report = (payload.get("analysis") or {}).get("report") or {}
print("   report:", json.dumps({k: report.get(k) for k in ("filename", "download_url", "skipped")}))
assert report.get("download_url"), report
status, _, _ = call(report["download_url"])
print("   report download -> HTTP", status)
assert status == 200

print()
print("=== 5. queue history, filters and detail ===")
listing = queue()
print("   stats:", json.dumps(listing["queue"]))
for job in listing["jobs"]:
    print("     #%-2s %-22s %-9s dur=%-8s http=%-4s %s"
          % (job["seq"], job["endpoint"], job["status"], job["duration_seconds"], job["http_status"], job["error"] or ""))
status, running, _ = call("/queue?status=running")
print("   ?status=running ->", status, len(running["jobs"]), "job(s)")
status, bad, _ = call("/queue?status=nope")
print("   ?status=nope ->", status, bad["error"])
status, one, _ = call("/queue/" + second["job_id"])
print("   detail: response present:", one["response"] is not None,
      "truncated:", one["response_truncated"], "bytes:", one["response_bytes"])
assert one["response"] is not None or one["response_preview"]
endpoints = {job["endpoint"] for job in listing["jobs"]}
print("   endpoints recorded:", sorted(endpoints))
assert {"/analyze-image", "/analyze-video-detailed", "/detect-objects"} <= endpoints, endpoints

print()
print("E2E OK")
