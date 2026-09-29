"""Live check of the queue actions: progress, cancel, reload, delete, clear."""
import json
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://192.168.1.16:5009"


def maybe_json(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw[:200]


def call(path, data=None, ctype=None, method=None):
    request = urllib.request.Request(BASE + path, data=data, method=method or ("POST" if data is not None else "GET"))
    if ctype:
        request.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, maybe_json(response.read())
    except urllib.error.HTTPError as error:
        return error.code, maybe_json(error.read())


def upload(path, filename, data, content_type):
    boundary = "----qa" + uuid.uuid4().hex
    body = b""
    body += ("--%s\r\n" % boundary).encode()
    body += ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % filename).encode()
    body += ("Content-Type: %s\r\n\r\n" % content_type).encode()
    body += data + b"\r\n"
    body += ("--%s--\r\n" % boundary).encode()
    return call(path, body, "multipart/form-data; boundary=%s" % boundary)


def job(job_id):
    return call("/queue/" + job_id)[1]


def wait_until(job_id, statuses, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = job(job_id)
        if record["status"] in statuses:
            return record
        time.sleep(0.5)
    raise AssertionError(f"{job_id} never reached {statuses}")


print("=== 1. an upload job is replayable and reload queues it again ===")
image = open("test_image_real.jpg", "rb").read()
status, accepted = upload("/detect-objects", "test_image_real.jpg", image, "image/jpeg")
assert status == 202, (status, accepted)
original = accepted["job_id"]
finished = wait_until(original, {"completed", "failed"})
print("   original:", finished["status"], "http", finished["http_status"],
      "progress", finished["progress"], "replayable", finished["replayable"], "deletable", finished["deletable"])
assert finished["status"] == "completed", finished
assert finished["replayable"], finished
status, replayed = call("/queue/%s/reload" % original, b"", "application/json")
assert status == 202, (status, replayed)
print("   reload -> HTTP", status, "new job", replayed["job_id"][:8], "position", replayed["queue_position"])
replayed_done = wait_until(replayed["job_id"], {"completed", "failed"})
print("   replay finished:", replayed_done["status"], "http", replayed_done["http_status"], "in", replayed_done["duration_seconds"], "s")
assert replayed_done["status"] == finished["status"], (replayed_done, finished)

print()
print("=== 2. progress is published while a long job runs, and it can be cancelled ===")
status, long_job = call("/detect-objects", json.dumps({"id": 999901}).encode(), "application/json")
assert status == 202, (status, long_job)
held = None
deadline = time.time() + 30
samples = []
while time.time() < deadline:
    record = job(long_job["job_id"])
    if record["progress_detail"] and record["progress_detail"] not in samples:
        samples.append(record["progress_detail"])
    if record["status"] == "running" and len(samples) >= 3:
        break
    time.sleep(1.5)
print("   progress samples:", samples[:5])
assert samples, "no progress was published"

# A second job waits behind it, and cancelling it while queued is immediate.
status, queued_job = call("/queue/%s/reload" % original, b"", "application/json")
assert status == 202, (status, queued_job)
assert job(queued_job["job_id"])["status"] == "queued", job(queued_job["job_id"])
assert job(queued_job["job_id"])["cancellable"] is True
status, cancelled_queued = call("/queue/%s/cancel" % queued_job["job_id"], b"", "application/json")
print("   cancel a queued job -> HTTP", status, cancelled_queued["status"], "|", cancelled_queued["error"])
assert status == 200 and cancelled_queued["status"] == "cancelled", cancelled_queued

status, cancelled_running = call("/queue/%s/cancel" % long_job["job_id"], b"", "application/json")
print("   cancel a running job -> HTTP", status)
stopped = wait_until(long_job["job_id"], {"completed", "failed", "cancelled"}, timeout=90)
print("   running job ended as:", stopped["status"], "http", stopped["http_status"],
      "progress", stopped["progress"], "|", stopped["error"])
assert stopped["status"] == "cancelled", stopped
assert "Cancelled" in (stopped["error"] or ""), stopped

print()
print("=== 3. deleting respects the job state ===")
status, again = call("/queue/%s/cancel" % long_job["job_id"], b"", "application/json")
print("   cancel a cancelled job ->", status, again["error"])
assert status == 409, (status, again)

# Occupy the worker so the next job really is still waiting in the queue.
status, held_job = call("/detect-objects", json.dumps({"id": 999901}).encode(), "application/json")
assert status == 202, (status, held_job)
wait_until(held_job["job_id"], {"running"}, timeout=60)
status, pending = call("/queue/%s/reload" % replayed["job_id"], b"", "application/json")
assert status == 202, (status, pending)
assert job(pending["job_id"])["status"] == "queued", job(pending["job_id"])
status, refused = call("/queue/" + pending["job_id"], method="DELETE")
print("   delete a queued job ->", status, refused["error"])
assert status == 409, (status, refused)
status, cancelled = call("/queue/%s/cancel" % pending["job_id"], b"", "application/json")
assert status == 200 and cancelled["status"] == "cancelled", cancelled

status, deleted = call("/queue/" + original, method="DELETE")
print("   delete a finished job ->", status, deleted)
assert status == 200, (status, deleted)
status, gone = call("/queue/%s/reload" % original, b"", "application/json")
print("   reload after delete ->", status, gone["error"])
assert status == 404, (status, gone)

print()
print("=== 4. stats count cancelled jobs, and clearing keeps pending work ===")
before = call("/queue")[1]
print("   stats before clear:", json.dumps(before["queue"]))
assert before["queue"]["cancelled"] >= 2, before["queue"]
assert before["queue"]["pending"] >= 1, before["queue"]
status, cleared = call("/queue", method="DELETE")
print("   clear -> HTTP", status, "deleted", cleared["deleted"], "remaining:", json.dumps(cleared["queue"]))
assert status == 200
assert job(held_job["job_id"])["status"] in {"queued", "running"}, "pending work must survive a clear"
call("/queue/%s/cancel" % held_job["job_id"], b"", "application/json")
stopped = wait_until(held_job["job_id"], {"completed", "failed", "cancelled"}, timeout=90)
print("   blocker ended as:", stopped["status"], "| progress", stopped["progress"])
assert stopped["status"] == "cancelled", stopped

print()
print("=== 5. an image OpenCV cannot decode is re-encoded instead of failing ===")
# test_image.png is a valid PNG that PIL reads and OpenCV refuses (libpng "bad
# parameters to zlib"); the detector used to fail it with an opaque numpy error.
broken = open("test_image.png", "rb").read()
status, accepted = upload("/detect-objects", "test_image.png", broken, "image/png")
assert status == 202, (status, accepted)
recovered = wait_until(accepted["job_id"], {"completed", "failed"})
print("   non-OpenCV PNG ->", recovered["status"], "http", recovered["http_status"], "|", recovered["error"])
assert recovered["status"] == "completed", recovered

status, healthy = call("/health")
print("   health ->", status, healthy["status"])
assert status == 200

print()
print("=== 6. the queue is still healthy afterwards ===")
status, final = call("/queue")
print("   stats:", json.dumps(final["queue"]))
print("   rows still listed:", len(final["jobs"]))
assert status == 200
print()
print("ACTIONS OK")
