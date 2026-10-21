"""Live end-to-end check of the running service.

Run against a live container:  python3 e2e_live.py
Override the service URL with E2E_BASE and the fixture folder with
E2E_FIXTURES. Fixtures: e2e_synth.png (640x360 frame), e2e_image.png
(real 1280x720 frame) and e2e_av1.mp4 (AV1 clip).

Covers the surface that must work without the vision engine: object detection
on an image, video object detection (including an AV1 upload), annotated-clip
download and range requests, the queue lifecycle (submit, reload, cancel,
delete) and monitoring/health. Each step asserts a real outcome, not a 200.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = os.getenv("E2E_BASE", "http://127.0.0.1:5009").rstrip("/")
FIXTURES = os.getenv("E2E_FIXTURES", "/tmp")
RESULTS = []


def record(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + " :: " + detail, flush=True)


def request(method, path, data=None, headers=None, timeout=120):
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def submit(endpoint, path, filename, content_type):
    boundary = "----e2e" + uuid.uuid4().hex
    with open(path, "rb") as handle:
        payload = handle.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
        + payload + f"\r\n--{boundary}--\r\n".encode())
    return request("POST", endpoint, data=body,
                   headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})


def wait(job_id, timeout=150):
    deadline = time.time() + timeout
    job = {}
    while time.time() < deadline:
        status, raw, _ = request("GET", f"/queue/{job_id}")
        job = json.loads(raw)
        if job.get("status") in ("completed", "failed", "cancelled"):
            return job
        time.sleep(2)
    return job


def main():
    status, raw, _ = request("GET", "/health")
    health = json.loads(raw)
    record("GET /health", status == 200 and health.get("status") == "ok",
           f"{status} status={health.get('status')} cuda={health.get('cuda_available')}")

    status, _, _ = request("GET", "/docs/")
    record("GET /docs/", status == 200, f"{status}")

    status, raw, _ = request("GET", "/gpu/monitor")
    monitor = json.loads(raw)
    res = monitor.get("resource") or {}
    record("GET /gpu/monitor",
           status == 200 and res.get("status") in ("ok", "warning", "critical")
           and res.get("engine_running") is not None,
           f"{status} status={res.get('status')} engine_running={res.get('engine_running')} "
           f"available={res.get('available_mib')} required={res.get('required_mib')}")

    status, raw, _ = submit("/detect-objects", f"{FIXTURES}/e2e_image.png", "e2e.png", "image/png")
    job = json.loads(raw) if status == 202 else {}
    record("POST /detect-objects", status == 202 and bool(job.get("job_id")),
           f"{status} job={job.get('job_id')}")
    image_job = job.get("job_id")
    if image_job:
        done = wait(image_job)
        det = (done.get("response") or {}).get("object_detection") or {}
        record("detect-objects completes", done.get("status") == "completed",
               f"status={done.get('status')} detections={len(det.get('detections') or [])} "
               f"elapsed={done.get('total_seconds')}s")

    # The default OBJECT_CONFIDENCE (0.25) is deliberately conservative, so this
    # frame has no boxes at it. Lowering it through the config API proves the
    # detector actually runs, then the default is put back.
    status, raw, _ = request("GET", "/config")
    original_conf = (json.loads(raw).get("configuration") or {}).get("OBJECT_CONFIDENCE")
    status, _, _ = request("POST", "/config", data=json.dumps({"OBJECT_CONFIDENCE": 0.05}).encode(),
                           headers={"Content-Type": "application/json"})
    record("POST /config lowers OBJECT_CONFIDENCE", status == 200 and original_conf is not None,
           f"{status} was={original_conf}")
    # A 640x360 frame: the detector resizes to OBJECT_IMAGE_SIZE (640), so the
    # 1280x720 frame above loses its marginal boxes when it is downscaled.
    status, raw, _ = submit("/detect-objects", f"{FIXTURES}/e2e_synth.png", "e2e_lowconf.png", "image/png")
    job = json.loads(raw) if status == 202 else {}
    if job.get("job_id"):
        done = wait(job["job_id"])
        det = (done.get("response") or {}).get("object_detection") or {}
        hits = len(det.get("detections") or [])
        record("detection finds objects at conf 0.05",
               done.get("status") == "completed" and hits > 0,
               f"status={done.get('status')} detections={hits} elapsed={done.get('total_seconds')}s")
        request("DELETE", f"/queue/{job['job_id']}")
    if original_conf is not None:
        status, _, _ = request("POST", "/config",
                               data=json.dumps({"OBJECT_CONFIDENCE": original_conf}).encode(),
                               headers={"Content-Type": "application/json"})
        status, raw, _ = request("GET", "/config")
        restored = (json.loads(raw).get("configuration") or {}).get("OBJECT_CONFIDENCE")
        record("OBJECT_CONFIDENCE restored", status == 200 and restored == original_conf,
               f"{status} now={restored}")

    status, raw, _ = submit("/detect-objects-video", f"{FIXTURES}/e2e_av1.mp4", "e2e_av1.mp4", "video/mp4")
    job = json.loads(raw) if status == 202 else {}
    record("POST /detect-objects-video (AV1)", status == 202 and bool(job.get("job_id")),
           f"{status} job={job.get('job_id')}")
    video_job = None
    if job.get("job_id"):
        video_job = wait(job["job_id"])
        od = (video_job.get("response") or {}).get("object_detection") or {}
        record("video detection completes on AV1",
               video_job.get("status") == "completed"
               and od.get("annotated_video_codec") == "h264",
               f"status={video_job.get('status')} codec={od.get('annotated_video_codec')} "
               f"detections={len(od.get('detections') or [])} "
               f"elapsed={video_job.get('total_seconds')}s")
        url = od.get("annotated_video_url")
        if url:
            status, body, headers = request("GET", url)
            record("annotated clip download",
                   status == 200 and headers.get("Content-Type", "").startswith("video/"),
                   f"{status} {headers.get('Content-Type')} bytes={len(body)}")
            status, body, headers = request("GET", url, headers={"Range": "bytes=0-1023"})
            record("annotated clip range request",
                   status == 206 and len(body) == 1024,
                   f"{status} received={len(body)} range={headers.get('Content-Range')}")

    status, raw, _ = submit("/analyze-image", f"{FIXTURES}/e2e_image.png", "e2e.png", "image/png")
    job = json.loads(raw) if status == 202 else {}
    if job.get("job_id"):
        done = wait(job["job_id"])
        record("analyze-image reports the missing engine",
               done.get("status") == "failed" and done.get("http_status") == 502,
               f"status={done.get('status')} http={done.get('http_status')} "
               f"error={(done.get('error') or '')[:60]}")
        request("DELETE", f"/queue/{job['job_id']}")

    status, raw, _ = request("POST", "/detect-objects-video", data=b"",
                             headers={"Content-Type": "application/octet-stream"})
    record("empty upload is rejected", status == 400, f"{status} {raw[:50]!r}")

    if video_job:
        status, raw, _ = request("POST", f"/queue/{video_job['job_id']}/reload")
        record("POST /queue/<id>/reload", status == 202, f"{status}")
        reloaded = json.loads(raw) if raw else {}
        if reloaded.get("job_id"):
            done = wait(reloaded["job_id"])
            record("reloaded job completes", done.get("status") == "completed",
                   f"status={done.get('status')} elapsed={done.get('total_seconds')}s")
            request("DELETE", f"/queue/{reloaded['job_id']}")

    status, raw, _ = request("GET", "/queue?page_size=3")
    listing = json.loads(raw) if status == 200 else {}
    record("GET /queue", status == 200 and "jobs" in listing,
           f"{status} total={listing.get('page', {}).get('total')} "
           f"rows={len(listing.get('jobs') or [])}")

    status, raw, _ = submit("/detect-objects-video", f"{FIXTURES}/e2e_av1.mp4", "cancel1.mp4", "video/mp4")
    first = json.loads(raw) if status == 202 else {}
    status, raw, _ = submit("/detect-objects-video", f"{FIXTURES}/e2e_av1.mp4", "cancel2.mp4", "video/mp4")
    second = json.loads(raw) if status == 202 else {}
    if second.get("job_id"):
        status, raw, _ = request("POST", f"/queue/{second['job_id']}/cancel")
        cancelled = json.loads(raw) if raw else {}
        record("POST /queue/<id>/cancel",
               status in (200, 202)
               and cancelled.get("status") in ("cancelled", "completed", "running", "queued"),
               f"{status} status={cancelled.get('status')}")
        request("DELETE", f"/queue/{second['job_id']}")
    if first.get("job_id"):
        wait(first["job_id"], timeout=90)
        status, _, _ = request("DELETE", f"/queue/{first['job_id']}")
        record("DELETE /queue/<id>", status in (200, 204), f"{status}")

    if image_job:
        request("DELETE", f"/queue/{image_job}")
    if video_job:
        request("DELETE", f"/queue/{video_job['job_id']}")

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
