"""Drive the live API and save one real response per tab.

Each heavy endpoint answers 202 with a job id; we poll /queue/<id> and record
progress samples so the dashboard progress bar and the analytics renderer can be
checked against genuine payloads rather than hand-written stubs.
"""
import json
import sys
import time

import requests

BASE = "http://192.168.1.16:5009"
POLL = 3.0
DEADLINE = 1800.0

CASES = [
    ("image", "/analyze-image", "test_image_real.jpg", "payload_image.json"),
    ("video", "/analyze-video", "test_clip.mp4", "payload_video.json"),
    ("detection", "/detect-objects", "test_image_real.jpg", "payload_detection.json"),
    ("detailed", "/analyze-video-detailed", "test_clip.mp4", "payload_detailed.json"),
]


def submit(endpoint, path):
    with open(path, "rb") as handle:
        files = {"file": (path, handle)}
        response = requests.post(BASE + endpoint, files=files, timeout=120)
    return response


def wait(job_id, label):
    started = time.time()
    seen = []
    last = None
    while time.time() - started < DEADLINE:
        job = requests.get(f"{BASE}/queue/{job_id}", timeout=30).json()
        marker = job.get("progress")
        detail = job.get("progress_detail")
        if marker != last:
            seen.append((job.get("status"), marker, detail))
            last = marker
        if job.get("status") in ("completed", "failed", "cancelled"):
            job["_progress_samples"] = seen
            return job
        time.sleep(POLL)
    raise SystemExit(f"{label}: timed out after {DEADLINE:.0f}s")


def main():
    wanted = sys.argv[1:] or [name for name, *_ in CASES]
    summary = []
    for name, endpoint, media, outfile in CASES:
        if name not in wanted:
            continue
        print(f"=== {name} -> {endpoint} ({media})", flush=True)
        response = submit(endpoint, media)
        print(f"    submit HTTP {response.status_code} {response.text[:160]}", flush=True)
        if response.status_code != 202:
            summary.append((name, response.status_code, "rejected", 0))
            continue
        body = response.json()
        job = wait(body["job_id"], name)
        payload = job.get("response") if job.get("response") is not None else job.get("response_preview")
        with open(outfile, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        samples = job.get("_progress_samples") or []
        print(f"    final status={job['status']} http={job['http_status']} "
              f"progress={job.get('progress')} duration={job.get('duration_seconds')}", flush=True)
        if job.get("error"):
            print(f"    error={job['error']}", flush=True)
        print(f"    progress samples ({len(samples)}):", flush=True)
        for status, percent, detail in samples:
            print(f"      {status:<9} {percent:>5}  {detail}", flush=True)
        print(f"    saved {outfile} ({len(json.dumps(payload))} chars)", flush=True)
        summary.append((name, job["status"], job.get("http_status"), len(samples)))
    print("\n=== summary")
    for name, status, http, samples in summary:
        print(f"  {name:<10} {status:<10} http={http} progress_updates={samples}")


if __name__ == "__main__":
    main()
