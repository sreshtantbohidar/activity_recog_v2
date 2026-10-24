#!/usr/bin/env python3
"""Run labeled videos through /analyze-video and report classification accuracy."""

import argparse
import csv
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def submit(base_url, video):
    output = subprocess.check_output(
        ["curl", "-fsS", "-F", f"file=@{video}", f"{base_url}/analyze-video"], text=True
    )
    return json.loads(output)["job_id"]


def expected_label(path):
    return {"political dipomacy": "political diplomacy"}.get(
        path.stem.lower().replace("_", " "), path.stem.lower().replace("_", " ")
    )


def result_for(base_url, video):
    job_id = submit(base_url, video)
    while True:
        job = get_json(f"{base_url}/queue/{job_id}")
        if job.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(5)
    response = job.get("response") or job.get("result") or {}
    analysis = response.get("analysis") or {}
    overall = analysis.get("overall_analysis") or {}
    predicted = overall.get("dominant_activity")
    summary = analysis.get("activity_summary") or {}
    return {
        "filename": video.name,
        "expected": expected_label(video),
        "predicted": predicted,
        "correct": predicted == expected_label(video),
        "status": job.get("status"),
        "job_id": job_id,
        "execution_seconds": job.get("execution_seconds"),
        "basis": overall.get("dominant_activity_basis"),
        "detected_activities": summary.get("detected_activities", []),
        "confidence": (overall.get("video_level_reasoning") or {}).get("confidence"),
        "error": job.get("error"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:5009")
    parser.add_argument("--output", type=Path, default=Path("activity-evaluation.json"))
    args = parser.parse_args()
    videos = sorted(args.videos.glob("*.mp4"))
    rows = []
    for index, video in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] {video.name}", flush=True)
        try:
            row = result_for(args.base_url, video)
        except Exception as exc:
            row = {"filename": video.name, "expected": expected_label(video), "predicted": None,
                   "correct": False, "status": "client_error", "error": str(exc)}
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)
    csv_path = args.output.with_suffix(".csv")
    keys = ["filename", "expected", "predicted", "correct", "status", "execution_seconds",
            "confidence", "basis", "detected_activities", "job_id", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "detected_activities": ",".join(row.get("detected_activities", []))})
    completed = [row for row in rows if row.get("status") == "completed"]
    correct = sum(bool(row.get("correct")) for row in completed)
    print(f"RESULT {correct}/{len(completed)} correct; {len(rows) - len(completed)} failed", flush=True)


if __name__ == "__main__":
    main()
