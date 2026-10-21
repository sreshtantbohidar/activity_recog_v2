"""Verify a running deployment still serves the features this service ships.

Run it after every deploy:

    python3 verify_deployment.py --url http://127.0.0.1:5009

It checks the HTTP surface only (no GPU work), and exits non-zero when the
Monitor screen, its endpoint, or the video-detection routes are missing. This is
the check that would have caught the 2026-10-17 deploy, where a stale file set
replaced app.py and the dashboard: the container stayed healthy, so nothing
else reported the loss.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

CHECKS = []


def fetch(url: str, timeout: float = 20.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", "replace")


def record(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check a running activity-recognition deployment")
    parser.add_argument("--url", default="http://127.0.0.1:5009", help="service base URL")
    parser.add_argument("--timeout", type=float, default=20.0)
    arguments = parser.parse_args(argv)
    base = arguments.url.rstrip("/")

    try:
        status, body = fetch(base + "/health", arguments.timeout)
        record("GET /health", status == 200, f"HTTP {status}")
    except Exception as exc:
        record("GET /health", False, str(exc))
        body = ""

    monitor = {}
    try:
        status, payload = fetch(base + "/gpu/monitor", arguments.timeout)
        monitor = json.loads(payload)
        record("GET /gpu/monitor", status == 200, f"HTTP {status}")
        record("monitor payload has device/resource",
               all(key in monitor for key in ("available", "device", "processes", "resource")),
               ", ".join(sorted(monitor))[:120])
        device = monitor.get("device") or {}
        record("monitor reports the card", bool(monitor.get("available")) and bool(device.get("name")),
               f"{device.get('name')} {device.get('used_mib')}/{device.get('total_mib')} MiB"
               if device else "GPU not readable from the service")
    except Exception as exc:
        record("GET /gpu/monitor", False, str(exc))

    try:
        status, html = fetch(base + "/", arguments.timeout)
        markers = ('data-mode="monitor"', 'id="monitorPanel"', 'id="vramChart"',
                   'id="resourceLight"', "fetch('/gpu/monitor')")
        missing = [marker for marker in markers if marker not in html]
        record("dashboard ships the Monitor tab", status == 200 and not missing,
               f"HTTP {status}" if missing else "all markers present")
        record("dashboard ships video object detection", 'data-mode="objectsVideo"' in html)
    except Exception as exc:
        record("GET /", False, str(exc))

    try:
        status, payload = fetch(base + "/apispec.json", arguments.timeout)
        spec = json.loads(payload)
        paths = spec.get("paths", {})
        for path in ("/gpu/monitor", "/detect-objects-video"):
            record(f"Swagger documents {path}", path in paths)
    except Exception as exc:
        record("GET /apispec.json", False, str(exc))

    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"RESULT: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
