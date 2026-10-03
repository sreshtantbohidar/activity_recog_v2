"""E2E probe: run one video through /analyze-video-detailed and print a summary.

Usage (inside the container):  python /tmp/e2e_probe.py /tmp/elint.mp4 elint
Saves the full response next to the summary as /tmp/e2e_<name>_full.json.
"""
import json
import sys
import time

import requests

path = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else path.split("/")[-1]
host = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:5009"

t0 = time.time()
with open(path, "rb") as fh:
    r = requests.post(host + "/analyze-video-detailed",
                      files={"file": (name + ".mp4", fh, "video/mp4")}, timeout=3600)
elapsed = time.time() - t0

out = {"expected_activity": name, "http": r.status_code, "elapsed_sec": round(elapsed, 1)}
try:
    j = r.json()
except Exception:
    out["body"] = r.text[:800]
    print(json.dumps(out, indent=2))
    sys.exit(0)

out["top_level_keys"] = sorted(j.keys())
if "error" in j:
    out["error"] = j["error"]
    print(json.dumps(out, indent=2))
    sys.exit(0)

with open("/tmp/e2e_%s_full.json" % name, "w") as fh:
    json.dump(j, fh)

a = j.get("analysis") or {}
ov = (a.get("detailed_report") or {}).get("executive_overview") or {}
au = a.get("audio_analysis") or {}
out.update({
    "dominant_activity": ov.get("dominant_activity"),
    "basis": ov.get("dominant_activity_basis"),
    "activities": ov.get("activities"),
    "frames_analyzed": ov.get("frames_analyzed"),
    "persistent_tracks": ov.get("persistent_tracks"),
    "audio_available": ov.get("audio_available"),
    "backends_used": a.get("backends_used"),
    "transcript_segments": len(au.get("transcript") or []),
    "transcript_preview": " ".join(s.get("text", "") for s in (au.get("transcript") or []))[:400],
    "sound_events": [e.get("label") for e in (au.get("sound_events") or [])][:12],
    "sound_events_count": len(au.get("sound_events") or []),
    "audio_warnings": au.get("warnings"),
    "report": (a.get("report") or {}).get("filename"),
})
out["match"] = out.get("dominant_activity") == name
print(json.dumps(out, indent=2))
