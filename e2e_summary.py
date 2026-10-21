"""Summarize all e2e_<name>.json probe outputs written inside the container.

Two signals per video:
  top1    - the dominant activity equals the expected label
  detected - the expected label appears anywhere in the video's activity set
"""
import json
import os

NAMES = [
    "elint", "disposition", "deployment", "foreign_relation", "infra",
    "internal_security", "joint_exercise", "miscellaneous", "mobilization",
    "movement", "operation", "patrolling", "political_dipomacy", "procurement",
    "technology", "training", "visit",
]

# The filename for political_dipomacy is misspelt, so map it to its real label.
EXPECTED = {"political_dipomacy": "political diplomacy"}


def norm(value):
    return str(value or "").strip().lower().replace("_", " ")


top1_hits = detected_hits = seen = 0
for n in NAMES:
    p = "/tmp/e2e_%s.json" % n
    if not os.path.exists(p):
        print("%-20s -- pending" % n)
        continue
    try:
        d = json.load(open(p))
    except Exception as e:
        print("%-20s ERR %s" % (n, e))
        continue
    if d.get("http") != 200:
        print("%-20s HTTP %s" % (n, d.get("http")))
        seen += 1
        continue
    expected = EXPECTED.get(n, n.replace("_", " "))
    dom = norm(d.get("dominant_activity"))
    acts = [norm(a) for a in (d.get("activities") or [])]
    full = {}
    try:
        full = json.load(open("/tmp/e2e_%s_full.json" % n))
    except Exception:
        pass
    for frame in ((full.get("analysis") or {}).get("frame_analysis") or []):
        acts.extend(norm(a) for a in (frame.get("detected_activity") or []))
    top1 = dom == expected
    detected = expected in acts
    top1_hits += top1
    detected_hits += detected
    seen += 1
    print("%-20s top1=%-5s detected=%-5s dom=%-22s expected=%-20s frames=%s audio_events=%s" % (
        n, top1, detected, dom, expected, d.get("frames_analyzed"), d.get("sound_events_count")))
print("---")
print("top1 %d/%d   detected-anywhere %d/%d" % (top1_hits, seen, detected_hits, seen))
