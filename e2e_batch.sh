#!/bin/sh
# Sequential E2E batch runner: one video at a time through /analyze-video-detailed.
# Usage: sh /tmp/e2e_batch.sh <name1> <name2> ...
API=http://127.0.0.1:5009

wait_health() {
  i=0
  while [ $i -lt 72 ]; do
    if curl -sf "$API/health" >/dev/null 2>&1; then
      if curl -s "$API/health" 2>/dev/null | grep -q '"model_available":true'; then
        return 0
      fi
    fi
    sleep 5
    i=$((i+1))
  done
  return 1
}

for n in "$@"; do
  echo "=== START $n $(date -u +%H:%M:%S) ==="
  wait_health || echo "WARN: health timeout before $n"
  python /tmp/e2e_probe.py "/testvideos/$n.mp4" "$n" > "/tmp/e2e_$n.json" 2>&1
  rc=$?
  echo "=== DONE $n rc=$rc $(date -u +%H:%M:%S) ==="
done
echo "=== ALL DONE $(date -u +%H:%M:%S) ==="
