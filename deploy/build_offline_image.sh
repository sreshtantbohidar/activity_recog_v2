#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/offline-deliverable}"
IMAGE="activity-recognition:2.0.0"
cd "$ROOT"

for file in \
  hf_models/Qwen3-VL-8B-Instruct-AWQ-4bit/config.json \
  hf_models/Qwen2.5-Omni-3B/config.json \
  whisper_models/faster-whisper-large-v3/model.bin \
  object_detect_model_256_class.pt app.py entrypoint.sh Dockerfile.ubuntu; do
  [[ -s "$file" ]] || { echo "missing or empty: $file" >&2; exit 1; }
done

docker build -f Dockerfile.ubuntu -t "$IMAGE" .
docker run --rm --network none --entrypoint python "$IMAGE" \
  /app/deploy/verify_ubuntu_image.py

rm -rf "$OUT"
mkdir -p "$OUT/deploy"
docker save -o "$OUT/activity-recognition-2.0.0-offline.tar" "$IMAGE"
(
  cd "$OUT"
  sha256sum activity-recognition-2.0.0-offline.tar > SHA256SUMS
)
cp deploy/deploy_ubuntu.sh deploy/docker-compose.yml \
  deploy/docker-compose.ubuntu.yml deploy/.env.ubuntu deploy/entrypoint.sh \
  deploy/verify_ubuntu_image.py deploy/UBUNTU_DEPLOY.md Dockerfile.ubuntu \
  "$OUT/deploy/"
echo "Offline production bundle: $OUT"

