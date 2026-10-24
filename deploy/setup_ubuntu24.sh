#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPU_INDEX="${GPU_DEVICE_INDEX:-}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --gpu-util) GPU_UTIL="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "Docker is not installed" >&2; exit 1; }
docker compose version >/dev/null
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not installed" >&2; exit 1; }
docker info 2>/dev/null | grep -qi nvidia || {
  echo "NVIDIA Container Toolkit is not configured for Docker" >&2; exit 1;
}

mkdir -p reports config secrets testvideos logs
[[ -f .env ]] || cp deploy/.env.ubuntu .env

if [[ -z "$GPU_INDEX" ]]; then
  GPU_INDEX="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
FREE_MIB="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ -z "$GPU_UTIL" ]]; then
  if (( FREE_MIB >= 40000 )); then GPU_UTIL=0.85
  elif (( FREE_MIB >= 14000 )); then GPU_UTIL=0.92
  elif (( FREE_MIB >= 10000 )); then GPU_UTIL=0.60
  else echo "GPU $GPU_INDEX has only ${FREE_MIB} MiB free" >&2; exit 1
  fi
fi

sed -i "s/^GPU_DEVICE_INDEX=.*/GPU_DEVICE_INDEX=$GPU_INDEX/" .env
sed -i "s/^VLLM_GPU_MEMORY_UTILIZATION=.*/VLLM_GPU_MEMORY_UTILIZATION=$GPU_UTIL/" .env

COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.ubuntu.yml)
"${COMPOSE[@]}" config >/dev/null
"${COMPOSE[@]}" build
"${COMPOSE[@]}" up -d --remove-orphans

for _ in {1..120}; do
  if curl -fsS "http://127.0.0.1:${ACTIVITY_PORT:-5009}/health" >/dev/null 2>&1; then
    echo "Activity Recognition is healthy on port ${ACTIVITY_PORT:-5009}"
    exit 0
  fi
  sleep 5
done
"${COMPOSE[@]}" logs --tail=100
exit 1

