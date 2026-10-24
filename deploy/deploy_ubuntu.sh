#!/usr/bin/env bash
# Deploy the offline activity-recognition stack on the Ubuntu host.
#
# Runs ON the Ubuntu machine, from the deliverable folder that arrived on the
# external HDD (image tar + checksum + deploy assets). The image is loaded from
# the tar automatically, so no internet and no docker-save pipeline are needed.
#
# Usage:
#   ./deploy_ubuntu.sh [--yes] [--gpu-index N] [--target DIR] [--gpu-util F] [--tar FILE]
#   --tar FILE  image tar to load (default: activity-recognition-*.tar next to
#               this script or one level up)
#
# GPU portability (any NVIDIA GPU works; validated for RTX A4000 16 GB,
# RTX A6000 48 GB, RTX PRO 4000 Blackwell 24 GB):
#   - picks the card with the most free VRAM (or --gpu-index to pin one)
#   - sizes VLLM_GPU_MEMORY_UTILIZATION from the detected VRAM
#   - bf16-capable cards (Ampere+, all three above) run the engine in bf16
#   - keeps containers that use the GPU healthy: if another container (e.g.
#     ollama) is holding VRAM, it is reported and the deploy stops
set -euo pipefail

TARGET="/home/demo/deliverable/actvity_recog_v2"
GPU_INDEX=""          # empty = auto-pick the card with most free VRAM
ASSUME_YES=0
OVERRIDE_UTIL=""
TAR_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)       ASSUME_YES=1; shift ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --gpu-util)  OVERRIDE_UTIL="$2"; shift 2 ;;
    --target)    TARGET="$2"; shift 2 ;;
    --tar)       TAR_FILE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die()  { printf '\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# 0. Host prerequisites ---------------------------------------------------------
say "Checking host prerequisites"
command -v docker >/dev/null || die "docker not installed"
docker info >/dev/null 2>&1 || die "docker daemon not reachable"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found - install the NVIDIA driver"
nvidia-smi >/dev/null 2>&1 || die "NVIDIA driver not responding"
info "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
docker info 2>/dev/null | grep -qi "nvidia" || die "nvidia-container-toolkit not wired into docker (run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker)"

# 1. GPU detection ----------------------------------------------------------------
say "Detecting GPU"
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
info "NVIDIA GPUs found: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

if [[ -n "$GPU_INDEX" ]]; then
  IDX="$GPU_INDEX"
else
  # Pick the GPU with the most free memory right now.
  IDX=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi
info "using GPU index: $IDX"
nvidia-smi -i "$IDX" >/dev/null 2>&1 || die "GPU index $IDX does not exist"

VRAM_MIB=$(nvidia-smi -i "$IDX" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
VRAM_USED=$(nvidia-smi -i "$IDX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
VRAM_FREE=$((VRAM_MIB - VRAM_USED))
GPU_NAME=$(nvidia-smi -i "$IDX" --query-gpu=name --format=csv,noheader)
info "GPU $IDX: $GPU_NAME  ${VRAM_MIB} MiB total, ${VRAM_FREE} MiB free"

# Size the engine from free VRAM: weights ~7.5 GB + KV cache + audio headroom.
# On a 16 GB card that means the whole card (0.92 = 14.35 GiB of 15.6 GiB): the
# weights alone load 7.37 GiB and the memory profile peaks at ~13.75 GiB, which
# leaves only 0.60 GiB of KV cache. The engine therefore cannot start while the
# API's audio stack (~5 GB, loaded in the API process on the same card) is
# resident, so engine_power.sleep_now() releases it before the entrypoint
# supervisor's restart - that release path is what keeps the service up.
if [[ -n "$OVERRIDE_UTIL" ]]; then
  GPU_UTIL="$OVERRIDE_UTIL"
elif   (( VRAM_FREE >= 40000 )); then GPU_UTIL=0.85     # A6000 48 GB
elif   (( VRAM_FREE >= 22000 )); then GPU_UTIL=0.80     # Blackwell 24 GB
elif   (( VRAM_FREE >= 14000 )); then GPU_UTIL=0.92     # A4000 16 GB; engine owns the card
elif   (( VRAM_FREE >= 10000 )); then GPU_UTIL=0.60     # shared / small card
else die "GPU $IDX has only ${VRAM_FREE} MiB free; the engine alone needs >= 10 GB. Stop the other GPU processes and retry."
fi
info "VLLM_GPU_MEMORY_UTILIZATION=$GPU_UTIL (from ${VRAM_FREE} MiB free)"

# 2. Image: load from the deliverable tar if not already loaded --------------------
say "Loading the offline image"
IMAGE="activity-recognition:2.0.0"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  info "already loaded: $(docker image inspect "$IMAGE" --format '{{.Id}}')"
else
  if [[ -z "$TAR_FILE" ]]; then
    HERE_TAR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    TAR_FILE=$(ls "$HERE_TAR"/activity-recognition-*.tar "$HERE_TAR"/../activity-recognition-*.tar 2>/dev/null | head -1 || true)
  fi
  [[ -n "$TAR_FILE" && -f "$TAR_FILE" ]] || die "image not loaded and no tar found - copy activity-recognition-*.tar from the deliverable folder and pass --tar FILE"
  CHECKSUM_FILE="$(dirname "$TAR_FILE")/SHA256SUMS"
  [[ -f "$CHECKSUM_FILE" ]] || die "missing SHA256SUMS next to the image tar"
  info "verifying SHA-256"
  (cd "$(dirname "$TAR_FILE")" && sha256sum -c SHA256SUMS) || die "tar checksum mismatch - re-copy from the HDD"
  info "loading $(basename "$TAR_FILE") (several minutes)"
  docker load -i "$TAR_FILE" || die "docker load failed"
fi
info "$(docker image inspect "$IMAGE" --format 'image: {{.Id}} ({{.Size}} bytes)')"

# 3. Prepare the target directory ----------------------------------------------------
say "Preparing $TARGET"
mkdir -p "$TARGET/reports" "$TARGET/config" "$TARGET/secrets" "$TARGET/testvideos" "$TARGET/logs"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Deploy assets are copied next to this script by transfer_image.ps1.
for f in docker-compose.yml docker-compose.ubuntu.yml Dockerfile.ubuntu .env.ubuntu verify_ubuntu_image.py; do
  [[ -f "$HERE/$f" ]] && cp -f "$HERE/$f" "$TARGET/" && info "copied $f"
done
[[ -f "$HERE/entrypoint.sh" ]] && cp -f "$HERE/entrypoint.sh" "$TARGET/"
[[ -f "$TARGET/secrets/id_ed25519" ]] || info "NOTE: no $TARGET/secrets/id_ed25519 - SFTP delivery (/detect-objects) will return 502 until added"

# 4. Environment -----------------------------------------------------------------------
say "Writing .env"
WRITE_ENV=0
if [[ -f "$TARGET/.env" && $ASSUME_YES -eq 0 ]]; then
  read -rp "   .env exists. Overwrite with ubuntu defaults (GPU/settings re-detected)? [y/N] " a
  [[ "${a:-n}" =~ ^[Yy] ]] && WRITE_ENV=1
else
  WRITE_ENV=1
fi
if [[ $WRITE_ENV -eq 1 ]]; then
  cp -f "$TARGET/.env.ubuntu" "$TARGET/.env"
  # Persist the GPU decisions so 'docker compose up' reuses them verbatim.
  sed -i "s/^GPU_DEVICE_INDEX=.*/GPU_DEVICE_INDEX=$IDX/" "$TARGET/.env"
  sed -i "/^VLLM_GPU_MEMORY_UTILIZATION=/d" "$TARGET/.env"
  echo "VLLM_GPU_MEMORY_UTILIZATION=$GPU_UTIL" >> "$TARGET/.env"
  info "wrote .env (GPU_DEVICE_INDEX=$IDX, VLLM_GPU_MEMORY_UTILIZATION=$GPU_UTIL)"
else
  info "keeping the existing .env"
fi
[[ -f "$TARGET/.env" ]] || die ".env missing in $TARGET"

# 5. Stop the old broken stack if it owns port 5009 -------------------------------------
say "Freeing port 5009"
OLD=$(docker ps -a --filter "publish=5009" --format '{{.Names}}' | grep -v activity_recog_v2 || true)
if [[ -n "$OLD" ]]; then
  info "removing container(s): $OLD (they are the old v3 stack, unhealthy since the vLLM crash)"
  docker rm -f $OLD
fi

# 6. Launch -----------------------------------------------------------------------------
cd "$TARGET"
say "Starting the stack"
docker compose -f docker-compose.yml -f docker-compose.ubuntu.yml up -d

# 7. Wait for health ---------------------------------------------------------------------
say "Waiting for /health (engine load can take several minutes)"
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5009/health || true)
  if [[ "$code" == "200" ]]; then
    info "healthy after ~$((i * 15))s"
    break
  fi
  [[ $i -eq 60 ]] && { echo "still not healthy; last code: $code" >&2
    docker compose -f docker-compose.yml -f docker-compose.ubuntu.yml logs --tail=50
    die "service did not become healthy" ; }
  sleep 15
done

# 8. Smoke test -----------------------------------------------------------------------------
say "Smoke test: /analyze-video-detailed on a test clip (if present)"
CLIP=$(ls "$TARGET"/testvideos/*.mp4 2>/dev/null | head -1 || true)
if [[ -n "$CLIP" ]]; then
  JOB=$(curl -s -F "file=@$CLIP" http://127.0.0.1:5009/analyze-video-detailed | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
  info "queued job $JOB; poll: curl -s http://127.0.0.1:5009/queue/$JOB"
else
  info "no clip in $TARGET/testvideos - copy one there and re-run this step manually"
fi

say "Deployment complete"
info "API:      http://$(hostname -I | awk '{print $1}'):5009"
info "Docs:     http://127.0.0.1:5009/docs/"
info "Queue UI: http://127.0.0.1:5009/"
info "Logs:     cd $TARGET && docker compose -f docker-compose.yml -f docker-compose.ubuntu.yml logs -f"
