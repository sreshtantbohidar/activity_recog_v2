#!/bin/sh
# Single-container entrypoint.
#
# Starts the in-container vLLM engine serving Qwen3-VL-8B (int4 AWQ) on the
# loopback interface, waits until it is ready, then execs Gunicorn.
#
# The engine may take most of the card: it is put to sleep (weights offloaded to
# CPU, KV cache discarded) before the audio stage runs, so vision and
# audio never need to be resident at the same time (see engine_power.py).
#
# Sleep mode is enabled so the API can also release the GPU while idle:
# POST /sleep?level=1 offloads the weights and parks the engine, POST /wake_up
# brings it back. Those control endpoints need vLLM's dev mode, which is scoped
# to the container's loopback interface.
#
# The engine is SUPERVISED: a CUDA fault on a particular multimodal input can
# kill the EngineCore process, and without a supervisor the API would be left
# without vision until the container was restarted by hand. A restart also
# waits for the engine's GPU memory budget first (gpu_vram_gate.py): while
# another process holds the card, loading 7.4 GiB of weights would only fail
# vLLM's free-memory check again.
set -u

MODEL="${VLLM_MODEL_ID:-cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit}"
HOST="127.0.0.1"
PORT="8001"

# Enables /sleep and /wake_up. Port 8001 is never published outside the container.
export VLLM_SERVER_DEV_MODE=1

ENGINE_PID=""
SHUTDOWN=0

# dtype auto keeps the image GPU-model-agnostic: Ampere+ runs bfloat16, older
# cards fall back to float16 instead of failing to start.
start_engine() {
  vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$MODEL" \
    --dtype "${VLLM_DTYPE:-auto}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --limit-mm-per-prompt '{"video": 1, "image": 4}' \
    --enforce-eager \
    --enable-sleep-mode \
    --no-enable-prefix-caching \
    --mm-processor-kwargs "{\"max_pixels\": ${VLLM_MM_MAX_PIXELS:-200704}, \"nframes\": ${VLLM_MM_NFRAMES:-8}}" &
  ENGINE_PID=$!
}

stop_engine() {
  SHUTDOWN=1
  if [ -n "$ENGINE_PID" ]; then kill "$ENGINE_PID" 2>/dev/null || true; fi
}
trap stop_engine INT TERM

# Wait until the card can hold the engine. The gate opens as soon as the
# memory is free again and never gives up, so a restart is attempted only
# when it can succeed. A container without the gate script still starts.
wait_for_vram() {
  [ -f /app/gpu_vram_gate.py ] || return 0
  python3 /app/gpu_vram_gate.py --util "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}" \
    --interval 5 --max-wait 0
}

# Supervise the engine in the background: restart it whenever it exits so a
# crashed EngineCore is replaced automatically instead of taking vision down.
supervise() {
  while :; do
    if ! wait_for_vram; then break; fi
    start_engine
    wait "$ENGINE_PID" 2>/dev/null || true
    if [ "$SHUTDOWN" = "1" ]; then break; fi
    echo "[entrypoint] vLLM engine exited; restarting in 5s" >&2
    sleep 5
  done
}
supervise &
SUPERVISOR_PID=$!

# How long the API waits for the engine before serving without it. The YOLO
# endpoints (object detection, video object detection, the queue and the
# dashboard) never touch the engine, so holding them hostage to a model load,
# or to a VRAM gate that is waiting for another container to release the card,
# costs functionality for nothing. 0 starts Gunicorn at once; the supervised
# engine still joins whenever the gate opens, and the analysis endpoints answer
# 502 until it does (/health reports it).
GRACE="${ENGINE_STARTUP_GRACE_SECONDS:-360}"
case "$GRACE" in
  ''|*[!0-9]*) GRACE=360 ;;
esac
if [ "$GRACE" -gt 0 ]; then
  echo "[entrypoint] waiting up to ${GRACE}s for the vLLM engine on ${HOST}:${PORT} ..."
  ATTEMPTS=0
  MAX_ATTEMPTS=$((GRACE / 4 + 1))
  until curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ "$ATTEMPTS" -ge "$MAX_ATTEMPTS" ]; then
      echo "[entrypoint] engine still not ready after ${GRACE}s; starting the API anyway" >&2
      break
    fi
    sleep 4
  done
else
  echo "[entrypoint] not waiting for the vLLM engine (ENGINE_STARTUP_GRACE_SECONDS=0)" >&2
fi
echo "[entrypoint] starting Gunicorn"

exec gunicorn --bind 0.0.0.0:5009 --worker-class gthread --workers 1 --threads 4 \
  --timeout 600 --graceful-timeout 30 --access-logfile - --error-logfile - app:app
