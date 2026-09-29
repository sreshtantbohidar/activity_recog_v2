#!/bin/sh
# Single-container entrypoint.
#
# Starts the in-container vLLM engine serving Qwen3-VL-8B (int4 AWQ) on the
# loopback interface, waits until it is ready, then execs Gunicorn.
#
# The engine may take most of the card: it is put to sleep (weights offloaded to
# CPU, KV cache discarded) before the Qwen2-Audio stage runs, so vision and
# audio never need to be resident at the same time (see engine_power.py).
#
# Sleep mode is enabled so the API can also release the GPU while idle:
# POST /sleep?level=1 offloads the weights and parks the engine, POST /wake_up
# brings it back. Those control endpoints need vLLM's dev mode, which is scoped
# to the container's loopback interface.
#
# The engine is SUPERVISED: a CUDA fault on a particular multimodal input can
# kill the EngineCore process, and without a supervisor the API would be left
# without vision until the container was restarted by hand.
set -u

MODEL="${VLLM_MODEL_ID:-cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit}"
HOST="127.0.0.1"
PORT="8001"

# Enables /sleep and /wake_up. Port 8001 is never published outside the container.
export VLLM_SERVER_DEV_MODE=1

ENGINE_PID=""
SHUTDOWN=0

start_engine() {
  vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$MODEL" \
    --dtype bfloat16 \
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

# Supervise the engine in the background: restart it whenever it exits so a
# crashed EngineCore is replaced automatically instead of taking vision down.
supervise() {
  while :; do
    start_engine
    wait "$ENGINE_PID" 2>/dev/null || true
    if [ "$SHUTDOWN" = "1" ]; then break; fi
    echo "[entrypoint] vLLM engine exited; restarting in 5s" >&2
    sleep 5
  done
}
supervise &
SUPERVISOR_PID=$!

echo "[entrypoint] waiting for the vLLM engine on ${HOST}:${PORT} ..."
ATTEMPTS=0
until curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; do
  ATTEMPTS=$((ATTEMPTS + 1))
  # Do not block startup forever: after ~6 minutes start the API anyway so
  # /health can report the engine as unreachable while it keeps retrying.
  if [ "$ATTEMPTS" -ge 90 ]; then
    echo "[entrypoint] engine still not ready; starting Gunicorn anyway" >&2
    break
  fi
  sleep 4
done
echo "[entrypoint] vLLM engine ready; starting Gunicorn"

exec gunicorn --bind 0.0.0.0:5009 --worker-class gthread --workers 1 --threads 4 \
  --timeout 600 --graceful-timeout 30 --access-logfile - --error-logfile - app:app
