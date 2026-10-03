FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface TRANSFORMERS_NO_ADVISORY_WARNINGS=1 ACTIVITY_VISION_BACKEND=vllm \
    VLLM_VISION_HOST=http://127.0.0.1:8001 \
    VLLM_MODEL_ID=cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit \
    VLLM_VISION_MODEL=cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit \
    QWEN_AUDIO_MODEL_ID=Qwen/Qwen2.5-Omni-3B \
    MPLCONFIGDIR=/tmp/matplotlib YOLO_CONFIG_DIR=/tmp/ultralytics \
    TRITON_CACHE_DIR=/tmp/triton HOME=/home/app
WORKDIR /app
# gcc/g++ are required by Triton's JIT: without a C compiler the vLLM engine
# dies with "Failed to find C compiler" while building a kernel at load time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Order matters: the API stack is installed first, then vLLM last so the engine's
# own pins (numpy/opencv/transformers) win the shared dependencies.
# vLLM 0.11.0 pins torch==2.8.0, so install the matching cu128 torch/torchvision
# first (0.23.0 pairs with 2.8.0); an older torchvision would break YOLO.
RUN python -m pip install --upgrade pip==24.3.1 \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 \
    && python -m pip install -r requirements.txt \
    && python -m pip install "vllm==0.11.0" "transformers>=4.57,<5"
# paramiko is deliberately installed in its own layer instead of being appended to
# requirements.txt: requirements.txt feeds the single torch/vLLM RUN above, so any
# edit there invalidates it and forces a multi-gigabyte re-download. It powers the
# SFTP delivery used by the DB-driven /detect-objects mode.
RUN python -m pip install "paramiko>=3.4,<5"
# Dedicated ASR for speech transcription. Installed in its own layer (like
# paramiko) so the multi-gigabyte torch/vLLM layer above is never invalidated.
# The large-v3 weights are baked into /opt/whisper -- NOT /models/huggingface,
# which the runtime named volume masks -- so transcription works offline.
RUN python -m pip install "faster-whisper>=1.0,<2" \
    && mkdir -p /opt/whisper \
    && python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8', download_root='/opt/whisper')" \
    && chown -R 65532:65532 /opt/whisper
ARG PRELOAD_MODELS=0
# Model weights come from the build context (deploy/prefetch_models.py fills
# ./hf_models on the build host with resumable downloads), so the image is
# assembled without touching the network -- a build can no longer stall or fail
# on a flaky HuggingFace connection. The RUN guards below are the fallback for
# repositories missing from ./hf_models (e.g. a hand-rolled context).
COPY hf_models /models/huggingface/hub
RUN if [ "$PRELOAD_MODELS" = "1" ] && [ ! -d /models/huggingface/hub/models--cyankiwi--Qwen3-VL-8B-Instruct-AWQ-4bit ]; then python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit')"; fi
RUN if [ "$PRELOAD_MODELS" = "1" ] && [ ! -d /models/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B ]; then python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-Omni-3B')"; fi
COPY . .
# Passwd/group entries for the runtime uid: torch's inductor cache setup calls
# getpass.getuser(), which does a pwd lookup that fails for an unlisted uid.
RUN mkdir -p /app/logs /app/tmp /app/reports /app/config /app/segments /app/secrets /app/data /home/app /tmp/triton /models/huggingface \
    && chown -R 65532:65532 /app /models /home/app /tmp/triton \
    && chmod +x /app/entrypoint.sh \
    && echo "app:x:65532:65532:Activity Service:/home/app:/bin/false" >> /etc/passwd \
    && echo "app:x:65532:" >> /etc/group
# HOME/USER are set so tools that read them skip the pwd lookup entirely.
ENV HOME=/home/app USER=app
USER 65532:65532
EXPOSE 5009
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 CMD curl -fsS http://127.0.0.1:5009/health || exit 1
# entrypoint.sh starts the in-container vLLM engine, waits for readiness, then
# launches Gunicorn. One container, one GPU: vision (vLLM) and audio
# (Qwen2.5-Omni) are sequenced so they never run together.
ENTRYPOINT ["/app/entrypoint.sh"]
