FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HOME=/app/hf_cache \
    HUGGINGFACE_HUB_CACHE=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# ------------------------------------------------------------
# System runtime dependencies
# ------------------------------------------------------------
# The base image uses Ubuntu repositories. Some environments
# block HTTP/port 80, so convert Ubuntu repositories to HTTPS.
RUN set -eux; \
    find /etc/apt -type f \
        \( -name "*.list" -o -name "*.sources" \) \
        -print0 | \
        xargs -0 -r sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g'; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libsndfile1 \
        libgomp1 \
        ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade \
        pip \
        setuptools \
        wheel && \
    python -m pip install --no-cache-dir \
        -r /app/requirements.txt

# ------------------------------------------------------------
# Application source + local offline models
# ------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------
# Local model locations
# ------------------------------------------------------------
ENV VLLM_MODEL_PATH=/app/hf_models/Qwen3-VL-8B-Instruct-AWQ-4bit \
    QWEN_OMNI_MODEL_PATH=/app/hf_models/Qwen2.5-Omni-3B \
    WHISPER_MODEL_PATH=/app/whisper_models/faster-whisper-large-v3 \
    YOLO_MODEL_PATH=/app/object_detect_model_256_class.pt

# ------------------------------------------------------------
# Verify that all required offline models are included
# ------------------------------------------------------------
RUN set -eux; \
    echo "============================================"; \
    echo "Checking offline models..."; \
    echo "============================================"; \
    test -d "/app/hf_models/Qwen3-VL-8B-Instruct-AWQ-4bit"; \
    echo "[OK] Qwen3-VL-8B-Instruct-AWQ-4bit"; \
    test -d "/app/hf_models/Qwen2.5-Omni-3B"; \
    echo "[OK] Qwen2.5-Omni-3B"; \
    test -d "/app/whisper_models/faster-whisper-large-v3"; \
    echo "[OK] faster-whisper-large-v3"; \
    test -f "/app/object_detect_model_256_class.pt"; \
    echo "[OK] object_detect_model_256_class.pt"; \
    echo "============================================"; \
    echo "Offline model verification PASSED"; \
    echo "============================================"

# ------------------------------------------------------------
# Writable directories
# ------------------------------------------------------------
# Do NOT recursively chown /app.
# The model files are ~22 GB and recursively changing ownership
# makes the image build unnecessarily slow.
RUN mkdir -p \
        /app/logs \
        /app/tmp \
        /app/reports \
        /app/config \
        /app/segments \
        /app/secrets \
        /app/data \
        /app/hf_cache \
        /home/app \
        /tmp/triton && \
    chown -R 65532:65532 \
        /app/logs \
        /app/tmp \
        /app/reports \
        /app/config \
        /app/segments \
        /app/secrets \
        /app/data \
        /app/hf_cache \
        /home/app \
        /tmp/triton

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
RUN chmod +x /app/entrypoint.sh

# ------------------------------------------------------------
# Run application as non-root
# ------------------------------------------------------------
USER 65532:65532

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
