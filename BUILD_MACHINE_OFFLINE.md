# Offline build-machine update

These files replace the corresponding files in the existing project. The build machine must have Internet access for Docker base images, Python packages and Hugging Face model downloads.

## 1. Install build-host tools

```bash
python3 -m pip install -U huggingface_hub hf_transfer
```

Docker must also be installed and running.

## 2. Prefetch all runtime models

```bash
python3 deploy/prefetch_models.py
```

This downloads exactly:

- `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit`
- `Qwen/Qwen2.5-Omni-3B`
- `Systran/faster-whisper-large-v3`

The local YOLO model `object_detect_model_256_class.pt` is already part of the project.

## 3. Build and verify the offline image

```bash
chmod +x deploy/build_offline_image.sh
deploy/build_offline_image.sh
```

The script:

1. downloads/resumes all three models;
2. builds the Docker image;
3. verifies the image with `--network none` and without GPU inference;
4. exports `activity-recognition-2.0.0.tar`;
5. generates `SHA256SUMS`;
6. prepares an offline deployment directory.

The build does not require an NVIDIA GPU.

## 4. Important

Do not run `docker compose ... up` on the 8 GB development GPU merely to prepare the offline package. The image can be verified without starting vLLM.

The target Ubuntu 24.04 machine only needs Docker + NVIDIA driver + NVIDIA Container Toolkit. It does not need Python, pip, Hugging Face access or Internet access.
