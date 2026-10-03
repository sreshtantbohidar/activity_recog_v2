# DevOps Runbook — Offline Deliverable & Ubuntu Deployment

End-to-end flow: **build on the Windows host → transfer the image → deploy on
Ubuntu**, three scripts, no manual steps, fully offline at the far end.

```
Windows (.16, build host)                external HDD          Ubuntu (any host)
┌────────────────────────────┐                                ┌──────────────────────────────┐
│ deploy\build_offline_...ps1│   copy deliverable\ folder     │ deliverable\deploy\...       │
│  1. prefetch models        │  ────────── HDD ──────────►    │ deploy_ubuntu.sh             │
│  2. docker build (COPY)    │   (~20 GB tar + md5 + assets)  │  loads tar, detects GPU,     │
│  3. verify offline         │                                │  writes .env, waits health   │
│  4. export deliverable\    │                                └──────────────────────────────┘
└────────────────────────────┘
```

## What the deliverable image contains

`activity-recognition:2.0.0` is **all-in-one** — every model and library is a
layer of the image, so `docker load` is the only thing the Ubuntu host needs:

| Component | Baked at | Purpose |
|---|---|---|
| vLLM 0.11 + torch 2.8 (cu128) | image packages | vision engine runtime |
| Qwen3-VL-8B-Instruct-AWQ-4bit | HF cache layer | frame analysis (vLLM) |
| Qwen2.5-Omni-3B | HF cache layer | sound description (Thinker, text-only) |
| faster-whisper large-v3 | `/opt/whisper` | speech transcription (ASR) |
| YOLO 256-class weights | `/app/object_detect_model_256_class.pt` | object detection |

The build script refuses to bless the image until `verify_offline_image.py`
passes **inside it**: library imports, the Omni classes, the whisper files, all
three HF checkpoints, and offline (`HF_HUB_OFFLINE=1`) resolution of each.

## Step 1 — Build (Windows host)

```powershell
cd C:\Users\Hp\projects\activity_recog_v2
powershell -ExecutionPolicy Bypass -File deploy\build_offline_image.ps1
```

- Idempotent: code-only changes reuse every cached layer (seconds).
- First run or Dockerfile change: ~30-60 min (model downloads).
- Fails loudly if the offline verification fails — never ship a red image.

## Step 2 — Move the deliverable (external HDD, manual)

`build_offline_image.ps1` leaves everything in `<project>\deliverable\`:

```
deliverable\
├── activity-recognition-2.0.0-offline.tar        (~20 GB image, all models)
├── activity-recognition-2.0.0-offline.tar.md5    checksum the deploy verifies
└── deploy\
    ├── deploy_ubuntu.sh        <- run this on the Ubuntu host
    ├── docker-compose.yml
    ├── docker-compose.ubuntu.yml   (pins the image, no build, no HF volume)
    ├── .env.ubuntu
    ├── entrypoint.sh
    ├── verify_offline_image.py
    └── UBUNTU_DEPLOY.md           (this runbook)
```

Copy the whole `deliverable\` folder onto the external HDD and from there onto
the Ubuntu machine, e.g. to `/home/demo/deliverable/`. No internet is needed on
the Ubuntu side; the tar checksum is verified before `docker load` runs.

> Optional LAN shortcut (when a direct copy is convenient instead of the HDD):
> `docker save activity-recognition:2.0.0 | ssh demo@<host> 'docker load'` —
> then deploy from the `deploy\` assets the same way.

## Step 3 — Deploy (Ubuntu host)

```bash
cd /home/demo/deliverable/deploy     # wherever the HDD folder was copied
chmod +x deploy_ubuntu.sh
./deploy_ubuntu.sh --yes
# options: --gpu-index N | --gpu-util 0.80 | --target /path | --tar FILE | --yes
```

The script loads `activity-recognition-2.0.0-offline.tar` from the deliverable
folder (verifying its md5 first), then continues automatically.

What it does, in order:

1. **Host prerequisites** — docker, NVIDIA driver, nvidia-container-toolkit.
2. **GPU detection (portable)** — lists every NVIDIA GPU and picks the one with
   the most **free** VRAM (`--gpu-index N` to pin one). Validated for:
   | GPU | VRAM | engine setting chosen |
   |---|---|---|
   | NVIDIA RTX A4000 | 16 GB | `VLLM_GPU_MEMORY_UTILIZATION=0.80` |
   | NVIDIA RTX PRO 4000 Blackwell | 24 GB | `0.80` |
   | NVIDIA RTX A6000 | 48 GB | `0.85` |
   Any other NVIDIA card works: utilisation is derived from free VRAM and the
   engine dtype is `auto` (bf16 on Ampere+, float16 fallback on older cards).
   If another container (e.g. ollama) holds VRAM, the deploy stops and says so
   instead of deploying a service that cannot load its engine.
3. **Image check** — confirms `activity-recognition:2.0.0` is loaded.
4. **Target dir** — creates `reports/ config/ secrets/ testvideos/ logs/`,
   copies deploy assets into place, warns if the SFTP key is missing.
5. **`.env`** — written from `.env.ubuntu` with the detected GPU index and
   utilisation persisted (an existing `.env` is kept unless confirmed).
6. **Old stack** — any container squatting on port 5009 (the old v3 stack) is
   removed.
7. **Launch** — `docker compose -f docker-compose.yml -f
   docker-compose.ubuntu.yml up -d` (the override pins the transferred image
   and cancels any rebuild).
8. **Health wait** — polls `/health` up to 15 min; dumps engine logs on failure.
9. **Smoke test** — queues `/analyze-video-detailed` on the first clip in
   `testvideos/` and prints the polling URL for the job.

Copy test clips into `<target>/testvideos/` before step 3 for the smoke test.

## Post-deploy verification

```bash
curl -s http://127.0.0.1:5009/health | python3 -m json.tool   # engine + CUDA
curl -s http://127.0.0.1:5009/health/components               # per-component green/red
# Dashboard: http://<host>:5009/  → "Health" tab and "UT" tab
```

The dashboard **Health** tab renders the component checks green/red; the **UT**
tab runs the unit-test suite inside the container and shows each case green/red.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health` 503; logs show `vLLM control call /sleep failed: Connection refused` | engine crashed at startup — check `docker logs`; verify `GPU_DEVICE_INDEX` in `.env` matches the intended card (`nvidia-smi`) |
| `NVMLError_InvalidArgument` in `get_device_capability` | old vLLM vs newer driver mismatch — the shipped vLLM 0.11.0 handles driver 595; if a swapped-in image still crashes, rebuild it |
| deploy stops: "GPU has only X MiB free" | another container holds VRAM (ollama held 11 GB on .125) — stop it or `--gpu-index` a free card |
| `/detect-objects` returns 502 | SFTP key missing at `secrets/id_ed25519` or `SFTP_HOST` unreachable — the Health tab's `sftp` row shows this in red |
| disk fills during transfer | streaming needs no temp tar, but Docker storage does — `docker system df`, then `docker builder prune -af` (reclaimed ~150 GB on .125) |
| `KeyError: 'qwen2_5_omni'` at audio load | image predates the Omni-capable transformers pin — rebuild (Dockerfile pins `transformers>=4.57,<5`) |

## Rollback

```bash
cd <target>
docker rm -f activity_recog_v2-activity-recognition-1 2>/dev/null || true
# restore the previous stack, e.g.:
docker start <old-container-name>
# or redeploy an older image tag the same way (the image name is a compose param)
```
