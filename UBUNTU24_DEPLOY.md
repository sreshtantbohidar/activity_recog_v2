# Ubuntu 24.04 deployment

## Requirements
- Ubuntu 24.04 LTS
- NVIDIA GPU with a current NVIDIA driver
- Docker Engine + Docker Compose v2
- NVIDIA Container Toolkit
- At least ~10 GB free VRAM for the Qwen3-VL AWQ engine; 16 GB+ is recommended
- Disk space for the Docker image/model layers

## Install host GPU runtime
Verify:
```bash
nvidia-smi
docker --version
docker compose version
```

If Docker cannot see NVIDIA:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i nvidia
```

## Source build
The first source build downloads/bakes the vision and audio models into the Docker image, so it needs Internet access and substantial disk space. Later starts are local/offline.

```bash
cd actvity_recog_v2
cp deploy/.env.ubuntu .env
nano .env
chmod +x deploy/setup_ubuntu24.sh
./deploy/setup_ubuntu24.sh
```

The script auto-selects the NVIDIA GPU with the most free VRAM. Use `--gpu-index N` to pin one.

## Offline deployment
If you have `activity-recognition-*.tar`:
```bash
cd deploy
chmod +x deploy_ubuntu.sh
./deploy_ubuntu.sh --yes
```

## Operations
```bash
docker compose -f docker-compose.yml -f docker-compose.linux.yml ps
docker compose -f docker-compose.yml -f docker-compose.linux.yml logs -f
docker compose -f docker-compose.yml -f docker-compose.linux.yml restart
curl http://127.0.0.1:5009/health
```

Dashboard: `http://<ubuntu-host>:5009/`
Swagger: `http://<ubuntu-host>:5009/docs/`

## SFTP
Place the user's key manually:
```bash
chmod 700 secrets
chmod 600 secrets/id_ed25519
```
Then set `SFTP_HOST`, `SFTP_USER`, `SFTP_REMOTE_DIR`, and related values in `.env`.

No Windows `.ps1` or `.cmd` script is required on Ubuntu.
