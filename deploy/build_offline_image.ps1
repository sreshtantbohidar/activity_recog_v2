# Build the fully offline deliverable on the Windows host (.16).
#
# Produces, in <project>\deliverable\, everything the Ubuntu machine needs:
#   activity-recognition-2.0.0-offline.tar   docker image, every model baked in
#   activity-recognition-2.0.0-offline.tar.md5
#   deploy\                                  deploy_ubuntu.sh + compose + env +
#                                            verify script + runbook
# Copy that folder to an external HDD; on the Ubuntu host run
# deploy\deploy_ubuntu.sh from it (see UBUNTU_DEPLOY.md).
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File deploy\build_offline_image.ps1
#
# The script is idempotent: the model prefetch resumes where it stopped, and
# finished docker layers are cached, so code-only rebuilds take seconds.

$ErrorActionPreference = "Stop"

# 0. Preconditions -------------------------------------------------------------
docker info *> $null
if ($LASTEXITCODE -ne 0) { throw "Docker is not running. Start Docker Desktop first." }

# 1. Prefetch the checkpoints on the host (resumable; skips finished ones) ------
#    The build then COPYs ./hf_models into the image without any network access,
#    so a flaky HuggingFace connection can no longer stall a build.
Write-Host "== Prefetching model checkpoints into .\hf_models ==" -ForegroundColor Cyan
python deploy\prefetch_models.py
if ($LASTEXITCODE -ne 0) { throw "Model prefetch incomplete - rerun this script to resume." }

# 2. Build -----------------------------------------------------------------------
Write-Host "== Building activity-recognition:2.0.0 ==" -ForegroundColor Cyan
docker compose build --build-arg PRELOAD_MODELS=1 2>&1 | Tee-Object -FilePath build_offline.log
if ($LASTEXITCODE -ne 0) { throw "Image build failed - see build_offline.log" }

# 3. Verify the image is truly offline-capable (no GPU needed) -----------------
Write-Host "== Verifying image contents (libraries + every checkpoint) ==" -ForegroundColor Cyan
docker run --rm --entrypoint python activity-recognition:2.0.0 /app/deploy/verify_offline_image.py
if ($LASTEXITCODE -ne 0) { throw "Offline verification FAILED - do not ship this image." }

# 4. Export the deliverable folder for the external HDD --------------------------
Write-Host "== Exporting deliverable ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path deliverable\deploy | Out-Null
docker save -o deliverable\activity-recognition-2.0.0-offline.tar activity-recognition:2.0.0
if ($LASTEXITCODE -ne 0) { throw "docker save failed." }
# Windows-native checksum (md5sum is not always present):
$hash = (Get-FileHash deliverable\activity-recognition-2.0.0-offline.tar -Algorithm MD5).Hash.ToLower()
"  $hash  activity-recognition-2.0.0-offline.tar" |
    Out-File -Encoding ascii deliverable\activity-recognition-2.0.0-offline.tar.md5
Copy-Item deploy\deploy_ubuntu.sh, deploy\docker-compose.yml, deploy\docker-compose.ubuntu.yml, `
    deploy\.env.ubuntu, deploy\entrypoint.sh, deploy\verify_offline_image.py, deploy\UBUNTU_DEPLOY.md `
    deliverable\deploy\ -Force

Write-Host "== Deliverable ready ==" -ForegroundColor Green
Get-ChildItem deliverable | Format-Table Name, @{n='GB';e={[math]::Round($_.Length/1GB,2)}}
Write-Host "Copy the deliverable folder to the external HDD, then on the Ubuntu host:" -ForegroundColor Yellow
Write-Host "  cd <deliverable>/deploy && ./deploy_ubuntu.sh --yes"
