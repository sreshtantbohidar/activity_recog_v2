# Build the fully offline deliverable on the Windows host (.16).
#
# Produces, in <project>\deliverable\, everything the Ubuntu machine needs:
#   activity-recognition-2.0.0-offline.tar   docker image, every model baked in
#   SHA256SUMS
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

# 1. Build -----------------------------------------------------------------------
Write-Host "== Building activity-recognition:2.0.0 ==" -ForegroundColor Cyan
docker build -f Dockerfile.ubuntu -t activity-recognition:2.0.0 . 2>&1 | Tee-Object -FilePath build_offline.log
if ($LASTEXITCODE -ne 0) { throw "Image build failed - see build_offline.log" }

# 3. Verify the image is truly offline-capable (no GPU needed) -----------------
Write-Host "== Verifying image contents (libraries + every checkpoint) ==" -ForegroundColor Cyan
docker run --rm --network none --entrypoint python activity-recognition:2.0.0 /app/deploy/verify_ubuntu_image.py
if ($LASTEXITCODE -ne 0) { throw "Offline verification FAILED - do not ship this image." }

# 4. Export the deliverable folder for the external HDD --------------------------
Write-Host "== Exporting deliverable ==" -ForegroundColor Cyan
if (Test-Path deliverable) { Remove-Item -Recurse -Force deliverable }
New-Item -ItemType Directory -Force -Path deliverable\deploy | Out-Null
docker save -o deliverable\activity-recognition-2.0.0-offline.tar activity-recognition:2.0.0
if ($LASTEXITCODE -ne 0) { throw "docker save failed." }
# Windows-native SHA-256 checksum.
$hash = (Get-FileHash deliverable\activity-recognition-2.0.0-offline.tar -Algorithm SHA256).Hash.ToLower()
"$hash  activity-recognition-2.0.0-offline.tar" |
    Out-File -Encoding ascii deliverable\SHA256SUMS
Copy-Item deploy\deploy_ubuntu.sh, deploy\setup_ubuntu24.sh, deploy\docker-compose.yml, `
    deploy\docker-compose.ubuntu.yml, Dockerfile.ubuntu, deploy\.env.ubuntu, `
    deploy\entrypoint.sh, deploy\verify_offline_image.py, deploy\verify_ubuntu_image.py, `
    deploy\build_offline_image.sh, deploy\UBUNTU_DEPLOY.md `
    deliverable\deploy\ -Force

Write-Host "== Deliverable ready ==" -ForegroundColor Green
Get-ChildItem deliverable | Format-Table Name, @{n='GB';e={[math]::Round($_.Length/1GB,2)}}
Write-Host "Copy the deliverable folder to the external HDD, then on the Ubuntu host:" -ForegroundColor Yellow
Write-Host "  cd <deliverable>/deploy && ./deploy_ubuntu.sh --yes"
