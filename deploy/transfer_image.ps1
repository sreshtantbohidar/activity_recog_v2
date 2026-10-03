# Transfer the offline image from the Windows build host to the Ubuntu host.
#
# Streams docker save output straight into docker load over SSH, so no ~20 GB
# tar file ever has to exist on either disk. Falls back to a temp tar file when
# direct streaming is blocked. Also copies every deployment asset the Ubuntu
# deploy script needs.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File deploy\transfer_image.ps1 `
#       [-UbuntuHost 192.168.1.125] [-UbuntuUser demo] [-TargetDir <dir>]
param(
    [string]$UbuntuHost = "192.168.1.125",
    [string]$UbuntuUser = "demo",
    [string]$TargetDir  = "/home/demo/Music/activity-code-v3/activity_recog_v2"
)

$ErrorActionPreference = "Stop"
$image = "activity-recognition:2.0.0"

docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) { throw "Image $image not found. Run deploy\build_offline_image.ps1 first." }

Write-Host "== Checking disk space on $UbuntuHost ==" -ForegroundColor Cyan
$freeKb = (ssh "$UbuntuUser@$UbuntuHost" "df -kPk $TargetDir 2>/dev/null || df -kPk /home" | Select-Object -Skip 1) |
    ForEach-Object { ($_ -split "\s+")[3] } | Select-Object -First 1
if (-not $freeKb) { throw "Could not read free disk space on $UbuntuHost (is sshpass/key auth set up?)." }
$freeGb = [math]::Round([long]$freeKb / 1MB, 1)
Write-Host "   free at ${TargetDir}: $freeGb GB"
if ($freeGb -lt 25) { throw "Only $freeGb GB free - need >= 25 GB for the image tar + workspace." }

Write-Host "== Streaming $image to $UbuntuUser@$UbuntuHost ==" -ForegroundColor Cyan
Write-Host "   (~20 GB; several minutes on LAN; no temp file on either host)"
cmd /c "docker save $image | ssh $UbuntuUser@$UbuntuHost 'docker load'"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Streamed load failed - falling back to a temporary tar file."
    $tar = Join-Path $env:TEMP "activity-recognition-2.0.0.tar"
    docker save -o $tar $image
    if ($LASTEXITCODE -ne 0) { throw "docker save failed." }
    scp $tar "${UbuntuUser}@${UbuntuHost}:/tmp/"
    if ($LASTEXITCODE -ne 0) { throw "scp failed." }
    ssh "$UbuntuUser@$UbuntuHost" "docker load -i /tmp/activity-recognition-2.0.0.tar && rm /tmp/activity-recognition-2.0.0.tar"
    if ($LASTEXITCODE -ne 0) { throw "docker load failed." }
    Remove-Item $tar
}

Write-Host "== Copying deploy assets (compose + env + scripts + runbook) ==" -ForegroundColor Cyan
ssh "$UbuntuUser@$UbuntuHost" "mkdir -p $TargetDir/deploy"
scp deploy\docker-compose.yml, deploy\docker-compose.ubuntu.yml, deploy\.env.ubuntu, `
    deploy\entrypoint.sh, deploy\verify_offline_image.py, deploy\deploy_ubuntu.sh, deploy\UBUNTU_DEPLOY.md `
    "${UbuntuUser}@${UbuntuHost}:${TargetDir}/deploy/"
if ($LASTEXITCODE -ne 0) { throw "Copying deploy assets failed." }

Write-Host "== Verifying the loaded image on $UbuntuHost ==" -ForegroundColor Cyan
ssh "$UbuntuUser@$UbuntuHost" "docker image inspect $image --format 'loaded: {{.Id}} ({{.Size}} bytes)'"
if ($LASTEXITCODE -ne 0) { throw "Image missing on $UbuntuHost after transfer." }

Write-Host "Image transferred. Next: ssh $UbuntuUser@$UbuntuHost and run deploy/deploy_ubuntu.sh" -ForegroundColor Green
