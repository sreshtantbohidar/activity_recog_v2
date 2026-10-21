@echo off
cd /d C:\Users\Hp\projects\activity_recog_v2
echo == verify offline image ==
docker run --rm --entrypoint python -v "C:/Users/Hp/projects/activity_recog_v2/deploy/verify_offline_image.py:/app/deploy/verify_offline_image.py" activity-recognition:2.0.0 /app/deploy/verify_offline_image.py
if errorlevel 1 goto :fail
if not exist deliverable mkdir deliverable
echo == docker save ==
docker save -o deliverable\activity-recognition-2.0.0-offline.tar activity-recognition:2.0.0
if errorlevel 1 goto :fail
echo == md5 ==
powershell -NoProfile -Command "$h=(Get-FileHash 'deliverable\activity-recognition-2.0.0-offline.tar' -Algorithm MD5).Hash.ToLower(); Set-Content -Path 'deliverable\activity-recognition-2.0.0-offline.tar.md5' -Value ('  '+$h+'  activity-recognition-2.0.0-offline.tar') -Encoding ascii"
echo == copy deploy assets ==
if not exist deliverable\deploy mkdir deliverable\deploy
copy /y deploy\deploy_ubuntu.sh deliverable\deploy\ >NUL
copy /y deploy\docker-compose.yml deliverable\deploy\ >NUL
copy /y deploy\docker-compose.ubuntu.yml deliverable\deploy\ >NUL
copy /y deploy\.env.ubuntu deliverable\deploy\ >NUL
copy /y deploy\entrypoint.sh deliverable\deploy\ >NUL
copy /y deploy\verify_offline_image.py deliverable\deploy\ >NUL
copy /y deploy\UBUNTU_DEPLOY.md deliverable\deploy\ >NUL
echo EXPORT_OK
goto :eof
:fail
echo EXPORT_FAILED
