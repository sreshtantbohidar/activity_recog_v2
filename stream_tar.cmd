@echo off
rem Stream the built image straight to the Ubuntu target over SSH.
rem No local tar is written, so the tight C: drive is never touched.
docker save activity-recognition:2.0.0 | ssh -i C:\Users\Hp\.ssh\arv2_ed25519 -o StrictHostKeyChecking=no -o BatchMode=yes demo@192.168.1.125 "cat > /home/demo/deliverable/activity-recognition-2.0.0-offline.tar"
if errorlevel 1 goto :fail
echo STREAM_OK
goto :eof
:fail
echo STREAM_FAILED
