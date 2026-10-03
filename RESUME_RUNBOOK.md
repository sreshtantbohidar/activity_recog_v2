# Resume Runbook — activity_recog_v2 offline deliverable (Oct 3, 2026)

## Hosts & credentials
| Host | OS | User / Password | Role |
|---|---|---|---|
| 192.168.1.16 | Windows (cmd.exe shell) | hp / 123456 | Build host. Project: `C:\Users\Hp\projects\activity_recog_v2` |
| 192.168.1.125 | Ubuntu, RTX A4000 16GB idx0 | demo / Tec#2309@Tum | PROD target. Project: `/home/demo/Music/activity-code-v3/activity_recog_v2` |

- Dev box CANNOT reach .125 directly (VPN). Reach .125 only from .16 or via scripts run on .16.
- Remote shell on .16 is cmd.exe: use `dir`, `findstr`, `copy`, `certutil`, `schtasks`, `type`. NO tail/ls/grep/head.
- DB creds (provision via API, never bake into files): host 192.168.1.125, db isquare, user postgres, password TeCtUmFaTbOy321, port 5432.
- SFTP on .125: user demo, key auth via `/app/secrets/id_ed25519` (mounted from `secrets/`), remote dir `/home/demo/Documents/fatboy/fatboy/ftpserver/media_analytics/output`, public base `https://192.168.1.125:9000`, public path `/media_analytics/output`.

## Where things stand (as of this runbook)
- Image `activity-recognition:2.0.0` (ID f4e6a6bc76a3, 35.6GB) = validated build with Omni+Whisper baked. A REBUILD was started to pick up the latest code (provisioning feature, Health/UT fixes). It is in the "exporting layers" phase — takes 1–2h. Watch:
  ```
  ssh hp@192.168.1.16                       (pw 123456)
  powershell -NoProfile -Command "Get-Content C:\Users\Hp\preload_build.log -Tail 5"
  # done when the log contains BUILD_EXIT_0
  docker images activity-recognition --format "{{.Tag}} {{.ID}} {{.CreatedSince}}"   # tag must show a NEW id
  ```
- All code fixes are in `C:\Users\Hp\projects\activity_recog_v2\` (pushed) and validated in the RUNNING container (UT 108/108, Health 9/9, E2E ok: operation.mp4 transcribes 5 segments; deployment.mp4 empty transcript + engine sounds).
- Scheduled task `ExportDeliverable` exists but its cmd path had a hiccup — prefer running the export manually (Step 3).

## Step 1 — After BUILD_EXIT_0: free disk (C: is tight, ~60GB free!)
```
docker builder prune -af
docker image prune -f
docker images   # verify ~100+ GB would be free:  powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,1)"
```
Do NOT prune while the build is running.

## Step 2 — Confirm the new image is good (2 min, no GPU needed)
```
docker run --rm --entrypoint python -v "C:/Users/Hp/projects/activity_recog_v2/deploy/verify_offline_image.py:/app/deploy/verify_offline_image.py" activity-recognition:2.0.0 /app/deploy/verify_offline_image.py
# expect: RESULT: all 10 checks passed -- image is fully offline-capable
```

## Step 3+4+5 — Export & transfer (DONE automatically via scheduled task StreamTar)
The tar never touches C: (too small). `C:\Users\Hp\stream_tar.cmd` streams docker save over SSH
using key auth (C:\Users\Hp\.ssh\arv2_ed25519 -> demo@192.168.1.125 authorized_keys).
Check progress on .125:  ls -la /home/demo/deliverable/
Done when size stabilizes at ~35GB AND docker save task (StreamTar) reports STREAM_OK in
its Last Result / or the size stops growing for 5+ minutes.
Afterwards compute md5 ON .125:
  md5sum /home/demo/deliverable/activity-recognition-2.0.0-offline.tar > /home/demo/deliverable/activity-recognition-2.0.0-offline.tar.md5
Deploy assets are already on .125 in /home/demo/deliverable/ (deploy_ubuntu.sh, composes, .env.ubuntu,
entrypoint.sh, verify_offline_image.py, UBUNTU_DEPLOY.md) — push again from .16 if ever missing:
  scp -i C:\Users\Hp\.ssh\arv2_ed25519 C:\Users\Hp\projects\activity_recog_v2\deploy\* demo@192.168.1.125:/home/demo/deliverable/

## Step 6 — Deploy on .125
```
ssh demo@192.168.1.125                # pw Tec#2309@Tum
docker rm -f activity-recognition-v4  # legacy container holds port 5009 (unhealthy)
docker rm -f activity-recog-v3 activity-recognition-v3 2>/dev/null
cd ~/deliverable/deploy && chmod +x deploy_ubuntu.sh entrypoint.sh
./deploy_ubuntu.sh --yes
# does: md5sum -c -> docker load -> GPU pick (A4000 idx0) -> writes .env -> compose up -> waits /health (up to 15 min) -> smoke test
```

## Step 7 — Provision credentials on .125 (applies instantly, persists to config/runtime.json)
```
curl -s -X POST -H "Content-Type: application/json" -d '{"DB_HOST":"192.168.1.125","DB_PORT":"5432","DB_NAME":"isquare","DB_USER":"postgres","DB_PASSWORD":"TeCtUmFaTbOy321","SFTP_HOST":"192.168.1.125","SFTP_PORT":"22","SFTP_USER":"demo","SFTP_PRIVATE_KEY_PATH":"/app/secrets/id_ed25519","SFTP_REMOTE_DIR":"/home/demo/Documents/fatboy/fatboy/ftpserver/media_analytics/output","OUTPUT_PUBLIC_BASE":"https://192.168.1.125:9000","OUTPUT_PUBLIC_PATH":"/media_analytics/output"}' http://localhost:5009/config/credentials
# then live-test both:
curl -s -X POST http://localhost:5009/config/credentials/test
# expect {"postgres":{"ok":true...},"sftp":{"ok":true...},"status":"ok"}
```
(Secrets are never echoed back by GET; saving again with a real value replaces them. The dashboard Config tab has a Connections section that does the same.)

## Step 8 — Final validation on .125
```
curl -s http://localhost:5009/health/components          # expect 9/9 ok (or 8/9 if no DB_HOST yet -> provision first)
curl -s http://localhost:5009/unit-tests                  # expect 108/108 passed
# E2E speech clip (Whisper transcript expected):
curl -s -F "file=@/home/demo/Music/activity-code-v3/activity_recog_v2/testvideos/operation.mp4" http://localhost:5009/analyze-video
# poll with the returned job_id:  curl -s "http://localhost:5009/queue/<JOB_ID>?include_response=1"
# expect 5 transcript segments ("Command to Bravo, eyes on target." etc), no warnings
# E2E ambient clip (EMPTY transcript expected):
curl -s -F "file=@/home/demo/Music/activity-code-v3/activity_recog_v2/testvideos/deployment.mp4" http://localhost:5009/analyze-video
# expect transcript segments: 0, sounds describing truck engine/door, no warnings
```
Dashboard: http://192.168.1.125:5009/ (tabs: Analysis, Config, Queue, Health, UT).

## Known-good references
- Verified working container on .16 (job seq 71 + f083d3db): Whisper large-v3 + Qwen2.5-Omni-3B, text-only Thinker, `return_audio=False`, no-speech filter 0.6.
- Swagger docs must cover every route (test enforces it): `/health/components`, `/unit-tests`, `/config/credentials`, `/config/credentials/test` are in the expected set.
- If compose on .125 mounts `huggingface-cache` volume → baked weights get masked by an empty volume. The ubuntu override compose (used by deploy_ubuntu.sh) deliberately omits it. Don't re-add.
- .125 disk: 111GB free before the tar (~35GB) + loaded image (~35GB) — fine, but keep an eye; `docker builder prune -af` is the emergency valve.
- Old legacy container on .125 crash-loops with NVMLError_InvalidArgument (old vLLM vs new driver) — irrelevant once removed; new image has vLLM 0.11.0.

## Housekeeping (optional, after go-live)
- Delete stray files on .16: `C:\Users\Hp\_probe_export.ps1`, `_push_clips.py` (used for clip upload), old `push_clips.log`.
- On .16 dev box keep container running with devtest override: `docker compose -f docker-compose.yml -f docker-compose.devtest.yml up -d` (drops HF volume so baked weights are used).
- model_store.py stays UNWIRED (parked feature per user decision).
