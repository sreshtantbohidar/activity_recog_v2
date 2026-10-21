# Activity Recognition service

A single container runs the whole service on one GPU:

- **Vision** — an in-container vLLM engine serving `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit` (int4 AWQ build of Qwen3-VL-8B-Instruct, ~6 GB weights, vision tower left unquantized). It runs with `enforce_eager`, a bounded pixel budget, 8 keyframes per window and a ~11 GB VRAM cap (0.45 of a 24 GB card). It listens on `127.0.0.1:8001`; that port is never published.
- **Audio** — `Qwen/Qwen2.5-Omni-3B` loaded by the API process (Thinker branch, text-only output) for non-speech sound description; speech transcription runs on faster-whisper large-v3 with the audio model's own speech pass as fallback.
- **Objects** — the dedicated YOLO checkpoints remain in use for `/detect-objects`, `/detect-objects-video`, and detailed-video tracking.

There is **no fallback backend**: a failed vision request returns an error instead of silently switching models.

## Vision and audio never run together

Both models share the one GPU and stay resident, and requests are strictly
sequenced by the API's stage locks:

1. Every frame/window inference completes against the vLLM engine.
2. Only then does the audio stage run; it holds `AUDIO_STAGE_LOCK` and the vision
   lock, so a concurrent request cannot start vLLM inference while the audio model
   is generating.

Memory fits without a sleep/wake cycle because the engine is capped near 11 GB
(`VLLM_GPU_MEMORY_UTILIZATION=0.45` on a 24 GB card) and the 4-bit audio model
uses about 6 GB — roughly 17 GB peak.

## Releasing the GPU when idle

When no API request has arrived for `VLLM_IDLE_SLEEP_SECONDS` (default 60), the
service gives the GPU back to the host so other containers can use it:

1. the vLLM engine is put to sleep — its weights are offloaded to CPU RAM and
   the KV cache is discarded, returning roughly 10 GB of VRAM;
2. the audio model is unloaded.

The next request wakes the engine and reloads the audio model transparently.
`/health` is exempt, so the container healthcheck does not keep the GPU awake;
in-flight requests are never interrupted. Set `VLLM_IDLE_SLEEP_SECONDS=0` to
disable idling and keep both models permanently resident.

GPU inference is required. The API returns unavailable/backend errors when CUDA is not exposed to the container; it never silently performs model inference on CPU. Media decoding, preprocessing, HTTP handling and transfers into VRAM still require bounded CPU and system RAM.

## Request queue

Concurrent callers must not be able to run inference at the same time: two video
jobs decoding frames and calling the engine together is what exhausts system RAM
and VRAM. The five heavy endpoints — `/analyze-image`, `/analyze-video`,
`/analyze-video-detailed`, `/detect-objects` and `/detect-objects-video` —
therefore hand their work to an internal FIFO queue instead of executing it
inline:

1. the upload is validated and staged *on the request*, so a bad or oversized
   file still fails immediately with the same 400/413 as before;
2. the work is enqueued and the API answers `202` with a `job_id`, a
   `queue_position` and a `status_url` (also sent as the `Location` header);
3. a single worker thread runs jobs in arrival order, one at a time;
4. the caller polls `GET /queue/<job_id>` for the recorded status, response and
   timings.

`/fetch-media` is deliberately **not** queued and still runs inline.

`/detect-objects-video` runs the YOLO checkpoint over every frame with persistent
tracking and writes an annotated MP4 under `REPORTS_DIR/object-videos`. The job
response carries the track summary plus a bounded sample of per-frame detections
and an `annotated_video_url` served by `GET /object-videos/<filename>`. The
newest `OBJECT_VIDEO_RETENTION` clips (default 25) are kept, so the reports
volume cannot grow without bound; an aged-out URL returns `404`. Clips are
transcoded to H.264 when ffmpeg is available (`OBJECT_VIDEO_TRANSCODE=1`) so the
dashboard can play them inline; if transcoding fails the mp4v original is kept
and remains downloadable.

```sh
curl -s -F file=@clip.mp4 localhost:5009/detect-objects-video
curl -s localhost:5009/queue/<job_id> | jq '.response.object_detection | {frames_sampled, detected_object_counts, annotated_video_url}'
```

Statuses are `queued`, `running`, `completed` and `failed`. A job is `failed`
when its work raised or produced a non-2xx response, with the reason in `error`.
The request that submitted a job has already returned, so the engine is held
awake for the job's whole duration: the idle timer cannot park the GPU in the
middle of a queued analysis.

```sh
curl -s localhost:5009/queue | jq '.queue'
curl -s 'localhost:5009/queue?status=running'
curl -s localhost:5009/queue/<job_id> | jq '{status, duration_seconds, response}'
```

History lives in SQLite at `JOB_DB_PATH` (`/app/data/jobs.sqlite3`, on the
`activity-data` volume) and keeps the newest `JOB_HISTORY_LIMIT`
completed/failed entries; queued and running jobs are never pruned. A restart
cannot resume in-process work, so anything still `queued`/`running` is marked
`failed` ("Interrupted by a service restart") on startup and its staged uploads
are deleted.

`JOB_MAX_PENDING` (default 32) caps the backlog: past it the API sheds load with
`503` instead of accepting work it cannot start soon. Responses are stored up to
`JOB_RESPONSE_MAX_CHARS`; a longer body is recorded as a `response_preview` with
`response_truncated: true`, so a base64-laden detector response cannot grow the
queue database without bound.

Every job also records a `progress` percentage (0-100) and a `progress_detail`
string, updated while it runs: image inference, the per-frame loop for video and
object annotation, object tracking, the audio stage and PDF rendering. A
completed job is 100%; a failed one keeps whatever it had reached, which is more
informative than a reset. The dashboard renders this as a progress bar and
highlights the queued/running rows.

Finished history can be pruned without touching pending work:

```sh
curl -s -X DELETE localhost:5009/queue/<job_id>   # one completed/failed job
curl -s -X DELETE localhost:5009/queue            # every completed/failed job
```

A queued or running job is refused with `409`, and the bulk clear only ever
deletes finished rows, so clearing history can never destroy work in progress.

A job can also be cancelled or reloaded:

```sh
curl -s -X POST localhost:5009/queue/<job_id>/cancel
curl -s -X POST localhost:5009/queue/<job_id>/reload
```

Cancellation is **cooperative**. A job still in the queue is marked `cancelled`
immediately and is never run. A running job stops at its next checkpoint —
between stages and on every frame of a video or annotation loop — so a step
already inside a single model call (image inference, the audio stage, PDF
rendering) finishes first and the job is then recorded as `completed`. A job
cancelled mid-run keeps the progress percentage it had reached.

Reload queues the same request again as a **new** job, rebuilt from what the
original recorded: the `media_analysis.id` for the DB-driven mode, or the
retained upload for a file upload. Retained uploads are bounded by
`JOB_REPLAY_RETENTION` (newest 10 jobs) and `JOB_REPLAY_MAX_BYTES` (1 GiB
in total), and are released when their row is deleted, pruned, or the service
restarts. A job whose input is gone reports `409` and the dashboard disables its
reload button rather than queueing an analysis of nothing.

The dashboard's **Queue** tab lists the depth and a page of jobs with their
status, progress, timings and response, and offers per-row cancel, reload and
delete plus a *Clear history* button.

## Start

Copy `.env.example` to `.env`, set database credentials if the database-backed route is needed, then run:

```sh
docker compose up --build -d
docker compose ps
```

The first start downloads the Qwen3-VL and Qwen2.5-Omni checkpoints into the persistent `huggingface-cache` volume, so the health check has a long `start-period` (10 minutes). The dashboard is at `http://localhost:5009/` and Swagger UI is at `http://localhost:5009/docs/`.

The default build is lean. Set `PRELOAD_MODELS=1` only when producing a fully offline deployment image on a build host with ample RAM and disk.

## Verify after Docker is running

```sh
docker compose exec activity-recognition python -m unittest discover -v
docker compose logs --tail=100 activity-recognition
```

Image and video requests can be exercised from the dashboard. The integration suite verifies `/health`, `/`, and every documented Swagger operation against the running container.

## Deploying changes safely

Deploy by rebuilding the image, never by copying source into a running container:

```sh
python3 -m unittest test_feature_guard -v
docker compose up --build -d
python3 verify_deployment.py --url http://127.0.0.1:5009
```

`docker cp` leaves the image behind, so the next `docker compose up` or a container recreate
silently reverts to the old files. That is how the Monitor tab and the idle engine recycle
disappeared on 2026-10-17: the container stayed healthy, so nothing reported the loss.
`verify_deployment.py` now fails a deploy when `/gpu/monitor`, the Monitor tab or the
video-detection routes are missing from the running service.

The source is under git, so a clean `git status` is part of the deploy. If a file is ever
overwritten again, restore the last known-good baseline with:

```sh
git checkout -- .
```

A portable copy of that baseline is kept outside the tree:
`/home/demo/activity-recog-v2-baseline.bundle` (`git clone <bundle> <dir>`).

## Performance controls

- `MAX_VIDEO_FRAMES` bounds expensive VLM calls (default 24).
- `FRAME_INTERVAL_SECONDS` controls temporal sampling (default 2 seconds).
- `VLLM_GPU_MEMORY_UTILIZATION` caps the engine (default 0.45, about 11 GB on a 24 GB card).
- `QWEN_AUDIO_LOAD_IN_4BIT=1` keeps the audio model inside a tighter VRAM budget (off by default for the 3B Omni model).
- One Gunicorn worker owns the GPU model; four threads keep health/UI requests responsive, and the internal job queue serialises the actual inference so two requests cannot exhaust VRAM at once.
- Database connections and YOLO weights are initialized only when their endpoints are used.
- Audio analysis is opt-in (`ENABLE_AUDIO_ANALYSIS=1`).
