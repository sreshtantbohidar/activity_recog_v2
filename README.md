# activity_recog_v2

A single-container, single-GPU activity-recognition service. It answers four
analysis endpoints — image analysis, video analysis, detailed video analysis
(with a PDF report) and object detection — plus an internal FIFO **request
queue** that makes concurrency safe on one GPU.

- **Vision** — an in-container vLLM engine serving
  `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit` on `127.0.0.1:8001` (never published).
- **Audio** — `Qwen/Qwen2-Audio-7B-Instruct` loaded 4-bit by the API process for
  speech transcription and non-speech sound description.
- **Objects** — YOLO checkpoints for `/detect-objects` and detailed-video tracking.

There is **no fallback backend**: a failed vision request returns an error rather
than silently switching models. GPU inference is required; the service never
falls back to CPU inference.

## Request queue

The four heavy endpoints hand their work to an in-process FIFO queue instead of
running it inline, so two callers can never drive inference at the same time.

1. The upload is validated and staged *on the request* — a bad or oversized file
   still fails immediately with the same `400`/`413` as before.
2. The work is enqueued and the API answers **`202`** with a `job_id`, a
   `queue_position` and a `status_url` (also sent as the `Location` header).
3. One worker thread runs jobs in arrival order, one at a time.
4. The caller polls `GET /queue/<job_id>` for status, progress, response and timings.

`/fetch-media` is deliberately **not** queued and still runs inline.

Statuses are `queued`, `running`, `completed`, `failed` and `cancelled`. A job is
`failed` when its work raised or produced a non-2xx response, with the reason in
`error`. Every job also records a `progress` percentage (0-100) and a
`progress_detail` string, updated as it runs; a completed job is 100% and a
failed/cancelled one keeps whatever it had reached.

```sh
curl -s localhost:5009/queue | jq '.queue'
curl -s 'localhost:5009/queue?status=running'
curl -s localhost:5009/queue/<job_id> | jq '{status, progress, duration_seconds, response}'
```

History lives in SQLite at `JOB_DB_PATH` (on the `activity-data` volume) and keeps
the newest `JOB_HISTORY_LIMIT` finished entries; queued and running jobs are never
pruned. A restart cannot resume in-process work, so anything still queued/running
is marked `failed` on startup and its staged uploads are deleted.

`JOB_MAX_PENDING` (default 32) caps the backlog: past it the API sheds load with
`503` instead of accepting work it cannot start soon. Responses are stored up to
`JOB_RESPONSE_MAX_CHARS`; a longer body is recorded as a `response_preview` with
`response_truncated: true`.

Finished history can be pruned without touching pending work, and a job can be
cancelled or reloaded:

```sh
curl -s -X DELETE localhost:5009/queue/<job_id>        # one finished job (409 while pending)
curl -s -X DELETE localhost:5009/queue                 # every finished job
curl -s -X POST   localhost:5009/queue/<job_id>/cancel # cooperative cancellation
curl -s -X POST   localhost:5009/queue/<job_id>/reload # re-queue the same request
```

Cancellation is **cooperative**: a queued job is cancelled immediately and never
runs, while a running job stops at its next checkpoint (between stages, and on
every frame of a video or annotation loop). Reload rebuilds the original request —
the `media_analysis.id` for the DB-driven mode, or the retained upload for a file
upload — as a **new** job. Retained uploads are bounded by `JOB_REPLAY_RETENTION`
(newest 10 jobs) and `JOB_REPLAY_MAX_BYTES` (1 GiB total), and a job whose input is
gone reports `409`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Dashboard (upload UI, configuration, queue tab) |
| `GET`/`POST` | `/config` | Read/update runtime analysis settings |
| `GET` | `/health` | Health and backend status (exempt from GPU idle sleeping) |
| `GET`/`POST` | `/vision-backend` | Inspect/select the vision backend |
| `GET` | `/reports/<path:filename>` | Serve generated PDF reports |
| `POST` | `/analyze-image` | Vision analysis of an uploaded image *(queued)* |
| `POST` | `/analyze-video` | Vision + audio analysis of an uploaded video *(queued)* |
| `POST` | `/analyze-video-detailed` | Detailed video analysis with tracking and PDF *(queued)* |
| `POST` | `/detect-objects` | YOLO detection, by upload or by `media_analysis.id` *(queued)* |
| `POST` | `/fetch-media` | Pull and analyze media from a remote source *(inline)* |
| `GET` | `/queue` | Queue depth plus a page of jobs |
| `GET` | `/queue/<job_id>` | One job's status, progress, timings and response |
| `DELETE` | `/queue/<job_id>` | Delete one finished job |
| `DELETE` | `/queue` | Clear all finished jobs |
| `POST` | `/queue/<job_id>/cancel` | Cancel a queued or running job |
| `POST` | `/queue/<job_id>/reload` | Re-queue a job from its recorded input |

Swagger UI is served at `/docs/`, and `swagger_docs.py` holds the operation
definitions.

## Dashboard

The dashboard at `http://localhost:5009/` has a tab per upload mode; detailed video
analysis has no tab and runs through the API only. Each tab renders the analytics payload visually by type — KPI tiles, tables, chips,
figures and a progress bar — with the raw JSON in a collapsible section beneath.
The **Queue** tab shows queue depth and a paginated table of jobs (status,
progress, input, timings, HTTP status, error) with a JSON viewer and per-row
cancel / reload / delete, plus *Clear history* and live polling.

## Quick start

Copy `.env.example` to `.env`, fill in the database credentials if the
database-backed route is needed, then:

```sh
docker compose up --build -d
docker compose ps
```

The first start downloads the Qwen3-VL and Qwen2-Audio checkpoints into the
persistent `huggingface-cache` volume, so the health check has a long
`start-period` (10 minutes). Set `PRELOAD_MODELS=1` only when producing a fully
offline deployment image on a build host with ample RAM and disk.

`OPERATIONS.md` covers the GPU idle-sleep behaviour, the stage locks that keep
vision and audio from running together, and the VRAM budget in detail.

## Configuration

Runtime analysis settings are edited from the dashboard (`/config`) and stored in
`config/runtime.json`. The queue and service limits are environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `JOB_DB_PATH` | `/app/data/jobs.sqlite3` | SQLite queue database |
| `JOB_FILES_DIR` | `/app/data/uploads` | Staged uploads retained for reload |
| `JOB_HISTORY_LIMIT` | `200` | Finished jobs kept |
| `JOB_MAX_PENDING` | `32` | Backlog cap before `503` |
| `JOB_RESPONSE_MAX_CHARS` | `262144` | Response body kept in full |
| `JOB_RESPONSE_PREVIEW_CHARS` | `2000` | Preview kept when truncated |
| `JOB_REPLAY_RETENTION` | `10` | Newest jobs whose upload is retained |
| `JOB_REPLAY_MAX_BYTES` | `1073741824` | Total budget for retained uploads |
| `JOB_PAGE_SIZE` | `25` | Default rows per queue page |
| `JOB_PAGE_SIZE_MAX` | `200` | Largest accepted page size |
| `MAX_UPLOAD_BYTES` | `209715200` | Upload size limit |
| `MAX_VIDEO_DURATION_SECONDS` | `300` | Rejection threshold for long video |
| `VLLM_IDLE_SLEEP_SECONDS` | `60` | Idle time before the GPU is released |
| `ENGINE_STARTUP_GRACE_SECONDS` | `360` | How long the API waits for the engine at startup (`0` = serve immediately) |
| `MEDIA_TLS_VERIFY` | `0` | Verify TLS for remote media fetches |
| `LOG_LEVEL` | `INFO` | Log verbosity |

## Tests

```sh
python -m unittest discover -v                            # unit + schema tests
docker compose exec activity-recognition python -m unittest discover -v
```

`test_job_queue.py` covers the queue in isolation. `test_swagger_docs.py`,
`test_response_schema.py` and `test_removed_models.py` are static.

`e2e_live.py` is the end-to-end check and needs a **running service**: it
submits image and video object detection (including an AV1 upload), follows the
jobs through the queue, downloads the annotated clip and asks for a byte range,
round-trips a config value, exercises reload/cancel/delete and confirms the
analysis endpoints report a missing engine. It asserts real outcomes, not
status codes. Point it at a deployment with `E2E_BASE` and a fixture folder with
`E2E_FIXTURES` (a 640x360 frame, a 1280x720 frame and an AV1 clip):

```sh
python3 e2e_live.py            # 19 checks against http://127.0.0.1:5009
```

`check_analytics.py`, `make_harness.py` and `run_harness.py` are development
helpers: the first captures one real response per endpoint from a live service,
the other two render a captured payload through the dashboard's own JavaScript in
headless Chrome.

## Layout

| Path | Role |
| --- | --- |
| `app.py` | Flask application: routes, staging, queue wiring, progress hooks |
| `job_queue.py` | SQLite-backed FIFO queue with one worker thread |
| `swagger_docs.py` | Swagger/OpenAPI operation definitions |
| `engine_power.py` | vLLM idle sleep/wake and stage locks |
| `response_schema.py` | Envelope schema validation and repair |
| `media_utils.py` | Media probing, frame sampling, audio extraction |
| `pdf_report.py` | Detailed report rendering |
| `sftp_upload.py` | SFTP delivery of annotated artefacts |
| `templates/dashboard.html` | Dashboard UI (queue table, analytics renderer) |
