"""OpenAPI operation documents kept separate from request handlers."""

from response_schema import (DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA, IMAGE_ANALYSIS_ENVELOPE_SCHEMA,
    VIDEO_ANALYSIS_ENVELOPE_SCHEMA)

DJANGO_VIDEO_OBJECT_DETECTION_TAG_DESCRIPTION = r"""END-TO-END ORDER for a video that the React application uploads and this service detects. Every step states who performs it, so the UI team can stop at step 6 and the Django team knows what it owes.

WHO DOES WHAT
- REACT (UI): step 1 (upload), step 4 (poll), step 6 (render). Nothing else.
- DJANGO: step 2 (accept the upload, create the row, queue this service, answer 201) and step 5 (serve the row back to the UI).
- THIS SERVICE: step 3 (infer, annotate, save) plus the clip download used in step 6.
- POSTGRESQL: public.media_analysis, written by step 3 and read by steps 4-7.

The contract in one sentence: Django answers 201 as soon as the job is queued, this service processes the clip asynchronously and writes the result back to the same media_analysis row, so the UI learns the outcome by polling that row. A 201 never contains detections.

URL MAP, IN ORDER (who calls what)
- [REACT]  POST  https://<django-host>/ai-analysis/media-analysis/upload/  ->  201 with data.id and data.detection
- [DJANGO] POST  http://<this-host>:5009/detect-objects-video  {"id": <row id>}  ->  202 with job_id (timeout 5 s connect / 30 s read)
- [REACT]  GET   https://<django-host>/ai-analysis/media-analysis/retrive/{id}/  ->  status 0 (wait) or 1 (final JSON)
- [REACT]  GET   <row.output_file_path>  ->  the annotated clip (published host, no token)
- [SERVER ONLY] GET  http://<this-host>:5009/queue/{job_id}  ->  progress and failure (no CORS, so not from the browser)

Django makes exactly ONE call to this service in this flow: POST /detect-objects-video. It never polls /queue and never fetches /object-videos; the worker writes the row directly. That job_id is forwarded to React only so a server-side component can poll when it needs explicit failure reporting.

STEP 1 - REACT (UI): upload the video through Django
The UI posts the file to Django, not to this service, and must not wait for detection.

POST https://<django-host>/ai-analysis/media-analysis/upload/
Authorization: Token <django-token>
Content-Type: multipart/form-data
  file=@clip.mp4                required - the video
  category=object recognition   required - selects the detection branch; any other value skips detection
  media_type=video              optional - detected from the file and overwritten
  title=<label>                 optional - shown in the UI list

```js
const form = new FormData();
form.append("file", file);
form.append("category", "object recognition");
form.append("media_type", "video");
const res = await fetch(`${DJANGO}/ai-analysis/media-analysis/upload/`, {
  method: "POST",
  headers: { Authorization: `Token ${token}` },   // do NOT set Content-Type: the browser adds the boundary
  body: form,
});
const body = await res.json();                    // HTTP 201
const id = body.data.id;                          // media_analysis row the result will land on
const jobId = body.data.detection?.job_id;        // undefined => nothing was queued (see step 2)
```

STEP 2 - DJANGO: accept, persist, queue, answer 201
Django does all of this inside the same request; the UI only sees the 201.
1. Validates the token and the file, writes the clip to SFTP at media_analytics/input/video/<uuid>.<ext>.
2. Creates the media_analysis row: status=0, media_type=video, file_path=<SFTP URL>.
3. Calls this service: POST /detect-objects-video {"id": <row id>} (timeout: 5 s connect, 30 s read).
4. Returns 201 with the row id and the job reference:

{"success": true, "message": "File uploaded successfully",
 "data": {"id": 646,
          "file_path": "https://.../media_analytics/input/video/<uuid>.mp4",
          "detection": {"job_id": "5970887b...", "status_url": "/queue/5970887b..."}}}

Two facts the UI must handle:
- data.detection.job_id is the job to poll; data.detection.status_url is RELATIVE to THIS service, not to Django, so it needs this service's base URL.
- If data.detection is absent, this service did NOT accept the job (queue full, rejected clip, or the call timed out). Django still returned 201, the row stays status=0 and nothing will be processed. The UI should show a retry instead of polling forever.

Django's request time covers only the upload and the queue acceptance; detection runs after the 201.

STEP 3 - THIS SERVICE: infer, annotate, save (no client action)
The queue answers 202 immediately, runs one job at a time, and the worker:
1. downloads the clip named by media_analysis.file_path,
2. runs the YOLO detector with persistent tracking over every frame,
3. annotates every frame, transcodes a web-playable clip, publishes it to SFTP,
4. updates the same row: analysis_summary (complete result), response_data (same object, legacy column), output_file_path (published clip URL), status=1.

Measured on this deployment: about 10-16 s for a 10 s 720p clip after the job starts; the five-minute upload ceiling bounds the worst case. Progress is visible only through the job queue (step 4, option B), not through the row.

STEP 4 - REACT (UI): poll until the row is ready
Option A (recommended for the React UI): poll Django - same origin, same token, no CORS problem.

GET https://<django-host>/ai-analysis/media-analysis/retrive/{id}/
Authorization: Token <django-token>

- status == 0 means not finished. status == 1 means the result is on the row.
- Recommended schedule: first poll after 1-2 s, then every 2 s; stop after about 10 minutes and show a failure.
- The row has no progress or error field. A failed job also leaves status=0, so the timeout is what distinguishes "failed" for the UI. If the UI can be told the job id (or a server-side proxy exposes it), option B reports failure explicitly.

```js
const readRow = () => fetch(`${DJANGO}/ai-analysis/media-analysis/retrive/${id}/`,
  { headers: { Authorization: `Token ${token}` } }).then((r) => r.json());

let record = await readRow();
for (let i = 0; record.status !== 1 && i < 300; i++) {   // 300 x 2 s = 10 min ceiling
  await new Promise((r) => setTimeout(r, 2000));
  record = await readRow();
}
if (record.status !== 1) throw new Error("detection did not finish");
```

Option B (server-side or dashboard, when this service is reachable): GET /queue/{job_id}
- No authentication. Reports status, progress (0-100), progress_detail, queue_position, queued_seconds, duration_seconds, total_seconds, http_status, error.
- Poll every 2 s; stop on completed, failed or cancelled. Never poll faster than once a second.
- A poll never changes the job and never writes the row.
- CORS: this service enables CORS only for /analyze-*; a browser page served by Django cannot read /queue cross-origin. Use option A from the UI, or proxy option B on the Django server.

STEP 5 - DJANGO (DATABASE): the result the UI reads
No extra call and no extra work: step 3 already committed the row. Django returns it verbatim through the endpoint in step 4, including analysis_summary, response_data, output_file_path and status. Nothing has to be copied between services.

STEP 6 - REACT (UI): render the detections and the annotated clip
Read from the row returned in step 4:
- analysis_summary.detected_object_counts - {"CLASS": count} of tracked objects, the headline number.
- analysis_summary.tracks[] - track_id, class, first_seen_sec, last_seen_sec, observations, max_confidence.
- analysis_summary.frames_analyzed / frames_annotated / frames_sampled, duration_sec, fps, sample_interval_sec.
- analysis_summary.detections[] - sampled evidence, one entry per sample_interval_sec, bounded by OBJECT_VIDEO_MAX_SAMPLED_FRAMES and OBJECT_VIDEO_MAX_DETECTIONS. detections_truncated / tracks_truncated say when the clip held more than the response reports; the annotated clip is never truncated.
- output_file_path - the published annotated clip (absolute, playable directly, no auth needed). analysis_summary.annotated_video_url is the same clip relative to THIS service: prefix this service's base URL to play it, and add ?download=1 to download it. (annotated_video_download_url is only filled by the direct-upload mode of POST /detect-objects-video, not by this Django-driven flow.)

```js
const summary = record.analysis_summary ?? {};
const counts = summary.detected_object_counts ?? {};
const clip = record.output_file_path ?? null;   // put it in <video src>
```

Clips are retained newest-first: only the newest OBJECT_VIDEO_RETENTION files (default 25) are kept, so an old URL answers 404 even though the row still lists it. Render the counts even when the clip has aged out.

STEP 7 - OTHER CONSUMERS (optional): read the row directly
Any product with database access can read the result without calling either HTTP service:
SELECT status, output_file_path, analysis_summary, response_data
  FROM public.media_analysis WHERE id = <row id>;
status=1 is the completion flag; analysis_summary and response_data carry the same detection object.

FAILURE HANDLING - WHO DOES WHAT
- UI: 201 without data.detection means Django did not queue the job; show "not started" and offer a retry.
- UI: polling timed out with status still 0 means the job failed or was cancelled; the row has no error text, so the message is generic unless the job id is available.
- DJANGO / ops: re-run without a new upload with POST /detect-objects-video {"id": <row id>} on this service, or POST /queue/{job_id}/reload when the job id is known (a new job from the recorded input).
- Failure never writes a partial row: a failed job leaves status=0 and no output_file_path.
- 404 from /queue/{job_id} means the job fell out of the history (JOB_HISTORY_LIMIT, default 200 finished jobs); the row keeps a completed result.

SEQUENCE
```
REACT (UI)          DJANGO                        THIS SERVICE           POSTGRES
   |  POST upload     |                                |                     |
   |----------------->|  INSERT row (status=0)         |                     |
   |                  |----------------------------------------------------->|
   |                  |  POST /detect-objects-video {"id": n}                |
   |                  |------------------------------->|                     |
   |                  |<----------- 202 job_id --------|                     |
   |<---- 201 --------|                                |  detect + annotate  |
   |  GET retrive/{id} every 2 s                      |  UPDATE row ------->|
   |----------------->|  SELECT row ---------------------------------------->|
   |<--- status 0/1 --|                                |                     |
   |  status=1 -> play output_file_path, show counts   |                     |
```

REFERENCE - the two poll targets side by side
- Django row: GET /ai-analysis/media-analysis/retrive/{id}/ - needs Token auth, same origin, status 0/1, full result on the row.
- Job queue: GET /queue/{job_id} - no auth, progress and explicit failure, body capped at 262144 characters (response_truncated=true means read the row for the full JSON).

REFERENCE - server-side curl run (same order, for integrators and tests)
```bash
DJANGO=https://<django-host>; AI=http://<this-host>:5009; TOKEN=<django-token>

# STEP 1-2: React would do this upload through Django
RESP=$(curl -sS -X POST "$DJANGO/ai-analysis/media-analysis/upload/" \
  -H "Authorization: Token $TOKEN" \
  -F file=@clip.mp4 -F category="object recognition")
ID=$(echo "$RESP" | jq -r .data.id)
JOB=$(echo "$RESP" | jq -r .data.detection.job_id)

# STEP 4 option B: poll the job on this service
while :; do
  J=$(curl -sS "$AI/queue/$JOB"); S=$(echo "$J" | jq -r .status)
  echo "$S $(echo "$J" | jq -r '.progress // 0')%"
  case "$S" in completed|failed|cancelled) break;; esac
  sleep 2
done

# STEP 4 option A + STEP 6: the row the React UI polls
curl -sS "$DJANGO/ai-analysis/media-analysis/retrive/$ID/" \
  -H "Authorization: Token $TOKEN" \
  | jq '{status, output_file_path, counts: .analysis_summary.detected_object_counts}'
```

"""

DJANGO_VIDEO_OBJECT_DETECTION_TAG = {
    "name": "Django Video Object Detection",
    "description": DJANGO_VIDEO_OBJECT_DETECTION_TAG_DESCRIPTION.replace("\r\n", "\n"),
}

ERROR = {"type": "object", "properties": {"error": {"type": "string"}}}
QUEUED_JOB_SCHEMA = {"type": "object", "properties": {
    "job_id": {"type": "string"}, "status": {"type": "string", "enum": ["queued"]},
    "status_url": {"type": "string", "description": "Relative queue URL to poll for this job"},
    "seq": {"type": "integer"}, "queue_position": {"type": "integer"},
    "endpoint": {"type": "string"}, "media_type": {"type": "string"},
    "detail": {"type": "string"}}}

HEALTH_SWAGGER_DOC = {
    "tags": ["Health"], "summary": "Service liveness and backend state",
    "responses": {"200": {"description": "Service is accepting requests"},
                  "503": {"description": "Service is unavailable", "schema": ERROR}},
}
HEALTH_COMPONENTS_SWAGGER_DOC = \
{"tags": ["Health"], "summary": "Per-component health for the DevOps dashboard",
 "responses": {
     "200": {"description": "Every component check passed",
             "schema": {"type": "object", "properties": {
                 "status": {"type": "string"}, "total": {"type": "integer"}, "ok_count": {"type": "integer"},
                 "components": {"type": "array", "items": {"type": "object", "properties": {
                     "component": {"type": "string"}, "ok": {"type": "boolean"}, "detail": {}}}}}}},
     "503": {"description": "At least one component failed", "schema": ERROR}}}
GPU_MONITOR_SWAGGER_DOC = \
{"tags": ["Health"], "summary": "Live GPU memory per process and whether the service can still run",
 "description": ("One frame per call: every compute process on the card with its share of the device total, "
                 "and the resource verdict. The engine needs gpu_memory_utilization of the card in addition "
                 "to whatever other processes hold, so the verdict is critical when a restart could not get "
                 "its budget and warning when the margin is thin. The dashboard polls this route."),
 "responses": {
     "200": {"description": "Device totals, per-process usage, and the resource verdict",
             "schema": {"type": "object", "properties": {
                 "available": {"type": "boolean"}, "timestamp": {"type": "string"},
                 "utilization": {"type": "number"},
                 "device": {"type": "object", "properties": {
                     "name": {"type": "string"}, "total_mib": {"type": "integer"},
                     "used_mib": {"type": "integer"}, "free_mib": {"type": "integer"},
                     "used_percent": {"type": "number"}, "free_percent": {"type": "number"}}},
                 "processes": {"type": "array", "items": {"type": "object", "properties": {
                     "pid": {"type": "integer"}, "name": {"type": "string"},
                     "used_mib": {"type": "integer"}, "used_percent": {"type": "number"},
                     "app": {"type": "boolean"}}}},
                 "resource": {"type": "object", "properties": {
                     "ok": {"type": "boolean"}, "status": {"type": "string"}, "detail": {"type": "string"},
                     "required_mib": {"type": "integer"}, "available_mib": {"type": "integer"},
                     "shortfall_mib": {"type": "integer"}, "margin_mib": {"type": "integer"},
                     "required_percent": {"type": "number"}, "audio_ok": {"type": "boolean"}}}}}}}}

UNIT_TESTS_SWAGGER_DOC = \
{"tags": ["Health"], "summary": "Run the unit-test suite and report pass/fail per case",
 "responses": {
     "200": {"description": "All tests passed",
             "schema": {"type": "object", "properties": {
                 "status": {"type": "string"}, "ran": {"type": "integer"}, "passed": {"type": "integer"},
                 "failures": {"type": "integer"}, "errors": {"type": "integer"}, "skipped": {"type": "integer"}}}},
     "500": {"description": "At least one test failed or errored", "schema": ERROR}}}
CREDENTIAL_FIELDS_SCHEMA = {"type": "object", "properties": {
    "DB_HOST": {"type": "string"}, "DB_PORT": {"type": "integer"}, "DB_NAME": {"type": "string"},
    "DB_USER": {"type": "string"}, "DB_PASSWORD": {"type": "string", "description": "write-only; never echoed"},
    "DB_POOL_MAX": {"type": "integer"}, "DB_SSLMODE": {"type": "string"},
    "SFTP_HOST": {"type": "string"}, "SFTP_PORT": {"type": "integer"}, "SFTP_USER": {"type": "string"},
    "SFTP_PASSWORD": {"type": "string", "description": "write-only; alternative to key auth"},
    "SFTP_PRIVATE_KEY_PATH": {"type": "string"}, "SFTP_REMOTE_DIR": {"type": "string"},
    "OUTPUT_PUBLIC_BASE": {"type": "string"}, "OUTPUT_PUBLIC_PORT": {"type": "integer"},
    "OUTPUT_PUBLIC_PATH": {"type": "string"}}}
GET_CREDENTIALS_SWAGGER_DOC = {"tags": ["Configuration"], "summary": "Get provisioned PostgreSQL and SFTP credentials (secrets masked)",
    "responses": {"200": {"description": "Provisioned credential state", "schema": {"type": "object"}}}}
SET_CREDENTIALS_SWAGGER_DOC = {"tags": ["Configuration"], "summary": "Provision PostgreSQL / SFTP credentials at runtime (applied immediately, persisted)",
    "parameters": [{"name": "body", "in": "body", "required": True, "schema": CREDENTIAL_FIELDS_SCHEMA}],
    "responses": {"200": {"description": "Credentials applied and persisted"},
                  "400": {"description": "Unknown or invalid credential fields", "schema": ERROR},
                  "500": {"description": "Applied in memory but not persisted", "schema": ERROR}}}
TEST_CREDENTIALS_SWAGGER_DOC = {"tags": ["Configuration"], "summary": "Live-test the provisioned PostgreSQL and SFTP connections",
    "responses": {"200": {"description": "Both targets reachable (or not configured)"},
                  "503": {"description": "At least one target failed", "schema": ERROR}}}
GET_VISION_BACKEND_SWAGGER_DOC = {
    "tags": ["Backend"], "summary": "Get the active vision and audio models",
    "responses": {"200": {"description": "Backend state", "schema": {"type": "object", "properties": {
        "vision_backend": {"type": "string"}, "vision_model": {"type": "string"},
        "audio_model": {"type": "string"},
        "available_backends": {"type": "array", "items": {"type": "string"}}}}}},
}
SET_VISION_BACKEND_SWAGGER_DOC = {
    "tags": ["Backend"], "summary": "Set the primary vision backend",
    "parameters": [{"name": "body", "in": "body", "required": True, "schema": {"type": "object",
        "required": ["backend"], "properties": {"backend": {"type": "string", "enum": ["vllm"]}}}}],
    "responses": {"200": {"description": "Backend activated"}, "400": {"description": "Invalid backend", "schema": ERROR},
                  "502": {"description": "Backend could not be loaded", "schema": ERROR}},
}

def upload_doc(media):
    # Queued endpoints answer 202 immediately: the analysis runs later and is read
    # back from the job's status_url, so no inference failure is a synchronous code.
    return {"tags": ["Media Analysis"], "summary": f"Queue an uploaded {media}", "consumes": ["multipart/form-data"],
        "parameters": [{"name": "file", "in": "formData", "type": "file", "required": True}],
        "responses": {"202": {"description": "Job accepted; poll the returned status_url"},
                      "400": {"description": f"Invalid {media}", "schema": ERROR},
                      "413": {"description": "Upload or video exceeds limits", "schema": ERROR},
                      "503": {"description": "The analysis queue is at capacity", "schema": ERROR}}}

ANALYZE_IMAGE_UPLOAD_SWAGGER_DOC = upload_doc("image")
ANALYZE_VIDEO_UPLOAD_SWAGGER_DOC = upload_doc("video")
ANALYZE_VIDEO_DETAILED_SWAGGER_DOC = upload_doc("video with temporal classification, object tracking, and audio")
# Attach the enforced response schema so the contract is visible in Swagger.
ANALYZE_IMAGE_UPLOAD_SWAGGER_DOC["responses"]["202"]["schema"] = IMAGE_ANALYSIS_ENVELOPE_SCHEMA
ANALYZE_VIDEO_UPLOAD_SWAGGER_DOC["responses"]["202"]["schema"] = VIDEO_ANALYSIS_ENVELOPE_SCHEMA
ANALYZE_VIDEO_DETAILED_SWAGGER_DOC["responses"]["202"]["schema"] = DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA
FETCH_MEDIA_SWAGGER_DOC = {
    "tags": ["Media Analysis"], "summary": "Analyze media referenced by a database record",
    "parameters": [{"name": "body", "in": "body", "required": True, "schema": {"type": "object", "required": ["id"],
        "properties": {"id": {"type": "integer"}}}}],
    "responses": {"200": {"description": "Analysis completed"}, "400": {"description": "Invalid request", "schema": ERROR},
                  "413": {"description": "Media exceeds limits", "schema": ERROR}, "502": {"description": "Inference failed", "schema": ERROR}},
}
DETECT_OBJECTS_SWAGGER_DOC = {
    "tags": ["Object Detection"], "summary": "Queue object detection for an uploaded image or a media_analysis record",
    "consumes": ["multipart/form-data", "application/json"],
    "parameters": [
        {"name": "file", "in": "formData", "type": "file", "required": False,
         "description": ("Image to detect objects in (multipart mode). An optional id form field "
                         "saves the run on that media_analysis row")},
        {"name": "id", "in": "body", "required": False,
         "schema": {"type": "object", "properties": {"id": {"type": "integer"}},
                    "description": "media_analysis.id to resolve, annotate, and publish instead of an upload"}}],
    "responses": {"202": {"description": "Job accepted; poll the returned status_url"},
                  "400": {"description": "Invalid image", "schema": ERROR},
                  "413": {"description": "Upload or video exceeds limits", "schema": ERROR},
                  "503": {"description": "The analysis queue is at capacity", "schema": ERROR}},
}
DETECT_OBJECTS_VIDEO_SWAGGER_DOC = {
    "tags": ["Django Video Object Detection"],
    "summary": "Queue video object detection for an uploaded clip or a media_analysis record",
    "description": ("Runs the dedicated YOLO model over every frame with persistent tracking. "
                    "Every frame is annotated in the clip served by annotated_video_url; the "
                    "detections returned in the job response are sampled and bounded so a long "
                    "clip cannot inflate the queue history.\n\n"
                    "WHO CALLS IT: Django, inside step 2 of the 'Django Video Object Detection' "
                    "section - the React client never calls this endpoint in that flow. Django "
                    "sends {\"id\": n} and this service performs step 3 (infer, annotate, save). "
                    "Direct calls from a script or the dashboard are supported too.\n\n"
                    "WORKFLOW (one POST is enough to start and persist the run):\n"
                    "1. POST this endpoint. JSON body {\"id\": n} resolves media_analysis.file_path, "
                    "downloads the clip and saves the result on row n. Multipart mode takes file "
                    "(and an optional id form field to save on a row). The response is 202 with a "
                    "job_id and status_url; inference has NOT finished yet.\n"
                    "2. Poll GET /queue/{job_id} (the returned status_url) until status is completed, "
                    "failed or cancelled. Polling only reports the job; it does not write the result.\n"
                    "3. On completed, the worker has ALREADY updated the named media_analysis row: "
                    "analysis_summary and response_data (the full result: detections, tracks, counts, "
                    "timings), output_file_path (published SFTP URL) and status=1. No second API call is "
                    "needed for the database write; if the other application cannot poll the queue, "
                    "it can poll media_analysis.status/analysis_summary for that id instead.\n\n"
                    "SWAGGER UI: choose application/json in 'Parameter content type' to send the id "
                    "body. Multipart/form-data is the file-upload mode and rejects a request without "
                    "a file.\n\n"
                    "The save is best-effort, so a database problem is reported as "
                    "persistence_warning instead of failing a detection that already finished."),
    # JSON first so Swagger UI defaults to the id mode; the file mode is still
    # selectable in Parameter content type.
    "consumes": ["application/json", "multipart/form-data"],
    "parameters": [
        {"name": "file", "in": "formData", "type": "file", "required": False,
         "description": ("Video to detect objects in (maximum 5 minutes, multipart mode). "
                         "Add an optional id form field to save the run on that media_analysis row")},
        {"name": "id", "in": "body", "required": False,
         "schema": {"type": "object", "properties": {"id": {"type": "integer"}},
                    "description": "media_analysis.id to resolve, annotate, and save instead of an upload"}}],
    "responses": {"202": {"description": "Job accepted; inference runs asynchronously. Poll the returned status_url; the row is updated when the job completes.",
                             "schema": QUEUED_JOB_SCHEMA},
                  "400": {"description": "Invalid video", "schema": ERROR},
                  "404": {"description": "No media_analysis record with that id", "schema": ERROR},
                  "413": {"description": "Upload or video exceeds limits", "schema": ERROR},
                  "503": {"description": "The analysis queue is at capacity", "schema": ERROR}},
}
DOWNLOAD_OBJECT_VIDEO_SWAGGER_DOC = {
    "tags": ["Django Video Object Detection"], "summary": "Download an annotated video-detection clip",
    "description": ("Step 6 of the 'Django Video Object Detection' flow. The filename is the "
                    "last segment of annotated_video_url in the job response or analysis_summary "
                    "(annotated_video_download_url, direct-upload mode only, already carries it plus "
                    "?download=1). Only the newest OBJECT_VIDEO_RETENTION clips are "
                    "kept, so an aged-out clip answers 404 even though the row still lists its URL. "
                    "Clients that should not reach this service play the row's output_file_path "
                    "instead; it is the same clip on the published host."),
    "parameters": [{"name":"filename","in":"path","required":True,"type":"string"},
                   {"name":"download","in":"query","required":False,"type":"string","enum":["1"],
                    "description":"Send the clip as an attachment instead of inline"}],
    "produces": ["video/mp4"],
    "responses": {"200":{"description":"Annotated video clip"},
                  "404":{"description":"Clip not found or aged out of retention","schema":ERROR}},
}
DOWNLOAD_REPORT_SWAGGER_DOC = {
    "tags": ["Reports"], "summary": "Download a generated detailed-analysis PDF",
    "parameters": [{"name":"filename","in":"path","required":True,"type":"string"}],
    "produces": ["application/pdf"],
    "responses": {"200":{"description":"PDF report"},"404":{"description":"Report not found","schema":ERROR}},
}
GET_CONFIG_SWAGGER_DOC = {"tags":["Configuration"],"summary":"Get persisted inference configuration",
    "responses":{"200":{"description":"Current configuration"}}}
SET_CONFIG_SWAGGER_DOC = {"tags":["Configuration"],"summary":"Validate, apply, and persist inference configuration",
    "parameters":[{"name":"body","in":"body","required":True,"schema":{"type":"object"}}],
    "responses":{"200":{"description":"Configuration saved"},"400":{"description":"Invalid setting","schema":ERROR}}}
JOB_STATUS_VALUES = ["queued", "running", "completed", "failed", "cancelled"]
GET_QUEUE_SWAGGER_DOC = {
    "tags": ["Queue"], "summary": "Queue depth and recent job history",
    "description": ("Heavy requests run one at a time in arrival order. Newest jobs first. Responses are "
                    "omitted unless include_response=1, because the dashboard polls this route; fetch one job for its body."),
    "parameters": [
        {"name": "page", "in": "query", "type": "integer", "required": False, "default": 1,
         "description": "Page number, newest jobs first"},
        {"name": "page_size", "in": "query", "type": "integer", "required": False, "default": 25,
         "description": "Jobs per page (1-200)"},
        {"name": "status", "in": "query", "type": "string", "required": False, "enum": JOB_STATUS_VALUES,
         "description": "Only return jobs in this state"},
        {"name": "include_response", "in": "query", "type": "string", "required": False, "enum": ["0", "1"],
         "description": "Embed each job's recorded response instead of only its summary"}],
    "responses": {"200": {"description": "Queue summary, a page of jobs, and the pagination state"},
                  "400": {"description": "Invalid page, page_size, or status filter", "schema": ERROR}},
}
GET_QUEUE_JOB_SWAGGER_DOC = {
    "tags": ["Queue"], "summary": "Status, response, progress, and timings of one job",
    "description": ("Poll the status_url returned by a queued request. Recommended schedule: first "
                    "poll after 1-2 s, then every 2 s, backing off to 5 s after about a minute; stop on "
                    "completed, failed or cancelled. A poll never changes the job, so do not poll faster "
                    "than once a second.\n\n"
                    "Fields that answer 'how long?': queue_position (jobs ahead, including the running "
                    "one), queued_seconds (wait before work started), duration_seconds (work), "
                    "total_seconds (submission to result), progress and progress_detail (current phase).\n\n"
                    "For /detect-objects-video and /detect-objects runs that named a media_analysis.id, "
                    "a completed job means that row has already been updated (analysis_summary, "
                    "response_data, output_file_path, status=1); this GET does not perform the save. A "
                    "failed job leaves the row unchanged (status remains 0) and the reason is in error. "
                    "The React UI normally polls the Django row instead, because this route has no CORS.\n\n"
                    "The stored body is capped at 262144 characters: when response_truncated is true, "
                    "response is null and only response_preview is returned - read the row for the "
                    "complete result."),
    "parameters": [{"name": "job_id", "in": "path", "required": True, "type": "string"}],
    "responses": {"200": {"description": "Job record, including its response once finished"},
                  "404": {"description": "No such job", "schema": ERROR}},
}
DELETE_QUEUE_JOB_SWAGGER_DOC = {
    "tags": ["Queue"], "summary": "Delete one finished job from the history",
    "description": "Only completed and failed jobs can be removed; a queued or running job is refused.",
    "parameters": [{"name": "job_id", "in": "path", "required": True, "type": "string"}],
    "responses": {"200": {"description": "Job deleted"},
                  "404": {"description": "No such job", "schema": ERROR},
                  "409": {"description": "The job is still queued or running", "schema": ERROR}},
}
CLEAR_QUEUE_SWAGGER_DOC = {
    "tags": ["Queue"], "summary": "Clear the finished job history",
    "description": ("Deletes every completed, failed and cancelled job. Queued and running jobs are "
                    "never removed, so clearing history cannot cancel work in progress."),
    "responses": {"200": {"description": "How many jobs were deleted, with the remaining queue state"}},
}
CANCEL_QUEUE_JOB_SWAGGER_DOC = {
    "tags": ["Queue"], "summary": "Cancel a queued or running job",
    "description": ("A still-queued job is marked cancelled immediately. A running job stops at its "
                    "next checkpoint, so a step already inside a single model call finishes first."),
    "parameters": [{"name": "job_id", "in": "path", "required": True, "type": "string"}],
    "responses": {"200": {"description": "Job cancelled; the updated job record is returned"},
                  "404": {"description": "No such job", "schema": ERROR},
                  "409": {"description": "The job has already finished", "schema": ERROR}},
}
RELOAD_QUEUE_JOB_SWAGGER_DOC = {
    "tags": ["Queue"], "summary": "Queue the same request again from a recorded job",
    "description": ("Rebuilds the original request from what the job stored -- the media_analysis id "
                    "for the DB-driven mode, or the retained upload for a file upload -- and queues it "
                    "as a new job. Retained uploads are bounded, so an old one may no longer be replayable."),
    "parameters": [{"name": "job_id", "in": "path", "required": True, "type": "string"}],
    "responses": {"202": {"description": "Replay accepted; poll the returned status_url"},
                  "404": {"description": "No such job", "schema": ERROR},
                  "409": {"description": "The job's input is no longer available", "schema": ERROR},
                  "503": {"description": "The analysis queue is at capacity", "schema": ERROR}},
}
