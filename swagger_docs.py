"""OpenAPI operation documents kept separate from request handlers."""

from response_schema import (DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA, IMAGE_ANALYSIS_ENVELOPE_SCHEMA,
    VIDEO_ANALYSIS_ENVELOPE_SCHEMA)

ERROR = {"type": "object", "properties": {"error": {"type": "string"}}}

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
         "description": "Image to detect objects in (multipart mode)"},
        {"name": "id", "in": "body", "required": False,
         "schema": {"type": "object", "properties": {"id": {"type": "integer"}},
                    "description": "media_analysis.id to resolve, annotate, and publish instead of an upload"}}],
    "responses": {"202": {"description": "Job accepted; poll the returned status_url"},
                  "400": {"description": "Invalid image", "schema": ERROR},
                  "413": {"description": "Upload or video exceeds limits", "schema": ERROR},
                  "503": {"description": "The analysis queue is at capacity", "schema": ERROR}},
}
DETECT_OBJECTS_VIDEO_SWAGGER_DOC = {
    "tags": ["Object Detection"],
    "summary": "Queue video object detection and return an annotated clip",
    "description": ("Runs the dedicated YOLO model over every frame with persistent tracking. "
                    "Detections returned in the job response are sampled and bounded so a long clip "
                    "cannot inflate the queue history; every frame is still annotated in the clip "
                    "served by annotated_video_url."),
    "consumes": ["multipart/form-data"],
    "parameters": [{"name": "file", "in": "formData", "type": "file", "required": True,
                    "description": "Video to detect objects in (maximum 5 minutes)"}],
    "responses": {"202": {"description": "Job accepted; poll the returned status_url"},
                  "400": {"description": "Invalid video", "schema": ERROR},
                  "413": {"description": "Upload or video exceeds limits", "schema": ERROR},
                  "503": {"description": "The analysis queue is at capacity", "schema": ERROR}},
}
DOWNLOAD_OBJECT_VIDEO_SWAGGER_DOC = {
    "tags": ["Object Detection"], "summary": "Download an annotated video-detection clip",
    "parameters": [{"name":"filename","in":"path","required":True,"type":"string"}],
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
