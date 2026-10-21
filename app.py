"""Single-container activity recognition: Qwen3-VL-8B via vLLM (vision),
Qwen2.5-Omni-3B (audio) and YOLO (objects). Vision and audio never run together.

Heavy requests are serialised through an internal FIFO job queue (see
job_queue.py) so a burst of callers cannot contend for the single GPU: they are
answered with 202 + a job id and polled through /queue.
"""
from __future__ import annotations

import base64, gc, json, logging, os, signal, subprocess, sys, tempfile, threading, time, unittest as unittest_result, uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from decord import VideoReader, cpu
from flask import Flask, g, jsonify, make_response, render_template, request, send_from_directory
from flask_cors import CORS
from flasgger import Swagger, swag_from
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from audio_analysis import (AUDIO_STAGE_LOCK, QWEN_AUDIO_MODEL_ID, analyze_video_audio,
    cadence_activity_bias, release_qwen_audio, sound_event_cadence_bias)
from engine_power import EnginePowerManager
from gpu_monitor import build_snapshot as gpu_snapshot
from gpu_monitor import engine_memory_mib, find_engine_server_pid
import audio_analysis
from job_queue import (JobCancelled, JobQueue, JobQueueFull, STATUSES as QUEUE_STATUSES,
    report_progress)
from pdf_report import create_video_analysis_pdf
from media_utils import ACTIVITY_LABELS, ACTIVITY_PROMPT as PROMPT
from media_utils import CONFIDENCE_LEVELS, MediaProbeError, VideoTooLongError, analyze_motion_burstiness, build_multimodal_summary_prompt, build_video_activity_prompt, compute_frame_change_scores, confidence_label, motion_activity_bias, normalize_activity_result, normalize_confidence, parse_json_text, validate_video_duration
from swagger_docs import (ANALYZE_IMAGE_UPLOAD_SWAGGER_DOC, ANALYZE_VIDEO_UPLOAD_SWAGGER_DOC,
    ANALYZE_VIDEO_DETAILED_SWAGGER_DOC,
    DOWNLOAD_REPORT_SWAGGER_DOC, DOWNLOAD_OBJECT_VIDEO_SWAGGER_DOC,
    DETECT_OBJECTS_VIDEO_SWAGGER_DOC,
    GET_CONFIG_SWAGGER_DOC, SET_CONFIG_SWAGGER_DOC,
    DETECT_OBJECTS_SWAGGER_DOC, FETCH_MEDIA_SWAGGER_DOC, GET_VISION_BACKEND_SWAGGER_DOC,
    GET_QUEUE_SWAGGER_DOC, GET_QUEUE_JOB_SWAGGER_DOC, GPU_MONITOR_SWAGGER_DOC,
    DELETE_QUEUE_JOB_SWAGGER_DOC, CLEAR_QUEUE_SWAGGER_DOC,
    CANCEL_QUEUE_JOB_SWAGGER_DOC, RELOAD_QUEUE_JOB_SWAGGER_DOC,
    HEALTH_SWAGGER_DOC, SET_VISION_BACKEND_SWAGGER_DOC,
    HEALTH_COMPONENTS_SWAGGER_DOC, UNIT_TESTS_SWAGGER_DOC,
    GET_CREDENTIALS_SWAGGER_DOC, SET_CREDENTIALS_SWAGGER_DOC, TEST_CREDENTIALS_SWAGGER_DOC)
from response_schema import (DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA, IMAGE_ANALYSIS_ENVELOPE_SCHEMA,
    VIDEO_ANALYSIS_ENVELOPE_SCHEMA, validate_and_repair)
from sftp_upload import SftpUploadError, publish
from video_object_detection import annotate_with_detections
from vision_backend import (SUPPORTED_VISION_BACKENDS, VisionBackendError,
    analyze_image_with_vllm, check_vllm_server, complete_text_with_vllm, get_vision_backend,
    engine_is_sleeping, sleep_vllm, wake_vllm)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.getenv("RUNTIME_CONFIG_PATH", str(BASE_DIR/"config"/"runtime.json")))
if CONFIG_PATH.exists():
    try:
        for _key, _value in json.loads(CONFIG_PATH.read_text(encoding="utf-8")).items(): os.environ[str(_key)] = str(_value)
    except Exception as _config_exc:
        logging.getLogger("activity-recognition").warning("Could not load runtime config: %s", _config_exc)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("activity-recognition")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# The shared media/file server presents a self-signed certificate. The legacy
# service downloaded media with TLS verification disabled, so that stays the
# default here; set MEDIA_TLS_VERIFY=1 to enforce verification (and point
# REQUESTS_CA_BUNDLE at the server's CA) once the certificate is trusted.
MEDIA_TLS_VERIFY = os.getenv("MEDIA_TLS_VERIFY", "0").strip().lower() not in {"", "0", "false", "no"}
if not MEDIA_TLS_VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
MAX_VIDEO_DURATION_SECONDS = float(os.getenv("MAX_VIDEO_DURATION_SECONDS", "300"))
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "24"))
FRAME_INTERVAL_SECONDS = float(os.getenv("FRAME_INTERVAL_SECONDS", "2"))
VLLM_VISION_HOST = os.getenv("VLLM_VISION_HOST", "http://127.0.0.1:8001")
VLLM_VISION_MODEL = os.getenv("VLLM_VISION_MODEL", "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit")

class UploadTooLargeError(ValueError): pass

class VisionManager:
    """Thread-safe client for the single in-container vLLM engine (no fallback)."""
    def __init__(self):
        self.primary = get_vision_backend()
        self.device = "cuda" if torch.cuda.is_available() else "unavailable"
        self.lock = threading.RLock()
        self.last_inference_ms = None

    @staticmethod
    def _normalize(text):
        parsed = parse_json_text(text)
        if "error" in parsed:
            # Keep a short slice of the raw answer: without it a malformed frame
            # is impossible to diagnose from the API response alone.
            raise VisionBackendError(
                f"Model returned invalid JSON: {parsed['error']} :: raw={str(parsed.get('raw'))[:200]}")
        return normalize_activity_result(parsed)

    def _vllm(self, path):
        """Thermal-efficient default: the resident vLLM server (Qwen3-VL-8B)."""
        return self._normalize(analyze_image_with_vllm(path, PROMPT, host=VLLM_VISION_HOST,
            model=VLLM_VISION_MODEL, timeout=float(os.getenv("VLLM_VISION_TIMEOUT_SECONDS", "180"))))

    def analyze(self, path):
        """Classify one image with the resident vLLM engine. There is no fallback:
        a failure here is reported to the caller instead of silently degrading."""
        started = time.perf_counter()
        with self.lock:
            engine_power.ensure_awake()  # restore the engine if it was parked while idle
            result, used = self._vllm(path), "vllm"
        self.last_inference_ms = round((time.perf_counter()-started)*1000, 1)
        return result, used

    def set_primary(self, backend):
        self.primary = get_vision_backend(backend)

vision = VisionManager()
# Idle power management: after VLLM_IDLE_SLEEP_SECONDS without a request the
# engine is parked (weights offloaded to CPU, VRAM released) and the audio model
# is unloaded, so other containers can use the GPU. The next request wakes it.
def _engine_ready(host=None) -> bool:
    """Wait until the vLLM engine answers, or give up after the timeout.

    Used before waking the engine: a recycle (see below) stops it and the
    entrypoint starts it again a few seconds later, so for a short window the
    engine is loading rather than parked. Waking a starting engine would fail
    the request. The engine answers /v1/models while it is parked, so this
    returns at once in the normal idle case.
    """
    base = host or VLLM_VISION_HOST
    deadline = time.monotonic() + float(os.getenv("VLLM_ENGINE_READY_TIMEOUT_SECONDS", "180"))
    while True:
        if check_vllm_server(host=base, model=VLLM_VISION_MODEL, timeout=3).get("reachable"):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(2)

def _stop_idle_engine(host=None) -> bool:
    """Stop the supervised vLLM server so an idle card can be released fully.

    Parking only returns what vLLM tracks, so an engine that has served for a
    while keeps several GB (measured on this deployment: 5.7 GB after a day of
    jobs against 0.66 GB for a fresh engine). The EngineCore child holds it and
    dies with the server; the entrypoint's supervisor starts a fresh one.
    """
    pid = find_engine_server_pid()
    if pid is None:
        logger.warning("Could not find the vLLM server process, so it cannot be recycled")
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as exc:
        logger.warning("Could not stop the vLLM server (pid %s): %s", pid, exc)
        return False
    logger.info("Stopped the idle vLLM engine (pid %s); the entrypoint restarts it", pid)
    return True

def _release_object_detector() -> None:
    """Drop the object detector when the service goes idle.

    The audio models are released on every park, because a parked engine is
    woken again for the next request; the detector is only worth keeping while
    requests are being served and is reloaded lazily. ``detector`` is created
    further down this module and is resolved when the idle path runs.
    """
    detector.release()

engine_power = EnginePowerManager(sleep_vllm, wake_vllm, release_qwen_audio, host=VLLM_VISION_HOST,
    idle_seconds=float(os.getenv("VLLM_IDLE_SLEEP_SECONDS", "60")),
    release_idle_fn=_release_object_detector, probe_fn=engine_is_sleeping,
    recycle_after_idle_seconds=float(os.getenv("VLLM_RECYCLE_AFTER_IDLE_SECONDS", "300")),
    recycle_vram_mib=int(os.getenv("VLLM_RECYCLE_VRAM_MIB", "1536")),
    engine_vram_fn=engine_memory_mib, recycle_fn=_stop_idle_engine, ready_fn=_engine_ready)
# Heavy endpoints enqueue their work here rather than running it inline: a burst
# of concurrent requests would otherwise decode frames and run inference at the
# same time on one GPU and freeze the host. Jobs run strictly one at a time in
# arrival order and their status/response/timings are kept in SQLite, so the
# queue can be inspected (and survives a restart) without holding a request open.
# Queue listing page sizes: the default the UI asks for and the ceiling the API
# will serve, so a hand-written ?page_size=100000 cannot scan the whole history.
QUEUE_PAGE_SIZE = int(os.getenv("JOB_PAGE_SIZE", "25"))
QUEUE_PAGE_SIZE_MAX = int(os.getenv("JOB_PAGE_SIZE_MAX", "200"))
JOBS = JobQueue(db_path=os.getenv("JOB_DB_PATH", str(BASE_DIR/"data"/"jobs.sqlite3")),
    files_dir=os.getenv("JOB_FILES_DIR", str(BASE_DIR/"data"/"uploads")),
    history_limit=int(os.getenv("JOB_HISTORY_LIMIT", "200")),
    max_pending=int(os.getenv("JOB_MAX_PENDING", "32")),
    response_max_chars=int(os.getenv("JOB_RESPONSE_MAX_CHARS", "262144")),
    preview_chars=int(os.getenv("JOB_RESPONSE_PREVIEW_CHARS", "2000")),
    # Retained inputs are what makes a job reloadable, and they are bounded: the
    # newest few uploads, never more than this many bytes in total.
    replay_retention=int(os.getenv("JOB_REPLAY_RETENTION", "10")),
    replay_max_bytes=int(os.getenv("JOB_REPLAY_MAX_BYTES", str(1024 * 1024 * 1024))))
_db_pool, _db_lock = None, threading.Lock()

# External service credentials (PostgreSQL + SFTP delivery target). These are
# provisionable at runtime: set through POST /config/credentials, persisted to
# config/runtime.json, and applied to os.environ immediately. get_db_pool() and
# sftp_upload read the environment per call, so a provisioned change takes
# effect without a restart; only the DB connection pool must be reset.
PROVISION_FIELDS = {
    "DB_HOST": str, "DB_PORT": int, "DB_NAME": str, "DB_USER": str, "DB_PASSWORD": str,
    "DB_POOL_MAX": int, "DB_SSLMODE": str,
    "SFTP_HOST": str, "SFTP_PORT": int, "SFTP_USER": str, "SFTP_PASSWORD": str,
    "SFTP_PRIVATE_KEY_PATH": str, "SFTP_REMOTE_DIR": str,
    "OUTPUT_PUBLIC_BASE": str, "OUTPUT_PUBLIC_PORT": int, "OUTPUT_PUBLIC_PATH": str,
}
# Values that must never be echoed back by GET endpoints.
PROVISION_SECRETS = {"DB_PASSWORD", "SFTP_PASSWORD"}

def reset_db_pool() -> None:
    """Drop the cached psycopg2 pool so the next call re-reads the credentials."""
    global _db_pool
    with _db_lock:
        if _db_pool is not None:
            try: _db_pool.closeall()
            except Exception: logger.warning("Closing the old DB pool failed; continuing", exc_info=True)
            _db_pool = None

def object_video_dir() -> Path:
    """Directory holding annotated video-detection artefacts.

    It lives under REPORTS_DIR by default, so the Docker reports volume keeps the
    clips downloadable after the job that produced them has finished.
    """
    return Path(os.getenv("OBJECT_VIDEO_OUTPUT_DIR",
        str(Path(os.getenv("REPORTS_DIR", str(BASE_DIR/"reports")))/"object-videos")))

def _transcode_for_web(path: Path) -> tuple[Path, str]:
    """Re-encode the annotated clip as H.264 so browsers can play it inline.

    OpenCV only writes the MPEG-4 Part 2 ``mp4v`` profile in this image, which
    Chrome/Edge cannot decode. ffmpeg is already installed for the audio stage,
    so the clip is transcoded once before it is advertised. A transcode failure
    is non-fatal: the mp4v original is still downloadable, just not playable in
    every browser.
    """
    if os.getenv("OBJECT_VIDEO_TRANSCODE", "1").strip().lower() in {"0", "false", "no"}:
        return path, "mp4v"
    target = path.with_name(f"{path.stem}_h264.mp4")
    try:
        completed = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
            "-i", str(path), "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(target)], text=True, capture_output=True, timeout=900, check=False)
        if completed.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            detail = (completed.stderr or completed.stdout or "ffmpeg produced no output").strip()
            logger.warning("Could not transcode %s for web playback: %s", path, detail[-500:])
            target.unlink(missing_ok=True)
            return path, "mp4v"
    except Exception as exc:
        logger.warning("Could not transcode %s for web playback: %s", path, exc)
        target.unlink(missing_ok=True)
        return path, "mp4v"
    path.unlink(missing_ok=True)
    return target, "h264"

def _prune_object_videos(directory: Path, keep: int | None = None) -> None:
    """Keep only the newest annotated clips so disk use stays bounded.

    Keeping the newest N is deliberate: the queue holds downloadable URLs for
    recent jobs, and an old clip that has aged out is reported as a 404 rather
    than growing the reports volume without limit.
    """
    retention = int(os.getenv("OBJECT_VIDEO_RETENTION", "25")) if keep is None else int(keep)
    try:
        clips = sorted(directory.glob("detected_*.mp4"),
            key=lambda path: path.stat().st_mtime, reverse=True)
        for stale in clips[max(1, retention):]:
            stale.unlink(missing_ok=True)
    except OSError:
        # Pruning is housekeeping: a host permission problem must not turn a
        # completed detection into a failed job.
        logger.warning("Could not prune annotated video clips in %s", directory, exc_info=True)

class ObjectDetector:
    """Lazy YOLO owner; keeps weights resident and avoids slowing API startup."""
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self.path = Path(os.getenv("OBJECT_MODEL_PATH", str(BASE_DIR/"object_detect_model_256_class.pt")))

    def release(self):
        """Drop the detector so an idle service holds no model VRAM.

        It is only kept warm while requests are being served; the next
        detection loads it again lazily.
        """
        with self.lock:
            model, self.model = self.model, None
        if model is None:
            return
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("CUDA cache scrub after releasing the object detector failed", exc_info=True)

    def detect(self, path):
        with self.lock:
            if self.model is None:
                if not self.path.exists(): raise VisionBackendError(f"Object model not found: {self.path}")
                from ultralytics import YOLO
                self.model = YOLO(str(self.path))
            results = self.model(path, conf=float(os.getenv("OBJECT_CONFIDENCE","0.25")),
                imgsz=int(os.getenv("OBJECT_IMAGE_SIZE","640")), verbose=False)
        result = results[0]
        rendered = result.plot()
        import cv2
        ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
        detections = []
        for box, cls, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            detections.append({"class": result.names[int(cls)], "confidence": round(float(confidence),4),
                "bbox": {"x1":round(box[0],2),"y1":round(box[1],2),"x2":round(box[2],2),"y2":round(box[3],2)}})
        counts = {}
        for item in detections: counts[item["class"]] = counts.get(item["class"],0)+1
        return {"detections":detections,"detected_object_counts":counts,
            "annotated_image_base64":base64.b64encode(encoded).decode("ascii") if ok else None}

    def detect_video(self, source_path):
        """Detect and track objects in every frame, returning a downloadable clip.

        This is the video counterpart of :meth:`detect`: the upload API needs a
        result for a clip it owns, so the annotated artefact is written to the
        reports volume instead of being published over SFTP. Only a bounded
        sample of per-frame detections is returned; every frame is still
        annotated, and the queue response stays small enough for SQLite.
        """
        directory = object_video_dir()
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory/f"detected_{uuid.uuid4().hex}.mp4"
        with self.lock:
            if self.model is None:
                if not self.path.exists(): raise VisionBackendError(f"Object model not found: {self.path}")
                from ultralytics import YOLO
                self.model = YOLO(str(self.path))
            # A fresh predictor prevents track ids leaking from a previous request.
            self.model.predictor = None
            try:
                result = annotate_with_detections(self.model, source_path, artifact,
                    progress=report_progress,
                    sample_interval_seconds=max(0.0, float(os.getenv("OBJECT_VIDEO_SAMPLE_INTERVAL_SECONDS", "1"))),
                    max_sampled_frames=max(1, int(os.getenv("OBJECT_VIDEO_MAX_SAMPLED_FRAMES", "30"))),
                    max_detections=max(1, int(os.getenv("OBJECT_VIDEO_MAX_DETECTIONS", "300"))),
                    max_tracks=max(1, int(os.getenv("OBJECT_VIDEO_MAX_TRACKS", "500"))),
                    confidence=float(os.getenv("OBJECT_CONFIDENCE", "0.25")),
                    image_size=int(os.getenv("OBJECT_IMAGE_SIZE", "640")),
                    device=int(os.getenv("OBJECT_DEVICE", "0")))
            finally:
                # Video inference is the memory peak of this path; release the
                # tracker/weights rather than holding them between requests.
                self.model = None
                gc.collect(); torch.cuda.empty_cache()
        report_progress(92, "encoding web-ready clip")
        artifact, codec = _transcode_for_web(artifact)
        _prune_object_videos(directory)
        result["annotated_video_path"] = str(artifact)
        result["annotated_video_url"] = f"/object-videos/{artifact.name}"
        result["annotated_video_codec"] = codec
        result["annotated_video_filename"] = artifact.name
        try:
            result["annotated_video_size_bytes"] = artifact.stat().st_size
        except OSError:
            result["annotated_video_size_bytes"] = None
        return result

    def track_frames(self, paths, timestamps):
        """Track objects across sampled frames and return compact trajectories."""
        tracks, observations = {}, []
        with self.lock:
            if self.model is None:
                if not self.path.exists(): raise VisionBackendError(f"Object model not found: {self.path}")
                from ultralytics import YOLO
                self.model = YOLO(str(self.path))
            # A fresh predictor prevents IDs leaking from a previous request.
            self.model.predictor = None
            for frame_index, (path, timestamp) in enumerate(zip(paths, timestamps)):
                result = self.model.track(path, persist=True, tracker="bytetrack.yaml",
                    conf=float(os.getenv("OBJECT_CONFIDENCE","0.25")),
                    imgsz=int(os.getenv("OBJECT_IMAGE_SIZE","640")), device=0, verbose=False)[0]
                frame_items = []
                if result.boxes is not None and len(result.boxes):
                    ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(result.boxes)
                    for track_id, cls, confidence, box in zip(ids, result.boxes.cls.cpu().tolist(),
                            result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist()):
                        label = result.names[int(cls)]
                        item = {"track_id":int(track_id), "class":label,
                            "confidence":round(float(confidence),4),
                            "bbox":[round(float(value),2) for value in box]}
                        frame_items.append(item)
                        if track_id >= 0:
                            key = str(track_id)
                            track = tracks.setdefault(key, {"track_id":int(track_id), "class":label,
                                "first_seen_sec":timestamp, "last_seen_sec":timestamp,
                                "observations":0, "max_confidence":0.0})
                            track["last_seen_sec"], track["observations"] = timestamp, track["observations"] + 1
                            track["max_confidence"] = max(track["max_confidence"], round(float(confidence),4))
                observations.append({"frame_index":frame_index,"timestamp_sec":timestamp,"objects":frame_items})
        # Tracking weights are cheap to reload and must not compete with later audio stages.
        with self.lock: self.model = None
        gc.collect(); torch.cuda.empty_cache()
        return {"tracks":list(tracks.values()),"frame_observations":observations}

    def annotate_video(self, source_path, destination_path):
        """Write an overlay-annotated copy of a video for DB delivery.

        One output frame per input frame keeps the artefact's timing identical to
        the source. Nothing is truncated here: callers enforce the duration limit
        (MAX_VIDEO_DURATION_SECONDS, 5 minutes) before calling, so the bound on
        this work is video length in seconds rather than a frame count.
        """
        import cv2
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened(): raise MediaProbeError("Video contains no decodable frames")
        writer, annotated = None, 0
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(destination_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened(): raise VisionBackendError("Could not open the annotated video writer")
            with self.lock:
                if self.model is None:
                    if not self.path.exists(): raise VisionBackendError(f"Object model not found: {self.path}")
                    from ultralytics import YOLO
                    self.model = YOLO(str(self.path))
                try:
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    while True:
                        ok, frame = capture.read()
                        if not ok: break
                        writer.write(self.model(frame, conf=float(os.getenv("OBJECT_CONFIDENCE","0.25")),
                            imgsz=int(os.getenv("OBJECT_IMAGE_SIZE","640")), verbose=False)[0].plot())
                        annotated += 1
                        # The frame loop is the whole cost of annotating, so it is
                        # what the queue's progress bar follows.
                        if total_frames > 0:
                            report_progress(20 + 70*annotated/total_frames,
                                f"annotating frame {annotated}/{total_frames}")
                finally:
                    if writer: writer.release()
        finally:
            capture.release()
        return {"frames_annotated":annotated,"annotated_video_path":str(destination_path)}

detector = ObjectDetector()
_detailed_video_lock = threading.Lock()

def _run_audio_stage(video_path, audio):
    """Run the audio stage strictly after all vLLM vision work.

    Vision and audio share one (already partly occupied) GPU and cannot both be
    resident, so the engine is parked for the audio stage and restored after.
    This runs under AUDIO_STAGE_LOCK and the vision lock, so a concurrent
    request cannot start vLLM frame inference while the audio model is generating.
    """
    with AUDIO_STAGE_LOCK, vision.lock:
        try:
            # Park the engine so the audio model gets the whole card; the audio
            # model is released and the engine restored in the finally block.
            engine_power.park()
            handle, output_path = tempfile.mkstemp(prefix="audio_result_", suffix=".json")
            os.close(handle)
            try:
                completed = subprocess.run(
                    [sys.executable, "/app/audio_worker.py", str(video_path), output_path],
                    text=True, capture_output=True, timeout=600, check=False)
                if completed.returncode != 0:
                    detail = completed.stderr.strip() or completed.stdout.strip() or "audio worker failed"
                    raise RuntimeError(detail[-2000:])
                return json.loads(Path(output_path).read_text(encoding="utf-8"))
            finally:
                Path(output_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.exception("Audio analysis failed")
            failed = dict(audio)
            failed["warnings"] = list(failed.get("warnings", [])) + [str(exc)]
            return failed
        finally:
            engine_power.restore()

def _analyze_frame_resilient(frame, index):
    """Classify one frame, retrying once before giving up on it.

    A single frame the model answers with malformed JSON must not abort the
    whole video: the frame is retried once and, if it still fails, skipped and
    reported as a warning so the remaining evidence is still analysed. Returns
    (result, backend, warning); result is None when the frame was skipped.
    """
    error = None
    for _ in range(2):
        try:
            result, backend = vision.analyze(frame)
            return result, backend, None
        except VisionBackendError as exc:
            error = exc
    return None, None, f"frame {index} skipped after retry: {error}"

def get_db_pool():
    global _db_pool
    if _db_pool is None:
        with _db_lock:
            if _db_pool is None:
                from psycopg2.pool import ThreadedConnectionPool
                params = dict(dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
                    host=os.environ["DB_HOST"], port=int(os.getenv("DB_PORT", "5432")), connect_timeout=5)
                # A password is optional (trust/peer auth on private networks); only
                # pass one when provisioned so an empty string never overrides the
                # libpg default lookup.
                if os.environ.get("DB_PASSWORD"):
                    params["password"] = os.environ["DB_PASSWORD"]
                if os.getenv("DB_SSLMODE"):
                    params["sslmode"] = os.getenv("DB_SSLMODE")
                _db_pool = ThreadedConnectionPool(1, int(os.getenv("DB_POOL_MAX", "10")), **params)
    return _db_pool

@contextmanager
def db_cursor():
    pool, conn = get_db_pool(), None
    conn = pool.getconn()
    try:
        with conn.cursor() as cursor: yield conn, cursor
    except Exception:
        conn.rollback(); raise
    finally: pool.putconn(conn)

def save_upload(directory=None):
    upload = request.files.get("file")
    if not upload or not upload.filename: raise ValueError("Missing non-empty 'file' field")
    if request.content_length and request.content_length > MAX_UPLOAD_BYTES: raise UploadTooLargeError("Upload is too large")
    handle, path = tempfile.mkstemp(prefix="activity_", suffix=Path(upload.filename).suffix.lower(), dir=directory); os.close(handle)
    upload.save(path)
    if Path(path).stat().st_size > MAX_UPLOAD_BYTES:
        Path(path).unlink(missing_ok=True); raise UploadTooLargeError("Upload is too large")
    return path, Path(upload.filename).name

def _decode_frames_to_disk(reader, indices, prefix="frame_", quality=85):
    """Decode the given frame indices in bounded batches and save them as JPEGs.

    Batching keeps peak RAM flat even when the frame budget is unlimited and
    the video is long.
    """
    batch_size = max(1, int(os.getenv("DECORD_BATCH_FRAMES", "48")))
    paths = []
    for start in range(0, len(indices), batch_size):
        for frame in reader.get_batch([int(i) for i in indices[start:start + batch_size]]).asnumpy():
            handle, frame_path = tempfile.mkstemp(prefix=prefix, suffix=".jpg"); os.close(handle)
            Image.fromarray(frame).save(frame_path, quality=quality, optimize=True); paths.append(frame_path)
    return paths

def sample_video_frames(path):
    reader = VideoReader(path, ctx=cpu(0), num_threads=max(1, int(os.getenv("DECORD_THREADS","2"))))
    count, fps = len(reader), float(reader.get_avg_fps())
    if count == 0 or fps <= 0: raise MediaProbeError("Video contains no decodable frames")
    max_frames = int(os.getenv("MAX_VIDEO_FRAMES", str(MAX_VIDEO_FRAMES)))
    interval = float(os.getenv("FRAME_INTERVAL_SECONDS", str(FRAME_INTERVAL_SECONDS)))
    desired = max(1, int(np.ceil((count/fps)/interval)))
    if max_frames > 0: desired = min(max_frames, desired)  # 0 or negative = unlimited
    indices = np.unique(np.linspace(0, count-1, desired, dtype=np.int64))
    paths = _decode_frames_to_disk(reader, indices)
    return paths, [round(float(i)/fps,2) for i in indices]

def sample_detailed_video_frames(path):
    """Combine uniform coverage with high-change frames that may contain brief events."""
    reader = VideoReader(path, ctx=cpu(0), num_threads=max(1, int(os.getenv("DECORD_THREADS","2"))))
    count, fps = len(reader), float(reader.get_avg_fps())
    if count == 0 or fps <= 0: raise MediaProbeError("Video contains no decodable frames")
    duration = count / fps
    max_frames = int(os.getenv("MAX_VIDEO_FRAMES", str(MAX_VIDEO_FRAMES)))
    interval = float(os.getenv("FRAME_INTERVAL_SECONDS", str(FRAME_INTERVAL_SECONDS)))
    uniform_count = max(1, int(np.ceil(duration / interval)))
    if max_frames > 0: uniform_count = min(max_frames, uniform_count)  # 0 or negative = unlimited
    uniform = np.unique(np.linspace(0, count - 1, uniform_count, dtype=np.int64))
    candidate_count = min(count, max(uniform_count * 4, 12))
    candidates = np.unique(np.linspace(0, count - 1, candidate_count, dtype=np.int64))
    scores = []
    previous = None
    candidate_batch = max(1, int(os.getenv("DECORD_BATCH_FRAMES", "48")))
    for start in range(0, len(candidates), candidate_batch):
        candidate_frames = reader.get_batch([int(i) for i in candidates[start:start + candidate_batch]]).asnumpy()
        for index, frame in zip(candidates[start:start + candidate_batch], candidate_frames):
            gray = np.asarray(Image.fromarray(frame).resize((96, 54)).convert("L"), dtype=np.float32)
            score = 0.0 if previous is None else float(np.mean(np.abs(gray - previous)) / 255.0)
            scores.append((score, int(index))); previous = gray
        del candidate_frames
    crucial_limit = int(os.getenv("MAX_CRUCIAL_FRAMES", "6"))
    minimum_gap = max(1, int(fps * float(os.getenv("CRUCIAL_FRAME_MIN_GAP_SECONDS", "0.75"))))
    crucial = []
    for score, index in sorted(scores, reverse=True):
        if score < float(os.getenv("CRUCIAL_FRAME_SCORE_THRESHOLD", "0.08")): break
        if all(abs(index - selected[1]) >= minimum_gap for selected in crucial):
            crucial.append((score, index))
        if len(crucial) >= crucial_limit: break
    crucial_scores = {index:score for score,index in crucial}
    indices = sorted(set(uniform.tolist()) | set(crucial_scores))
    paths, selection = [], []
    batch_size = max(1, int(os.getenv("DECORD_BATCH_FRAMES", "48")))
    uniform_set = set(uniform.tolist())
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        for index, frame in zip(batch_indices, reader.get_batch([int(i) for i in batch_indices]).asnumpy()):
            handle, frame_path = tempfile.mkstemp(prefix="frame_", suffix=".jpg"); os.close(handle)
            Image.fromarray(frame).save(frame_path, quality=88, optimize=True); paths.append(frame_path)
            reasons = []
            if index in uniform_set: reasons.append("uniform_timeline")
            if index in crucial_scores: reasons.append("scene_change_or_motion_peak")
            selection.append({"reasons":reasons,"change_score":round(crucial_scores.get(index,0.0),4)})
    scores = [round(float(score), 4) for score in compute_frame_change_scores(paths)]
    return paths, [round(float(i)/fps,2) for i in indices], selection, scores

def _decide_dominant_activity(activity_counts, evidence_biases):
    """Fuse visual frame votes with cadence/motion biases from other modalities.

    Unanimous non-visual evidence can confirm the visual winner, override a
    split vote, or decide when frames carry no usable labels; conflicting
    evidence defers to the visual counts.
    """
    dominant = max(activity_counts, key=activity_counts.get) if activity_counts else "miscellaneous"
    votes = {}
    for bias in evidence_biases:
        if bias: votes[bias] = votes.get(bias, 0) + 1
    if not votes or len(votes) > 1: return dominant, "visual_frame_votes"
    best_bias = next(iter(votes))
    if not activity_counts: return best_bias, "multimodal_evidence_consensus"
    strongest = max(activity_counts.items(), key=lambda item: item[1])
    if strongest[0] == best_bias: return dominant, "multimodal_evidence_agreement"
    # A single visual label is consistent evidence, even when it appears in
    # only a few sampled frames. Motion/audio heuristics may corroborate it but
    # must not replace it with a different activity.
    if len(activity_counts) == 1: return dominant, "unanimous_visual_evidence"
    runner_up = sorted(activity_counts.values(), reverse=True)[1] if len(activity_counts) >= 2 else 0
    if (strongest[1] - runner_up) <= 2: return best_bias, "multimodal_evidence_override"
    return dominant, "visual_frame_votes"

def _video_level_activity(frames, audio, activity_counts):
    """Decide the whole-video activity with one reasoning pass over all evidence.

    Per-frame voting loses mission context (a video of an active operation is
    dominated by frames that only show transport, so the vote returns the
    generic label). This pass reviews every frame observation plus the transcript
    and sound events and returns one label. It returns None when the pass is
    unusable, so callers keep their vote-based answer.
    """
    if not frames:
        return None
    transcript = " ".join(segment.get("text", "") for segment in (audio.get("transcript") or []))
    prompt = build_video_activity_prompt(frames, activity_counts, transcript,
        audio.get("sound_events") or [])
    try:
        engine_power.ensure_awake()
        raw = complete_text_with_vllm(prompt, host=VLLM_VISION_HOST, model=VLLM_VISION_MODEL,
            max_tokens=int(os.getenv("VLLM_VIDEO_LEVEL_MAX_TOKENS", "320")))
    except Exception as exc:
        logger.warning("Video-level reasoning unavailable: %s", exc)
        return None
    parsed = parse_json_text(raw)
    activity = str(parsed.get("activity", "")).strip().lower()
    if activity not in ACTIVITY_LABELS:
        logger.warning("Video-level reasoning returned an unusable label: %r", activity)
        return None
    return {"activity": activity,
        "confidence": normalize_confidence(parsed.get("confidence")),
        "reasoning": parsed.get("reasoning")}


def _refine_video_level_activity(video_level, frames, activity_counts, audio=None):
    """Correct a few labels only when the full timeline supplies strong evidence."""
    if not video_level:
        return video_level
    activity = video_level.get("activity")
    text = " ".join(
        f"{frame.get('scene_summary', '')} {frame.get('reasoning', '')}" for frame in frames
    ).lower()

    # Construction work is infrastructure even when the temporary structures
    # could be interpreted as a military deployment.  Require an overwhelming
    # frame consensus so an actual deployment containing incidental earthworks
    # is left unchanged.
    infra_count = activity_counts.get("infra", 0)
    deployment_count = activity_counts.get("deployment", 0)
    if (activity == "deployment" and infra_count >= max(4, deployment_count * 2)
            and infra_count / max(1, len(frames)) >= 0.60):
        return {"activity": "infra",
            "confidence": max(0.85, normalize_confidence(video_level.get("confidence"))),
            "reasoning": "Infrastructure/construction evidence dominates at least 60% of the "
                "timeline and outnumbers deployment evidence by at least two to one.",
            "basis": "temporal_evidence_refinement"}

    # Do not let a few radar/HUD details replace the main activity when nearly
    # the entire timeline explicitly shows forces positioned in formation.
    disposition_count = activity_counts.get("disposition", 0)
    elint_count = activity_counts.get("elint", 0)
    if (activity == "elint" and disposition_count >= max(6, elint_count * 3)
            and disposition_count / max(1, len(frames)) >= 0.75):
        return {"activity": "disposition",
            "confidence": max(0.85, normalize_confidence(video_level.get("confidence"))),
            "reasoning": "Stationary force arrangement dominates at least 75% of the timeline; "
                "isolated radar or HUD details are secondary evidence.",
            "basis": "temporal_evidence_refinement"}

    if activity not in {"disposition", "training"}:
        return video_level

    # A real operation alternates command-and-control scenes with personnel
    # executing the mission in the field.  This combination distinguishes it
    # from static disposition footage and from a field-only training exercise.
    transcript = " ".join(
        segment.get("text", "") for segment in ((audio or {}).get("transcript") or [])
    ).lower()
    command_control = any(term in text for term in (
        "command center", "command centre", "control room", "operations room",
        "tactical map", "surveillance monitors", "surveillance screens"))
    armed_field_team = (
        any(term in text for term in ("armed personnel", "armed team", "soldier", "squad"))
        and any(term in text for term in (
            "advancing", "advance", "moving", "perimeter", "entrance", "aiming")))
    mission_terms = {
        term for term in ("target", "perimeter", "initiate", "phase", "in position",
                          "synchronization", "objective", "secure", "command")
        if term in transcript
    }
    if command_control and armed_field_team and len(mission_terms) >= 3:
        return {"activity": "operation",
            "confidence": max(0.85, normalize_confidence(video_level.get("confidence"))),
            "reasoning": "The timeline alternates command-and-control monitoring with an armed "
                "field team executing sequenced mission commands.",
            "basis": "multimodal_mission_sequence_refinement"}

    combined_arms = "tank" in text and any(term in text for term in ("infantry", "soldier"))
    simulated = any(term in text for term in (
        "hud", "heads-up display", "simulation", "simulated", "digital overlay",
        "tactical display", "targeting display"))
    if combined_arms and simulated and activity_counts.get("training", 0) >= 3:
        return {"activity": "joint exercise",
            "confidence": max(0.85, normalize_confidence(video_level.get("confidence"))),
            "reasoning": "Combined tanks and infantry maneuver under repeated simulation/HUD cues, "
                "with exercise labels across the timeline; this supports a joint exercise more "
                "strongly than a static disposition.",
            "basis": "temporal_evidence_refinement"}
    return video_level


def _generate_cross_modal_summary(prompt):
    """Best-effort cross-modal synthesis on the vLLM engine; None when unavailable."""
    try:
        engine_power.ensure_awake()
        return complete_text_with_vllm(prompt, host=VLLM_VISION_HOST, model=VLLM_VISION_MODEL)
    except Exception as exc:
        logger.warning("Cross-modal summary unavailable: %s", exc)
        return None

def _build_conclusion_summary(metadata, frames, audio, dominant, basis, confidence_summary):
    """Compose a concise conclusion from the frame and audio evidence.

    Deterministic: it only recombines fields the pipeline already computed, so
    it never triggers another model call and adds no GPU contention.
    """
    audio = audio if isinstance(audio, dict) else {}
    duration = metadata.get("duration_sec") if isinstance(metadata, dict) else None
    duration_text = f"{duration:.2f}s" if isinstance(duration, (int, float)) else "an unknown duration"
    sentences = [f"Analyzed {len(frames)} sampled frame(s) over {duration_text}; "
        f"the dominant activity is {dominant}."]
    mean_confidence = (confidence_summary or {}).get("mean")
    if mean_confidence is not None:
        sentences.append(f"Mean frame confidence is {mean_confidence:.2f} ({confidence_label(mean_confidence)}).")
    sentences.append(f"Conclusion basis: {basis}.")
    transcript = " ".join(segment.get("text", "") for segment in (audio.get("transcript") or [])).strip()
    cues = sorted({event.get("label") for event in (audio.get("sound_events") or []) if event.get("label")})
    if audio.get("has_audio"):
        if transcript:
            excerpt = transcript if len(transcript) <= 160 else transcript[:157].rstrip() + "..."
            sentences.append(f'Speech detected: "{excerpt}".')
        if cues:
            sentences.append(f"Acoustic cues: {', '.join(cues)}.")
        if not transcript and not cues:
            sentences.append("Audio was present but no speech or sound events were recognized.")
    else:
        sentences.append("No audio track was available; this conclusion rests on visual evidence only.")
    return " ".join(sentences)

def analyze_video_file(path):
    metadata = validate_video_duration(path, float(os.getenv("MAX_VIDEO_DURATION_SECONDS", str(MAX_VIDEO_DURATION_SECONDS))))
    report_progress(5, "probing media")
    paths, timestamps = sample_video_frames(path); results, backends = [], set()
    frame_warnings = []
    change_scores = []
    sampled = max(1, len(paths))
    report_progress(12, f"sampled {len(paths)} frame(s)")
    try:
        for index, (frame, timestamp) in enumerate(zip(paths, timestamps)):
            result, backend, warning = _analyze_frame_resilient(frame, index)
            report_progress(12 + 58*(index + 1)/sampled, f"frame {index + 1}/{sampled}")
            if warning: frame_warnings.append(warning); continue
            backends.add(backend)
            results.append({"frame_index":index,"timestamp_sec":timestamp,**result})
        change_scores = compute_frame_change_scores(paths)
    finally:
        for frame in paths: Path(frame).unlink(missing_ok=True)
    if not results:
        detail = frame_warnings[-1] if frame_warnings else "no frames produced a usable result"
        raise VisionBackendError(
            f"Video analysis failed because every sampled frame was rejected; {detail}"
        )
    report_progress(72, "merging frame evidence")
    # Strict stage ordering: all vision inference (vLLM server calls) is already
    # complete here; the audio stage runs next, never concurrently with it.
    audio = {"has_audio":metadata["has_audio"],"transcript":[],"sound_events":[],"warnings":[]}
    if metadata["has_audio"] and os.getenv("ENABLE_AUDIO_ANALYSIS","1") == "1":
        # Vision finished above; the audio stage sleeps the vLLM engine first.
        # One long model call, so it moves the bar in a single step.
        report_progress(76, "audio analysis")
        audio = _run_audio_stage(path, audio)
    report_progress(90, "summarising evidence")
    cadence = audio.get("firing_cadence") if isinstance(audio, dict) else None
    ast_cadence = audio.get("sound_event_cadence") if isinstance(audio, dict) else None
    motion = analyze_motion_burstiness(change_scores)
    evidence_biases = [cadence_activity_bias(cadence), sound_event_cadence_bias(ast_cadence), motion_activity_bias(motion)]
    activities = sorted({a for frame in results for a in frame.get("detected_activity",[])})
    summary = {"detected_activities":activities or ["miscellaneous"],"frames_analyzed":len(results),"duration_sec":metadata["duration_sec"]}
    activity_counts, confidence_counts = {}, {level:0 for level in CONFIDENCE_LEVELS}
    confidence_values = []
    for frame in results:
        for activity in frame.get("detected_activity",[]):
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
        confidence = normalize_confidence(frame.get("confidence"))
        confidence_values.append(confidence)
        confidence_counts[confidence_label(confidence)] += 1
    confidence_summary = {
        "mean": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
        "min": round(min(confidence_values), 4) if confidence_values else None,
        "max": round(max(confidence_values), 4) if confidence_values else None,
        "frames": len(confidence_values),
    }
    dominant, basis = _decide_dominant_activity(activity_counts, evidence_biases)
    # The whole-video reasoning pass is the final authority: it sees every frame
    # plus the audio evidence and is not fooled by a dominant generic frame label.
    video_level = _video_level_activity(results, audio, activity_counts)
    video_level = _refine_video_level_activity(video_level, results, activity_counts, audio)
    if video_level:
        dominant = video_level["activity"]
        basis = video_level.get("basis", "video_level_reasoning")
    transcript = " ".join(segment.get("text","") for segment in audio.get("transcript",[])).strip()
    sound_cues = sorted({event.get("label") for event in audio.get("sound_events",[]) if event.get("label")})
    timeline = [{"timestamp_sec":frame["timestamp_sec"],"activities":frame.get("detected_activity",[]),
        "scene":frame.get("scene_summary",""),"confidence":frame.get("confidence",0.0)}
        for frame in results]
    if basis in {"multimodal_evidence_override", "multimodal_evidence_consensus"}:
        reasons = []
        if cadence_activity_bias(cadence): reasons.append(f"the audio firing rhythm ({cadence.get('classification')} discharges, median gap {cadence.get('median_gap_sec')}s)")
        if sound_event_cadence_bias(ast_cadence): reasons.append(f"acoustic event spacing ({ast_cadence.get('classification')})")
        if motion_activity_bias(motion): reasons.append(f"the inter-frame motion rhythm ({motion.get('classification')})")
        narrative = (f"Analyzed {len(results)} sampled frames across {metadata['duration_sec']:.2f} seconds. "
            f"Frame-level visual evidence was split between {', '.join(sorted(activity_counts)) or 'unclassified scenes'}; "
            + " and ".join(reasons) + f" identified {dominant} as the dominant activity. "
            "See the timeline and modality fields for the supporting evidence and uncertainty.")
    else:
        narrative = (f"Analyzed {len(results)} sampled frames across {metadata['duration_sec']:.2f} seconds. "
            f"The most frequently observed activity was {dominant}, supported in "
            f"{activity_counts.get(dominant,0)} frame(s). "
            f"Audio was {'available and analyzed' if audio.get('has_audio') else 'not available'}. "
            "See the timeline and modality fields for the supporting evidence and uncertainty.")
    # A model-written counterpart to the deterministic conclusion above. It is
    # best-effort: when the engine is unavailable it is None and clients keep
    # using conclusion_summary.
    llm_conclusion = _generate_cross_modal_summary(
        build_multimodal_summary_prompt(results, audio.get("transcript",[]), audio.get("sound_events",[])))
    overall = {"narrative":narrative,"dominant_activity":dominant,"dominant_activity_basis":basis,
        "video_level_reasoning":video_level,
        "frame_warnings":frame_warnings,
        "firing_cadence":cadence if isinstance(cadence, dict) else None,
        "motion_burstiness":motion,"evidence_biases":[bias for bias in evidence_biases if bias],
        "activity_frequency":activity_counts,"confidence_distribution":confidence_counts,"confidence_summary":confidence_summary,
        "conclusion_summary":_build_conclusion_summary(metadata, results, audio, dominant, basis, confidence_summary),
        "conclusion_summary_llm":llm_conclusion,
        "timeline":timeline,"transcript":transcript or None,"sound_cues":sound_cues,
        "audio_warnings":audio.get("warnings",[]),"backends_used":sorted(backends)}
    return {"frame_analysis":results,"audio_analysis":audio,"activity_summary":summary,
        "overall_analysis":overall,"backends_used":sorted(backends)}

def _build_detailed_report(metadata, frames, tracking, audio):
    activities = {}
    for frame in frames:
        for label in frame.get("detected_activity", []):
            activities.setdefault(label, []).append(frame["timestamp_sec"])
    evidence = [{"activity":label,"first_seen_sec":min(times),"last_seen_sec":max(times),
        "supporting_frames":len(times)} for label,times in sorted(activities.items())]
    chronology = [{"timestamp_sec":frame["timestamp_sec"],"scene":frame.get("scene_summary",""),
        "activities":frame.get("detected_activity",[]),"confidence":frame.get("confidence",0.0)}
        for frame in frames]
    transcript = " ".join(segment.get("text", "") for segment in audio.get("transcript",[])).strip()
    confidences = [normalize_confidence(frame.get("confidence")) for frame in frames]
    confidence_distribution = {level:0 for level in CONFIDENCE_LEVELS}
    for value in confidences: confidence_distribution[confidence_label(value)] += 1
    confidence_summary = {"mean": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "min": round(min(confidences), 4) if confidences else None,
        "max": round(max(confidences), 4) if confidences else None, "frames": len(confidences)}
    return {        "executive_overview":{
            "duration_sec":metadata["duration_sec"],"frames_analyzed":len(frames),
            "persistent_tracks":len(tracking.get("tracks",[])),
            "activities":list(activities) or ["miscellaneous"],
            "confidence_distribution":confidence_distribution,
            "confidence_summary":confidence_summary,
            "audio_available":audio.get("has_audio",False)},
        "chronology":chronology,"activity_evidence":evidence,
        "spoken_content":transcript or None,
        "cross_modal_notes":[
            "Visual activity labels come from timestamped frame evidence.",
            "Speech is reported as transcript and is not treated as visual proof.",
            "Sound-event labels are acoustic cues and may require human confirmation."],
        "limitations":["Sampling can miss events occurring between analyzed frames.",
            "Track IDs may change after occlusion or major camera cuts.",
            "Model predictions are decision support and require review for high-impact use."]}

def analyze_video_detailed_file(path, original_filename=None):
    """Run all video modalities serially to keep GPU memory predictable."""
    with _detailed_video_lock:
        metadata = validate_video_duration(path, float(os.getenv("MAX_VIDEO_DURATION_SECONDS", str(MAX_VIDEO_DURATION_SECONDS))))
        report_progress(4, "probing media")
        paths, timestamps, selections, change_scores = sample_detailed_video_frames(path)
        frames, backends = [], set()
        frame_warnings = []
        sampled = max(1, len(paths))
        report_progress(10, f"sampled {len(paths)} frame(s)")
        try:
            for index, (frame, timestamp, selection) in enumerate(zip(paths, timestamps, selections)):
                result, backend, warning = _analyze_frame_resilient(frame, index)
                report_progress(10 + 45*(index + 1)/sampled, f"frame {index + 1}/{sampled}")
                if warning: frame_warnings.append(warning); continue
                backends.add(backend)
                frames.append({"frame_index":index,"timestamp_sec":timestamp,
                    "frame_selection":selection,**result})
            # Every frame failing means the engine is unusable; report that
            # plainly instead of returning an all-"miscellaneous" report.
            if not frames:
                raise VisionBackendError("No frame could be analyzed: " +
                    (frame_warnings[0] if frame_warnings else "unknown vision error"))
            report_progress(58, "object tracking")
            tracking = detector.track_frames(paths, timestamps)
            audio = {"has_audio":metadata["has_audio"],"transcript":[],"sound_events":[],"warnings":[]}
            if metadata["has_audio"] and os.getenv("ENABLE_AUDIO_ANALYSIS","1") == "1":
                # Vision (frames + tracking) finished above; the audio stage now
                # sleeps the vLLM engine so the audio model gets the GPU to itself.
                report_progress(68, "audio analysis")
                audio = _run_audio_stage(path, audio)
            report_progress(84, "summarising evidence")
            motion = analyze_motion_burstiness(change_scores)
            ast_cadence = audio.get("sound_event_cadence") if isinstance(audio, dict) else None
            activity_counts = {}
            for frame in frames:
                for label in frame.get("detected_activity", []):
                    activity_counts[label] = activity_counts.get(label, 0) + 1
            dominant, basis = _decide_dominant_activity(activity_counts,
                [cadence_activity_bias(audio.get("firing_cadence")), sound_event_cadence_bias(ast_cadence), motion_activity_bias(motion)])
            # Final authority: one reasoning pass over all frames + the audio
            # evidence, so a dominant generic frame label cannot decide alone.
            video_level = _video_level_activity(frames, audio, activity_counts)
            video_level = _refine_video_level_activity(video_level, frames, activity_counts, audio)
            if video_level:
                dominant = video_level["activity"]
                basis = video_level.get("basis", "video_level_reasoning")
            report = _build_detailed_report(metadata, frames, tracking, audio)
            report["video_level_reasoning"] = video_level
            report["motion_burstiness"] = motion
            report["executive_overview"]["dominant_activity"] = dominant
            report["executive_overview"]["dominant_activity_basis"] = basis
            conclusion = _build_conclusion_summary(metadata, frames, audio, dominant, basis,
                report["executive_overview"]["confidence_summary"])
            report["executive_overview"]["conclusion_summary"] = conclusion
            report["cross_modal_summary"] = _generate_cross_modal_summary(
                build_multimodal_summary_prompt(frames, audio.get("transcript",[]), audio.get("sound_events",[])))
            # The model-written summary already produced above; exposed next to
            # the deterministic conclusion without a second inference call.
            report["executive_overview"]["conclusion_summary_llm"] = report["cross_modal_summary"]
            if frame_warnings: report.setdefault("limitations", []).extend(frame_warnings)
            analysis = {"metadata":metadata,"detailed_report":report,"frame_analysis":frames,
                "frame_warnings":frame_warnings,
                "object_tracking":tracking,
                "overall_analysis":{"conclusion_summary":conclusion,"dominant_activity":dominant,
                    "dominant_activity_basis":basis,
                    "conclusion_summary_llm":report["cross_modal_summary"]},
                "audio_analysis":audio,"motion_burstiness":motion,
                "backends_used":sorted(backends)}
            # Report rendering is opt-out: skipping it removes PDF build time
            # (and its reportlab layout work) from the request path entirely.
            if os.getenv("REPORTING_ENABLED", "1") == "1":
                report_progress(92, "building PDF report")
                analysis["report"] = create_video_analysis_pdf(analysis, paths,
                    original_filename or Path(path).name)
            else:
                analysis["report"] = {"skipped": True,
                    "reason": "REPORTING_ENABLED is off; PDF rendering was skipped to reduce latency"}
            return analysis
        finally:
            for frame in paths: Path(frame).unlink(missing_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR/"templates")); app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app, resources={r"/analyze-*":{"origins":os.getenv("CORS_ORIGINS","*")}})
Swagger(app, template={"swagger":"2.0","info":{"title":"Activity Recognition API","version":"2.0.0",
    "description":"Single-container activity recognition: in-container vLLM (Qwen3-VL-8B int4) and Qwen2.5-Omni audio"}}, config={"headers":[],"specs":[{"endpoint":"apispec",
    "route":"/apispec.json","rule_filter":lambda rule:True,"model_filter":lambda tag:True}],"swagger_ui":True,
    "specs_route":"/docs/","static_url_path":"/flasgger_static"})

# Read-only endpoints the console polls while it is open. They must not count
# as GPU activity: an open tab polls /queue every few seconds, which kept the
# engine awake (and its VRAM held) for as long as the console stayed open, so
# an idle service never released the card.
_PASSIVE_GET_PREFIXES = ("/health", "/queue", "/config", "/reports/", "/object-videos/",
                         "/vision-backend", "/gpu/", "/docs", "/apispec.json", "/flasgger_static")

@app.before_request
def _track_request_activity():
    # Only requests that can start GPU work count as activity: the container
    # healthcheck and the console's status polling are excluded. /unit-tests is a
    # GET that runs the suite in-process, so a GET is passive only on the
    # read-only endpoints above.
    if request.method == "GET":
        path = request.path
        if path == "/" or any(path == prefix or path.startswith(prefix)
                              for prefix in _PASSIVE_GET_PREFIXES):
            return
    g.power_tracked = True
    engine_power.enter()

@app.teardown_request
def _untrack_request_activity(exc=None):
    if getattr(g, "power_tracked", False):
        g.power_tracked = False
        engine_power.exit()

engine_power.start()
JOBS.start()

@app.route("/", methods=["GET"])
def dashboard():
    response = make_response(render_template("dashboard.html", max_upload_mb=MAX_UPLOAD_BYTES//1048576))
    # The dashboard is the surface the queue UI ships in; never let a browser
    # keep a stale copy after the container is redeployed.
    response.headers["Cache-Control"] = "no-store"
    return response

CONFIG_FIELDS = {
    "MAX_VIDEO_FRAMES": (int, 0, 100000), "FRAME_INTERVAL_SECONDS": (float, 0.25, 30),
    "MAX_CRUCIAL_FRAMES": (int, 0, 24),
    "CRUCIAL_FRAME_SCORE_THRESHOLD": (float, 0, 1),
    "CRUCIAL_FRAME_MIN_GAP_SECONDS": (float, 0.1, 30),
    "AUDIO_EVENT_WINDOW_SECONDS": (int, 1, 60), "AUDIO_EVENT_HOP_SECONDS": (int, 1, 60),
    "AUDIO_EVENT_CONFIDENCE_THRESHOLD": (float, 0, 1),
    "OBJECT_CONFIDENCE": (float, 0, 1), "OBJECT_IMAGE_SIZE": (int, 256, 1280),
    "OBJECT_VIDEO_SAMPLE_INTERVAL_SECONDS": (float, 0.1, 30),
    "OBJECT_VIDEO_MAX_SAMPLED_FRAMES": (int, 1, 500),
    "OBJECT_VIDEO_MAX_DETECTIONS": (int, 1, 5000),
    "OBJECT_VIDEO_MAX_TRACKS": (int, 1, 10000),
    "OBJECT_VIDEO_RETENTION": (int, 1, 1000),
    "MAX_VIDEO_DURATION_SECONDS": (float, 1, 3600),
    "REPORTING_ENABLED": (int, 0, 1),
}
CONFIG_DEFAULTS = {"MAX_VIDEO_FRAMES":0,"FRAME_INTERVAL_SECONDS":1,
    "MAX_CRUCIAL_FRAMES":6,"CRUCIAL_FRAME_SCORE_THRESHOLD":0.08,"CRUCIAL_FRAME_MIN_GAP_SECONDS":0.75,
    "AUDIO_EVENT_WINDOW_SECONDS":10,"AUDIO_EVENT_HOP_SECONDS":5,
    "AUDIO_EVENT_CONFIDENCE_THRESHOLD":0.35,"OBJECT_CONFIDENCE":0.25,
    "OBJECT_IMAGE_SIZE":640,"OBJECT_VIDEO_SAMPLE_INTERVAL_SECONDS":1,
    "OBJECT_VIDEO_MAX_SAMPLED_FRAMES":30,"OBJECT_VIDEO_MAX_DETECTIONS":300,
    "OBJECT_VIDEO_MAX_TRACKS":500,"OBJECT_VIDEO_RETENTION":25,
    "MAX_VIDEO_DURATION_SECONDS":300,"REPORTING_ENABLED":1}

def current_config():
    values = {key:caster(os.getenv(key, str(CONFIG_DEFAULTS[key]))) for key,(caster,_,_) in CONFIG_FIELDS.items()}
    values["AUDIO_EVENT_OVERLAP_SECONDS"] = max(0, values["AUDIO_EVENT_WINDOW_SECONDS"] - values["AUDIO_EVENT_HOP_SECONDS"])
    return values

@app.route("/config", methods=["GET"])
@swag_from(GET_CONFIG_SWAGGER_DOC)
def get_config(): return jsonify({"configuration":current_config(),"persisted_path":str(CONFIG_PATH)})

@app.route("/config", methods=["POST"])
@swag_from(SET_CONFIG_SWAGGER_DOC)
def set_config():
    payload = request.get_json(silent=True) or {}
    if "AUDIO_EVENT_OVERLAP_SECONDS" in payload:
        payload["AUDIO_EVENT_HOP_SECONDS"] = float(payload.get("AUDIO_EVENT_WINDOW_SECONDS",
            current_config()["AUDIO_EVENT_WINDOW_SECONDS"])) - float(payload.pop("AUDIO_EVENT_OVERLAP_SECONDS"))
    unknown = set(payload) - set(CONFIG_FIELDS)
    if unknown: return jsonify({"error":f"Unknown settings: {sorted(unknown)}"}),400
    saved = current_config(); saved.pop("AUDIO_EVENT_OVERLAP_SECONDS", None)
    try:
        for key,value in payload.items():
            caster,minimum,maximum = CONFIG_FIELDS[key]; parsed = caster(value)
            if not minimum <= parsed <= maximum: raise ValueError(f"{key} must be between {minimum} and {maximum}")
            saved[key] = parsed
        if saved["AUDIO_EVENT_HOP_SECONDS"] > saved["AUDIO_EVENT_WINDOW_SECONDS"]:
            raise ValueError("Audio overlap cannot be negative; hop must not exceed window")
    except (TypeError,ValueError) as exc: return jsonify({"error":str(exc)}),400
    # Provisioned credentials share runtime.json but are managed by
    # /config/credentials; carry them over so saving inference settings
    # never silently drops them.
    try:
        _existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    except Exception:
        _existing = {}
    for _key, _value in _existing.items():
        if _key in PROVISION_FIELDS and _key not in saved: saved[_key] = _value
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(saved, indent=2, sort_keys=True), encoding="utf-8")
    for key,value in saved.items(): os.environ[key] = str(value)
    return jsonify({"status":"saved","configuration":current_config(),"persisted_path":str(CONFIG_PATH)})

def _provision_status() -> dict:
    """Snapshot of provisioned credentials; secrets are masked, never echoed."""
    status: dict = {}
    for key in PROVISION_FIELDS:
        raw = os.getenv(key, "")
        if key in PROVISION_SECRETS:
            # Never echo a secret back, not even masked: a client that echoes
            # form values could otherwise persist the mask as the real value.
            status[key] = {"configured": bool(raw), "value": ""}
        else:
            status[key] = {"configured": bool(raw), "value": raw}
    status["_postgres"] = {"host": os.getenv("DB_HOST", ""), "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", ""), "user": os.getenv("DB_USER", ""),
        "password_configured": bool(os.getenv("DB_PASSWORD"))}
    key_path = os.getenv("SFTP_PRIVATE_KEY_PATH", "")
    auth = "password" if os.getenv("SFTP_PASSWORD") else ("key" if key_path else "none")
    status["_sftp"] = {"host": os.getenv("SFTP_HOST", ""), "port": os.getenv("SFTP_PORT", "22"),
        "user": os.getenv("SFTP_USER", ""), "remote_dir": os.getenv("SFTP_REMOTE_DIR", ""),
        "auth": auth, "key_present": bool(key_path and Path(key_path).exists())}
    return status

@app.route("/config/credentials", methods=["GET"])
@swag_from(GET_CREDENTIALS_SWAGGER_DOC)
def get_credentials():
    return jsonify({"credentials": _provision_status(), "persisted_path": str(CONFIG_PATH),
        "note": "secret values are masked; POST the full value again to replace it"})

@app.route("/config/credentials", methods=["POST"])
@swag_from(SET_CREDENTIALS_SWAGGER_DOC)
def set_credentials():
    """Provision PostgreSQL / SFTP credentials at runtime.

    Applied to os.environ immediately (get_db_pool() and sftp_upload read the
    environment per call), persisted to config/runtime.json for restarts, and
    the DB pool is reset so new Postgres settings are picked up. An empty
    string clears a field.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Provide a JSON object of credential fields to provision"}), 400
    unknown = set(payload) - set(PROVISION_FIELDS)
    if unknown:
        return jsonify({"error": f"Unknown credential fields: {sorted(unknown)}"}), 400
    applied, errors = {}, []
    for key, value in payload.items():
        text = "" if value is None else str(value).strip()
        # The historical masked-echo sentinel: treat it as "keep current" so a
        # stale client that round-trips the mask cannot clobber the secret.
        if text == "***":
            continue
        if PROVISION_FIELDS[key] is int and text:
            try:
                text = str(int(text))
            except ValueError:
                errors.append(f"{key} must be an integer")
                continue
        applied[key] = text
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400
    for key, value in applied.items():
        if value == "": os.environ.pop(key, None)
        else: os.environ[key] = value
    if any(key.startswith("DB_") for key in applied):
        reset_db_pool()
    try:
        persisted = {}
        if CONFIG_PATH.exists():
            try: persisted = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception: persisted = {}
        for key, value in applied.items():
            if value == "": persisted.pop(key, None)
            else: persisted[key] = value
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(persisted, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        return jsonify({"error": f"Applied in memory but could not persist: {exc}"}), 500
    snapshot = _provision_status()
    return jsonify({"status": "saved", "credentials": snapshot,
        "postgres": snapshot["_postgres"], "sftp": snapshot["_sftp"],
        "persisted_path": str(CONFIG_PATH)})

@app.route("/config/credentials/test", methods=["POST"])
@swag_from(TEST_CREDENTIALS_SWAGGER_DOC)
def test_credentials():
    """Live-connect to the provisioned PostgreSQL and SFTP targets."""
    summary: dict = {"postgres": None, "sftp": None}
    if os.getenv("DB_HOST"):
        try:
            pool = get_db_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                conn.commit()
            finally:
                pool.putconn(conn)
            summary["postgres"] = {"ok": True,
                "detail": f"reachable at {os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '5432')}"}
        except Exception as exc:
            summary["postgres"] = {"ok": False, "detail": str(exc)}
    else:
        summary["postgres"] = {"ok": True, "detail": "no DB_HOST configured - skipped"}
    if os.getenv("SFTP_HOST"):
        try:
            import socket
            host, port = os.getenv("SFTP_HOST", ""), int(os.getenv("SFTP_PORT", "22"))
            with socket.create_connection((host, port), timeout=3):
                pass
            summary["sftp"] = {"ok": True, "detail": f"{host}:{port} accepting connections"}
        except Exception as exc:
            summary["sftp"] = {"ok": False, "detail": str(exc)}
    else:
        summary["sftp"] = {"ok": True, "detail": "no SFTP_HOST configured - skipped"}
    ok = all(entry["ok"] for entry in summary.values() if entry)
    return jsonify({"status": "ok" if ok else "failed", **summary}), 200 if ok else 503

@app.route("/reports/<path:filename>", methods=["GET"])
@swag_from(DOWNLOAD_REPORT_SWAGGER_DOC)
def download_report(filename):
    return send_from_directory(os.getenv("REPORTS_DIR", "/app/reports"), filename,
        as_attachment=True, download_name=filename)

@app.route("/health", methods=["GET"])
@swag_from(HEALTH_SWAGGER_DOC)
def health():
    cuda_ready = torch.cuda.is_available()
    vllm_status = check_vllm_server(host=VLLM_VISION_HOST, model=VLLM_VISION_MODEL, timeout=1.5)
    payload = {"status":"ok" if cuda_ready else "unavailable","service":"activity-recognition","version":"2.0.0",
        "primary_backend":vision.primary,"vision_model":VLLM_VISION_MODEL,
        "audio_model":QWEN_AUDIO_MODEL_ID,"device":vision.device,"vllm_server":vllm_status,
        "cuda_required":True,"cuda_available":cuda_ready}
    return jsonify(payload), 200 if cuda_ready else 503

def _component(name: str, ok: bool, detail: Any) -> Dict[str, Any]:
    """Uniform health-check component record (one list entry per product part)."""
    return {"component": name, "ok": bool(ok), "detail": detail}


@app.route("/health/components", methods=["GET"])
@swag_from(HEALTH_COMPONENTS_SWAGGER_DOC)
def health_components():
    """Check every product component individually for the DevOps Health tab.

    Each entry answers one question about one moving part; ``ok`` is false when
    that part would break an inference request right now. Heavy parts (engine,
    audio models, weights) are probed without loading anything new into VRAM.
    """
    components: list = []

    # CUDA + driver ----------------------------------------------------------
    try:
        cuda_ok = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda_ok else None
        free, total = (torch.cuda.mem_get_info(0) if cuda_ok else (0, 0))
        components.append(_component("cuda", cuda_ok, {
            "device": device, "free_mib": round(free/1048576), "total_mib": round(total/1048576)}
            if cuda_ok else "torch reports no CUDA device"))
    except Exception as exc:
        components.append(_component("cuda", False, str(exc)))

    # vLLM vision engine (asleep is fine: /sleep keeps /v1/models served) -----
    try:
        vllm_status = check_vllm_server(host=VLLM_VISION_HOST, model=VLLM_VISION_MODEL, timeout=2.0)
        ok = bool(vllm_status.get("reachable")) and bool(vllm_status.get("model_available"))
        components.append(_component("vllm_vision_engine", ok, vllm_status))
    except Exception as exc:
        components.append(_component("vllm_vision_engine", False, str(exc)))

    # Audio stack configuration ----------------------------------------------
    components.append(_component("audio_transcription_backend", True, {
        "backend": audio_analysis.TRANSCRIPTION_BACKEND,
        "whisper_model": audio_analysis.WHISPER_MODEL_ID,
        "whisper_weights_baked": Path(audio_analysis.WHISPER_DOWNLOAD_ROOT).exists(),
        "understanding_model": QWEN_AUDIO_MODEL_ID}))

    # faster-whisper availability (import only; weights are baked) -------------
    try:
        import faster_whisper  # noqa: F401
        components.append(_component("whisper_asr", True, {"package": "faster-whisper",
            "weights_dir": audio_analysis.WHISPER_DOWNLOAD_ROOT}))
    except Exception as exc:
        components.append(_component("whisper_asr", False, f"import failed: {exc}"))

    # Qwen2.5-Omni checkpoint present either as a direct offline model folder
    # (Ubuntu deliverable) or in the Hugging Face cache (Windows build).
    try:
        for repo in (QWEN_AUDIO_MODEL_ID,):
            model_path = Path(repo)
            if model_path.is_dir():
                size = sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())
                ok = (model_path / "config.json").is_file() and bool(list(model_path.glob("*.safetensors")))
                components.append(_component(f"checkpoint {model_path.name}", ok,
                    {"path": str(model_path), "size_gb": round(size/1e9, 2)}))
            else:
                from huggingface_hub import scan_cache_dir
                cached = {entry.repo_id: entry.size_on_disk for entry in scan_cache_dir().repos}
                size = cached.get(repo)
                components.append(_component(f"checkpoint {repo}", size is not None,
                    {"size_gb": round(size/1e9, 2)} if size else "not in the local HF cache"))
    except Exception as exc:
        components.append(_component("hf_cache", False, str(exc)))

    # YOLO object detector weights ----------------------------------------------
    detector_path = Path(os.getenv("OBJECT_MODEL_PATH", str(BASE_DIR/"object_detect_model_256_class.pt")))
    components.append(_component("object_detector_weights", detector_path.exists(), str(detector_path)))

    # Job queue worker + database ----------------------------------------------
    try:
        stats = JOBS.stats()
        components.append(_component("job_queue", bool(stats.get("worker_alive", True)), stats))
    except Exception as exc:
        components.append(_component("job_queue", False, str(exc)))

    # PostgreSQL (used by /detect-objects and /fetch-media) ---------------------
    db_required = bool(os.getenv("DB_HOST"))
    try:
        if not db_required:
            components.append(_component("postgres", True, "no DB_HOST configured - skipped"))
        else:
            pool = get_db_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            finally:
                pool.putconn(conn)
            components.append(_component("postgres", True, {"host": os.getenv("DB_HOST"),
                "port": int(os.getenv("DB_PORT", "5432")), "database": os.getenv("DB_NAME"),
                "user": os.getenv("DB_USER")}))
    except Exception as exc:
        components.append(_component("postgres", False, f"unreachable: {exc}"))

    # SFTP delivery target -------------------------------------------------------
    sftp_required = bool(os.getenv("SFTP_HOST"))
    try:
        if not sftp_required:
            components.append(_component("sftp", True, "no SFTP_HOST configured - skipped"))
        else:
            import socket
            host, port = os.getenv("SFTP_HOST", ""), int(os.getenv("SFTP_PORT", "22"))
            with socket.create_connection((host, port), timeout=3):
                pass
            key_path = os.getenv("SFTP_PRIVATE_KEY_PATH", "")
            auth = "password" if os.getenv("SFTP_PASSWORD") else ("key" if key_path else "none")
            components.append(_component("sftp", True, {"host": host, "port": port,
                "user": os.getenv("SFTP_USER", ""), "auth": auth,
                "key_present": bool(key_path and Path(key_path).exists())}))
    except Exception as exc:
        components.append(_component("sftp", False, f"{os.getenv('SFTP_HOST')} unreachable: {exc}"))

    all_ok = all(item["ok"] for item in components)
    return jsonify({"status": "ok" if all_ok else "degraded", "components": components,
        "ok_count": sum(1 for item in components if item["ok"]), "total": len(components)}), 200 if all_ok else 503


# Serializes /unit-tests runs; a test run must never interleave with GPU work.
_UNIT_TEST_LOCK = threading.Lock()
_UNIT_TEST_MODULES = ("test_audio_analysis", "test_job_queue", "test_media_utils",
    "test_engine_power", "test_vram_gate", "test_gpu_monitor", "test_vision_backend_vllm",
    "test_response_schema", "test_video_object_detection", "test_feature_guard",
    "test_swagger_docs")


class _CaseRecorder(unittest_result.TestResult):
    """TestResult that remembers every case so the UI can list green/red rows."""
    def __init__(self, stream=None, descriptions=None, verbosity=None):
        # TextTestRunner calls resultclass(stream, descriptions, verbosity);
        # accepting the standard args keeps it a drop-in TestResult.
        super().__init__(stream, descriptions, verbosity)
        self.cases: list = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.cases.append({"id": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.cases.append({"id": test.id(), "status": "failed"})

    def addError(self, test, err):
        super().addError(test, err)
        self.cases.append({"id": test.id(), "status": "error"})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.cases.append({"id": test.id(), "status": "skipped", "detail": str(reason)})


@app.route("/unit-tests", methods=["GET"])
@swag_from(UNIT_TESTS_SWAGGER_DOC)
def unit_tests():
    """Run the product's unit-test suite in-process and report pass/fail per case.

    One run at a time (serialized by a lock) so a test run can never interleave
    with GPU inference bookkeeping; the response lists every case with its
    outcome so the dashboard can render green/red rows.
    """
    from io import StringIO
    from unittest import TestLoader, TextTestRunner

    with _UNIT_TEST_LOCK:
        stream = StringIO()
        suite = TestLoader().loadTestsFromNames(list(_UNIT_TEST_MODULES))
        runner = TextTestRunner(stream=stream, verbosity=0)
        runner.resultclass = _CaseRecorder
        result = runner.run(suite)
        output = stream.getvalue()

    skipped = [case for case in result.cases if case["status"] == "skipped"]
    payload = {"status": "ok" if result.wasSuccessful() else "failed",
        "ran": result.testsRun, "passed": result.testsRun - len(result.failures) - len(result.errors) - len(skipped),
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(skipped),
        "modules": list(_UNIT_TEST_MODULES), "cases": result.cases,
        "summary": output.strip().splitlines()[-1] if output.strip() else ""}
    return jsonify(payload), 200 if result.wasSuccessful() else 500


@app.route("/gpu/monitor", methods=["GET"])
@swag_from(GPU_MONITOR_SWAGGER_DOC)
def gpu_monitor_view():
    """Live VRAM usage: every process on the card, and whether we can still run.

    Read-only and cheap: the Monitor screen polls it while it is open, which is
    why it is a passive GET (it must not keep the engine awake).
    """
    return jsonify(gpu_snapshot(float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.80"))))

@app.route("/vision-backend", methods=["GET"])
@swag_from(GET_VISION_BACKEND_SWAGGER_DOC)
def backend_status():
    return jsonify({"vision_backend":vision.primary,"vision_model":VLLM_VISION_MODEL,
        "audio_model":QWEN_AUDIO_MODEL_ID,"available_backends":sorted(SUPPORTED_VISION_BACKENDS),
        "last_inference_ms":vision.last_inference_ms})

@app.route("/vision-backend", methods=["POST"])
@swag_from(SET_VISION_BACKEND_SWAGGER_DOC)
def set_backend():
    backend = (request.get_json(silent=True) or {}).get("backend")
    if not backend: return jsonify({"error":"Missing 'backend'","available_backends":sorted(SUPPORTED_VISION_BACKENDS)}),400
    try: vision.set_primary(backend)
    except VisionBackendError as exc: return jsonify({"error":str(exc)}),400
    except Exception as exc:
        logger.exception("Backend activation failed"); return jsonify({"error":f"Unable to activate {backend}: {exc}"}),502
    return backend_status()

# --------------------------------------------------
# REQUEST QUEUE
# --------------------------------------------------
# One heavy job at a time, in arrival order, so concurrent callers cannot
# contend for the GPU. Handlers stage their upload, enqueue the work and answer
# 202 with a job id the caller polls on /queue/<job_id>.
def _translate_job_errors(work):
    """Run a handler body in the worker, reproducing the synchronous contract.

    A job runs outside a request, so Flask's errorhandlers cannot see it; the
    same statuses and bodies are produced here instead.
    """
    try:
        return work()
    except JobCancelled:
        # Cancellation is a queue-level outcome, not a job failure: it must reach
        # the worker as an exception rather than become a 500 response body.
        raise
    except HTTPException as exc:
        return jsonify({"error":exc.description}), exc.code
    except (UploadTooLargeError, VideoTooLongError) as exc:
        return jsonify({"error":str(exc)}), 413
    except (VisionBackendError, SftpUploadError) as exc:
        return jsonify({"error":str(exc)}), 502
    except (MediaProbeError, ValueError) as exc:
        return jsonify({"error":str(exc)}), 400
    except Exception:
        logger.exception("Queued job failed")
        return jsonify({"error":"Internal server error"}), 500

def _job_body(work):
    """Adapt a handler body into a queue callable.

    The worker is not a request, so an application context is pushed for
    jsonify, and the engine is held awake for the job's whole duration: the
    request that queued it has already returned, so the idle timer would
    otherwise park the GPU in the middle of a queued analysis.
    """
    def run():
        with app.app_context():
            engine_power.enter()
            try:
                return _translate_job_errors(work)
            finally:
                engine_power.exit()
    return run

def _enqueue(endpoint, work, filename=None, media_type=None, detail=None, input_path=None, input_ref=None):
    """Queue work and answer 202 with the job record; 503 when the backlog is full."""
    try:
        record = JOBS.submit(endpoint, _job_body(work), filename=filename, media_type=media_type,
            client=request.remote_addr, detail=detail, input_path=input_path, input_ref=input_ref)
    except JobQueueFull as exc:
        # No row owns the staged upload now, so it would otherwise be orphaned.
        if input_path: Path(input_path).unlink(missing_ok=True)
        return jsonify({"error":str(exc),"queue":JOBS.stats()}),503
    response = jsonify(record)
    response.headers["Location"] = record["status_url"]
    return response, 202

def _stage_upload(extensions, rejection):
    """Stage an upload in the queue's own directory and check its extension.

    Validation stays on the request so malformed input still fails immediately
    with the same 400/413 as before instead of being queued only to fail. The
    staged file belongs to the queued job, which deletes it when the job ends.
    """
    path, name = save_upload(directory=str(JOBS.files_dir))
    if Path(path).suffix.lower() not in extensions:
        Path(path).unlink(missing_ok=True)
        raise ValueError(rejection)
    return path, name

def _stage_image_upload():
    path, name = _stage_upload(IMAGE_EXTENSIONS, "Unsupported image format")
    try:
        with Image.open(path) as image: image.verify()
    except UnidentifiedImageError:
        Path(path).unlink(missing_ok=True)
        raise ValueError("Invalid or unreadable image")
    return path, name

def _stage_video_upload():
    path, name = _stage_upload(VIDEO_EXTENSIONS, "Unsupported video format")
    try:
        validate_video_duration(path, float(os.getenv("MAX_VIDEO_DURATION_SECONDS", str(MAX_VIDEO_DURATION_SECONDS))))
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise
    return path, name

def _queued_image_analysis(path, name):
    """Queued body for /analyze-image.

    The queue owns the staged upload, not this function: keeping it is what makes
    the job replayable, and the queue releases it when the row goes.
    """
    report_progress(20, "loading image")
    result, backend = vision.analyze(path)
    report_progress(85, "validating response")
    payload = {"filename":name,"media_type":"image","backend_used":backend,"analysis":result}
    return jsonify(validate_and_repair(payload, IMAGE_ANALYSIS_ENVELOPE_SCHEMA))

def _queued_video_analysis(path, name):
    """Queued body for /analyze-video; the queue owns the staged upload."""
    payload = {"filename":name,"media_type":"video","analysis":analyze_video_file(path)}
    return jsonify(validate_and_repair(payload, VIDEO_ANALYSIS_ENVELOPE_SCHEMA))

def _queued_detailed_video_analysis(path, name):
    """Queued body for /analyze-video-detailed; the queue owns the staged upload."""
    payload = {"filename":name,"media_type":"video","analysis":analyze_video_detailed_file(path, name)}
    return jsonify(validate_and_repair(payload, DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA))

def _queued_object_detection(path, name):
    """Queued body for the upload mode of /detect-objects; the queue owns the upload."""
    report_progress(20, "detecting objects")
    result = detector.detect(path)
    report_progress(90, "encoding annotation")
    return jsonify({"filename":name,"object_detection":result})

def _absolute_base_url(base_url=None):
    """Base URL for links inside a queued response.

    The queue worker runs outside a request context, so the submitter's origin is
    captured in the route and passed in. PUBLIC_BASE_URL takes precedence: set it
    when clients reach the service through a proxy or a name the request does not
    carry, and for replayed jobs (which have no request). Otherwise the origin the
    client used to submit the job is returned.
    """
    configured = os.getenv("PUBLIC_BASE_URL", "").strip() or (base_url or "").strip()
    return configured.rstrip("/")

def _queued_video_object_detection(path, name, base_url=None):
    """Queued body for /detect-objects-video; the queue owns the upload.

    The response carries the annotated clip three ways: the container path
    (``annotated_video_path``), a host-relative URL for the dashboard
    (``annotated_video_url``), and an absolute HTTP URL that fetches the file as
    an attachment (``annotated_video_download_url``). Clients that do not know
    the service origin use the last one.
    """
    report_progress(10, "opening video")
    result = detector.detect_video(path)
    report_progress(95, "finalising detections")
    relative = result.get("annotated_video_url")
    if relative:
        result["annotated_video_download_url"] = \
            f"{_absolute_base_url(base_url)}{relative}?download=1"
    return jsonify({"filename":name,"media_type":"video","object_detection":result})

# How a replay rebuilds a request from a retained upload, per endpoint.
_REPLAY_BODIES = {"/analyze-image":_queued_image_analysis, "/analyze-video":_queued_video_analysis,
    "/analyze-video-detailed":_queued_detailed_video_analysis, "/detect-objects":_queued_object_detection,
    "/detect-objects-video":_queued_video_object_detection}

@app.route("/queue", methods=["GET"])
@swag_from(GET_QUEUE_SWAGGER_DOC)
def get_queue():
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", QUEUE_PAGE_SIZE))
    except ValueError:
        return jsonify({"error":"'page' and 'page_size' must be integers"}),400
    if page < 1 or page_size < 1:
        return jsonify({"error":"'page' and 'page_size' must be positive"}),400
    page_size = min(page_size, QUEUE_PAGE_SIZE_MAX)
    status = (request.args.get("status") or "").strip().lower() or None
    if status and status not in QUEUE_STATUSES:
        return jsonify({"error":f"'status' must be one of {sorted(QUEUE_STATUSES)}"}),400
    include_response = (request.args.get("include_response","0").strip().lower() in {"1","true","yes"})
    total = JOBS.count_jobs(status=status)
    return jsonify({"queue":JOBS.stats(),
        "page":{"number":page,"size":page_size,"total":total,
                "pages":max(1, -(-total // page_size))},
        "jobs":JOBS.list_jobs(limit=page_size, offset=(page - 1)*page_size, status=status,
            include_response=include_response)})

@app.route("/queue/<job_id>", methods=["GET"])
@swag_from(GET_QUEUE_JOB_SWAGGER_DOC)
def get_queue_job(job_id):
    record = JOBS.get(job_id)
    if record is None: return jsonify({"error":"No such job"}),404
    return jsonify(record)

@app.route("/queue", methods=["DELETE"])
@swag_from(CLEAR_QUEUE_SWAGGER_DOC)
def clear_queue_history():
    # Pending work is never touched, so clearing the history can never destroy a
    # result that is still being produced.
    return jsonify({"deleted":JOBS.clear_finished(),"queue":JOBS.stats()})

@app.route("/queue/<job_id>", methods=["DELETE"])
@swag_from(DELETE_QUEUE_JOB_SWAGGER_DOC)
def delete_queue_job(job_id):
    outcome = JOBS.delete(job_id)
    if outcome == "missing": return jsonify({"error":"No such job"}),404
    if outcome == "pending":
        return jsonify({"error":"A queued or running job must be cancelled before it can be deleted"}),409
    return jsonify({"deleted":job_id})

@app.route("/queue/<job_id>/cancel", methods=["POST"])
@swag_from(CANCEL_QUEUE_JOB_SWAGGER_DOC)
def cancel_queue_job(job_id):
    outcome = JOBS.cancel(job_id)
    if outcome == "missing": return jsonify({"error":"No such job"}),404
    if outcome == "finished": return jsonify({"error":"The job has already finished"}),409
    return jsonify(JOBS.get(job_id))

@app.route("/queue/<job_id>/reload", methods=["POST"])
@swag_from(RELOAD_QUEUE_JOB_SWAGGER_DOC)
def reload_queue_job(job_id):
    """Queue the same request again from what the original job recorded."""
    spec = JOBS.replay_input(job_id)
    if spec is None: return jsonify({"error":"No such job"}),404
    endpoint, input_path, input_ref, filename, media_type = spec
    if input_ref:
        # Only the DB-driven mode stores a rebuildable input, and it is just an id.
        try: record_id = json.loads(input_ref).get("id")
        except (TypeError, ValueError): record_id = None
        if record_id is None: return jsonify({"error":"This job cannot be replayed"}),409
        return _enqueue(endpoint, lambda: detect_objects_by_id(record_id), media_type=media_type,
            detail=f"media_analysis.id={record_id}", input_ref=input_ref)
    if not input_path or not Path(input_path).exists():
        return jsonify({"error":"The original upload is no longer retained, so this job cannot be replayed"}),409
    body = _REPLAY_BODIES.get(endpoint)
    if body is None: return jsonify({"error":f"{endpoint} cannot be replayed"}),409
    return _enqueue(endpoint, lambda: body(input_path, filename), filename=filename,
        media_type=media_type, input_path=input_path)

@app.route("/analyze-image", methods=["POST"])
@swag_from(ANALYZE_IMAGE_UPLOAD_SWAGGER_DOC)
def analyze_image_upload():
    path, name = _stage_image_upload()
    return _enqueue("/analyze-image", lambda: _queued_image_analysis(path, name),
        filename=name, media_type="image", input_path=path)

@app.route("/analyze-video", methods=["POST"])
@swag_from(ANALYZE_VIDEO_UPLOAD_SWAGGER_DOC)
def analyze_video_upload():
    path, name = _stage_video_upload()
    return _enqueue("/analyze-video", lambda: _queued_video_analysis(path, name),
        filename=name, media_type="video", input_path=path)

@app.route("/analyze-video-detailed", methods=["POST"])
@swag_from(ANALYZE_VIDEO_DETAILED_SWAGGER_DOC)
def analyze_video_detailed_upload():
    path, name = _stage_video_upload()
    return _enqueue("/analyze-video-detailed", lambda: _queued_detailed_video_analysis(path, name),
        filename=name, media_type="video", input_path=path)

def _fetch_media_inline(record_id):
    """Body of /fetch-media: resolve the record, analyse it, persist the result.

    It keeps the legacy synchronous contract -- the caller receives the analysis
    in this very response -- while /fetch-media itself hands it to
    JobQueue.record_inline so the request also shows up on the queue page.
    """
    with db_cursor() as (conn,cursor):
        cursor.execute("SELECT file_path FROM public.media_analysis WHERE id=%s",(record_id,)); row=cursor.fetchone()
        if not row: return jsonify({"error":"No record found"}),404
        value = row[0][0] if isinstance(row[0],list) else row[0]; parsed=urlparse(str(value))
        if parsed.scheme not in {"http","https"} or not parsed.hostname: raise ValueError("Invalid media URL")
        handle,path=tempfile.mkstemp(prefix="remote_",suffix=Path(parsed.path).suffix.lower()); os.close(handle)
        try:
            with requests.get(str(value),stream=True,timeout=(5,60),verify=MEDIA_TLS_VERIFY) as response:
                response.raise_for_status(); size=0
                with open(path,"wb") as target:
                    for chunk in response.iter_content(1048576):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES: raise UploadTooLargeError("Remote media is too large")
                        target.write(chunk)
            if Path(path).suffix.lower() in VIDEO_EXTENSIONS: result=analyze_video_file(path)
            else:
                result,backend=vision.analyze(path); result={**result,"backend_used":backend}
            cursor.execute("UPDATE public.media_analysis SET analysis_summary=%s,status=%s WHERE id=%s",(json.dumps(result),1,record_id)); conn.commit()
            return jsonify({"id":record_id,"status":1,"analysis":result})
        finally: Path(path).unlink(missing_ok=True)

@app.route("/fetch-media", methods=["POST"])
@swag_from(FETCH_MEDIA_SWAGGER_DOC)
def fetch_media():
    record_id = (request.get_json(silent=True) or {}).get("id")
    if record_id is None: return jsonify({"error":"Missing id"}),400
    # The caller needs the analysis in this response, so the work cannot be
    # handed to the queue worker -- but it is still recorded on the queue page,
    # which is what makes these API calls visible alongside the dashboard's own.
    return JOBS.record_inline("/fetch-media", lambda: _fetch_media_inline(record_id),
        media_type="db-record", detail=f"media_analysis.id={record_id}",
        client=request.remote_addr)

# --------------------------------------------------
# LEGACY DB-DRIVEN OBJECT DETECTION
# --------------------------------------------------
def _fetch_remote_media(media_url, suffix):
    """Download remote media into a temp file, honouring the upload size cap."""
    handle, local_path = tempfile.mkstemp(prefix="remote_", suffix=suffix)
    os.close(handle)
    try:
        with requests.get(str(media_url), stream=True, timeout=(5, 120), verify=MEDIA_TLS_VERIFY) as response:
            response.raise_for_status()
            size = 0
            with open(local_path, "wb") as target:
                for chunk in response.iter_content(1048576):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES: raise UploadTooLargeError("Remote media is too large")
                    target.write(chunk)
    except Exception:
        Path(local_path).unlink(missing_ok=True)
        raise
    logger.info("Fetched remote media %s (%s bytes) -> %s", media_url, size, local_path)
    return local_path

def _normalize_for_detection(source_path, directory=None):
    """Re-encode media as a plain RGB JPEG via PIL.

    Media can be alpha/16-bit PNGs that OpenCV's decoder rejects outright
    (libpng "bad parameters to zlib"); PIL reads them fine, so normalizing here
    keeps detection independent of the source encoding.
    """
    target = Path(directory or tempfile.gettempdir())/f"normalized_{uuid.uuid4().hex}.jpg"
    with Image.open(source_path) as image:
        image.convert("RGB").save(target, quality=92)
    return str(target)

def _ensure_opencv_readable(path):
    """Return a path OpenCV can decode, re-encoding through PIL when it cannot.

    A PNG OpenCV refuses to read leaves the detector's pre-transform batch empty,
    which surfaces as an opaque "need at least one array to stack" instead of a
    result. The DB path always normalises; this gives an upload the same handling,
    but only when OpenCV actually fails, so ordinary images are untouched.
    """
    import cv2
    if cv2.imread(str(path)) is not None:
        return path
    logger.warning("OpenCV could not decode %s; re-encoding it through PIL", path)
    normalized = _normalize_for_detection(path, directory=str(JOBS.files_dir))
    Path(path).unlink(missing_ok=True)
    return normalized

def _write_annotated_artifact(source_path):
    """Render the annotated artefact for `source_path`; returns (path, result)."""
    logger.info("Detecting objects in %s (%s bytes)", source_path, os.path.getsize(source_path))
    if Path(source_path).suffix.lower() in VIDEO_EXTENSIONS:
        artifact = Path(tempfile.gettempdir())/f"detected_{uuid.uuid4().hex}.mp4"
        # annotate_video drives the bar across the frame loop (20-90%).
        return str(artifact), detector.annotate_video(source_path, artifact)
    report_progress(35, "detecting objects")
    normalized = _normalize_for_detection(source_path)
    try:
        result = detector.detect(normalized)
    finally:
        Path(normalized).unlink(missing_ok=True)
    report_progress(85, "encoding annotation")
    encoded = result.get("annotated_image_base64")
    if not encoded: raise VisionBackendError("Object detector returned no annotated image")
    artifact = Path(tempfile.gettempdir())/f"detected_{uuid.uuid4().hex}.jpg"
    artifact.write_bytes(base64.b64decode(encoded))
    return str(artifact), result

def detect_objects_by_id(record_id):
    """Resolve a media_analysis row by id, detect objects, publish the annotated
    artefact over SFTP and record its URL: the legacy /detect-objects contract."""
    source = artifact = None
    with db_cursor() as (conn, cursor):
        cursor.execute("SELECT file_path, media_type FROM public.media_analysis WHERE id = %s", (record_id,))
        row = cursor.fetchone()
        if not row: return jsonify({"error":"No record found"}), 404
        media_value, media_type = row[0], row[1] or "image"
        if isinstance(media_value, list): media_value = media_value[0] if media_value else None
        parsed = urlparse(str(media_value))
        if parsed.scheme not in {"http","https"} or not parsed.hostname: raise ValueError("Invalid media URL")
        try:
            report_progress(4, "downloading media")
            source = _fetch_remote_media(media_value, Path(parsed.path).suffix.lower())
            report_progress(15, "media downloaded")
            if Path(source).suffix.lower() in VIDEO_EXTENSIONS:
                # The same ceiling the upload endpoints enforce. The limit is purely
                # duration-based so an over-long video is rejected outright instead of
                # yielding a silently truncated annotation.
                validate_video_duration(source, float(os.getenv("MAX_VIDEO_DURATION_SECONDS", str(MAX_VIDEO_DURATION_SECONDS))))
            artifact, result = _write_annotated_artifact(source)
            report_progress(93, "publishing artefact")
            output_url = publish(artifact, media_type)
            cursor.execute("UPDATE public.media_analysis SET analysis_summary=%s, output_file_path=%s, status=%s WHERE id=%s",
                (json.dumps(result), output_url, 1, record_id))
            conn.commit()
        except Exception:
            logger.exception("DB-driven /detect-objects failed for media_analysis.id=%s", record_id)
            raise
        finally:
            for candidate in (source, artifact):
                if candidate: Path(candidate).unlink(missing_ok=True)
    return jsonify({"id":record_id,"status":1,"object_detection":result})

@app.route("/detect-objects", methods=["POST"])
@swag_from(DETECT_OBJECTS_SWAGGER_DOC)
def detect_objects_api():
    # Legacy DB-driven mode: {"id": <media_analysis.id>}. Handled ahead of the
    # upload path so the existing multipart behaviour is completely untouched.
    body = request.get_json(silent=True) if request.is_json else None
    if isinstance(body, dict) and body.get("id") is not None:
        record_id = body["id"]
        return _enqueue("/detect-objects", lambda: detect_objects_by_id(record_id),
            media_type="db-record", detail=f"media_analysis.id={record_id}",
            input_ref=json.dumps({"id":record_id}))
    path, name = _stage_image_upload()
    # Detection is the only cv2-based path, so the re-encode belongs here rather
    # than in the shared image staging used by the vLLM endpoints.
    path = _ensure_opencv_readable(path)
    return _enqueue("/detect-objects", lambda: _queued_object_detection(path, name),
        filename=name, media_type="image", input_path=path)

@app.route("/detect-objects-video", methods=["POST"])
@swag_from(DETECT_OBJECTS_VIDEO_SWAGGER_DOC)
def detect_objects_video_api():
    """Queue object detection/tracking for an uploaded video.

    The upload is validated (extension + five-minute duration ceiling) on the
    request so a bad clip still fails immediately; annotation and tracking run
    on the single queue worker.
    """
    path, name = _stage_video_upload()
    base_url = request.host_url
    return _enqueue("/detect-objects-video",
        lambda: _queued_video_object_detection(path, name, base_url),
        filename=name, media_type="video", input_path=path)

@app.route("/object-videos/<path:filename>", methods=["GET"])
@swag_from(DOWNLOAD_OBJECT_VIDEO_SWAGGER_DOC)
def download_object_video(filename):
    """Serve an annotated clip written by /detect-objects-video.

    ``?download=1`` sends it as an attachment, which is what the absolute
    ``annotated_video_download_url`` in the job response uses. Without the flag
    the clip is served inline so the dashboard can play it.
    """
    return send_from_directory(str(object_video_dir()), filename,
        as_attachment=request.args.get("download") == "1")

@app.errorhandler(UploadTooLargeError)
@app.errorhandler(VideoTooLongError)
def too_large(exc): return jsonify({"error":str(exc)}),413
@app.errorhandler(MediaProbeError)
@app.errorhandler(ValueError)
def bad_media(exc): return jsonify({"error":str(exc)}),400
@app.errorhandler(VisionBackendError)
def backend_failure(exc): return jsonify({"error":str(exc)}),502
@app.errorhandler(SftpUploadError)
def sftp_failure(exc): return jsonify({"error":str(exc)}),502
@app.errorhandler(413)
def flask_too_large(_): return jsonify({"error":"Upload is too large"}),413
@app.errorhandler(Exception)
def unhandled(exc):
    if isinstance(exc, HTTPException):
        return jsonify({"error":exc.description}),exc.code
    logger.exception("Unhandled request failure"); return jsonify({"error":"Internal server error"}),500

if __name__ == "__main__": app.run(host="0.0.0.0",port=5009,debug=False)
