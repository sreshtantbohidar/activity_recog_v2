"""In-process FIFO job queue with a SQLite-backed status history.

One container owns one GPU, so letting several requests decode frames or run
inference at the same time is what freezes the host. Heavy endpoints therefore
enqueue their work here instead of doing it inline: a single worker thread
executes jobs strictly in arrival order, and every job's status, response and
timings are recorded in a SQLite file so the queue can be inspected -- and
survives a restart -- even though the work itself is in-process.

Callers get HTTP 202 and a job id straight away and poll ``GET /queue/<job_id>``
for the recorded response. This module deliberately knows nothing about Flask:
the worker takes a zero-argument callable that returns a response object with a
``status_code`` and a ``get_json()``.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("activity-recognition.queue")

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
STATUSES = (QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED)
# Work that has stopped for good: safe to delete, prune and clear.
FINISHED = (COMPLETED, FAILED, CANCELLED)

# Only finished jobs are ever pruned, so a deep backlog can never be deleted
# out from under the worker.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id             TEXT    NOT NULL UNIQUE,
    endpoint           TEXT    NOT NULL,
    status             TEXT    NOT NULL,
    filename           TEXT,
    media_type         TEXT,
    client             TEXT,
    detail             TEXT,
    created_at         REAL    NOT NULL,
    started_at         REAL,
    finished_at        REAL,
    http_status        INTEGER,
    execution_seconds  REAL,
    wait_seconds       REAL,
    response           TEXT,
    response_bytes     INTEGER,
    response_truncated INTEGER NOT NULL DEFAULT 0,
    error              TEXT,
    progress           INTEGER NOT NULL DEFAULT 0,
    progress_detail    TEXT,
    input_path         TEXT,
    input_ref          TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_seq_idx ON jobs (status, seq);
"""

# Columns added after the first release, applied to an existing database file.
_ADDED_COLUMNS = (("progress", "INTEGER NOT NULL DEFAULT 0"),
                  ("progress_detail", "TEXT"),
                  ("input_path", "TEXT"),
                  ("input_ref", "TEXT"))

# Reports which job the worker is currently running, so the pipeline stages can
# publish progress without being handed a callback. The worker is the only
# thread that sets this, so a plain thread-local is enough.
_progress_state = threading.local()


class JobCancelled(RuntimeError):
    """Raised at a checkpoint by a job that was cancelled while it ran."""


def check_cancelled() -> None:
    """Raise :class:`JobCancelled` if the job on this thread was cancelled."""
    cancel = getattr(_progress_state, "cancel", None)
    if cancel is not None and cancel.is_set():
        raise JobCancelled("Cancelled by request")


def report_progress(percent: float, detail: str | None = None) -> None:
    """Publish progress for the job running on this thread.

    Doubles as the cancellation checkpoint: the pipelines call it at every safe
    point (each frame of a video or annotation loop, and between stages), which is
    what gives cancellation somewhere to take effect. It stays a no-op when the
    same pipeline runs inline -- /fetch-media is not queue-owned -- and the write
    is throttled, so a per-frame loop can call it freely.
    """
    queue = getattr(_progress_state, "queue", None)
    if queue is None:
        return
    check_cancelled()
    queue._record_progress(percent, detail)


class JobQueueFull(RuntimeError):
    """Raised by :meth:`JobQueue.submit` when the pending backlog is at its cap.

    The cap is the last line of defence for the hardware: past it the API sheds
    load with a 503 instead of accepting work it cannot hope to start soon.
    """


def _unwrap_response(response):
    """Split a handler's return value into (body, explicit status).

    A handler may return a plain response or pick up Flask's ``(body, status)``
    tuple convention -- which is what the error paths use. Without this the
    tuple's *body* would be read as the whole response, and a failed job would be
    recorded as a success with HTTP 200.
    """
    if isinstance(response, tuple):
        body = response[0] if response else None
        return body, (response[1] if len(response) > 1 else None)
    return response, None


def _iso(epoch: float | None) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class JobQueue:
    """Serialised execution of heavy API work, with a persisted job history."""

    def __init__(self, db_path, files_dir, history_limit: int = 200, max_pending: int = 32,
                 response_max_chars: int = 262144, preview_chars: int = 2000,
                 replay_retention: int = 10, replay_max_bytes: int = 1024 * 1024 * 1024):
        self.db_path = Path(db_path)
        self.files_dir = Path(files_dir)
        self.history_limit = int(history_limit)
        self.max_pending = int(max_pending)
        self.response_max_chars = int(response_max_chars)
        self.preview_chars = int(preview_chars)
        # How much is kept on disk so a job can be reloaded: the newest N
        # uploads, and never more than this many bytes in total.
        self.replay_retention = int(replay_retention)
        self.replay_max_bytes = int(replay_max_bytes)
        self._lock = threading.RLock()
        self._pending: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        # Cancel tokens for work that is still in memory, by job id.
        self._live: dict[str, threading.Event] = {}
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        # One shared connection is plenty for a handful of writes per job, and a
        # single lock keeps SQLite's threading rules trivially satisfied.
        self._conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Bring an older database file up to the current column set.

        CREATE TABLE IF NOT EXISTS leaves an existing table alone, so columns
        added later have to be applied explicitly.
        """
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        for column, declaration in _ADDED_COLUMNS:
            if column not in existing:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {declaration}")
        self._conn.commit()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Recover whatever a restart left behind, then serve the queue."""
        self._recover()
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, name="job-queue", daemon=True)
        self._worker.start()
        logger.info("Job queue started (history_limit=%s, max_pending=%s)", self.history_limit, self.max_pending)

    def stop(self, timeout: float = 10.0) -> None:
        # Only signal a worker that exists: leaving a stray sentinel behind would
        # make the next start() return immediately instead of serving the queue.
        if self._worker is None or not self._worker.is_alive():
            return
        self._pending.put(None)
        self._worker.join(timeout)

    def close(self, timeout: float = 10.0) -> None:
        """Stop the worker and release the database handle."""
        self.stop(timeout)
        with self._lock:
            self._conn.close()

    def _recover(self) -> None:
        """Jobs cannot outlive the process, so abandon theirs and purge uploads.

        Without this a restart would leave rows stuck in ``queued``/``running``
        forever, which reads as a hung service in the queue tab.
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, error=? WHERE status IN (?,?)",
                (FAILED, time.time(), "Interrupted by a service restart", QUEUED, RUNNING))
            self._conn.commit()
            interrupted = cursor.rowcount
        if interrupted:
            logger.warning("Marked %s queued/running job(s) as failed after a restart", interrupted)
        # The staged uploads live on the container filesystem, so the retained
        # copies are gone too: drop the references that now point nowhere, which
        # is what makes a reload report a missing input instead of failing oddly.
        for stale in self.files_dir.iterdir():
            if stale.is_file():
                try:
                    stale.unlink()
                except OSError:
                    logger.debug("Could not remove stale staged upload %s", stale)
        with self._lock:
            self._conn.execute("UPDATE jobs SET input_path=NULL WHERE input_path IS NOT NULL")
            self._conn.commit()

    # ------------------------------------------------------------- submission
    def submit(self, endpoint: str, work: Callable[[], Any], *, filename: str | None = None,
               media_type: str | None = None, client: str | None = None,
               detail: str | None = None, input_path: str | None = None,
               input_ref: str | None = None) -> dict:
        """Record and enqueue ``work``, returning the job as it stands right now.

        ``input_path`` and ``input_ref`` are what a reload replays: a staged
        upload that is kept on disk, or a small JSON blob of the arguments that
        rebuilt the request (the /detect-objects record id).
        """
        with self._lock:
            pending = self._conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status IN (?,?)", (QUEUED, RUNNING)).fetchone()["n"]
            if pending >= self.max_pending:
                raise JobQueueFull(f"The analysis queue is full ({pending} pending jobs)")
            job_id = uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO jobs (job_id, endpoint, status, filename, media_type, client, detail,"
                " created_at, input_path, input_ref) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (job_id, endpoint, QUEUED, filename, media_type, client, detail, time.time(),
                 input_path, input_ref))
            self._conn.commit()
            self._live[job_id] = threading.Event()
            token = self._live[job_id]
        self._pending.put({"job_id": job_id, "endpoint": endpoint, "work": work, "cancel": token})
        return self.get(job_id)

    # ------------------------------------------------------------- inspection
    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            position = self._position(job_id) if row is not None and row["status"] == QUEUED else None
        return self._record(row, position, include_response=True) if row is not None else None

    def list_jobs(self, limit: int = 25, offset: int = 0, status: str | None = None,
                  include_response: bool = False) -> list[dict]:
        """One page, newest first. Responses are omitted by default: the tab polls this."""
        statement = "SELECT * FROM jobs"
        parameters: list[Any] = []
        if status:
            statement += " WHERE status=?"
            parameters.append(status)
        statement += " ORDER BY seq DESC LIMIT ? OFFSET ?"
        parameters.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        with self._lock:
            rows = self._conn.execute(statement, parameters).fetchall()
        return [self._record(row, None, include_response=include_response) for row in rows]

    def count_jobs(self, status: str | None = None) -> int:
        """Total rows a listing would page through, so the UI can show page count."""
        statement = "SELECT COUNT(*) AS n FROM jobs"
        parameters: list[Any] = []
        if status:
            statement += " WHERE status=?"
            parameters.append(status)
        with self._lock:
            return self._conn.execute(statement, parameters).fetchone()["n"]

    def counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        by_status = {status: 0 for status in STATUSES}
        for row in rows:
            by_status[row["status"]] = row["n"]
        by_status["total"] = sum(row["n"] for row in rows)
        return by_status

    def delete(self, job_id: str) -> str:
        """Delete one job if it has finished.

        Returns ``deleted``, ``missing`` or ``pending``: recorded work must not be
        removable while it is still queued or running, and the history is only
        ever the record of what already happened.
        """
        with self._lock:
            row = self._conn.execute("SELECT status, input_path FROM jobs WHERE job_id=?",
                                     (job_id,)).fetchone()
            if row is None:
                return "missing"
            if row["status"] not in FINISHED:
                return "pending"
            self._conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
            self._conn.commit()
        self._discard_input(row["input_path"])
        logger.info("Deleted finished job %s", job_id)
        return "deleted"

    def clear_finished(self) -> int:
        """Delete every completed/failed job, returning how many rows went.

        Queued and running jobs are never touched, so clearing history can never
        destroy a job that is still going to produce a result.
        """
        with self._lock:
            doomed = self._conn.execute("SELECT input_path FROM jobs WHERE status IN (?,?,?)",
                                        FINISHED).fetchall()
            cursor = self._conn.execute("DELETE FROM jobs WHERE status IN (?,?,?)", FINISHED)
            self._conn.commit()
        for row in doomed:
            self._discard_input(row["input_path"])
        logger.info("Cleared %s finished job(s) from the queue history", cursor.rowcount)
        return cursor.rowcount

    def _record_progress(self, percent: float, detail: str | None) -> None:
        """Store progress for the running job, writing only when it changes."""
        job_id = getattr(_progress_state, "job_id", None)
        if job_id is None:
            return
        try:
            value = max(0, min(100, int(percent)))
        except (TypeError, ValueError):
            return
        text = None if detail is None else str(detail)[:200]
        with self._lock:
            row = self._conn.execute("SELECT progress, progress_detail FROM jobs WHERE job_id=?",
                                     (job_id,)).fetchone()
            if row is None or (row["progress"] == value and row["progress_detail"] == text):
                return
            self._conn.execute("UPDATE jobs SET progress=?, progress_detail=? WHERE job_id=?",
                               (value, text, job_id))
            self._conn.commit()

    def cancel(self, job_id: str) -> str:
        """Ask a queued or running job to stop.

        Returns ``cancelled``, ``finished`` or ``missing``. A queued job is marked
        cancelled straight away so the caller sees it at once; a running one stops
        at its next checkpoint, which means it can finish a step that is already
        inside a single model call.
        """
        with self._lock:
            row = self._conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return "missing"
            if row["status"] in FINISHED:
                return "finished"
            token = self._live.get(job_id)
            if token is None:
                # In flight but not owned by this process: leave it alone.
                return "missing"
            token.set()
            if row["status"] == QUEUED:
                self._conn.execute(
                    "UPDATE jobs SET status=?, finished_at=?, execution_seconds=0, error=? WHERE job_id=?",
                    (CANCELLED, time.time(), "Cancelled before it started", job_id))
                self._conn.commit()
        logger.info("Cancelled job %s", job_id)
        return "cancelled"

    def replay_input(self, job_id: str):
        """What a reload needs: (endpoint, input_path, input_ref, filename, media_type)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT endpoint, input_path, input_ref, filename, media_type FROM jobs WHERE job_id=?",
                (job_id,)).fetchone()
        if row is None:
            return None
        return (row["endpoint"], row["input_path"], row["input_ref"], row["filename"], row["media_type"])

    def _discard_input(self, path: str | None) -> None:
        """Remove a retained upload once no job row still references it.

        A reloaded job shares the path of the job it was replayed from, so the file
        can only go when the last row pointing at it does.
        """
        if not path:
            return
        with self._lock:
            used = self._conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE input_path=?",
                                      (path,)).fetchone()["n"]
        if used:
            return
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove retained upload %s", path)

    def _evict_inputs(self) -> None:
        """Bound what reload keeps on disk: the newest N inputs, within a byte cap."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id, input_path FROM jobs WHERE input_path IS NOT NULL ORDER BY seq DESC"
            ).fetchall()
        keep, drop, total = [], [], 0
        for row in rows:
            try:
                size = os.path.getsize(row["input_path"])
            except OSError:
                drop.append(row)  # already gone, so only the reference needs clearing
                continue
            if len(keep) < self.replay_retention and total + size <= self.replay_max_bytes:
                keep.append(row)
                total += size
            else:
                drop.append(row)
        if not drop:
            return
        with self._lock:
            for row in drop:
                self._conn.execute("UPDATE jobs SET input_path=NULL WHERE job_id=?", (row["job_id"],))
            self._conn.commit()
        for row in drop:
            self._discard_input(row["input_path"])

    def stats(self) -> dict:
        """Queue-level summary for the UI header and the 503 load-shed response."""
        counts = self.counts()
        return {
            "queued": counts[QUEUED], "running": counts[RUNNING],
            "completed": counts[COMPLETED], "failed": counts[FAILED],
            "cancelled": counts[CANCELLED], "total": counts["total"],
            "pending": counts[QUEUED] + counts[RUNNING],
            "max_pending": self.max_pending,
            "history_limit": self.history_limit,
            "concurrency": 1,
            "worker_alive": bool(self._worker is not None and self._worker.is_alive()),
        }

    def _position(self, job_id: str) -> int | None:
        """How many pending jobs -- including the running one -- are ahead.

        The running job counts because it still has to finish before this one can
        start, which is what a caller waiting in line actually cares about.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status IN (?,?) AND seq < (SELECT seq FROM jobs WHERE job_id=?)",
            (QUEUED, RUNNING, job_id)).fetchone()
        return None if row is None else row["n"] + 1

    def _record(self, row: sqlite3.Row, position: int | None, include_response: bool) -> dict:
        record = {
            "job_id": row["job_id"], "seq": row["seq"], "endpoint": row["endpoint"],
            "status": row["status"], "filename": row["filename"], "media_type": row["media_type"],
            "client": row["client"], "detail": row["detail"],
            "created_at": _iso(row["created_at"]), "started_at": _iso(row["started_at"]),
            "finished_at": _iso(row["finished_at"]),
            # queued_seconds is time spent waiting for a slot (the queue), while
            # duration_seconds is the actual work -- the "time taken" users ask about.
            "queued_seconds": row["wait_seconds"], "duration_seconds": row["execution_seconds"],
            # Wall-clock from submission to result, i.e. what the caller waited.
            "total_seconds": (round(row["finished_at"] - row["created_at"], 3)
                              if row["finished_at"] and row["created_at"] else None),
            "http_status": row["http_status"], "error": row["error"],
            "progress": row["progress"], "progress_detail": row["progress_detail"],
            "queue_position": position, "status_url": f"/queue/{row['job_id']}",
            "deletable": row["status"] in FINISHED, "cancellable": row["status"] in {QUEUED, RUNNING},
            # Reload needs the input the job was built from; uploads are retained
            # for the newest few jobs, record ids are kept with the row.
            "replayable": bool(row["input_ref"] or row["input_path"]),
            "response": None, "response_preview": None,
            "response_bytes": row["response_bytes"], "response_truncated": bool(row["response_truncated"]),
        }
        text = row["response"]
        if include_response and text is not None:
            record["response_preview"] = text[:self.preview_chars]
            # A truncated body is no longer valid JSON, so expose it as a preview
            # only rather than pretending to parse it.
            if not record["response_truncated"]:
                try:
                    record["response"] = json.loads(text)
                except json.JSONDecodeError:
                    record["response"] = None
        return record

    # ------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            job = self._pending.get()
            if job is None:
                return
            self._execute(job)

    def _execute(self, job: dict) -> None:
        job_id, endpoint, cancel = job["job_id"], job["endpoint"], job["cancel"]
        if cancel.is_set():
            # Cancelled while it was still queued: the row was already recorded by
            # cancel(), so there is nothing to run and nothing left to write.
            logger.info("Skipping cancelled job %s (%s)", job_id, endpoint)
            with self._lock:
                self._live.pop(job_id, None)
            return
        started = time.time()
        with self._lock:
            row = self._conn.execute("SELECT created_at FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            self._conn.execute(
                "UPDATE jobs SET status=?, started_at=?, wait_seconds=? WHERE job_id=?",
                (RUNNING, started, round(started - row["created_at"], 3) if row else None, job_id))
            self._conn.commit()
        logger.info("Running queued job %s (%s)", job_id, endpoint)

        http_status, payload, error = None, None, None
        # Publish the running job to the thread-local so the pipeline stages can
        # report progress without being handed a callback. They see nothing when
        # the same pipeline runs inline (/fetch-media), keeping it a no-op there.
        _progress_state.queue, _progress_state.job_id, _progress_state.cancel = self, job_id, cancel
        cancelled = False
        try:
            try:
                body, status = _unwrap_response(job["work"]())
                try:
                    http_status = int(status if status is not None else getattr(body, "status_code", 200))
                except (TypeError, ValueError):
                    http_status = int(getattr(body, "status_code", 500))
                get_json = getattr(body, "get_json", None)
                payload = get_json(silent=True) if callable(get_json) else None
                if payload is None:
                    payload = {"note": "The job returned a non-JSON response"}
            except JobCancelled as exc:
                logger.info("Queued job %s (%s) cancelled mid-run", job_id, endpoint)
                cancelled, error = True, str(exc)
                payload = {"error": str(exc), "cancelled": True}
            except Exception as exc:  # the callable is expected to translate its own errors
                logger.exception("Queued job %s (%s) raised", job_id, endpoint)
                http_status, error = 500, f"{type(exc).__name__}: {exc}"
                payload = {"error": "Internal server error"}
        finally:
            _progress_state.queue = _progress_state.job_id = _progress_state.cancel = None
            with self._lock:
                self._live.pop(job_id, None)

        if error is None and isinstance(payload, dict) and payload.get("error"):
            error = str(payload["error"])
        if cancelled:
            status = CANCELLED
        else:
            status = COMPLETED if http_status is not None and 200 <= http_status < 300 else FAILED

        text = None if payload is None else json.dumps(payload, default=str)
        truncated = bool(text) and len(text) > self.response_max_chars
        stored = text[:self.response_max_chars] if truncated else text
        finished = time.time()
        with self._lock:
            # A completed job is 100% done; a failed one keeps the percentage it
            # reached, which says far more than resetting it to zero.
            self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, http_status=?, execution_seconds=?, response=?,"
                " response_bytes=?, response_truncated=?, error=?, progress=COALESCE(?, progress)"
                " WHERE job_id=?",
                (status, finished, http_status, round(finished - started, 3), stored,
                 len(text) if text else None, int(truncated), error,
                 100 if status == COMPLETED else None, job_id))
            self._conn.commit()
            self._prune_locked()
        logger.info("Queued job %s (%s) finished as %s in %.1fs (HTTP %s)",
                    job_id, endpoint, status, finished - started, http_status)

    def _prune_locked(self) -> None:
        """Keep the newest ``history_limit`` finished jobs; never touch pending ones.

        The retained uploads of the rows that go are released with them, and the
        reload window is re-evaluated in the same pass.
        """
        doomed = self._conn.execute(
            "SELECT job_id, input_path FROM jobs WHERE status IN (?,?,?)"
            " ORDER BY seq DESC LIMIT -1 OFFSET ?",
            (*FINISHED, self.history_limit)).fetchall()
        if doomed:
            self._conn.execute(
                "DELETE FROM jobs WHERE job_id IN (%s)" % ",".join("?" * len(doomed)),
                [row["job_id"] for row in doomed])
            self._conn.commit()
            for row in doomed:
                self._discard_input(row["input_path"])
        self._evict_inputs()
