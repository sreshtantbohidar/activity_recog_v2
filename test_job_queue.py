import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from job_queue import (CANCELLED, COMPLETED, FAILED, FINISHED, QUEUED, RUNNING, JobQueue,
                       JobQueueFull, report_progress)


class FakeResponse:
    """Stand-in for a Flask response: the queue only reads these two members."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def get_json(self, silent=False):
        return self._payload


class FakeStreamedResponse:
    """A response whose body the queue cannot read (no JSON accessor)."""

    status_code = 200


def wait_until(predicate, timeout=5.0, description="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"{description} was not met within {timeout}s")


def wait_for(queue, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = queue.get(job_id)
        if record and record["status"] in FINISHED:
            return record
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


class JobQueueTests(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="jobqueue_test_"))
        self.started = []

    def tearDown(self):
        # close() also stops the worker, and releases the SQLite handle so the
        # temporary directory can be removed cleanly.
        for queue in self.started:
            try:
                queue.close()
            except sqlite3.Error:
                pass
        shutil.rmtree(self.workdir, ignore_errors=True)

    def make_queue(self, start=True, **overrides):
        options = {"db_path": self.workdir / "jobs.sqlite3", "files_dir": self.workdir / "uploads"}
        options.update(overrides)
        queue = JobQueue(**options)
        self.started.append(queue)
        if start:
            queue.start()
        return queue

    # ------------------------------------------------------------- happy path
    def test_jobs_run_in_arrival_order_and_record_their_response(self):
        queue = self.make_queue()
        order = []
        records = [
            queue.submit("/analyze-image", lambda index=index: (order.append(index), FakeResponse({"index": index}))[1])
            for index in range(3)
        ]
        for record in records:
            self.assertEqual(record["status"], QUEUED)
            self.assertEqual(record["status_url"], f"/queue/{record['job_id']}")
        finished = [wait_for(queue, record["job_id"]) for record in records]
        self.assertEqual(order, [0, 1, 2], "jobs must run in arrival order")
        self.assertEqual([job["status"] for job in finished], [COMPLETED] * 3)
        self.assertEqual(finished[1]["response"], {"index": 1})
        self.assertEqual(finished[0]["http_status"], 200)
        self.assertIsNotNone(finished[0]["duration_seconds"])

    def test_only_one_job_runs_at_a_time(self):
        queue = self.make_queue()
        lock = threading.Lock()
        state = {"running": 0, "peak": 0}

        def work():
            with lock:
                state["running"] += 1
                state["peak"] = max(state["peak"], state["running"])
            time.sleep(0.05)
            with lock:
                state["running"] -= 1
            return FakeResponse({"ok": True})

        records = [queue.submit("/analyze-video", work) for _ in range(4)]
        for record in records:
            wait_for(queue, record["job_id"])
        self.assertEqual(state["peak"], 1, "the worker must serialise jobs")

    def test_queue_position_counts_the_jobs_ahead(self):
        queue = self.make_queue(start=False)
        release = threading.Event()
        try:
            queue.start()
            first = queue.submit("/analyze-video", lambda: self._held(release))
            second = queue.submit("/analyze-video", lambda: self._held(release))
            third = queue.submit("/analyze-video", lambda: self._held(release))
            # A wait is only free of a race once the worker has taken the first job.
            wait_until(lambda: queue.get(first["job_id"])["status"] == RUNNING,
                       description="the first job to start")
            self.assertIsNone(queue.get(first["job_id"])["queue_position"], "a running job is not queued")
            self.assertEqual(queue.get(second["job_id"])["queue_position"], 2)
            self.assertEqual(queue.get(third["job_id"])["queue_position"], 3)
            release.set()
            wait_for(queue, third["job_id"])
            self.assertIsNone(queue.get(third["job_id"])["queue_position"])
        finally:
            release.set()

    @staticmethod
    def _held(release):
        release.wait(5)
        return FakeResponse({"ok": True})

    # ------------------------------------------------------------- failures
    def test_error_status_is_recorded_as_failed_with_its_message(self):
        queue = self.make_queue()
        record = queue.submit("/analyze-video",
            lambda: FakeResponse({"error": "Video duration is 1339.04 seconds"}, 413))
        finished = wait_for(queue, record["job_id"])
        self.assertEqual(finished["status"], FAILED)
        self.assertEqual(finished["http_status"], 413)
        self.assertEqual(finished["error"], "Video duration is 1339.04 seconds")

    def test_an_exception_in_the_job_is_recorded_and_does_not_stop_the_worker(self):
        queue = self.make_queue()

        def explode():
            raise RuntimeError("detector failed")

        broken = queue.submit("/detect-objects", explode)
        finished = wait_for(queue, broken["job_id"])
        self.assertEqual(finished["status"], FAILED)
        self.assertEqual(finished["http_status"], 500)
        self.assertIn("detector failed", finished["error"])
        healthy = queue.submit("/detect-objects", lambda: FakeResponse({"ok": True}))
        self.assertEqual(wait_for(queue, healthy["job_id"])["status"], COMPLETED)

    def test_flask_body_status_tuple_is_read_as_a_failure(self):
        # Flask's error paths return (body, status). Reading only the tuple's
        # first element would record a rejected job as a success with HTTP 200.
        queue = self.make_queue()
        record = queue.submit("/detect-objects",
            lambda: (FakeResponse({"error": "Video duration is 1339.04 seconds"}), 413))
        finished = wait_for(queue, record["job_id"])
        self.assertEqual(finished["status"], FAILED)
        self.assertEqual(finished["http_status"], 413)
        self.assertEqual(finished["error"], "Video duration is 1339.04 seconds")
        self.assertEqual(finished["response"], {"error": "Video duration is 1339.04 seconds"})

    def test_a_body_status_tuple_without_a_payload_still_records_the_status(self):
        queue = self.make_queue()
        record = queue.submit("/reports/x.pdf", lambda: (FakeStreamedResponse(), 404))
        finished = wait_for(queue, record["job_id"])
        self.assertEqual(finished["http_status"], 404)
        self.assertEqual(finished["status"], FAILED)

    def test_non_json_response_is_recorded_without_a_body(self):
        queue = self.make_queue()
        record = queue.submit("/reports/x.pdf", lambda: FakeStreamedResponse())
        finished = wait_for(queue, record["job_id"])
        self.assertEqual(finished["status"], COMPLETED)
        self.assertEqual(finished["response"], {"note": "The job returned a non-JSON response"})

    # ------------------------------------------------------------- responses
    def test_oversized_response_is_stored_as_a_preview_only(self):
        queue = self.make_queue(response_max_chars=200, preview_chars=50)
        record = queue.submit("/detect-objects", lambda: FakeResponse({"blob": "x" * 5000}))
        finished = wait_for(queue, record["job_id"])
        self.assertTrue(finished["response_truncated"])
        self.assertIsNone(finished["response"], "a truncated body is not valid JSON")
        self.assertEqual(len(finished["response_preview"]), 50)
        self.assertGreater(finished["response_bytes"], 200)

    def test_responses_are_omitted_from_the_listing_unless_requested(self):
        queue = self.make_queue()
        record = queue.submit("/analyze-image", lambda: FakeResponse({"detected_activity": ["infra"]}))
        wait_for(queue, record["job_id"])
        listed = queue.list_jobs()[0]
        self.assertIsNone(listed["response"])
        self.assertIsNone(listed["response_preview"])
        with_response = queue.list_jobs(include_response=True)[0]
        self.assertEqual(with_response["response"], {"detected_activity": ["infra"]})

    # ------------------------------------------------------------- listing
    def test_listing_is_newest_first_and_filterable(self):
        queue = self.make_queue()
        good = queue.submit("/analyze-image", lambda: FakeResponse({"ok": True}))
        bad = queue.submit("/analyze-video", lambda: FakeResponse({"error": "nope"}, 400))
        wait_for(queue, good["job_id"])
        wait_for(queue, bad["job_id"])
        self.assertEqual([job["job_id"] for job in queue.list_jobs()], [bad["job_id"], good["job_id"]])
        self.assertEqual([job["job_id"] for job in queue.list_jobs(status=FAILED)], [bad["job_id"]])
        self.assertEqual(queue.list_jobs(status=QUEUED), [])
        self.assertEqual(queue.list_jobs(limit=1000)[0]["seq"] > 0, True)

    def test_listing_pages_through_the_history(self):
        queue = self.make_queue()
        records = [queue.submit("/analyze-image", lambda index=index: FakeResponse({"index": index}))
                   for index in range(5)]
        wait_for(queue, records[-1]["job_id"])
        self.assertEqual(queue.count_jobs(), 5)
        self.assertEqual(queue.count_jobs(status=COMPLETED), 5)
        self.assertEqual(queue.count_jobs(status=FAILED), 0)
        first = queue.list_jobs(limit=2, include_response=True)
        self.assertEqual([job["response"] for job in first], [{"index": 4}, {"index": 3}])
        second = queue.list_jobs(limit=2, offset=2, include_response=True)
        self.assertEqual([job["response"] for job in second], [{"index": 2}, {"index": 1}])
        last = queue.list_jobs(limit=2, offset=4, include_response=True)
        self.assertEqual([job["response"] for job in last], [{"index": 0}])
        self.assertEqual(queue.list_jobs(limit=2, offset=99), [])

    def test_each_job_reports_the_wall_clock_time_it_cost(self):
        queue = self.make_queue()
        release = threading.Event()
        try:
            held = queue.submit("/analyze-video", lambda: self._held(release))
            self.assertIsNone(held["total_seconds"], "an unfinished job has no total time")
            release.set()
            finished = wait_for(queue, held["job_id"])
            self.assertIsNotNone(finished["total_seconds"])
            # Total is submission-to-result, so it also covers the queue wait.
            self.assertGreaterEqual(finished["total_seconds"], finished["duration_seconds"])
        finally:
            release.set()

    def test_listing_and_stats_report_the_queue_state(self):
        queue = self.make_queue()
        record = queue.submit("/analyze-image", lambda: FakeResponse({"ok": True}))
        wait_for(queue, record["job_id"])
        stats = queue.stats()
        self.assertEqual(stats["completed"], 1)
        self.assertEqual(stats["cancelled"], 0)
        self.assertEqual(stats["queued"], 0)
        self.assertEqual(stats["running"], 0)
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["pending"], 0)
        self.assertEqual(stats["concurrency"], 1)
        self.assertTrue(stats["worker_alive"])
        self.assertIsNone(queue.get("no-such-job"))

    def test_history_keeps_the_newest_finished_jobs_only(self):
        queue = self.make_queue(history_limit=2)
        records = [queue.submit("/analyze-image", lambda index=index: FakeResponse({"index": index}))
                   for index in range(4)]
        # Wait on the last job only: the first two are pruned as soon as they
        # finish, and completion is recorded in the same critical section as the
        # prune, so seeing the final job done means the history is settled.
        wait_for(queue, records[-1]["job_id"])
        remaining = queue.list_jobs(include_response=True)
        self.assertEqual([job["response"] for job in remaining],
                         [{"index": 3}, {"index": 2}])

    # ------------------------------------------------------------- capacity
    def test_submit_sheds_load_once_the_backlog_is_full(self):
        queue = self.make_queue(max_pending=1)
        release = threading.Event()
        try:
            held = queue.submit("/analyze-video", lambda: self._held(release))
            with self.assertRaises(JobQueueFull):
                queue.submit("/analyze-video", lambda: FakeResponse({"ok": True}))
            self.assertEqual(queue.stats()["pending"], 1)
            self.assertFalse(held["status"] in {COMPLETED, FAILED})
        finally:
            release.set()

    # ------------------------------------------------------------- progress
    def test_progress_is_stored_while_a_job_runs_and_lands_on_100(self):
        queue = self.make_queue()
        release = threading.Event()

        def work():
            report_progress(45, "half way")
            release.wait(5)
            return FakeResponse({"ok": True})

        record = queue.submit("/analyze-video", work)
        try:
            wait_until(lambda: queue.get(record["job_id"])["progress"] == 45,
                       description="the reported progress to be stored")
            midway = queue.get(record["job_id"])
            self.assertEqual(midway["status"], RUNNING)
            self.assertEqual(midway["progress_detail"], "half way")
        finally:
            release.set()
        self.assertEqual(wait_for(queue, record["job_id"])["progress"], 100)

    def test_progress_is_clamped_to_a_percentage(self):
        queue = self.make_queue()
        release = threading.Event()

        def work():
            report_progress(-40, "negative")
            report_progress(5000, "too far")
            release.wait(5)
            return FakeResponse({"ok": True})

        record = queue.submit("/analyze-video", work)
        try:
            wait_until(lambda: queue.get(record["job_id"])["progress_detail"] == "too far",
                       description="the clamped progress to be stored")
            self.assertEqual(queue.get(record["job_id"])["progress"], 100)
        finally:
            release.set()

    def test_a_failed_job_keeps_the_progress_it_reached(self):
        queue = self.make_queue()

        def work():
            report_progress(37, "annotating frame 3/8")
            raise RuntimeError("boom")

        finished = wait_for(queue, queue.submit("/detect-objects", work)["job_id"])
        self.assertEqual(finished["status"], FAILED)
        self.assertEqual(finished["progress"], 37)

    def test_reporting_progress_outside_the_worker_does_nothing(self):
        queue = self.make_queue()
        finished = wait_for(queue, queue.submit("/analyze-image", lambda: FakeResponse({"ok": True}))["job_id"])
        # The worker clears the thread-local when the job ends, and the test thread
        # never had one, so the pipelines that run inline stay untouched.
        report_progress(3, "nobody is listening")
        self.assertEqual(queue.get(finished["job_id"])["progress"], 100)

    def test_an_existing_database_gains_the_progress_columns(self):
        path = self.workdir / "legacy.sqlite3"
        legacy = sqlite3.connect(str(path))
        legacy.executescript("""
            CREATE TABLE jobs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL UNIQUE,
                endpoint TEXT NOT NULL, status TEXT NOT NULL, filename TEXT, media_type TEXT,
                client TEXT, detail TEXT, created_at REAL NOT NULL, started_at REAL,
                finished_at REAL, http_status INTEGER, execution_seconds REAL, wait_seconds REAL,
                response TEXT, response_bytes INTEGER, response_truncated INTEGER NOT NULL DEFAULT 0,
                error TEXT);
            INSERT INTO jobs (job_id, endpoint, status, created_at)
                VALUES ('legacy', '/analyze-image', 'completed', 1.0);
        """)
        legacy.commit()
        legacy.close()
        queue = self.make_queue(start=False, db_path=path, files_dir=self.workdir / "legacy-uploads")
        record = queue.get("legacy")
        self.assertIsNotNone(record)
        self.assertEqual(record["progress"], 0)
        self.assertIsNone(record["progress_detail"])
        self.assertTrue(record["deletable"])

    # ------------------------------------------------------------- deletion
    def test_only_finished_jobs_can_be_deleted(self):
        queue = self.make_queue()
        release = threading.Event()
        try:
            held = queue.submit("/analyze-video", lambda: self._held(release))
            self.assertEqual(queue.delete(held["job_id"]), "pending")
            self.assertIsNotNone(queue.get(held["job_id"]), "a pending job must survive a delete")
            self.assertEqual(queue.delete("no-such-job"), "missing")
        finally:
            release.set()
        wait_until(lambda: queue.get(held["job_id"])["status"] == COMPLETED,
                   description="the held job to finish")
        self.assertEqual(queue.delete(held["job_id"]), "deleted")
        self.assertIsNone(queue.get(held["job_id"]))
        self.assertEqual(queue.count_jobs(), 0)

    def test_clearing_history_keeps_pending_jobs(self):
        queue = self.make_queue()
        release = threading.Event()
        try:
            done = queue.submit("/analyze-image", lambda: FakeResponse({"ok": True}))
            wait_for(queue, done["job_id"])
            held = queue.submit("/analyze-video", lambda: self._held(release))
            queued = queue.submit("/analyze-image", lambda: FakeResponse({"error": "nope"}, 400))
            wait_until(lambda: queue.get(held["job_id"])["status"] == RUNNING,
                       description="the held job to start")
            self.assertEqual(queue.clear_finished(), 1, "only the finished job may go")
            self.assertIsNone(queue.get(done["job_id"]))
            self.assertIsNotNone(queue.get(held["job_id"]))
            self.assertEqual(queue.get(queued["job_id"])["status"], QUEUED)
        finally:
            release.set()
        wait_for(queue, queued["job_id"])
        self.assertEqual(queue.clear_finished(), 2)
        self.assertEqual(queue.count_jobs(), 0)

    # ------------------------------------------------------------- cancel
    def test_cancelling_a_running_job_stops_it_at_its_next_checkpoint(self):
        queue = self.make_queue()
        steps = []

        def work():
            for index in range(200):
                steps.append(index)
                report_progress(index % 100, f"step {index}")
                time.sleep(0.01)
            return FakeResponse({"ok": True})

        record = queue.submit("/analyze-video", work)
        wait_until(lambda: queue.get(record["job_id"])["status"] == RUNNING,
                   description="the job to start")
        self.assertEqual(queue.cancel(record["job_id"]), "cancelled")
        finished = wait_for(queue, record["job_id"])
        self.assertEqual(finished["status"], CANCELLED)
        self.assertIn("Cancelled", finished["error"])
        self.assertLess(len(steps), 200, "the loop must stop before running to completion")
        self.assertTrue(finished["deletable"])

    def test_cancelling_a_queued_job_marks_it_cancelled_at_once(self):
        queue = self.make_queue()
        release = threading.Event()
        ran = []
        try:
            held = queue.submit("/analyze-video", lambda: self._held(release))
            wait_until(lambda: queue.get(held["job_id"])["status"] == RUNNING,
                       description="the first job to start")
            queued = queue.submit("/analyze-image",
                lambda: (ran.append(1), FakeResponse({"ok": True}))[1])
            self.assertEqual(queued["status"], QUEUED)
            self.assertEqual(queue.cancel(queued["job_id"]), "cancelled")
            cancelled = queue.get(queued["job_id"])
            self.assertEqual(cancelled["status"], CANCELLED)
            self.assertIn("before it started", cancelled["error"])
        finally:
            release.set()
        wait_for(queue, held["job_id"])
        time.sleep(0.2)
        self.assertEqual(ran, [], "a job cancelled while queued must never run")
        self.assertEqual(queue.get(queued["job_id"])["status"], CANCELLED)

    def test_cancelling_refuses_finished_and_unknown_jobs(self):
        queue = self.make_queue()
        done = queue.submit("/analyze-image", lambda: FakeResponse({"ok": True}))
        wait_for(queue, done["job_id"])
        self.assertEqual(queue.cancel(done["job_id"]), "finished")
        self.assertEqual(queue.cancel("no-such-job"), "missing")

    def test_a_cancelled_job_does_not_stop_the_worker(self):
        queue = self.make_queue()

        def work():
            for _ in range(1000):
                report_progress(1, "working")
                time.sleep(0.02)
            return FakeResponse({"ok": True})

        blocked = queue.submit("/analyze-video", work)
        wait_until(lambda: queue.get(blocked["job_id"])["status"] == RUNNING,
                   description="the long job to start")
        queue.cancel(blocked["job_id"])
        self.assertEqual(wait_for(queue, blocked["job_id"])["status"], CANCELLED)
        healthy = queue.submit("/analyze-image", lambda: FakeResponse({"ok": True}))
        self.assertEqual(wait_for(queue, healthy["job_id"])["status"], COMPLETED)

    def test_a_step_inside_one_long_call_finishes_before_the_cancel_is_seen(self):
        # Cancellation is cooperative. The pipelines report progress between stages
        # and on every frame, so this only applies to work already inside a single
        # model call -- which the API documents rather than pretending otherwise.
        queue = self.make_queue()
        release = threading.Event()

        def work():
            release.wait(5)
            return FakeResponse({"ok": True})

        record = queue.submit("/analyze-video", work)
        wait_until(lambda: queue.get(record["job_id"])["status"] == RUNNING,
                   description="the job to start")
        self.assertEqual(queue.cancel(record["job_id"]), "cancelled")
        release.set()
        self.assertEqual(wait_for(queue, record["job_id"])["status"], COMPLETED)

    def test_clearing_history_includes_cancelled_jobs(self):
        queue = self.make_queue()
        release = threading.Event()
        try:
            held = queue.submit("/analyze-video", lambda: self._held(release))
            wait_until(lambda: queue.get(held["job_id"])["status"] == RUNNING,
                       description="the held job to start")
            queued = queue.submit("/analyze-image", lambda: FakeResponse({"ok": True}))
            queue.cancel(queued["job_id"])
            self.assertEqual(queue.stats()["cancelled"], 1)
            self.assertEqual(queue.clear_finished(), 1)
            self.assertIsNone(queue.get(queued["job_id"]))
        finally:
            release.set()

    # ------------------------------------------------------------- reload
    def test_replay_input_exposes_what_a_reload_needs(self):
        queue = self.make_queue()
        staged = self.workdir / "uploads" / "clip.mp4"
        staged.write_bytes(b"pretend video")
        record = queue.submit("/analyze-video", lambda: FakeResponse({"ok": True}),
            filename="clip.mp4", media_type="video", input_path=str(staged))
        self.assertTrue(record["replayable"])
        self.assertEqual(queue.replay_input(record["job_id"]),
                         ("/analyze-video", str(staged), None, "clip.mp4", "video"))
        self.assertIsNone(queue.replay_input("no-such-job"))
        wait_for(queue, record["job_id"])
        self.assertTrue(staged.exists(), "a finished job keeps its upload for reload")
        self.assertEqual(queue.delete(record["job_id"]), "deleted")
        self.assertFalse(staged.exists(), "deleting the row releases its retained upload")

    def test_a_record_backed_job_is_replayable_without_a_file(self):
        queue = self.make_queue()
        record = queue.submit("/detect-objects", lambda: FakeResponse({"ok": True}),
            input_ref='{"id": 999902}')
        self.assertTrue(record["replayable"])
        self.assertEqual(queue.replay_input(record["job_id"])[2], '{"id": 999902}')

    def test_retained_uploads_are_pruned_to_the_newest_few(self):
        queue = self.make_queue(replay_retention=2)
        staged = []
        for index in range(4):
            path = self.workdir / "uploads" / f"clip{index}.mp4"
            path.write_bytes(b"x" * 32)
            staged.append(path)
            record = queue.submit("/analyze-video", lambda: FakeResponse({"ok": True}),
                input_path=str(path))
            wait_for(queue, record["job_id"])
        self.assertFalse(staged[0].exists())
        self.assertFalse(staged[1].exists())
        self.assertTrue(staged[2].exists())
        self.assertTrue(staged[3].exists())
        # Only the retained input is released; the history itself stays.
        self.assertEqual(queue.count_jobs(), 4)
        self.assertFalse(queue.list_jobs(limit=4)[-1]["replayable"])

    def test_a_shared_retained_upload_outlives_the_row_it_was_replayed_from(self):
        queue = self.make_queue()
        staged = self.workdir / "uploads" / "shared.mp4"
        staged.write_bytes(b"video")
        first = queue.submit("/analyze-video", lambda: FakeResponse({"ok": True}), input_path=str(staged))
        wait_for(queue, first["job_id"])
        replayed = queue.submit("/analyze-video", lambda: FakeResponse({"ok": True}), input_path=str(staged))
        wait_for(queue, replayed["job_id"])
        self.assertEqual(queue.delete(first["job_id"]), "deleted")
        self.assertTrue(staged.exists(), "the replayed row still needs the upload")
        self.assertEqual(queue.delete(replayed["job_id"]), "deleted")
        self.assertFalse(staged.exists())

    def test_a_restart_releases_retained_uploads_and_their_references(self):
        queue = self.make_queue(start=False)
        staged = self.workdir / "uploads" / "clip.mp4"
        staged.write_bytes(b"video")
        record = queue.submit("/analyze-video", lambda: FakeResponse({"ok": True}), input_path=str(staged))
        recovered = JobQueue(db_path=self.workdir / "jobs.sqlite3", files_dir=self.workdir / "uploads")
        self.started.append(recovered)
        recovered.start()
        self.assertFalse(staged.exists())
        self.assertFalse(recovered.get(record["job_id"])["replayable"],
                         "a vanished upload must not be advertised as replayable")

    # ------------------------------------------------------------- restart
    def test_a_restart_fails_in_flight_jobs_and_purges_staged_uploads(self):
        queue = self.make_queue(start=False)
        staged = self.workdir / "uploads" / "activity_staged.mp4"
        staged.write_bytes(b"pretend video")
        queued = queue.submit("/analyze-video", lambda: FakeResponse({"ok": True}))
        self.assertEqual(queued["status"], QUEUED)
        recovered = JobQueue(db_path=self.workdir / "jobs.sqlite3", files_dir=self.workdir / "uploads")
        self.started.append(recovered)
        recovered.start()
        after = recovered.get(queued["job_id"])
        self.assertEqual(after["status"], FAILED)
        self.assertIn("restart", after["error"])
        self.assertFalse(staged.exists(), "uploads owned by an abandoned job must be removed")


    # ------------------------------------------------------------- inline jobs
    def test_record_inline_runs_in_the_caller_thread_and_is_listed(self):
        queue = self.make_queue()
        seen = []

        def work():
            seen.append(threading.get_ident())
            return FakeResponse({"id": 7, "status": 1})

        result = queue.record_inline("/fetch-media", work, media_type="db-record",
                                     detail="media_analysis.id=7", client="10.0.0.9")
        self.assertEqual(seen, [threading.get_ident()], "inline work must run in the caller's thread")
        self.assertEqual(result.get_json(), {"id": 7, "status": 1})
        record = queue.list_jobs(include_response=True)[0]
        self.assertEqual(record["endpoint"], "/fetch-media")
        self.assertEqual(record["status"], COMPLETED)
        self.assertEqual(record["response"], {"id": 7, "status": 1})
        self.assertEqual(record["client"], "10.0.0.9")
        self.assertEqual(record["http_status"], 200)
        self.assertIsNone(record["queue_position"], "an inline job never waited in line")
        self.assertTrue(record["deletable"])
        self.assertFalse(record["cancellable"])

    def test_record_inline_records_a_failure_and_re_raises(self):
        queue = self.make_queue()

        def explode():
            raise ValueError("bad media")

        with self.assertRaises(ValueError):
            queue.record_inline("/fetch-media", explode, media_type="db-record")
        record = queue.list_jobs()[0]
        self.assertEqual(record["status"], FAILED)
        self.assertIn("bad media", record["error"])

    def test_record_inline_reads_a_body_status_tuple_as_a_failure(self):
        queue = self.make_queue()
        result = queue.record_inline("/fetch-media",
                                     lambda: (FakeResponse({"error": "No record found"}), 404))
        self.assertEqual(result[1], 404, "the caller's response is returned unchanged")
        record = queue.list_jobs()[0]
        self.assertEqual(record["status"], FAILED)
        self.assertEqual(record["http_status"], 404)
        self.assertEqual(record["error"], "No record found")


if __name__ == "__main__":
    unittest.main()
