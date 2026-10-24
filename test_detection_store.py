"""The object-detection DB save, exercised with fakes.

Both detection endpoints share one contract: a finished run is written back to
``public.media_analysis`` (``analysis_summary``, ``response_data``,
``output_file_path``, ``status=1``) the way the legacy service did it for images
and videos. ``response_data`` mirrors the summary because the UI detail view
reads that column. These checks pin that contract, including the failure modes
that must not cost a caller its detection.
"""
from __future__ import annotations

import contextlib
import json
import unittest

from detection_store import UPDATE_SQL, save_detection


class FakeCursor:
    def __init__(self, row=("video",), error=None):
        self.row = row
        self.error = error
        self.statements = []

    def execute(self, sql, parameters=None):
        if self.error:
            raise self.error
        self.statements.append((sql, parameters))

    def fetchone(self):
        return self.row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor
        self.commits = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1


def factory(cursor, connection):
    @contextlib.contextmanager
    def cursor_factory():
        yield connection, cursor
    return cursor_factory


class SaveDetectionTests(unittest.TestCase):
    def test_existing_row_is_updated_with_summary_output_and_status(self):
        cursor, connection = FakeCursor(), FakeConnection(FakeCursor())
        connection.cursor_instance = cursor
        result = {"detected_object_counts": {"person": 1}}

        output_url, warning = save_detection(4, result, "/tmp/annotated.mp4",
            cursor_factory=factory(cursor, connection),
            publish_fn=lambda path, media_type: f"https://example.test/{media_type}/{path.split('/')[-1]}")

        self.assertIsNone(warning)
        self.assertEqual(output_url, "https://example.test/video/annotated.mp4")
        self.assertEqual(connection.commits, 1)
        update_sql, parameters = cursor.statements[1]
        self.assertEqual(update_sql, UPDATE_SQL)
        self.assertIn("response_data", update_sql)
        stored_summary, stored_response, stored_url, stored_status, stored_id = parameters
        self.assertEqual(json.loads(stored_summary), result)
        self.assertEqual(json.loads(stored_response), result)
        self.assertEqual(stored_url, output_url)
        self.assertEqual(stored_status, 1)
        self.assertEqual(stored_id, 4)

    def test_missing_row_warns_and_never_updates(self):
        cursor, connection = FakeCursor(row=None), FakeConnection(FakeCursor())
        connection.cursor_instance = cursor

        output_url, warning = save_detection(999404, {}, cursor_factory=factory(cursor, connection),
            publish_fn=lambda *_: "https://example.test/unused")

        self.assertIsNone(output_url)
        self.assertIn("does not exist", warning)
        self.assertEqual(len(cursor.statements), 1)  # the SELECT only
        self.assertEqual(connection.commits, 0)

    def test_publish_failure_still_saves_and_leaves_the_existing_url(self):
        cursor, connection = FakeCursor(), FakeConnection(FakeCursor())
        connection.cursor_instance = cursor

        class SftpError(RuntimeError):
            pass

        def explode(path, media_type):
            raise SftpError("SFTP is not configured")

        with self.assertLogs("detection_store", "WARNING"):
            output_url, warning = save_detection(7, {"detections": []}, "/tmp/clip.mp4",
                cursor_factory=factory(cursor, connection), publish_fn=explode)

        self.assertIsNone(output_url)
        self.assertIn("not published", warning)
        update_sql, parameters = cursor.statements[1]
        # COALESCE keeps the stored URL: a failed publish must not blank it.
        self.assertIn("COALESCE", update_sql)
        self.assertIsNone(parameters[2])
        self.assertEqual(connection.commits, 1)

    def test_unexpected_publish_error_is_also_best_effort(self):
        cursor, connection = FakeCursor(), FakeConnection(FakeCursor())
        connection.cursor_instance = cursor

        def explode(path, media_type):
            raise RuntimeError("boom")

        with self.assertLogs("detection_store", "WARNING"):
            output_url, warning = save_detection(7, {}, "/tmp/clip.mp4",
                cursor_factory=factory(cursor, connection), publish_fn=explode)

        self.assertIsNone(output_url)
        self.assertIn("not published", warning)
        self.assertEqual(connection.commits, 1)

    def test_database_failure_is_reported_not_raised(self):
        cursor = FakeCursor(error=RuntimeError("connection refused"))
        connection = FakeConnection(cursor)

        with self.assertLogs("detection_store", "WARNING"):
            output_url, warning = save_detection(4, {}, cursor_factory=factory(cursor, connection))

        self.assertIsNone(output_url)
        self.assertIn("not saved to PostgreSQL", warning)

    def test_no_artifact_skips_publishing(self):
        cursor, connection = FakeCursor(), FakeConnection(FakeCursor())
        connection.cursor_instance = cursor
        calls = []

        output_url, warning = save_detection(4, {}, None, cursor_factory=factory(cursor, connection),
            publish_fn=lambda *args: calls.append(args) or "https://example.test/x")

        self.assertIsNone(output_url)
        self.assertIsNone(warning)
        self.assertEqual(calls, [])
        self.assertEqual(connection.commits, 1)


class DetectionEndpointWiringTests(unittest.TestCase):
    """Both detection endpoints must reach the shared save, upload mode too."""

    def test_app_wires_the_store_into_both_detection_bodies(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parent/"app.py").read_text(errors="replace")
        self.assertIn("from detection_store import save_detection", source)
        for body in ("_queued_object_detection", "_queued_video_object_detection"):
            self.assertIn(f"def {body}(", source)
        # The save lives in one helper that both queued bodies call.
        self.assertEqual(source.count("payload.update(_persistence_fields("), 2)
        # /detect-objects-video exposes the DB-driven mode as well.
        self.assertIn('lambda: detect_objects_by_id(record_id)', source)

    def test_db_driven_video_uses_the_full_detection_result(self):
        """A video row must keep detections/tracks, not just a frame count.

        The DB-driven body shares _write_annotated_artifact with images. For a
        video that helper must run detect_video (evidence + retained clip), not
        annotate_video (annotation count only), otherwise the row saved to
        media_analysis.analysis_summary loses the whole result.
        """
        from pathlib import Path
        source = (Path(__file__).resolve().parent/"app.py").read_text(errors="replace")
        body = source.split("def _write_annotated_artifact", 1)[1] \
            .split("def detect_objects_by_id", 1)[0]
        self.assertIn("detector.detect_video(source_path)", body)
        self.assertNotIn("detector.annotate_video", body)
        self.assertIn("result[\"annotated_video_path\"]", body)
        # The DB-driven save also fills response_data, which the UI detail reads.
        save = source.split("def detect_objects_by_id", 1)[1].split("@app.route", 1)[0]
        self.assertIn("response_data=%s", save)


if __name__ == "__main__":
    unittest.main()
