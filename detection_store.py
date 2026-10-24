"""Persist object-detection results on ``public.media_analysis``.

The legacy service always wrote a finished detection back to its media row
(``analysis_summary``, ``response_data``, ``output_file_path``, ``status=1``) for
images and videos alike; the annotated artefact was published to SFTP first.
``response_data`` carries the same result for the UI, whose detail view reads
that column. The upload endpoints have no row of their own, so the caller names
one and the same update applies.

Only that contract lives here. The connection pool and the publisher are
injected, which keeps the module importable -- and testable -- without Flask,
psycopg2 or a database.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

UPDATE_SQL = ("UPDATE public.media_analysis SET analysis_summary=%s, response_data=%s,"
    " output_file_path=COALESCE(%s, output_file_path), status=%s WHERE id=%s")


def save_detection(record_id: int, result: Dict[str, Any], artifact: Optional[str] = None, *,
                   cursor_factory: Callable,
                   publish_fn: Optional[Callable] = None) -> Tuple[Optional[str], Optional[str]]:
    """Save a finished detection on its media row; returns ``(output_url, warning)``.

    ``cursor_factory`` is the app's pooled ``db_cursor`` context manager and
    ``publish_fn`` its SFTP publisher; both are injected so the contract can be
    exercised with fakes. Persistence is best-effort -- a completed detection is
    worth more than a failed job -- so every failure is reported as a warning
    rather than raised. ``output_file_path`` is only replaced when a new artefact
    was actually published, so a failed upload never blanks a working URL.
    """
    output_url, warning = None, None
    try:
        with cursor_factory() as (conn, cursor):
            cursor.execute("SELECT media_type FROM public.media_analysis WHERE id = %s",
                (record_id,))
            row = cursor.fetchone()
            if not row:
                return None, f"media_analysis.id={record_id} does not exist; result not saved"
            media_type = row[0] or "image"
            if artifact and publish_fn is not None:
                try:
                    output_url = publish_fn(artifact, media_type)
                except Exception as exc:
                    logger.warning("Publishing the annotated %s for id=%s failed",
                        media_type, record_id, exc_info=True)
                    warning = f"annotated artefact not published: {exc}"
            cursor.execute(UPDATE_SQL,
                (json.dumps(result), json.dumps(result), output_url, 1, record_id))
            conn.commit()
    except Exception as exc:
        logger.warning("Saving the detection for media_analysis.id=%s failed",
            record_id, exc_info=True)
        return None, f"result not saved to PostgreSQL: {exc}"
    return output_url, warning
