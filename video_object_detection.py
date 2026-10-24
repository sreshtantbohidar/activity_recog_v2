"""One-pass video object detection with a bounded JSON evidence trail.

The caller supplies a loaded Ultralytics-style model, so this module deliberately
does not import ultralytics and can be unit-tested with a fake model. Every frame
is annotated into the output clip, while the per-frame detections reported in the
response are sampled and capped: a long clip must not be able to inflate the
queued job response (and the SQLite history) without bound.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

DEFAULT_TRACKER = "bytetrack.yaml"


def record_track(tracks: dict, track_id: Any, label: str, confidence: float,
                 timestamp: float) -> None:
    """Fold one detection into the persistent-track summary.

    A tracker may report ``None`` or a negative id for a detection it could not
    associate. Those are still drawn and counted per frame, but they cannot form
    a trajectory, so they are not added to ``tracks``.
    """
    if track_id is None:
        return
    try:
        numeric_id = int(track_id)
    except (TypeError, ValueError):
        return
    if numeric_id < 0:
        return
    key = str(numeric_id)
    track = tracks.get(key)
    if track is None:
        tracks[key] = {
            "track_id": numeric_id,
            "class": label,
            "first_seen_sec": timestamp,
            "last_seen_sec": timestamp,
            "observations": 1,
            "max_confidence": confidence,
        }
        return
    track["last_seen_sec"] = timestamp
    track["observations"] += 1
    track["max_confidence"] = max(track["max_confidence"], confidence)


def summarize_tracks(tracks: dict) -> tuple[list, dict]:
    """Return tracks ordered by id plus one count per uniquely tracked class."""
    ordered = sorted(tracks.values(), key=lambda item: item["track_id"])
    counts: dict = {}
    for track in ordered:
        counts[track["class"]] = counts.get(track["class"], 0) + 1
    return ordered, counts


def _boxes(result: Any) -> list:
    """Flatten a result's boxes into (id, class, confidence, xyxy) rows."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    ids = boxes.id.int().cpu().tolist() if getattr(boxes, "id", None) is not None \
        else [-1] * len(boxes)
    return list(zip(ids, boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                    boxes.xyxy.cpu().tolist()))


def annotate_with_detections(model, source_path, destination_path, *,
        progress: Callable[[float, str], None] | None = None,
        sample_interval_seconds: float = 1.0, max_sampled_frames: int = 30,
        max_detections: int = 300, max_tracks: int = 500, confidence: float = 0.25,
        image_size: int = 640, device: int = 0,
        tracker: str = DEFAULT_TRACKER) -> dict:
    """Detect/track objects in every frame, annotate a copy, and summarise.

    ``progress`` doubles as the queue's cancellation checkpoint: it is called
    once per frame, so a cancelled job stops at the next decoded frame rather
    than finishing the whole clip. A failed or cancelled run removes its partial
    artefact so the reports directory never serves a truncated clip.
    """
    import cv2

    source_path, destination_path = Path(source_path), Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Video contains no decodable frames")

    writer = None
    tracks: dict = {}
    detections: list = []
    peak_counts: dict = {}
    frames = 0
    sampled_objects = 0
    truncated = False
    next_sample = 0.0
    success = False
    fps = 25.0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Video contains no decodable frames")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        writer = cv2.VideoWriter(str(destination_path),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not open the annotated video writer")

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.track(frame, persist=True, tracker=tracker,
                conf=confidence, imgsz=image_size, device=device, verbose=False)[0]
            writer.write(result.plot())

            timestamp = frames / fps
            rows = _boxes(result)
            if rows:
                concurrent: dict = {}
                for track_id, cls, score, box in rows:
                    label = result.names[int(cls)]
                    record_track(tracks, track_id, label, round(float(score), 4),
                        round(timestamp, 2))
                    concurrent[label] = concurrent.get(label, 0) + 1
                for label, count in concurrent.items():
                    peak_counts[label] = max(peak_counts.get(label, 0), count)

            if timestamp + 1e-9 >= next_sample:
                if len(detections) >= max_sampled_frames:
                    # This frame was due for sampling but the response budget is
                    # already spent; the video is still annotated below.
                    truncated = True
                else:
                    objects = []
                    for track_id, cls, score, box in rows:
                        if sampled_objects >= max_detections:
                            truncated = True
                            break
                        objects.append({
                            "track_id": int(track_id),
                            "class": result.names[int(cls)],
                            "confidence": round(float(score), 4),
                            "bbox": [round(float(value), 2) for value in box],
                        })
                        sampled_objects += 1
                    detections.append({"frame_index": frames,
                        "timestamp_sec": round(timestamp, 2), "objects": objects})
                if sample_interval_seconds > 0:
                    next_sample = timestamp + float(sample_interval_seconds)
                else:
                    next_sample = timestamp

            frames += 1
            if progress:
                percent = 15 + (75 * frames / total_frames) if total_frames > 0 else 15
                progress(percent, f"annotating frame {frames}/{total_frames or '?'}")
        if frames == 0:
            raise ValueError("Video contains no decodable frames")
        success = True
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        if not success:
            destination_path.unlink(missing_ok=True)

    track_list, counts = summarize_tracks(tracks)
    tracks_truncated = len(track_list) > max_tracks
    track_list = track_list[:max(1, max_tracks)]

    return {
        "frames_analyzed": frames,
        "frames_annotated": frames,
        "frames_sampled": len(detections),
        "duration_sec": round(frames / fps, 3),
        "fps": round(fps, 3),
        "sample_interval_sec": float(sample_interval_seconds),
        "detections_truncated": bool(truncated),
        "detections": detections,
        "tracks": track_list,
        "tracks_truncated": bool(tracks_truncated),
        "detected_object_counts": counts,
        "max_concurrent_counts": peak_counts,
        "annotated_video_path": str(destination_path),
    }
