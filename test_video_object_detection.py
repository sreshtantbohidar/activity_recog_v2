import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from video_object_detection import annotate_with_detections, record_track, summarize_tracks


class _Tensor:
    """Minimal tensor stand-in exposing the Ultralytics tolist() contract."""

    def __init__(self, values):
        self.values = list(values)

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _Boxes:
    def __init__(self, rows):
        self.id = _Tensor([row[0] for row in rows])
        self.cls = _Tensor([row[1] for row in rows])
        self.conf = _Tensor([row[2] for row in rows])
        self.xyxy = _Tensor([row[3] for row in rows])

    def __len__(self):
        return len(self.cls.values)


class _Result:
    names = {0: "person", 1: "car"}

    def __init__(self, frame, rows):
        self._frame = frame
        self.boxes = _Boxes(rows)

    def plot(self):
        return self._frame


class _FakeModel:
    names = {0: "person", 1: "car"}

    def __init__(self, rows_for_frame):
        self.rows_for_frame = rows_for_frame
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [_Result(frame, self.rows_for_frame(len(self.calls) - 1))]


def _write_clip(path, frames=5, fps=10.0, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV cannot create an mp4v clip in this environment")
    for index in range(frames):
        writer.write(np.full((size[1], size[0], 3), index * 30, np.uint8))
    writer.release()


def _frame_count(path):
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


class TrackSummaryTests(unittest.TestCase):
    def test_record_track_keeps_first_class_and_peak_confidence(self):
        tracks = {}
        record_track(tracks, 7, "person", 0.6, 0.0)
        record_track(tracks, 7, "car", 0.9, 0.5)
        record_track(tracks, -1, "person", 0.8, 0.5)
        record_track(tracks, None, "person", 0.8, 0.5)

        ordered, counts = summarize_tracks(tracks)
        self.assertEqual(counts, {"person": 1})
        self.assertEqual(len(ordered), 1)
        self.assertEqual(ordered[0]["class"], "person")
        self.assertEqual(ordered[0]["observations"], 2)
        self.assertEqual(ordered[0]["max_confidence"], 0.9)
        self.assertEqual(ordered[0]["first_seen_sec"], 0.0)
        self.assertEqual(ordered[0]["last_seen_sec"], 0.5)


class AnnotatedVideoTests(unittest.TestCase):
    def test_annotates_every_frame_and_samples_bounded_evidence(self):
        def rows(index):
            if index == 0:
                return [(1, 0, 0.9, [1.0, 1.0, 10.0, 10.0])]
            return [(1, 0, 0.8, [2.0, 2.0, 12.0, 12.0]),
                    (2, 1, 0.7, [20.0, 20.0, 30.0, 30.0])]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source.mp4"
            destination = Path(directory)/"annotated.mp4"
            _write_clip(source)
            model = _FakeModel(rows)
            progress = []

            result = annotate_with_detections(model, source, destination,
                progress=lambda percent, detail: progress.append((percent, detail)),
                sample_interval_seconds=0.1, max_sampled_frames=3,
                max_detections=10, confidence=0.3, image_size=320, device=-1)

            self.assertEqual(result["frames_analyzed"], 5)
            self.assertEqual(result["frames_annotated"], 5)
            self.assertEqual(result["frames_sampled"], 3)
            self.assertTrue(result["detections_truncated"])
            self.assertFalse(result["tracks_truncated"])
            self.assertEqual(len(model.calls), 5)
            self.assertTrue(destination.exists())
            self.assertEqual(_frame_count(destination), 5)
            self.assertEqual([detail for _, detail in progress],
                [f"annotating frame {index}/5" for index in range(1, 6)])

        self.assertEqual(result["detected_object_counts"], {"person": 1, "car": 1})
        self.assertEqual(result["max_concurrent_counts"], {"person": 1, "car": 1})
        tracks = {track["track_id"]: track for track in result["tracks"]}
        self.assertEqual(set(tracks), {1, 2})
        self.assertEqual(tracks[1]["observations"], 5)
        self.assertEqual(tracks[1]["last_seen_sec"], 0.4)
        self.assertEqual(tracks[1]["max_confidence"], 0.9)

    def test_detection_payload_is_capped_independently_of_annotation(self):
        def rows(index):
            return [(1, 0, 0.9, [1.0, 1.0, 10.0, 10.0]),
                    (2, 1, 0.8, [20.0, 20.0, 30.0, 30.0])]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source.mp4"
            destination = Path(directory)/"annotated.mp4"
            _write_clip(source)

            result = annotate_with_detections(_FakeModel(rows), source, destination,
                sample_interval_seconds=0.1, max_sampled_frames=10,
                max_detections=2, max_tracks=1, device=-1)

            self.assertEqual(result["frames_sampled"], 5)
            self.assertEqual(sum(len(frame["objects"]) for frame in result["detections"]), 2)
            self.assertTrue(result["detections_truncated"])
            # Counts are computed before the track list is capped, so the summary
            # still reports both objects even though only one trajectory is kept.
            self.assertEqual(result["detected_object_counts"], {"person": 1, "car": 1})
            self.assertEqual(len(result["tracks"]), 1)
            self.assertTrue(result["tracks_truncated"])

    def test_invalid_video_is_rejected_without_leaving_an_artefact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"empty.mp4"
            source.write_bytes(b"not a video")
            destination = Path(directory)/"annotated.mp4"
            with self.assertRaises(ValueError):
                annotate_with_detections(_FakeModel(lambda index: []), source, destination)
            self.assertFalse(destination.exists())

    def test_cancellation_removes_the_partial_clip(self):
        def rows(index):
            return [(1, 0, 0.9, [1.0, 1.0, 10.0, 10.0])]

        def progress(percent, detail):
            if detail == "annotating frame 2/5":
                raise RuntimeError("cancelled")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source.mp4"
            destination = Path(directory)/"annotated.mp4"
            _write_clip(source)
            with self.assertRaises(RuntimeError):
                annotate_with_detections(_FakeModel(rows), source, destination,
                    progress=progress, sample_interval_seconds=1.0, device=-1)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
