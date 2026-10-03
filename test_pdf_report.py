import importlib.util
import os
import tempfile
import unittest

os.environ.setdefault("REPORTS_DIR", tempfile.mkdtemp())

REPORTLAB_AVAILABLE = importlib.util.find_spec("reportlab") is not None
if REPORTLAB_AVAILABLE:
    from pdf_report import create_video_analysis_pdf, unique_frame_groups
else:
    unique_frame_groups = None
    create_video_analysis_pdf = None


@unittest.skipUnless(REPORTLAB_AVAILABLE, "reportlab is needed for report helpers")
class UniqueFrameGroupTests(unittest.TestCase):
    def test_collapses_consecutive_identical_observations(self):
        frames = [
            {"timestamp_sec": 0.0, "detected_activity": ["training"], "scene_summary": "range",
             "key_objects": ["rifle"]},
            {"timestamp_sec": 1.0, "detected_activity": ["training"], "scene_summary": "range",
             "key_objects": ["rifle"]},
            {"timestamp_sec": 2.0, "detected_activity": ["training"], "scene_summary": "range",
             "key_objects": ["rifle"]},
            {"timestamp_sec": 3.0, "detected_activity": ["operation"], "scene_summary": "range",
             "key_objects": ["rifle"]},
        ]
        groups = unique_frame_groups(frames)
        self.assertEqual(len(groups), 2)
        self.assertEqual((groups[0]["first_sec"], groups[0]["last_sec"], groups[0]["count"]),
            (0.0, 2.0, 3))
        self.assertEqual((groups[1]["first_sec"], groups[1]["count"]), (3.0, 1))

    def test_non_consecutive_repeat_is_kept_separate(self):
        frames = [
            {"timestamp_sec": 0.0, "detected_activity": ["training"], "scene_summary": "a", "key_objects": []},
            {"timestamp_sec": 1.0, "detected_activity": ["operation"], "scene_summary": "a", "key_objects": []},
            {"timestamp_sec": 2.0, "detected_activity": ["training"], "scene_summary": "a", "key_objects": []},
        ]
        groups = unique_frame_groups(frames)
        self.assertEqual(len(groups), 3)

    def test_object_change_breaks_the_group(self):
        frames = [
            {"timestamp_sec": 0.0, "detected_activity": [], "scene_summary": "s", "key_objects": ["truck"]},
            {"timestamp_sec": 1.0, "detected_activity": [], "scene_summary": "s", "key_objects": ["truck", "rifle"]},
        ]
        self.assertEqual(len(unique_frame_groups(frames)), 2)

    def test_empty_analysis_yields_no_groups(self):
        self.assertEqual(unique_frame_groups([]), [])
        self.assertEqual(unique_frame_groups(None), [])


@unittest.skipUnless(REPORTLAB_AVAILABLE, "reportlab is needed to build the PDF")
class ReportGenerationTests(unittest.TestCase):
    def test_pdf_builds_with_dedupe_and_overall_section(self):
        frames = [
            {"timestamp_sec": 0.0, "detected_activity": ["training"], "scene_summary": "range",
             "key_objects": ["rifle"], "confidence": 0.9, "scene_summary": "range",
             "reasoning": "target flags visible", "frame_selection": {"reasons": ["uniform_timeline"], "change_score": 0.1}},
            {"timestamp_sec": 1.0, "detected_activity": ["training"], "scene_summary": "range",
             "key_objects": ["rifle"], "confidence": 0.9,
             "reasoning": "target flags visible", "frame_selection": {"reasons": ["uniform_timeline"], "change_score": 0.1}},
        ]
        paths = []
        try:
            from reportlab.lib.utils import ImageReader  # noqa: F401
            from PIL import Image as PILImage
            for _ in frames:
                handle, path = tempfile.mkstemp(suffix=".jpg")
                os.close(handle)
                PILImage.new("RGB", (96, 54), (10, 20, 30)).save(path)
                paths.append(path)
            analysis = {
                "metadata": {"duration_sec": 2.0},
                "detailed_report": {"executive_overview": {"frames_analyzed": 2, "audio_available": False,
                        "dominant_activity": "training", "dominant_activity_basis": "visual_frame_votes",
                        "activities": ["training"], "persistent_tracks": 0},
                    "activity_evidence": [{"activity": "training", "first_seen_sec": 0.0,
                        "last_seen_sec": 1.0, "supporting_frames": 2}],
                    "cross_modal_notes": ["note"], "limitations": ["limit"]},
                "frame_analysis": frames,
                "object_tracking": {"tracks": []},
                "audio_analysis": {"has_audio": True, "transcript": [], "sound_events": [],
                    "firing_cadence": {"classification": "gapped", "suggested_activity_bias": "training",
                        "median_gap_sec": 4.8, "firing_coverage_pct": 12.0, "ast_gunfire_support": {"event_count": 2}},
                    "background_noise_analysis": {"detected_sounds": []}, "warnings": []},
                "overall_analysis": {"dominant_activity": "training",
                    "dominant_activity_basis": "multimodal_evidence_override",
                    "detected_activities": ["training"], "evidence_biases": ["training"],
                    "narrative": "Audio cadence identified training."},
                "motion_burstiness": {"classification": "bursty", "suggested_activity_bias": "training"},
                "backends_used": ["vllm"],
            }
            result = create_video_analysis_pdf(analysis, paths, "clip.mp4")
            # `local_path` is the in-container location; `path` is the host view
            # and intentionally may not exist inside the container.
            self.assertTrue(os.path.exists(result["local_path"]))
            self.assertGreater(os.path.getsize(result["local_path"]), 1000)
            self.assertTrue(result["path"].endswith(result["filename"]))
            self.assertTrue(result["download_url"].startswith("/reports/"))
        finally:
            for path in paths:
                os.path.exists(path) and os.remove(path)


if __name__ == "__main__":
    unittest.main()
