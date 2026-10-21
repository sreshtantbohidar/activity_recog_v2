import importlib.util
import json
import subprocess
import unittest
from unittest.mock import patch

from media_utils import (
    MediaProbeError,
    VideoTooLongError,
    analyze_motion_burstiness,
    build_multimodal_summary_prompt,
    confidence_label,
    motion_activity_bias,
    normalize_activity_result,
    normalize_confidence,
    parse_json_text,
    validate_video_duration,
)

PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


class MediaDurationTests(unittest.TestCase):
    @patch("media_utils.subprocess.run")
    def test_accepts_video_at_exact_five_minute_limit(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"format": {"duration": "300"}, "streams": [{"codec_type": "audio"}]}),
            stderr="",
        )
        result = validate_video_duration("clip.mp4")
        self.assertEqual(result, {"duration_sec": 300.0, "has_audio": True})

    @patch("media_utils.subprocess.run")
    def test_rejects_video_longer_than_five_minutes(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"format": {"duration": "300.01"}, "streams": []}),
            stderr="",
        )
        with self.assertRaises(VideoTooLongError):
            validate_video_duration("long.mp4")

    @patch("media_utils.subprocess.run")
    def test_invalid_probe_metadata_fails_closed(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="{}", stderr=""
        )
        with self.assertRaises(MediaProbeError):
            validate_video_duration("invalid.mp4")


class ActivityOutputTests(unittest.TestCase):
    def test_parses_json_with_surrounding_prose(self):
        self.assertEqual(parse_json_text('result: {"detected_activity": ["movement"]}'), {
            "detected_activity": ["movement"]
        })

    def test_normalizes_unknown_activity_to_miscellaneous(self):
        result = normalize_activity_result({
            "detected_activity": ["attack", "movement", "movement"],
            "confidence": "certain",
            "location_type": "classified",
        })
        self.assertEqual(result["detected_activity"], ["movement"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["location_type"], "unknown")

    def test_multimodal_prompt_contains_timestamped_visual_speech_and_sound(self):
        prompt = build_multimodal_summary_prompt(
            [{"timestamp_sec": 7.25, "detected_activity": ["movement"],
              "scene_summary": "Two vehicles on a road", "reasoning": "Vehicles appear displaced"}],
            [{"start_sec": 8, "end_sec": 9, "text": "Proceed"}],
            [{"start_sec": 5, "end_sec": 10, "label": "engine", "confidence": 0.8}],
        )
        self.assertIn("[7.2s]", prompt)
        self.assertIn("[8.0-9.0s] Proceed", prompt)
        self.assertIn("[5.0-10.0s] engine (confidence 0.80)", prompt)
        self.assertIn("not as proof that the described event is visible", prompt)

    def test_multimodal_prompt_stays_bounded_for_long_videos(self):
        frames = [
            {"timestamp_sec": float(i), "detected_activity": ["operation"],
             "scene_summary": "scene " + "x" * 400,
             "reasoning": "reason " + "y" * 400}
            for i in range(400)
        ]
        speech = [
            {"start_sec": i, "end_sec": i + 1, "text": "word " + "z" * 400}
            for i in range(400)
        ]
        sounds = [
            {"start_sec": i, "end_sec": i + 1,
             "label": "sound " + "w" * 400, "confidence": 0.5}
            for i in range(400)
        ]
        prompt = build_multimodal_summary_prompt(frames, speech, sounds)
        # Stays well within the vLLM 8k context (~4 chars/token) instead of
        # overflowing and producing "Cross-modal summary unavailable".
        self.assertLess(len(prompt), 20000)
        self.assertIn("more entr", prompt)

    def test_multimodal_prompt_clips_a_single_overlong_field(self):
        prompt = build_multimodal_summary_prompt(
            [{"timestamp_sec": 0, "detected_activity": ["movement"],
              "scene_summary": "s" * 5000, "reasoning": "r" * 5000}],
            [{"start_sec": 0, "end_sec": 1, "text": "t" * 5000}],
            [{"start_sec": 0, "end_sec": 1, "label": "l" * 5000, "confidence": 0.5}],
        )
        self.assertNotIn("s" * 1000, prompt)
        self.assertNotIn("r" * 1000, prompt)
        self.assertNotIn("t" * 1000, prompt)
        self.assertNotIn("l" * 1000, prompt)


class ConfidenceTests(unittest.TestCase):
    def test_numeric_confidence_is_kept_as_a_float(self):
        self.assertEqual(normalize_confidence(0.45), 0.45)

    def test_percent_forms_are_scaled_to_the_unit_interval(self):
        self.assertEqual(normalize_confidence("45%"), 0.45)
        self.assertEqual(normalize_confidence(45), 0.45)

    def test_out_of_range_and_invalid_values_are_clamped_or_defaulted(self):
        self.assertEqual(normalize_confidence(1.5), 1.0)
        self.assertEqual(normalize_confidence(-0.2), 0.0)
        self.assertEqual(normalize_confidence("unknown"), 0.0)
        self.assertEqual(normalize_confidence(None), 0.0)

    def test_confidence_label_buckets_numeric_values(self):
        self.assertEqual(confidence_label(0.85), "high")
        self.assertEqual(confidence_label(0.5), "medium")
        self.assertEqual(confidence_label(0.1), "low")


class MotionBurstinessTests(unittest.TestCase):
    def test_steady_motion_suggests_operation(self):
        result = analyze_motion_burstiness([0.30, 0.28, 0.32, 0.29, 0.31, 0.30])
        self.assertEqual(result["classification"], "consistent")
        self.assertEqual(result["suggested_activity_bias"], "operation")
        self.assertGreaterEqual(result["active_fraction"], 0.6)

    def test_bursty_motion_suggests_training(self):
        result = analyze_motion_burstiness([0.0, 0.35, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0])
        self.assertEqual(result["classification"], "bursty")
        self.assertEqual(result["suggested_activity_bias"], "training")
        self.assertLessEqual(result["active_fraction"], 0.4)

    def test_mixed_motion_is_ambiguous(self):
        result = analyze_motion_burstiness([0.0, 0.30, 0.02, 0.28, 0.0, 0.25])
        self.assertEqual(result["classification"], "ambiguous")
        self.assertIsNone(result["suggested_activity_bias"])

    def test_too_few_frames_is_insufficient(self):
        result = analyze_motion_burstiness([0.3, 0.3, 0.3])
        self.assertEqual(result["classification"], "insufficient_motion_evidence")
        self.assertIsNone(result["suggested_activity_bias"])

    def test_bias_mapping_rejects_unknown(self):
        self.assertEqual(motion_activity_bias({"classification": "consistent"}), "operation")
        self.assertEqual(motion_activity_bias({"classification": "bursty"}), "training")
        self.assertIsNone(motion_activity_bias({"classification": "ambiguous"}))
        self.assertIsNone(motion_activity_bias(None))

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is needed to decode saved frames")
    def test_change_scores_are_ordered_and_first_is_zero(self):
        import os
        import tempfile
        from PIL import Image

        paths = []
        try:
            for color in ((0, 0, 0), (255, 255, 255), (255, 255, 255), (0, 0, 0)):
                handle, path = tempfile.mkstemp(suffix=".png")
                os.close(handle)
                Image.new("RGB", (96, 54), color).save(path)
                paths.append(path)
            from media_utils import compute_frame_change_scores
            scores = compute_frame_change_scores(paths)
            self.assertEqual(len(scores), 4)
            self.assertEqual(scores[0], 0.0)
            self.assertGreater(scores[1], 0.5)
            self.assertEqual(scores[2], 0.0)
            self.assertGreater(scores[3], 0.5)
        finally:
            for path in paths:
                os.path.exists(path) and os.remove(path)


if __name__ == "__main__":
    unittest.main()
