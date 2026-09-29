import unittest

from response_schema import (
    DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA,
    IMAGE_ANALYSIS_ENVELOPE_SCHEMA,
    VIDEO_ANALYSIS_ENVELOPE_SCHEMA,
    normalize_response,
    validate_response,
)

DETAILED = DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA


def detailed_response(**overrides):
    response = {
        "filename": "clip.mp4",
        "media_type": "video",
        "analysis": {
            "metadata": {"duration_sec": 12.0, "has_audio": True},
            "frame_analysis": [
                {"frame_index": 0, "timestamp_sec": 0.0, "scene_summary": "trucks on a road",
                 "key_objects": ["truck"], "location_type": "rural",
                 "detected_activity": ["movement"], "confidence": 0.62, "reasoning": "vehicles moving"},
            ],
            "frame_warnings": [],
            "audio_analysis": {"has_audio": True, "transcript": [], "sound_events": [], "warnings": []},
            "backends_used": ["vllm"],
            "object_tracking": {"tracks": []},
            "motion_burstiness": {"classification": "ambiguous"},
            "detailed_report": {
                "executive_overview": {"dominant_activity": "movement"},
                "video_level_reasoning": {"activity": "movement", "confidence": 0.7, "reasoning": "transit"},
            },
            "report": {"filename": "x.pdf"},
        },
    }
    response.update(overrides)
    return response


class ValidResponseTests(unittest.TestCase):
    def test_well_formed_response_has_no_violations(self):
        self.assertEqual(validate_response(detailed_response(), DETAILED), [])

    def test_normalization_is_idempotent_on_a_valid_response(self):
        self.assertEqual(normalize_response(detailed_response(), DETAILED), detailed_response())


class RepairTests(unittest.TestCase):
    def frame(self, response):
        return response["analysis"]["frame_analysis"][0]

    def test_missing_confidence_is_defaulted_and_reported(self):
        response = detailed_response()
        del self.frame(response)["confidence"]
        problems = validate_response(response, DETAILED)
        self.assertTrue(any("confidence" in problem for problem in problems), problems)
        self.assertEqual(self.frame(normalize_response(response, DETAILED))["confidence"], 0.0)

    def test_out_of_range_confidence_is_clamped(self):
        response = detailed_response()
        self.frame(response)["confidence"] = 5.0
        repaired = normalize_response(response, DETAILED)
        self.assertEqual(self.frame(repaired)["confidence"], 1.0)

    def test_string_activity_is_wrapped_into_a_list(self):
        response = detailed_response()
        self.frame(response)["detected_activity"] = "movement"
        self.assertEqual(self.frame(normalize_response(response, DETAILED))["detected_activity"], ["movement"])

    def test_unknown_activity_label_becomes_miscellaneous(self):
        response = detailed_response()
        self.frame(response)["detected_activity"] = ["attack"]
        self.assertEqual(self.frame(normalize_response(response, DETAILED))["detected_activity"], ["miscellaneous"])

    def test_null_video_level_reasoning_is_allowed(self):
        response = detailed_response()
        response["analysis"]["detailed_report"]["video_level_reasoning"] = None
        repaired = normalize_response(response, DETAILED)
        self.assertEqual(validate_response(response, DETAILED), [])
        self.assertIsNone(repaired["analysis"]["detailed_report"]["video_level_reasoning"])

    def test_video_level_reasoning_missing_confidence_is_repaired(self):
        response = detailed_response()
        del response["analysis"]["detailed_report"]["video_level_reasoning"]["confidence"]
        repaired = normalize_response(response, DETAILED)
        self.assertEqual(repaired["analysis"]["detailed_report"]["video_level_reasoning"]["confidence"], 0.0)

    def test_wrong_typed_frame_analysis_is_replaced_with_an_empty_list(self):
        response = detailed_response()
        response["analysis"]["frame_analysis"] = {"not": "a list"}
        repaired = normalize_response(response, DETAILED)
        self.assertEqual(repaired["analysis"]["frame_analysis"], [])

    def test_invalid_media_type_falls_back_to_video(self):
        response = detailed_response(media_type="audio")
        repaired = normalize_response(response, DETAILED)
        self.assertEqual(repaired["media_type"], "video")
        self.assertTrue(any("media_type" in problem for problem in validate_response(response, DETAILED)))

    def test_unmodelled_keys_pass_through_untouched(self):
        response = detailed_response()
        response["analysis"]["custom_block"] = {"x": 1}
        repaired = normalize_response(response, DETAILED)
        self.assertEqual(repaired["analysis"]["custom_block"], {"x": 1})


class AudioSchemaTests(unittest.TestCase):
    def test_transcript_segment_fields_are_repaired(self):
        response = detailed_response()
        response["analysis"]["audio_analysis"]["transcript"] = [{"start_sec": 1.0, "end_sec": 2.0}]
        problems = validate_response(response, DETAILED)
        self.assertTrue(any("text" in problem for problem in problems), problems)
        segment = normalize_response(response, DETAILED)["analysis"]["audio_analysis"]["transcript"][0]
        self.assertEqual(segment["text"], "")

    def test_sound_event_confidence_is_clamped(self):
        response = detailed_response()
        response["analysis"]["audio_analysis"]["sound_events"] = [
            {"label": "gunfire", "start_sec": 0.0, "end_sec": 1.0, "confidence": 4.0}]
        repaired = normalize_response(response, DETAILED)
        self.assertEqual(repaired["analysis"]["audio_analysis"]["sound_events"][0]["confidence"], 1.0)

    def test_well_typed_audio_items_have_no_violations(self):
        response = detailed_response()
        response["analysis"]["audio_analysis"].update({
            "transcript": [{"start_sec": 0.5, "end_sec": 1.5, "text": "move up"}],
            "sound_events": [{"label": "engine", "start_sec": 0.0, "end_sec": 2.0, "confidence": 1.0}]})
        self.assertEqual(validate_response(response, DETAILED), [])


class ImageEnvelopeTests(unittest.TestCase):
    def test_image_response_does_not_gain_video_only_fields(self):
        response = {"filename": "a.jpg", "media_type": "image",
            "backend_used": "vllm",
            "analysis": {"scene_summary": "a truck", "key_objects": ["truck"], "location_type": "unknown",
                "detected_activity": ["movement"], "confidence": 0.5, "reasoning": "moving"}}
        self.assertEqual(validate_response(response, IMAGE_ANALYSIS_ENVELOPE_SCHEMA), [])
        repaired = normalize_response(response, IMAGE_ANALYSIS_ENVELOPE_SCHEMA)
        self.assertNotIn("frame_index", repaired["analysis"])
        self.assertNotIn("timestamp_sec", repaired["analysis"])


class VideoEnvelopeTests(unittest.TestCase):
    def test_overall_video_level_reasoning_is_validated(self):
        response = {"filename": "clip.mp4", "media_type": "video",
            "analysis": {"frame_analysis": [], "audio_analysis": {"has_audio": False},
                "overall_analysis": {"dominant_activity": "not-a-label",
                    "video_level_reasoning": {"activity": "training", "confidence": 0.9}}}}
        problems = validate_response(response, VIDEO_ANALYSIS_ENVELOPE_SCHEMA)
        self.assertTrue(any("not-a-label" in problem for problem in problems), problems)
        repaired = normalize_response(response, VIDEO_ANALYSIS_ENVELOPE_SCHEMA)
        self.assertEqual(repaired["analysis"]["overall_analysis"]["dominant_activity"], "miscellaneous")
        self.assertEqual(repaired["analysis"]["audio_analysis"]["transcript"], [])


if __name__ == "__main__":
    unittest.main()
