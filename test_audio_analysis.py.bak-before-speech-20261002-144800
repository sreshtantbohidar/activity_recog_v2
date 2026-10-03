import importlib.util
import unittest
from unittest.mock import MagicMock, patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

import audio_analysis


class UnifiedOutputParsingTests(unittest.TestCase):
    def test_parses_transcript_lines_and_bracketed_sounds(self):
        reply = ("TRANSCRIPT:\n"
                 "0.5-2.4: contact left\n"
                 "3.0-4.1: move up\n"
                 "SOUNDS:\n"
                 "[gunshot] [bird chirping]\n"
                 "[heavy vehicle engine]")
        transcript, sounds = audio_analysis._parse_unified_output(reply, 10.0, 40.0)
        self.assertEqual([(s["start_sec"], s["end_sec"], s["text"]) for s in transcript],
            [(10.5, 12.4, "contact left"), (13.0, 14.1, "move up")])
        self.assertEqual([s["label"] for s in sounds], ["gunshot", "bird chirping", "heavy vehicle engine"])
        self.assertEqual(sounds[0]["start_sec"], 10.0)
        self.assertEqual(sounds[0]["end_sec"], 40.0)

    def test_none_sections_yield_empty_lists(self):
        transcript, sounds = audio_analysis._parse_unified_output(
            "TRANSCRIPT: none\nSOUNDS: none", 0.0, 30.0)
        self.assertEqual(transcript, [])
        self.assertEqual(sounds, [])

    def test_sound_cap_per_window(self):
        reply = "SOUNDS:\n" + "\n".join(f"[sound {i}]" for i in range(20))
        _, sounds = audio_analysis._parse_unified_output(reply, 0.0, 30.0)
        self.assertEqual(len(sounds), audio_analysis.QWEN_AUDIO_MAX_SOUNDS_PER_WINDOW)

    def test_parses_numbered_section_prefixes(self):
        # The model tends to number its sections; a stray "1) " must not hide sounds.
        reply = ("1) TRANSCRIPT: none\n"
                 "2) SOUNDS:\n"
                 "- [radio chatter]\n"
                 "- [heavy vehicle engine]")
        transcript, sounds = audio_analysis._parse_unified_output(reply, 0.0, 10.0)
        self.assertEqual(transcript, [])
        self.assertEqual([s["label"] for s in sounds], ["radio chatter", "heavy vehicle engine"])

    def test_decimal_timestamps_survive_marker_stripping(self):
        transcript, _ = audio_analysis._parse_unified_output("0.5-2.4: contact left", 10.0, 40.0)
        self.assertEqual([(s["start_sec"], s["end_sec"], s["text"]) for s in transcript],
            [(10.5, 12.4, "contact left")])

    def test_parses_json_reply(self):
        reply = ('{"transcript": [{"start_sec": 1.0, "end_sec": 2.0, "text": "move up"}], '
                 '"sounds": [{"label": "radio chatter"}]}')
        transcript, sounds = audio_analysis._parse_unified_output(reply, 5.0, 15.0)
        self.assertEqual([(s["start_sec"], s["end_sec"], s["text"]) for s in transcript],
            [(6.0, 7.0, "move up")])
        self.assertEqual([s["label"] for s in sounds], ["radio chatter"])
        # Sounds without explicit spans cover the whole window.
        self.assertEqual((sounds[0]["start_sec"], sounds[0]["end_sec"]), (5.0, 15.0))

    def test_parses_typed_json_with_per_sound_spans(self):
        reply = ('{"transcript": [{"start_sec": 1.0, "end_sec": 2.5, "text": "move up"}], '
                 '"sounds": [{"start_sec": 3.0, "end_sec": 6.0, "label": "heavy vehicle engine"}]}')
        transcript, sounds = audio_analysis._parse_unified_output(reply, 10.0, 40.0)
        self.assertEqual([(s["start_sec"], s["end_sec"], s["text"]) for s in transcript],
            [(11.0, 12.5, "move up")])
        self.assertEqual([(s["start_sec"], s["end_sec"], s["label"]) for s in sounds],
            [(13.0, 16.0, "heavy vehicle engine")])

    def test_json_sound_spans_are_clamped_to_the_window(self):
        reply = '{"sounds": [{"start_sec": 90.0, "end_sec": 120.0, "label": "engine"}]}'
        _, sounds = audio_analysis._parse_unified_output(reply, 10.0, 40.0)
        self.assertEqual((sounds[0]["start_sec"], sounds[0]["end_sec"]), (40.0, 40.0))

    def test_json_with_bad_numbers_falls_back_to_defaults(self):
        reply = '{"sounds": [{"start_sec": "x", "end_sec": null, "label": "engine"}]}'
        _, sounds = audio_analysis._parse_unified_output(reply, 0.0, 30.0)
        self.assertEqual(sounds[0]["label"], "engine")
        self.assertEqual((sounds[0]["start_sec"], sounds[0]["end_sec"]), (0.0, 0.01))

    def test_parses_python_style_single_quoted_reply(self):
        reply = ("{'transcript': [{'start_sec': 1.0, 'end_sec': 2.0, 'text': 'move up'}], "
                 "'sounds': [{'start_sec': 3.0, 'end_sec': 6.0, 'label': 'engine'}]}")
        transcript, sounds = audio_analysis._parse_unified_output(reply, 10.0, 40.0)
        self.assertEqual([(s["start_sec"], s["end_sec"], s["text"]) for s in transcript], [(11.0, 12.0, "move up")])
        self.assertEqual([(s["start_sec"], s["end_sec"], s["label"]) for s in sounds], [(13.0, 16.0, "engine")])

    def test_sound_entry_accepts_text_key_as_label(self):
        reply = '{"sounds": [{"text": "gunfire", "start_sec": 1.0, "end_sec": 2.0}]}'
        _, sounds = audio_analysis._parse_unified_output(reply, 0.0, 10.0)
        self.assertEqual(sounds[0]["label"], "gunfire")

    def test_unparseable_json_shaped_reply_is_not_turned_into_a_label(self):
        transcript, sounds = audio_analysis._parse_unified_output("{this is not valid json}", 0.0, 10.0)
        self.assertEqual(transcript, [])
        self.assertEqual(sounds, [])

    def test_unified_prompt_specifies_typed_json_fields(self):
        prompt = audio_analysis.UNIFIED_AUDIO_PROMPT
        for token in ('"transcript"', '"sounds"', '"start_sec"', '"end_sec"', '"text"', '"label"'):
            self.assertIn(token, prompt)
        self.assertIn("number", prompt)
        self.assertIn("string", prompt)

    def test_unstructured_prose_is_kept_as_a_sound(self):
        transcript, sounds = audio_analysis._parse_unified_output(
            "I heard radio communication and several beeps of a radio transceiver.", 0.0, 10.0)
        self.assertEqual(transcript, [])
        self.assertEqual(len(sounds), 1)
        self.assertIn("radio communication", sounds[0]["label"])

    def test_firing_keywords_match_described_events(self):
        labels = ["tank or heavy vehicle engine", "Gunshot", "bird chirping"]
        support = [label for label in labels
                   if any(keyword in label.lower() for keyword in audio_analysis.FIRING_LABEL_KEYWORDS)]
        self.assertEqual(support, ["Gunshot"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is needed to exercise model inference")
class QwenAudioChunkTests(unittest.TestCase):
    def test_chunk_analysis_uses_unified_prompt_and_releases_inputs(self):
        import torch

        class Processor:
            def apply_chat_template(self, conversation, **kwargs):
                return "CHAT_TEMPLATE"

            def __call__(self, text=None, audio=None, **kwargs):
                return {"input_ids": torch.ones((1, 8), dtype=torch.long),
                        "input_features": torch.zeros((1, 4))}

            def batch_decode(self, generated, **kwargs):
                return ["TRANSCRIPT: 0.0-1.0: hello\nSOUNDS: [gunshot]"]

        class Model:
            def __init__(self):
                self.generate_kwargs = {}

            def parameters(self):
                return iter([torch.zeros(1, requires_grad=False)])

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                return torch.ones((1, 12), dtype=torch.long)

        model = Model()
        with patch.object(audio_analysis, "_load_qwen_audio", return_value=(Processor(), model)):
            transcript, sounds = audio_analysis._understand_chunk(Processor(), model, [0.0] * 16000, "cpu")
        self.assertEqual(transcript[0]["text"], "hello")
        self.assertEqual(sounds[0]["label"], "gunshot")
        self.assertFalse(model.generate_kwargs.get("do_sample"))
        self.assertEqual(model.generate_kwargs["max_new_tokens"], audio_analysis.QWEN_AUDIO_MAX_NEW_TOKENS)

    def test_chunk_timestamps_are_shifted_by_window_start(self):
        import torch

        class Processor:
            def apply_chat_template(self, conversation, **kwargs):
                return "CHAT_TEMPLATE"

            def __call__(self, text=None, audio=None, **kwargs):
                return {"input_ids": torch.ones((1, 8), dtype=torch.long),
                        "input_features": torch.zeros((1, 4))}

            def batch_decode(self, generated, **kwargs):
                return ["TRANSCRIPT: 1.0-2.0: hello\nSOUNDS: [gunshot]"]

        class Model:
            def parameters(self):
                return iter([torch.zeros(1)])

            def generate(self, **kwargs):
                return torch.ones((1, 12), dtype=torch.long)

        transcript, sounds = audio_analysis._understand_chunk(
            Processor(), Model(), [0.0] * 16000, "cpu", 30.0, 60.0)
        self.assertEqual((transcript[0]["start_sec"], transcript[0]["end_sec"]), (31.0, 32.0))
        self.assertEqual((sounds[0]["start_sec"], sounds[0]["end_sec"]), (30.0, 60.0))

    def test_multi_chunk_analysis_stays_on_the_video_timeline(self):
        import numpy as np
        import torch

        class Processor:
            def apply_chat_template(self, conversation, **kwargs):
                return "CHAT_TEMPLATE"

            def __call__(self, text=None, audio=None, **kwargs):
                return {"input_ids": torch.ones((1, 8), dtype=torch.long),
                        "input_features": torch.zeros((1, 4))}

            def batch_decode(self, generated, **kwargs):
                return ["TRANSCRIPT: 1.0-2.0: hello\nSOUNDS: [gunshot]"]

        class Model:
            def parameters(self):
                return iter([torch.zeros(1)])

            def generate(self, **kwargs):
                return torch.ones((1, 12), dtype=torch.long)

        # 61 s of audio -> full 30 s chunk at 0 s, 30 s chunk, 1 s tail chunk.
        samples = np.zeros(61 * 16000, dtype=np.float32)
        with patch.object(audio_analysis, "_load_qwen_audio", return_value=(Processor(), Model())):
            result = audio_analysis._analyze_samples(samples, "cpu")
        self.assertEqual([s["start_sec"] for s in result["transcript"]], [1.0, 31.0, 61.0])
        self.assertEqual([s["start_sec"] for s in result["sound_events"]], [0.0, 30.0, 60.0])
        self.assertTrue(all(s["start_sec"] >= 0.0 for s in result["transcript"]))


class StageOrderingTests(unittest.TestCase):
    def test_audio_stage_lock_exists_and_is_usable(self):
        self.assertTrue(hasattr(audio_analysis, "AUDIO_STAGE_LOCK"))
        with audio_analysis.AUDIO_STAGE_LOCK:
            pass

    def test_result_schema_is_preserved(self):
        with patch.object(audio_analysis, "extract_audio_track", side_effect=RuntimeError("no stream")):
            result = audio_analysis.analyze_video_audio("silent.mp4")
        for key in ("has_audio", "transcript", "sound_events", "firing_cadence",
                    "background_noise_analysis", "warnings"):
            self.assertIn(key, result)
        self.assertFalse(result["has_audio"])


if __name__ == "__main__":
    unittest.main()
