import importlib.util
import unittest
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

import audio_analysis


class UnifiedOutputParsingTests(unittest.TestCase):
    def test_legacy_combined_transcript_and_sound_parsing(self):
        reply = ("TRANSCRIPT:\n0.5-2.4: contact left\n3.0-4.1: move up\n"
                 "SOUNDS:\n[gunshot] [bird chirping]\n[heavy vehicle engine]")
        transcript, sounds = audio_analysis._parse_unified_output(reply, 10.0, 40.0)
        self.assertEqual([(s["start_sec"], s["end_sec"], s["text"]) for s in transcript],
            [(10.5, 12.4, "contact left"), (13.0, 14.1, "move up")])
        self.assertEqual([s["label"] for s in sounds], ["gunshot", "bird chirping", "heavy vehicle engine"])

    def test_legacy_json_and_python_literal_parse(self):
        reply = ("{'transcript': [{'start_sec': 1.0, 'end_sec': 2.0, 'text': 'move up'}], "
                 "'sounds': [{'label': 'radio chatter'}]}")
        transcript, sounds = audio_analysis._parse_unified_output(reply, 5.0, 15.0)
        self.assertEqual(transcript[0]["text"], "move up")
        self.assertEqual((transcript[0]["start_sec"], transcript[0]["end_sec"]), (6.0, 7.0))
        self.assertEqual(sounds[0]["label"], "radio chatter")

    def test_legacy_numbered_sound_sections_parse(self):
        _, sounds = audio_analysis._parse_unified_output(
            "1) TRANSCRIPT: none\n2) SOUNDS:\n- [radio chatter]\n- [heavy vehicle engine]", 0.0, 10.0)
        self.assertEqual([item["label"] for item in sounds], ["radio chatter", "heavy vehicle engine"])

    def test_unstructured_sound_reply_is_retained(self):
        _, sounds = audio_analysis._parse_unified_output(
            "I heard radio communication and several beeps of a radio transceiver.", 0.0, 10.0)
        self.assertEqual(len(sounds), 1)
        self.assertIn("radio communication", sounds[0]["label"])

    def test_separate_transcription_and_sound_parsers_keep_existing_shapes(self):
        transcript = audio_analysis._parse_spoken_transcript(
            "The transcript of the speech is: 'command to Bravo, eyes on target.'", 30.0, 40.0)
        sounds = audio_analysis._parse_background_sounds(
            "I heard radio communication and several beeps of a radio transceiver.", 30.0, 40.0)
        self.assertEqual(transcript, [{"start_sec": 30.0, "end_sec": 40.0,
            "text": "command to Bravo, eyes on target."}])
        self.assertEqual(set(sounds[0]), {"start_sec", "end_sec", "label", "confidence"})
        self.assertEqual(sounds[0]["start_sec"], 30.0)

    def test_json_string_transcript_and_no_speech_replies(self):
        self.assertEqual(audio_analysis._parse_spoken_transcript(
            '{"transcript": "hello there"}', 4.0, 10.0),
            [{"start_sec": 4.0, "end_sec": 10.0, "text": "hello there"}])
        self.assertEqual(audio_analysis._parse_spoken_transcript(
            "There is no speech in this audio; therefore, no transcript can be provided.", 0.0, 10.0), [])
        self.assertEqual(audio_analysis._parse_background_sounds("No non-speech sounds are audible.", 0.0, 10.0), [])
        self.assertEqual(audio_analysis._parse_background_sounds(
            "The audio contains static noise at intervals. There is no music present.", 0.0, 10.0)[0]["label"],
            "The audio contains static noise at intervals. There is no music present.")

    def test_instruction_echo_is_not_reported_as_speech(self):
        # The model echoes the instruction on chunks with no speech; on a long
        # video that echo used to become a transcript segment.
        self.assertEqual(audio_analysis._parse_spoken_transcript(
            "No speech is audible; return an empty transcript.", 0.0, 30.0), [])
        self.assertEqual(audio_analysis._parse_spoken_transcript(
            "The transcript of the speech in this audio is: 'hold the line.'", 30.0, 60.0),
            [{"start_sec": 30.0, "end_sec": 60.0, "text": "hold the line."}])

    def test_firing_keywords_match_described_events(self):
        labels = ["tank or heavy vehicle engine", "Gunshot", "bird chirping"]
        support = [label for label in labels
                   if any(keyword in label.lower() for keyword in audio_analysis.FIRING_LABEL_KEYWORDS)]
        self.assertEqual(support, ["Gunshot"])


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is needed to exercise the cadence analysis")
class SoundEventCadenceTests(unittest.TestCase):
    def test_contiguous_chunk_events_do_not_crash_and_stay_ambiguous(self):
        # Long videos tile the timeline with back-to-back 30s windows, so every
        # inter-event gap is zero. That used to raise "'<=' not supported between
        # instances of 'NoneType' and 'float'" and lose the whole cadence result.
        events = [{"start_sec": float(i * 30), "end_sec": float((i + 1) * 30),
                   "label": "radio chatter", "confidence": 1.0} for i in range(10)]
        result = audio_analysis.analyze_sound_event_cadence(events)
        self.assertEqual(result["classification"], "ambiguous")
        self.assertIsNone(result["suggested_activity_bias"])
        self.assertEqual(result["median_gap_sec"], 0.0)
        self.assertIsNone(result["gap_cv"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is needed to exercise model inference")
class QwenAudioChunkTests(unittest.TestCase):
    class FakeModel:
        def __init__(self):
            self.generate_kwargs = {}

        def parameters(self):
            import torch
            return iter([torch.zeros(1, requires_grad=False)])

        def generate(self, **kwargs):
            import torch
            self.generate_kwargs = kwargs
            return torch.ones((1, 12), dtype=torch.long)

    class FakeProcessor:
        def __init__(self, replies):
            self.replies = list(replies)
            self.prompts = []
            self.conversations = []

        def apply_chat_template(self, conversation, **kwargs):
            self.conversations.append(conversation)
            # The audio model is Qwen2.5-Omni: conversation[0] is the system
            # message, conversation[1] the user turn with the audio + prompt.
            prompt = conversation[1]["content"][1]["text"]
            self.prompts.append(prompt)
            return "CHAT_TEMPLATE"

        def __call__(self, text=None, audio=None, **kwargs):
            import torch
            return {"input_ids": torch.ones((1, 8), dtype=torch.long),
                    "input_features": torch.zeros((1, 4))}

        def batch_decode(self, generated, **kwargs):
            return [self.replies.pop(0)]

    def test_chunk_analysis_uses_separate_prompts_and_preserves_audio_records(self):
        processor = self.FakeProcessor([
            "The transcript of the speech is: 'command to bravo eyes on target'",
            "I heard radio communication and low engine noise.",
        ])
        model = self.FakeModel()
        with patch.object(audio_analysis, "_load_qwen_audio", return_value=(processor, model)):
            transcript, sounds = audio_analysis._understand_chunk(processor, model, [0.0] * 16000, "cpu")
        self.assertEqual(transcript[0]["text"], "command to bravo eyes on target")
        self.assertEqual(set(transcript[0]), {"start_sec", "end_sec", "text"})
        self.assertEqual(sounds[0]["label"], "I heard radio communication and low engine noise.")
        self.assertEqual(set(sounds[0]), {"start_sec", "end_sec", "label", "confidence"})
        self.assertEqual(len(processor.prompts), 2)
        self.assertIn("exact words spoken", processor.prompts[0])
        self.assertIn("What sounds can you hear", processor.prompts[1])
        # Qwen2.5-Omni: a system persona precedes each user turn, and generation
        # must stay on the text-only Thinker branch (no speech synthesis).
        self.assertEqual(processor.conversations[0][0]["role"], "system")
        self.assertIn("audio analysis engine", processor.conversations[0][0]["content"][0]["text"])
        self.assertIs(model.generate_kwargs.get("return_audio"), False)
        self.assertFalse(model.generate_kwargs.get("do_sample"))
        self.assertEqual(model.generate_kwargs["max_new_tokens"], audio_analysis.QWEN_AUDIO_MAX_NEW_TOKENS)

    def test_chunk_timestamps_are_shifted_by_window_start(self):
        processor = self.FakeProcessor(["TRANSCRIPT: 1.0-2.0: hello", "[gunshot]"])
        transcript, sounds = audio_analysis._understand_chunk(
            processor, self.FakeModel(), [0.0] * 16000, "cpu", 30.0, 60.0)
        self.assertEqual((transcript[0]["start_sec"], transcript[0]["end_sec"]), (31.0, 32.0))
        self.assertEqual((sounds[0]["start_sec"], sounds[0]["end_sec"]), (30.0, 60.0))

    def test_multi_chunk_analysis_stays_on_the_video_timeline(self):
        import numpy as np

        class RepeatedProcessor(self.FakeProcessor):
            def __init__(self):
                super().__init__(replies=())  # batch_decode is overridden below
                self.calls = 0

            def batch_decode(self, generated, **kwargs):
                self.calls += 1
                return ["TRANSCRIPT: 1.0-2.0: hello" if self.calls % 2 else "[gunshot]"]

        samples = np.zeros(61 * 16000, dtype=np.float32)
        with patch.object(audio_analysis, "_load_qwen_audio",
                          return_value=(RepeatedProcessor(), self.FakeModel())):
            result = audio_analysis._analyze_samples(samples, "cpu")
        self.assertEqual([s["start_sec"] for s in result["transcript"]], [1.0, 31.0, 61.0])
        self.assertEqual([s["start_sec"] for s in result["sound_events"]], [0.0, 30.0, 60.0])


class WhisperTranscriptionTests(unittest.TestCase):
    def test_whisper_segments_become_timed_transcript(self):
        class Segment:
            def __init__(self, start, end, text):
                self.start, self.end, self.text = start, end, text

        class FakeWhisper:
            def transcribe(self, samples, **kwargs):
                return ([Segment(0.5, 2.0, "  move up  "), Segment(2.0, 3.0, "   ")], object())

        with patch.object(audio_analysis, "_load_whisper", return_value=FakeWhisper()):
            out = audio_analysis._transcribe_with_whisper([0.0] * 16000)
        self.assertEqual(out, [{"start_sec": 0.5, "end_sec": 2.0, "text": "move up"}])

    def test_whisper_drops_likely_hallucinated_segments(self):
        class Segment:
            def __init__(self, start, end, text, no_speech_prob):
                self.start, self.end, self.text = start, end, text
                self.no_speech_prob = no_speech_prob

        class FakeWhisper:
            def transcribe(self, samples, **kwargs):
                # Real speech scores a low no-speech probability; the invented
                # line over engine noise scores high and must be dropped.
                return ([Segment(0.0, 2.0, " move up ", 0.18),
                    Segment(1.3, 2.3, " \u041f\u0440\u043e\u0434\u043e\u043b\u0436\u0435\u043d\u0438\u0435 \u0441\u043b\u0435\u0434\u0443\u0435\u0442... ", 0.79)], object())

        with patch.object(audio_analysis, "_load_whisper", return_value=FakeWhisper()):
            out = audio_analysis._transcribe_with_whisper([0.0] * 16000)
        self.assertEqual(out, [{"start_sec": 0.0, "end_sec": 2.0, "text": "move up"}])

    def test_transcription_backend_can_disable_whisper(self):
        with patch.object(audio_analysis, "TRANSCRIPTION_BACKEND", "qwen"):
            self.assertFalse(audio_analysis._whisper_available())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is needed to exercise model inference")
class SoundOnlyChunkTests(unittest.TestCase):
    def test_sound_only_chunk_skips_the_speech_pass(self):
        processor = QwenAudioChunkTests.FakeProcessor(["I heard an engine."])
        model = QwenAudioChunkTests.FakeModel()
        transcript, sounds = audio_analysis._understand_chunk(
            processor, model, [0.0] * 16000, "cpu", include_speech=False)
        self.assertEqual(transcript, [])
        self.assertEqual(len(processor.prompts), 1)
        self.assertIn("What sounds can you hear", processor.prompts[0])
        self.assertEqual(sounds[0]["label"], "I heard an engine.")


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
