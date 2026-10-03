import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch

import vllm_video_pipeline as vp

VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class WindowPlanningTests(unittest.TestCase):
    def test_five_minutes_yields_twenty_fifteen_second_windows(self):
        windows = vp.plan_windows(300.0, 15.0)
        self.assertEqual(len(windows), 20)
        self.assertEqual(windows[0], {"start_sec": 0.0, "end_sec": 15.0})
        self.assertEqual(windows[-1], {"start_sec": 285.0, "end_sec": 300.0})

    def test_partial_tail_window_is_clamped_to_duration(self):
        windows = vp.plan_windows(47.0, 15.0)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[-1]["end_sec"], 47.0)

    def test_overlap_shrinks_the_step(self):
        windows = vp.plan_windows(30.0, 15.0, 5.0)
        self.assertEqual([w["start_sec"] for w in windows], [0.0, 10.0, 20.0])

    def test_nonpositive_duration_yields_no_windows(self):
        self.assertEqual(vp.plan_windows(0.0), [])


@unittest.skipUnless(TORCH_AVAILABLE, "torch required for the loop test")
class SequentialLoopTests(unittest.TestCase):
    def test_loop_slices_infers_and_pauses_without_engine(self):
        events = []

        def fake_sleep(seconds):
            events.append(("sleep", seconds))

        engine = MagicMock()
        output = MagicMock()
        output.outputs = [MagicMock(text='{"detected_activity": ["training"], "confidence": 0.9}')]
        engine.generate.return_value = [output]

        with patch.object(vp, "plan_windows", return_value=[
                {"start_sec": 0.0, "end_sec": 15.0}, {"start_sec": 15.0, "end_sec": 30.0}]), \
             patch.object(vp, "validate_video_duration", return_value={
                 "duration_sec": 30.0, "has_audio": False}), \
             patch.object(vp, "extract_window_clip", side_effect=lambda path, start, end:
                 events.append(("clip", start, end)) or f"/tmp/win_{start}.mp4"), \
             patch.object(vp, "infer_window", side_effect=lambda llm, sp, window, prompt:
                 events.append(("infer", window)) or {"detected_activity": ["training"]}), \
             patch.object(vp.time, "sleep", side_effect=fake_sleep):
            result = vp.analyze_video_sequential("clip.mp4", llm=engine, sampling_params=object())

        self.assertEqual(result["windows_analyzed"], 2)
        infer_events = [item for item in events if item[0] == "infer"]
        self.assertEqual(len(infer_events), 2)
        sleeps = [item for item in events if item[0] == "sleep"]
        self.assertEqual(len(sleeps), 2, "pause after every inference step")
        self.assertTrue(all(secs == vp.COOLDOWN_SECONDS for _, secs in sleeps))


@unittest.skipUnless(VLLM_AVAILABLE and TORCH_AVAILABLE, "vLLM engine not installed in this image")
class EngineConfigurationTests(unittest.TestCase):
    def test_default_model_is_qwen3_vl_8b_int4(self):
        # int4 AWQ build: ~6 GB of weights instead of the ~17.5 GB bf16 release.
        self.assertEqual(vp.VLLM_MODEL_ID, "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit")

    def test_engine_uses_thermal_tight_kwargs(self):
        with patch("vllm.LLM") as llm_cls:
            vp.build_engine()
        kwargs = llm_cls.call_args.kwargs
        self.assertEqual(kwargs["dtype"], "bfloat16")
        self.assertEqual(kwargs["gpu_memory_utilization"], vp.GPU_MEMORY_UTILIZATION)
        self.assertEqual(kwargs["max_model_len"], vp.MAX_MODEL_LEN)
        self.assertTrue(kwargs["enforce_eager"])
        self.assertEqual(kwargs["limit_mm_per_prompt"], {"video": 1})
        self.assertEqual(kwargs["mm_processor_kwargs"]["max_pixels"], vp.MM_MAX_PIXELS)
        self.assertEqual(kwargs["mm_processor_kwargs"]["nframes"], vp.MM_NFRAMES)

    def test_sampling_params_temperature_and_budget(self):
        from vllm import SamplingParams
        params = vp.build_sampling_params()
        self.assertIsInstance(params, SamplingParams)
        self.assertEqual(params.temperature, vp.TEMPERATURE)
        self.assertEqual(params.max_tokens, vp.MAX_NEW_TOKENS)


if __name__ == "__main__":
    unittest.main()
