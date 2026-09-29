"""Guards the single-model design.

The service must use exactly two models and no fallback:
  * vision: cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit served by the in-container vLLM engine
  * audio:  Qwen/Qwen2-Audio-7B-Instruct (4-bit NF4)

Everything else (Ollama, the in-process PyTorch/InternVL vision loaders and
VideoMAE) must stay gone. This test reads source text only, so it runs without
any heavy dependency installed.
"""
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCANNED = ("app.py", "vision_backend.py", "audio_analysis.py", "pdf_report.py", "swagger_docs.py")
REMOVED_TOKENS = ("ollama", "videomae", "internvl", "pytorch", "mcg-nju", "qwen3-vl:8b-instruct")


class RemovedBackendGuardTests(unittest.TestCase):
    def test_no_references_to_removed_backends_or_models(self):
        for name in SCANNED:
            text = (HERE / name).read_text(encoding="utf-8").lower()
            for token in REMOVED_TOKENS:
                self.assertNotIn(token, text, f"{name} still references {token!r}")

    def test_the_two_models_are_declared(self):
        self.assertIn("cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
                      (HERE / "app.py").read_text(encoding="utf-8"))
        self.assertIn("Qwen/Qwen2-Audio-7B-Instruct",
                      (HERE / "audio_analysis.py").read_text(encoding="utf-8"))

    def test_vllm_is_the_only_vision_backend(self):
        import vision_backend
        self.assertEqual(vision_backend.SUPPORTED_VISION_BACKENDS, {"vllm"})
        self.assertFalse(hasattr(vision_backend, "analyze_image_with_ollama"))

    def test_audio_stage_is_serialized_against_vision(self):
        app = (HERE / "app.py").read_text(encoding="utf-8")
        self.assertIn("AUDIO_STAGE_LOCK", app)
        self.assertIn("vision.lock", app)
        # Idle periods release the GPU through the power manager.
        self.assertIn("engine_power", app)
        self.assertIn("VLLM_IDLE_SLEEP_SECONDS", app)


if __name__ == "__main__":
    unittest.main()
