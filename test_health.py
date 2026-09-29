import json
import unittest
from unittest.mock import patch

from vision_backend import check_ollama_model


class OllamaHealthTests(unittest.TestCase):
    def test_reports_model_available_when_exact_tag_is_installed(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"models": [{"name": "qwen3-vl:8b-instruct"}]}).encode()

        status = check_ollama_model(
            host="http://ollama:11434",
            model="qwen3-vl:8b-instruct",
            opener=lambda request, timeout: Response(),
        )
        self.assertEqual(status, {
            "reachable": True,
            "model": "qwen3-vl:8b-instruct",
            "model_available": True,
        })

    def test_reports_missing_model_when_ollama_is_reachable(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"models": [{"name": "qwen2.5:14b"}]}).encode()

        status = check_ollama_model(
            model="qwen3-vl:8b-instruct",
            opener=lambda request, timeout: Response(),
        )
        self.assertTrue(status["reachable"])
        self.assertFalse(status["model_available"])

    def test_reports_unreachable_ollama_without_raising(self):
        status = check_ollama_model(opener=lambda request, timeout: (_ for _ in ()).throw(OSError("offline")))
        self.assertFalse(status["reachable"])
        self.assertFalse(status["model_available"])
        self.assertIn("offline", status["error"])


if __name__ == "__main__":
    unittest.main()
