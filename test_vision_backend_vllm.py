import json
import os
import unittest
from unittest.mock import patch

import vision_backend as vb


class Response:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


class VllmClientTests(unittest.TestCase):
    def test_default_backend_is_vllm(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ACTIVITY_VISION_BACKEND", None)
            self.assertEqual(vb.get_vision_backend(), "vllm")

    def test_vllm_backend_is_supported_and_selectable(self):
        self.assertIn("vllm", vb.SUPPORTED_VISION_BACKENDS)
        self.assertEqual(vb.get_vision_backend("vllm"), "vllm")

    def test_check_vllm_server_reports_served_model(self):
        status = vb.check_vllm_server(
            host="http://thermal:8001", model="Qwen/Qwen3-VL-8B-Instruct",
            opener=lambda request, timeout: Response({"data": [{"id": "Qwen/Qwen3-VL-8B-Instruct"}]}))
        self.assertTrue(status["reachable"])
        self.assertTrue(status["model_available"])

    def test_check_vllm_server_reports_unreachable_without_raising(self):
        status = vb.check_vllm_server(opener=lambda request, timeout: (_ for _ in ()).throw(OSError("down")))
        self.assertFalse(status["reachable"])
        self.assertFalse(status["model_available"])
        self.assertIn("down", status["error"])

    def test_analyze_image_with_vllm_returns_content(self):
        import tempfile
        handle, image_path = tempfile.mkstemp(suffix=".jpg")
        os.write(handle, b"fake-bytes")
        os.close(handle)
        captured = {}

        def fake_opener(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode())
            return Response({"choices": [{"message": {"content": '{"detected_activity": ["training"]}'}}]})

        try:
            with patch.dict(os.environ, {"VLLM_VISION_HOST": "http://thermal:8001"}):
                text = vb.analyze_image_with_vllm(image_path, "classify", opener=fake_opener)
        finally:
            os.remove(image_path)
        self.assertIn("training", text)
        self.assertTrue(captured["url"].endswith("/v1/chat/completions"))
        # The served name is whatever the deployment configures (env wins over
        # the module default), so assert against that resolution rather than a
        # hardcoded checkpoint id.
        self.assertEqual(captured["body"]["model"],
            os.environ.get("VLLM_VISION_MODEL", vb.DEFAULT_VLLM_VISION_MODEL))
        self.assertEqual(vb.DEFAULT_VLLM_VISION_MODEL, "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit")
        self.assertEqual(captured["body"]["temperature"], 0.1)
        self.assertTrue(captured["body"]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg"))

    def test_analyze_image_with_vllm_wraps_transport_errors(self):
        with self.assertRaises(vb.VisionBackendError):
            vb.analyze_image_with_vllm("missing.jpg", "p", opener=lambda request, timeout: (_ for _ in ()).throw(OSError("no route")))

    def test_sleep_and_wake_call_the_control_endpoints(self):
        captured = []

        def fake_opener(request, timeout):
            captured.append((request.full_url, request.get_method()))
            return Response({})

        vb.sleep_vllm(host="http://thermal:8001", opener=fake_opener)
        vb.wake_vllm(host="http://thermal:8001", opener=fake_opener)
        self.assertTrue(captured[0][0].endswith("/sleep?level=1"))
        self.assertTrue(captured[1][0].endswith("/wake_up"))
        self.assertTrue(all(method == "POST" for _, method in captured))

    def test_vllm_is_the_only_backend_and_ollama_is_gone(self):
        self.assertEqual(vb.SUPPORTED_VISION_BACKENDS, {"vllm"})
        self.assertFalse(hasattr(vb, "analyze_image_with_ollama"))
        self.assertFalse(hasattr(vb, "check_ollama_model"))


if __name__ == "__main__":
    unittest.main()
