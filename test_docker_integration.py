"""Black-box tests. Run only against an already started Docker container."""
import io, json, os, unittest
from urllib.request import Request, urlopen

BASE = os.getenv("TEST_BASE_URL", "http://127.0.0.1:5009")

class DockerIntegrationTests(unittest.TestCase):
    def get(self, path):
        with urlopen(BASE + path, timeout=10) as response:
            return response.status, response.headers, response.read()

    def test_health_and_dashboard(self):
        status, _, body = self.get("/health")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        # The only backend is the in-container vLLM engine serving Qwen3-VL.
        self.assertEqual(payload["primary_backend"],
            os.getenv("ACTIVITY_VISION_BACKEND", "vllm"))
        self.assertEqual(payload["vision_model"],
            os.getenv("VLLM_VISION_MODEL", "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"))
        self.assertEqual(payload["audio_model"], "Qwen/Qwen2-Audio-7B-Instruct")
        self.assertNotIn("fallback_backend", payload)
        status, headers, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"Activity Recognition Console", body)

    def test_swagger_spec_has_all_public_operations(self):
        status, _, body = self.get("/apispec.json")
        self.assertEqual(status, 200)
        spec = json.loads(body)
        expected = {"/health", "/vision-backend", "/analyze-image", "/analyze-video",
                    "/analyze-video-detailed", "/fetch-media", "/detect-objects"}
        expected.add("/reports/{filename}")
        expected.add("/config")
        expected.add("/queue")
        expected.add("/queue/{job_id}")
        expected.add("/queue/{job_id}/cancel")
        expected.add("/queue/{job_id}/reload")
        self.assertTrue(expected.issubset(spec["paths"]))
        for path in expected:
            for operation in spec["paths"][path].values():
                self.assertTrue(operation.get("summary"), (path, operation))
                self.assertIn("responses", operation)

if __name__ == "__main__": unittest.main()
