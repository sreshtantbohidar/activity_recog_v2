"""Unit tests for the vLLM control client. No engine and no GPU needed."""
import unittest

import vision_backend as vb


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def recording_opener(body, calls):
    def opener(request, timeout=None):
        calls.append((request.get_method(), request.full_url, timeout))
        return FakeResponse(body)

    return opener


class EngineSleepStateTests(unittest.TestCase):
    def test_a_sleeping_engine_is_reported(self):
        calls = []
        opener = recording_opener(b'{"is_sleeping": true}', calls)
        self.assertIs(vb.engine_is_sleeping("http://127.0.0.1:8001", opener=opener), True)
        self.assertEqual(calls, [("GET", "http://127.0.0.1:8001/is_sleeping", 120.0)])

    def test_an_awake_engine_is_reported(self):
        opener = recording_opener(b'{"is_sleeping": false}', [])
        self.assertIs(vb.engine_is_sleeping("http://h:1", opener=opener), False)

    def test_an_engine_that_cannot_answer_reports_unknown(self):
        """An unreachable or older engine must not fail the caller's request."""
        def refused(request, timeout=None):
            raise OSError("connection refused")

        self.assertIsNone(vb.engine_is_sleeping("http://h:1", opener=refused))
        self.assertIsNone(vb.engine_is_sleeping("http://h:1", opener=recording_opener(b"", [])))
        self.assertIsNone(vb.engine_is_sleeping("http://h:1", opener=recording_opener(b"not json", [])))
        self.assertIsNone(vb.engine_is_sleeping("http://h:1", opener=recording_opener(b'{"other": 1}', [])))
        self.assertIsNone(vb.engine_is_sleeping("http://h:1",
                                                opener=recording_opener(b'{"is_sleeping": "yes"}', [])))


if __name__ == "__main__":
    unittest.main()
