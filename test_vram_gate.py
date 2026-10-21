"""Unit tests for the engine-start VRAM gate. No GPU or nvidia-smi needed."""
import unittest
from types import SimpleNamespace

import gpu_vram_gate as gate


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(free_values, total=16376, util=0.92, max_wait=0.0, should_stop=None, error=False):
    """A gate over a scripted free-memory sequence, with a virtual clock."""
    events = []
    clock = FakeClock()
    remaining = list(free_values)

    def free_fn():
        if error:
            raise RuntimeError("no driver")
        return remaining.pop(0) if remaining else free_values[-1]

    def sleep(seconds):
        events.append(("sleep", seconds))
        clock.now += seconds

    instance = gate.VramGate(util, total_fn=lambda: total, free_fn=free_fn, sleep=sleep,
                             clock=clock, interval=5.0, max_wait=max_wait, should_stop=should_stop)
    return instance, events, clock


class RequiredMemoryTests(unittest.TestCase):
    def test_required_mib_follows_the_engine_budget(self):
        # The same ratio vLLM checks, from the same total it reads.
        self.assertEqual(gate.required_mib(0.92, 16376), 15065)   # RTX A4000 16 GB
        self.assertEqual(gate.required_mib(0.80, 24576), 19660)   # 24 GB card
        self.assertEqual(gate.required_mib(0.85, 49152), 41779)   # A6000 48 GB


class VramGateTests(unittest.TestCase):
    def test_starts_immediately_when_the_card_can_hold_the_engine(self):
        instance, events, _ = make([16000])
        self.assertTrue(instance.wait())
        self.assertEqual(events, [])  # no polling at all

    def test_waits_for_the_other_process_to_release_the_card(self):
        """The usual case: another container holds the GPU and then lets go.

        Regression: the supervisor used to load the model immediately, fail
        vLLM's free-memory check, and retry that whole load every five seconds.
        """
        instance, events, clock = make([11000, 11000, 14000, 16000])
        self.assertTrue(instance.wait())
        self.assertEqual([event[1] for event in events], [5.0, 5.0, 5.0])
        self.assertGreaterEqual(clock.now, 15.0)

    def test_does_not_start_while_the_memory_is_short(self):
        instance, _, _ = make([11000] * 3, max_wait=10.0)
        self.assertFalse(instance.wait())

    def test_reports_what_it_is_waiting_for(self):
        instance, _, _ = make([11000, 16000])
        with self.assertLogs("vram_gate", level="WARNING") as captured:
            self.assertTrue(instance.wait())
        message = "\n".join(captured.output)
        self.assertIn("11000 MiB free", message)
        self.assertIn("15065 MiB needed", message)
        self.assertIn("another process", message)

    def test_reports_a_short_wait_only_once(self):
        """A one-second wait for a card freeing up is not worth a log line."""
        instance, _, _ = make([11000] * 20, max_wait=1.0)
        with self.assertLogs("vram_gate", level="WARNING") as captured:
            self.assertFalse(instance.wait())
        warnings = [line for line in captured.output if "Waiting for GPU memory" in line]
        self.assertEqual(len(warnings), 1)

    def test_unmeasurable_memory_does_not_block_startup(self):
        """Without nvidia-smi, vLLM's own check stays the authority."""
        instance, _, _ = make([0], total=None)
        self.assertTrue(instance.wait())
        instance, _, _ = make([11000], error=True)  # free_fn raises
        self.assertTrue(instance.wait())

    def test_a_shutdown_request_ends_the_wait(self):
        instance, events, _ = make([11000], should_stop=lambda: True)
        self.assertFalse(instance.wait())
        self.assertEqual(events, [])  # stopped before sleeping


class QueryTests(unittest.TestCase):
    def test_reads_the_first_device_only(self):
        """The container sees one GPU: its reservation, so line one is ours."""
        runner = lambda *a, **k: SimpleNamespace(returncode=0, stdout="16376\n24576\n")
        self.assertEqual(gate.query_mib("total", runner=runner), 16376)

    def test_returns_none_when_nvidia_smi_fails(self):
        runner = lambda *a, **k: SimpleNamespace(returncode=1, stdout="")
        self.assertIsNone(gate.query_mib("free", runner=runner))
        runner = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi"))
        self.assertIsNone(gate.query_mib("free", runner=runner))

    def test_returns_none_on_unparsable_output(self):
        runner = lambda *a, **k: SimpleNamespace(returncode=0, stdout="[N/A]\n")
        self.assertIsNone(gate.query_mib("free", runner=runner))


if __name__ == "__main__":
    unittest.main()
