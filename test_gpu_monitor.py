"""Unit tests for the GPU memory monitor. No GPU or nvidia-smi needed."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gpu_monitor as monitor


def runner(stdout, returncode=0):
    return lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout)


class ReaderTests(unittest.TestCase):
    def test_device_reading(self):
        device = monitor.read_device(runner("NVIDIA RTX A4000, 16376, 968, 15408\n"))
        self.assertEqual(device, {"name": "NVIDIA RTX A4000", "total_mib": 16376,
                                  "used_mib": 968, "free_mib": 15408})

    def test_device_reading_degrades_without_nvidia_smi(self):
        self.assertIsNone(monitor.read_device(runner("", returncode=1)))
        self.assertIsNone(monitor.read_device(lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())))
        self.assertIsNone(monitor.read_device(runner("NVIDIA RTX A4000, [N/A], [N/A], [N/A]\n")))

    def test_process_reading_keeps_ours_and_others(self):
        output = ("549132, VLLM::EngineCore, 630\n"
                  "549789, gunicorn: worker [app:app], 288\n"
                  "550517, python3, 6920\n")
        self.assertEqual(monitor.read_processes(runner(output)),
                         [{"pid": 549132, "name": "VLLM::EngineCore", "used_mib": 630},
                          {"pid": 549789, "name": "gunicorn: worker [app:app]", "used_mib": 288},
                          {"pid": 550517, "name": "python3", "used_mib": 6920}])

    def test_process_reading_skips_malformed_rows(self):
        self.assertEqual(monitor.read_processes(runner("not a row\n12, thing, 8\n")),
                         [{"pid": 12, "name": "thing", "used_mib": 8}])


class OwnershipTests(unittest.TestCase):
    def test_our_processes_are_recognised_by_command_line(self):
        ours = {"/opt/conda/bin/python /opt/conda/bin/vllm serve /app/hf_models/x",
                "gunicorn: worker [app:app]", "python3 /app/audio_worker.py /tmp/a.wav /tmp/o.json"}
        for command_line in ours:
            with patch.object(monitor, "process_command_line", return_value=command_line):
                self.assertTrue(monitor.is_app_process(4242), command_line)

    def test_another_containers_process_is_not_ours(self):
        with patch.object(monitor, "process_command_line", return_value="ollama serve"):
            self.assertFalse(monitor.is_app_process(4242))

    def test_a_dead_or_hidden_pid_is_not_ours(self):
        with patch.object(monitor, "process_command_line", return_value=""):
            self.assertFalse(monitor.is_app_process(999999))


class EngineLookupTests(unittest.TestCase):
    """Finding the engine's own processes, for the idle recycle."""
    commands = {549132: "VLLM::EngineCore",
                549789: "gunicorn: worker [app:app]",
                551002: "/opt/conda/bin/python /opt/conda/bin/vllm serve /app/hf_models/x"}

    def reader(self, pid):
        return self.commands.get(pid, "")

    def test_engine_memory_counts_the_engine_core_only(self):
        output = ("549132, VLLM::EngineCore, 5732\n"
                  "549789, gunicorn: worker [app:app], 256\n")
        self.assertEqual(monitor.engine_memory_mib(runner(output), self.reader), 5732)

    def test_engine_memory_sums_a_second_engine_process(self):
        output = ("549132, VLLM::EngineCore, 4996\n"
                  "551002, python3, 736\n")
        self.assertEqual(monitor.engine_memory_mib(runner(output), self.reader), 5732)

    def test_engine_memory_is_unknown_without_a_reading(self):
        """Unknown must never be mistaken for "the engine holds nothing"."""
        self.assertIsNone(monitor.engine_memory_mib(runner("", returncode=1), self.reader))
        self.assertIsNone(monitor.engine_memory_mib(
            runner("549789, gunicorn: worker [app:app], 256\n"), self.reader))

    def test_the_supervised_server_is_the_one_found(self):
        with patch.object(monitor.os, "listdir",
                          return_value=["1", "549132", "551002", "jit"]), \
             patch.object(monitor, "process_command_line", side_effect=self.reader):
            self.assertEqual(monitor.find_engine_server_pid(), 551002)

    def test_the_engine_core_alone_is_not_the_server(self):
        """Stopping the EngineCore would leave the server without a restart."""
        only_core = {549132: "VLLM::EngineCore", 549789: "gunicorn: worker [app:app]"}
        with patch.object(monitor.os, "listdir", return_value=["549132", "549789"]), \
             patch.object(monitor, "process_command_line", side_effect=only_core.get):
            self.assertIsNone(monitor.find_engine_server_pid())

    def test_a_host_without_proc_returns_nothing(self):
        with patch.object(monitor.os, "listdir", side_effect=OSError("no /proc")):
            self.assertIsNone(monitor.find_engine_server_pid())


class EvaluationTests(unittest.TestCase):
    total = 16376

    def evaluate(self, used_mib, processes, engine=None):
        return monitor.evaluate({"name": "NVIDIA RTX A4000", "total_mib": self.total,
                                 "used_mib": used_mib, "free_mib": self.total - used_mib},
                                processes, 0.92, engine)

    def test_ours_alone_leaves_room_to_restart(self):
        """The normal serving state: the engine holds the card, nothing else does."""
        verdict = self.evaluate(10400, [{"pid": 1, "name": "VLLM::EngineCore", "used_mib": 10094, "app": True},
                                        {"pid": 2, "name": "gunicorn: worker [app:app]", "used_mib": 288, "app": True}])
        self.assertEqual(verdict["status"], monitor.OK)
        self.assertTrue(verdict["ok"])
        # 16376 - 18 MiB of driver/desktop = everything else is ours.
        self.assertEqual(verdict["unattributed_mib"], 18)
        self.assertEqual(verdict["available_mib"], 16358)
        self.assertGreaterEqual(verdict["available_mib"], verdict["required_mib"])

    def test_another_process_holding_the_card_is_critical(self):
        """How the service went down: 6.9 GB taken by someone else."""
        verdict = self.evaluate(7893, [{"pid": 1, "name": "VLLM::EngineCore", "used_mib": 630, "app": True},
                                        {"pid": 2, "name": "gunicorn: worker [app:app]", "used_mib": 288, "app": True},
                                        {"pid": 3, "name": "python3", "used_mib": 6920, "app": False}])
        self.assertEqual(verdict["status"], monitor.CRITICAL)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["required_mib"], 15065)
        self.assertLess(verdict["available_mib"], verdict["required_mib"])
        self.assertEqual(verdict["shortfall_mib"], 15065 - verdict["available_mib"])
        self.assertIn("not enough vram", verdict["detail"].lower())

    def test_a_thin_margin_warns_before_it_fails(self):
        verdict = self.evaluate(2000, [{"pid": 1, "name": "gunicorn: worker [app:app]", "used_mib": 1000, "app": True},
                                        {"pid": 2, "name": "other", "used_mib": 1000, "app": False}])
        self.assertEqual(verdict["status"], monitor.WARNING)
        self.assertFalse(verdict["ok"])
        self.assertGreaterEqual(verdict["margin_mib"], 0)
        self.assertLess(verdict["margin_mib"], monitor.THIN_MARGIN_MIB)

    def test_desktop_and_driver_memory_counts_against_the_engine(self):
        """Memory no compute process owns still cannot be used by the engine."""
        verdict = self.evaluate(1400, [{"pid": 1, "name": "gunicorn: worker [app:app]", "used_mib": 100, "app": True}])
        self.assertEqual(verdict["unattributed_mib"], 1300)
        self.assertEqual(verdict["available_mib"], 16376 - 1300)

    def test_audio_stage_headroom_is_reported(self):
        roomy = self.evaluate(1000, [{"pid": 1, "name": "gunicorn: worker [app:app]", "used_mib": 900, "app": True}])
        self.assertTrue(roomy["audio_ok"])
        tight = self.evaluate(14000, [{"pid": 1, "name": "gunicorn: worker [app:app]", "used_mib": 900, "app": True},
                                      {"pid": 2, "name": "other", "used_mib": 13000, "app": False}])
        self.assertFalse(tight["audio_ok"])

    def test_a_serving_engine_is_not_reported_as_an_outage(self):
        """The card can be too full to restart while the service serves perfectly."""
        verdict = self.evaluate(12864, [{"pid": 1, "name": "VLLM::EngineCore", "used_mib": 10094, "app": True},
                                        {"pid": 2, "name": "python3", "used_mib": 2770, "app": False}],
                                engine={"reachable": True, "sleeping": False})
        self.assertEqual(verdict["status"], monitor.OK)
        self.assertTrue(verdict["ok"])
        self.assertTrue(verdict["engine_running"])
        self.assertFalse(verdict["restart_ok"])
        self.assertIn("engine is running", verdict["detail"].lower())
        self.assertNotIn("not enough vram", verdict["detail"].lower())

    def test_a_parked_engine_that_cannot_wake_is_critical(self):
        """Parked weights come back on the next request, so the budget is live again."""
        verdict = self.evaluate(3605, [{"pid": 2, "name": "python3", "used_mib": 2770, "app": False},
                                        {"pid": 3, "name": "vllm serve", "used_mib": 630, "app": True}],
                                engine={"reachable": True, "sleeping": True})
        self.assertEqual(verdict["status"], monitor.CRITICAL)
        self.assertFalse(verdict["restart_ok"])
        self.assertIn("wake the parked engine", verdict["detail"])

    def test_an_unreachable_engine_keeps_the_plain_budget_verdict(self):
        verdict = self.evaluate(3605, [{"pid": 2, "name": "python3", "used_mib": 2770, "app": False}],
                                engine={"reachable": False, "sleeping": None})
        self.assertEqual(verdict["status"], monitor.CRITICAL)
        self.assertFalse(verdict["engine_reachable"])
        self.assertIn("not enough vram", verdict["detail"].lower())


class SnapshotTests(unittest.TestCase):
    def snapshot(self, **kwargs):
        defaults = dict(
            device_reader=lambda: {"name": "NVIDIA RTX A4000", "total_mib": 16376,
                                   "used_mib": 968, "free_mib": 15408},
            process_reader=lambda: [{"pid": 1, "name": "VLLM::EngineCore", "used_mib": 630},
                                    {"pid": 2, "name": "gunicorn: worker [app:app]", "used_mib": 288},
                                    {"pid": 3, "name": "other", "used_mib": 50}],
            owner_check=lambda pid: pid in (1, 2),
            clock=lambda: 1_760_000_000.0)
        defaults.update(kwargs)
        return monitor.build_snapshot(0.92, **defaults)

    def test_percentages_are_of_the_device_and_rows_are_sorted(self):
        frame = self.snapshot()
        self.assertTrue(frame["available"])
        self.assertEqual(frame["device"]["used_percent"], 5.9)
        self.assertEqual([process["name"] for process in frame["processes"]],
                         ["VLLM::EngineCore", "gunicorn: worker [app:app]", "other"])
        self.assertEqual(frame["processes"][0]["used_percent"], 3.8)
        self.assertEqual(frame["processes"][0]["app"], True)
        self.assertEqual(frame["processes"][2]["app"], False)
        self.assertEqual(frame["resource"]["required_percent"], 92.0)

    def test_snapshot_carries_a_timestamp_and_the_engine_budget(self):
        frame = self.snapshot()
        self.assertTrue(frame["timestamp"].startswith("2025-"))
        self.assertEqual(frame["resource"]["required_mib"], 15065)
        self.assertEqual(frame["resource"]["utilization"], 0.92)

    def test_snapshot_without_a_readable_card_reports_it_instead_of_failing(self):
        frame = self.snapshot(device_reader=lambda: None)
        self.assertFalse(frame["available"])
        self.assertIsNone(frame["device"])
        self.assertEqual(frame["processes"], [])
        self.assertEqual(frame["resource"]["status"], monitor.WARNING)
        self.assertIsNone(frame["resource"]["required_mib"])

    def test_snapshot_carries_the_engine_state_it_was_given(self):
        frame = self.snapshot(engine_reader=lambda: {"reachable": True, "sleeping": False})
        self.assertEqual(frame["resource"]["status"], monitor.OK)
        self.assertTrue(frame["resource"]["engine_running"])

    def test_a_failing_engine_probe_does_not_break_the_frame(self):
        def boom():
            raise RuntimeError("engine probe exploded")
        frame = self.snapshot(engine_reader=boom)
        self.assertTrue(frame["available"])
        self.assertIsNone(frame["resource"]["engine_running"])


if __name__ == "__main__":
    unittest.main()
