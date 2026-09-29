"""Unit tests for the idle GPU power manager. No GPU or Flask needed."""
import unittest

import engine_power as ep


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class EnginePowerManagerTests(unittest.TestCase):
    def make(self, idle=60.0):
        events = []
        clock = FakeClock()
        manager = ep.EnginePowerManager(
            sleep_fn=lambda host=None: events.append(("sleep", host)),
            wake_fn=lambda host=None: events.append(("wake", host)),
            release_audio_fn=lambda: events.append(("release", None)),
            host="http://127.0.0.1:8001", idle_seconds=idle, clock=clock)
        return manager, events, clock

    def test_stays_awake_inside_the_idle_window(self):
        manager, events, clock = self.make(idle=60)
        clock.now = 30
        self.assertFalse(manager.maybe_sleep())
        self.assertTrue(manager.awake)
        self.assertEqual(events, [])

    def test_park_releases_engine_and_audio_model(self):
        manager, events, clock = self.make(idle=60)
        clock.now = 120
        self.assertTrue(manager.maybe_sleep())
        self.assertFalse(manager.awake)
        self.assertEqual(events, [("sleep", "http://127.0.0.1:8001"), ("release", None)])

    def test_in_flight_request_defers_the_park(self):
        manager, events, clock = self.make(idle=60)
        manager.enter()
        clock.now = 600
        self.assertFalse(manager.maybe_sleep())
        self.assertTrue(manager.awake)
        manager.exit()
        clock.now = 700  # the exit resets the idle timer
        self.assertTrue(manager.maybe_sleep())

    def test_next_request_wakes_the_engine(self):
        manager, events, clock = self.make(idle=60)
        clock.now = 120
        manager.maybe_sleep()
        manager.ensure_awake()
        self.assertTrue(manager.awake)
        self.assertIn(("wake", "http://127.0.0.1:8001"), events)

    def test_ensure_awake_is_a_noop_when_already_awake(self):
        manager, events, _ = self.make()
        manager.ensure_awake()
        self.assertEqual(events, [])

    def test_park_happens_only_once(self):
        manager, events, clock = self.make(idle=60)
        clock.now = 120
        self.assertTrue(manager.maybe_sleep())
        clock.now = 300
        self.assertFalse(manager.maybe_sleep())
        self.assertEqual([event for event in events if event[0] == "sleep"],
                         [("sleep", "http://127.0.0.1:8001")])

    def test_release_failure_does_not_undo_the_sleep(self):
        clock = FakeClock()

        def boom():
            raise RuntimeError("no cuda")

        manager = ep.EnginePowerManager(lambda host=None: None, lambda host=None: None, boom,
                                        host="h", idle_seconds=60, clock=clock)
        clock.now = 120
        self.assertTrue(manager.maybe_sleep())
        self.assertFalse(manager.awake)

    def test_disabled_idling_never_parks(self):
        manager, events, clock = self.make(idle=0)
        clock.now = 10_000
        self.assertFalse(manager.maybe_sleep())
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
