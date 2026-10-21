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

    def test_ensure_awake_releases_the_models_before_waking_the_engine(self):
        """Only one model may hold the card, so our own models go first.

        Regression: a unit-test run loaded the audio models in-process while the
        engine was parked. Waking it without releasing them made the engine
        allocate what was left of the card and OOM mid-request.
        """
        manager, events, clock = self.make()
        clock.now = 120
        manager.maybe_sleep()
        events.clear()
        manager.ensure_awake()
        self.assertEqual(events, [("release", None), ("wake", "http://127.0.0.1:8001")])
        self.assertTrue(manager.awake)

    def test_ensure_awake_on_an_awake_engine_still_releases_the_models(self):
        manager, events, _ = self.make()
        manager.ensure_awake()
        self.assertEqual(events, [("release", None)])
        self.assertTrue(manager.awake)

    def test_idle_park_also_drops_the_idle_only_models(self):
        """An idle service must not hold any model, not even a small warm one."""
        events = []
        clock = FakeClock()
        manager = ep.EnginePowerManager(
            sleep_fn=lambda host=None: events.append(("sleep", host)),
            wake_fn=lambda host=None: events.append(("wake", host)),
            release_audio_fn=lambda: events.append(("release", None)),
            host="h", idle_seconds=60, clock=clock,
            release_idle_fn=lambda: events.append(("release_idle", None)))
        clock.now = 120
        self.assertTrue(manager.maybe_sleep())
        self.assertEqual(events, [("sleep", "h"), ("release", None), ("release_idle", None)])
        self.assertFalse(manager.awake)

    def test_waking_the_engine_keeps_the_idle_only_models(self):
        """ensure_awake drops the audio models, not the lazily reloaded ones."""
        events = []
        clock = FakeClock()
        manager = ep.EnginePowerManager(
            sleep_fn=lambda host=None: events.append(("sleep", host)),
            wake_fn=lambda host=None: events.append(("wake", host)),
            release_audio_fn=lambda: events.append(("release", None)),
            host="h", idle_seconds=60, clock=clock,
            release_idle_fn=lambda: events.append(("release_idle", None)))
        clock.now = 120
        manager.maybe_sleep()
        events.clear()
        manager.ensure_awake()
        self.assertEqual(events, [("release", None), ("wake", "h")])

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

    def test_failed_park_still_releases_the_audio_model(self):
        """A crashed engine cannot answer /sleep; the audio model must drop anyway.

        Regression: the release used to be skipped when the engine could not be
        parked. The engine's controlled restart then failed its free-memory
        check forever, because the several GB the audio stack holds on the
        shared card were never returned - vLLM stayed down until the container
        was restarted by hand.
        """
        events = []
        clock = FakeClock()

        def unreachable(host=None):
            raise RuntimeError("connection refused")

        manager = ep.EnginePowerManager(unreachable, lambda host=None: None,
                                        lambda: events.append(("release", None)),
                                        host="h", idle_seconds=60, clock=clock)
        clock.now = 120
        self.assertFalse(manager.maybe_sleep())
        self.assertEqual(events, [("release", None)])
        # The engine was never parked, so it is still considered awake and the
        # next idle cycle retries the park.
        self.assertTrue(manager.awake)

    def test_parked_engine_still_drops_the_idle_models(self):
        """Object detection loads a model without waking the engine.

        Regression: the idle path returned early once the engine was parked, so
        that model stayed resident on an idle service indefinitely.
        """
        events = []
        clock = FakeClock()
        manager = ep.EnginePowerManager(
            sleep_fn=lambda host=None: events.append(("sleep", host)),
            wake_fn=lambda host=None: events.append(("wake", host)),
            release_audio_fn=lambda: events.append(("release", None)),
            host="h", idle_seconds=60, clock=clock,
            release_idle_fn=lambda: events.append(("release_idle", None)))
        clock.now = 120
        self.assertTrue(manager.maybe_sleep())
        events.clear()
        clock.now = 240
        self.assertFalse(manager.maybe_sleep())
        self.assertEqual(events, [("release", None), ("release_idle", None)])

    def test_disabled_idling_never_parks(self):
        manager, events, clock = self.make(idle=0)
        clock.now = 10_000
        self.assertFalse(manager.maybe_sleep())
        self.assertEqual(events, [])

    # -- the engine's own answer wins over the local flag ---------------------
    def make_with_probe(self, probe, idle=60.0):
        events = []
        clock = FakeClock()
        manager = ep.EnginePowerManager(
            sleep_fn=lambda host=None: events.append(("sleep", host)),
            wake_fn=lambda host=None: events.append(("wake", host)),
            release_audio_fn=lambda: events.append(("release", None)),
            host="h", idle_seconds=idle, clock=clock, probe_fn=lambda host=None: probe())
        return manager, events, clock

    def test_a_reloaded_worker_wakes_an_engine_that_is_already_asleep(self):
        """Regression: a fresh worker believes the engine is awake.

        After a reload the engine is usually still parked from the previous
        worker's idle cycle. Trusting the local flag sent vision work to a
        sleeping engine, which vLLM answers with a CUDA fault instead of a clean
        error: the request failed with 502 and the engine process died.
        """
        manager, events, _ = self.make_with_probe(lambda: True)
        self.assertTrue(manager.awake, "the constructor still assumes awake")
        manager.ensure_awake()
        self.assertEqual(events, [("release", None), ("wake", "h")])
        self.assertTrue(manager.awake)

    def test_an_engine_that_is_already_awake_is_not_woken_again(self):
        manager, events, _ = self.make_with_probe(lambda: False)
        manager.awake = False  # a stale flag must not cause a redundant wake
        manager.ensure_awake()
        self.assertEqual(events, [("release", None)])
        self.assertTrue(manager.awake)

    def test_park_does_not_sleep_an_engine_that_is_already_asleep(self):
        manager, events, _ = self.make_with_probe(lambda: True)
        self.assertFalse(manager.park())
        self.assertEqual(events, [])
        self.assertFalse(manager.awake, "the local flag is corrected from the engine")

    def test_restore_leaves_an_awake_engine_alone(self):
        manager, events, _ = self.make_with_probe(lambda: False)
        manager.awake = False
        manager.restore()
        self.assertEqual(events, [("release", None)])

    def test_sleep_now_skips_a_second_sleep_when_the_engine_reports_asleep(self):
        manager, events, _ = self.make_with_probe(lambda: True)
        self.assertFalse(manager.sleep_now())
        self.assertEqual([event for event in events if event[0] == "sleep"], [])
        self.assertFalse(manager.awake)

    def test_an_unanswered_probe_falls_back_to_the_local_flag(self):
        """An older engine without /is_sleeping must behave exactly as before."""
        manager, events, _ = self.make_with_probe(lambda: None)
        manager.ensure_awake()
        self.assertEqual(events, [("release", None)])
        parking, parking_events, clock = self.make_with_probe(lambda: None)
        clock.now = 120
        self.assertTrue(parking.maybe_sleep())
        self.assertIn(("sleep", "h"), parking_events)

    def test_a_probe_that_raises_does_not_break_the_wake(self):
        def boom():
            raise RuntimeError("engine unreachable")

        manager, events, _ = self.make_with_probe(boom)
        manager.awake = False
        manager.ensure_awake()
        self.assertIn(("wake", "h"), events)
        self.assertTrue(manager.awake)

    def test_without_a_probe_nothing_is_asked(self):
        manager, events, _ = self.make()
        manager.ensure_awake()
        self.assertEqual(events, [("release", None)])

    # -- deep idle: an engine whose VRAM never came back is restarted --------
    def make_recycler(self, held_mib=5732, ready=True, stopped=True, idle=60.0,
                      recycle_after=120.0, threshold=1536):
        """A parked manager whose engine footprint and restart are faked."""
        events = []
        clock = FakeClock()
        manager = ep.EnginePowerManager(
            sleep_fn=lambda host=None: events.append(("sleep", host)),
            wake_fn=lambda host=None: events.append(("wake", host)),
            release_audio_fn=lambda: events.append(("release", None)),
            host="h", idle_seconds=idle, clock=clock,
            recycle_after_idle_seconds=recycle_after, recycle_vram_mib=threshold,
            engine_vram_fn=lambda: (events.append(("vram", None)), held_mib)[1],
            recycle_fn=lambda host=None: (events.append(("recycle", host)), stopped)[1],
            ready_fn=lambda host=None: (events.append(("ready", host)), ready)[1])
        return manager, events, clock

    def park(self, manager, clock, idle=200.0):
        """Drive the manager into the parked state the recycle starts from."""
        clock.now = idle
        manager.maybe_sleep()

    def test_a_long_idle_engine_holding_vram_is_restarted(self):
        """Parking returns the weights and the KV cache, not the rest.

        Regression: an engine that had served for hours kept 5.7 GB of runtime
        memory that /sleep cannot release (a fresh one parks to 0.66 GB), so an
        idle service held several GB for the life of the container.
        """
        manager, events, clock = self.make_recycler(held_mib=5732)
        self.park(manager, clock)
        events.clear()
        clock.now = 350  # parked, then idle past the deep-idle window
        self.assertTrue(manager.maybe_recycle())
        self.assertEqual(events, [("vram", None), ("recycle", "h"), ("ready", "h")])
        self.assertTrue(manager.awake, "the replacement comes up awake")

    def test_a_fresh_footprint_is_left_alone(self):
        manager, events, clock = self.make_recycler(held_mib=676)
        self.park(manager, clock)
        events.clear()
        clock.now = 350
        self.assertFalse(manager.maybe_recycle())
        self.assertEqual(events, [("vram", None)])
        self.assertFalse(manager.awake)

    def test_no_recycle_before_the_deep_idle_window(self):
        manager, events, clock = self.make_recycler(held_mib=5732, recycle_after=600)
        self.park(manager, clock)
        events.clear()
        clock.now = 350
        self.assertFalse(manager.maybe_recycle())
        self.assertEqual(events, [], "the engine is not even measured yet")

    def test_no_recycle_while_a_request_is_in_flight(self):
        manager, events, clock = self.make_recycler(held_mib=5732)
        self.park(manager, clock)
        events.clear()
        manager.enter()
        clock.now = 900
        self.assertFalse(manager.maybe_recycle())
        self.assertEqual(events, [])
        self.assertFalse(manager.awake)

    def test_no_recycle_while_the_engine_is_awake(self):
        """Only a parked engine is safe to replace: nothing is in flight."""
        manager, events, clock = self.make_recycler(held_mib=5732)
        events.clear()
        clock.now = 900
        self.assertFalse(manager.maybe_recycle())
        self.assertEqual(events, [])
        self.assertTrue(manager.awake)

    def test_an_unmeasurable_footprint_never_restarts_a_running_engine(self):
        manager, events, clock = self.make_recycler(held_mib=None)
        self.park(manager, clock)
        events.clear()
        clock.now = 350
        self.assertFalse(manager.maybe_recycle())
        self.assertFalse(manager.awake)

    def test_recycling_is_off_by_default(self):
        manager, events, clock = self.make()  # no recycle callables at all
        clock.now = 120
        manager.maybe_sleep()
        clock.now = 10_000
        self.assertFalse(manager.maybe_recycle())
        disabled, disabled_events, disabled_clock = self.make_recycler(recycle_after=0)
        self.park(disabled, disabled_clock)
        disabled_events.clear()
        disabled_clock.now = 900
        self.assertFalse(disabled.maybe_recycle())
        self.assertEqual(disabled_events, [])

    def test_a_failed_stop_leaves_the_state_alone(self):
        manager, events, clock = self.make_recycler(held_mib=5732, stopped=False)
        self.park(manager, clock)
        events.clear()
        clock.now = 350
        self.assertFalse(manager.maybe_recycle())
        self.assertEqual(events, [("vram", None), ("recycle", "h")])
        self.assertFalse(manager.awake)

    def test_a_replacement_that_never_answers_is_reported_as_not_awake(self):
        manager, events, clock = self.make_recycler(held_mib=5732, ready=False)
        self.park(manager, clock)
        events.clear()
        clock.now = 350
        self.assertTrue(manager.maybe_recycle())
        self.assertFalse(manager.awake)

    def test_a_replaced_engine_needs_no_wake_call(self):
        """A replacement starts awake, so the next request must not wake it."""
        manager, events, clock = self.make_recycler(held_mib=5732)
        self.park(manager, clock)
        clock.now = 350
        manager.maybe_recycle()
        events.clear()
        manager.ensure_awake()
        self.assertEqual(events, [("release", None)])

    def test_a_request_waits_for_a_restarting_engine_before_waking_it(self):
        """A request arriving during the restart must wait, not fail.

        /wake_up against a starting engine is a connection error, and the old
        behaviour was to send it anyway and fail the request.
        """
        calls = []
        clock = FakeClock()
        manager = ep.EnginePowerManager(
            sleep_fn=lambda host=None: None, wake_fn=lambda host=None: calls.append(("wake", host)),
            release_audio_fn=lambda: None, host="h", idle_seconds=60, clock=clock,
            probe_fn=lambda host=None: True,  # parked
            ready_fn=lambda host=None: (calls.append(("ready", host)), True)[1])
        manager.awake = False
        manager.ensure_awake()
        self.assertEqual(calls, [("ready", "h"), ("wake", "h")])
        self.assertTrue(manager.awake)

    def test_a_request_wakes_a_parked_engine_without_waiting(self):
        """A parked engine answers /v1/models, so the waiter returns at once."""
        calls = []
        manager = ep.EnginePowerManager(
            sleep_fn=lambda host=None: None, wake_fn=lambda host=None: calls.append(("wake", host)),
            release_audio_fn=lambda: None, host="h", idle_seconds=60,
            probe_fn=lambda host=None: True,
            ready_fn=lambda host=None: True)
        manager.ensure_awake()
        self.assertEqual(calls, [("wake", "h")])


if __name__ == "__main__":
    unittest.main()
