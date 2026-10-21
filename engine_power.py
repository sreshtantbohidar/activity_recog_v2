"""Idle-driven GPU power management.

While the service is busy we keep the vLLM engine and the audio model resident
for speed. When no API request has run for a while, we release the GPU so other
containers on the host can use it:

  * the vLLM engine is put to sleep (weights offloaded to CPU RAM, KV cache
    discarded, ~10 GB of VRAM returned);
  * the audio model is unloaded.

Parking only releases what vLLM tracks (the weights and the KV cache). The
EngineCore keeps several GB of runtime memory outside those pools that ``/sleep``
cannot release, and that memory grows with use; a long idle period therefore also
restarts the engine, so the replacement's footprint starts from scratch.

The next request wakes/reloads them transparently. The manager is deliberately
dependency-light (callables are injected) so its behaviour is unit-testable
without a GPU.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class EnginePowerManager:
    """Puts the vision engine to sleep and drops every other model when idle."""

    def __init__(self, sleep_fn: Callable[..., object], wake_fn: Callable[..., object],
                 release_audio_fn: Callable[[], object], host: Optional[str] = None,
                 idle_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic,
                 release_idle_fn: Optional[Callable[[], object]] = None,
                 probe_fn: Optional[Callable[..., Optional[bool]]] = None,
                 recycle_after_idle_seconds: float = 0.0, recycle_vram_mib: int = 0,
                 engine_vram_fn: Optional[Callable[[], Optional[int]]] = None,
                 recycle_fn: Optional[Callable[..., bool]] = None,
                 ready_fn: Optional[Callable[..., bool]] = None):
        self._sleep_fn = sleep_fn
        self._wake_fn = wake_fn
        self._release_audio_fn = release_audio_fn
        self._release_idle_fn = release_idle_fn
        self._probe_fn = probe_fn
        self._engine_vram_fn = engine_vram_fn
        self._recycle_fn = recycle_fn
        self._ready_fn = ready_fn
        self._clock = clock
        self.host = host
        self.idle_seconds = float(idle_seconds or 0)
        # Idle recycling: parking cannot return the engine's non-pool memory, so
        # a long idle period also replaces the process. 0 seconds disables it.
        self.recycle_after_idle_seconds = float(recycle_after_idle_seconds or 0)
        self.recycle_vram_mib = int(recycle_vram_mib or 0)
        self.awake = True
        self.last_activity = self._clock()
        self._active = 0
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None

    # -- activity tracking ---------------------------------------------------
    def touch(self) -> None:
        self.last_activity = self._clock()

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.touch()

    def exit(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            self.touch()

    # -- power control -------------------------------------------------------
    def release_models(self, idle: bool = False) -> None:
        """Drop the models this process holds so the engine can have the card.

        ``idle`` also drops the models that are kept warm only while requests
        are being served, which is what leaves an idle service holding none.
        """
        try:
            self._release_audio_fn()
        except Exception:  # a failed unload must not undo the engine sleep
            logger.exception("Could not release the audio model")
        if idle and self._release_idle_fn is not None:
            try:
                self._release_idle_fn()
            except Exception:
                logger.exception("Could not release the idle models")

    def engine_sleeping(self) -> Optional[bool]:
        """Whether the engine reports being parked, or None if it cannot say.

        The local flag is only correct while this process is the one that parked
        the engine. A reloaded worker starts out believing the engine is awake,
        and sending vision work to a parked engine is a CUDA fault rather than a
        clean error, so the engine itself is asked before the decision is made.
        """
        if self._probe_fn is None:
            return None
        try:
            return self._probe_fn(host=self.host)
        except Exception:
            logger.exception("Could not ask the vLLM engine whether it is asleep")
            return None

    def engine_vram_mib(self) -> Optional[int]:
        """VRAM the engine still holds, or None when it cannot be measured.

        None means "unknown" and never "nothing": the recycle is skipped rather
        than acting on a number a failed nvidia-smi read made up.
        """
        if self._engine_vram_fn is None:
            return None
        try:
            return self._engine_vram_fn()
        except Exception:
            logger.exception("Could not measure the engine's VRAM use")
            return None

    def engine_ready(self) -> bool:
        """Whether the engine answers, waiting a bounded time for a restart.

        True when no waiter is wired in: the caller then simply tries to wake
        the engine, which is the behaviour before idle recycling existed.
        """
        if self._ready_fn is None:
            return True
        try:
            return bool(self._ready_fn(host=self.host))
        except Exception:
            logger.exception("Could not wait for the vLLM engine to become ready")
            return False

    def _sync_power_state(self) -> None:
        """Correct the local flag from the engine's own answer, when it gives one."""
        sleeping = self.engine_sleeping()
        if sleeping is not None:
            self.awake = not sleeping

    def ensure_awake(self) -> None:
        """Wake the engine if it was parked. Called before any vision work.

        This process's own models are released first: waking the engine needs
        the whole card back, and whatever loaded them in-process (an audio
        stage, or the unit-test suite) must not be resident at the same time.
        """
        self.release_models()
        with self._lock:
            self.touch()
            self._sync_power_state()
            if self.awake:
                return
            # A parked engine answers /is_sleeping, so this normally means it is
            # parked. It can also mean that an idle recycle (see maybe_recycle)
            # stopped the engine and the supervised replacement is still
            # loading: /wake_up against a starting engine is a connection error
            # that would fail the request. Wait for it to answer and look again,
            # because a replacement comes up awake and needs no wake call.
            if not self.engine_ready():
                logger.warning("The vLLM engine did not answer in time; waking it anyway")
            self._sync_power_state()
            if self.awake:
                return
            self._wake_fn(host=self.host)
            self.awake = True
            logger.info("Woke the vLLM engine for a new request")

    def park(self) -> bool:
        """Sleep the engine ahead of the audio stage. Returns True if parked.

        The vision engine and the audio model cannot both be resident on a
        shared card, so the audio stage parks the engine first.
        """
        with self._lock:
            self.touch()
            self._sync_power_state()
            if not self.awake:
                return False
            self._sleep_fn(host=self.host)
            self.awake = False
            return True

    def restore(self) -> None:
        """Release the audio model and wake the engine after the audio stage."""
        try:
            self._release_audio_fn()
        except Exception:
            logger.exception("Could not release the audio model")
        finally:
            with self._lock:
                self.touch()
                self._sync_power_state()
                if not self.awake:
                    try:
                        self._wake_fn(host=self.host)
                        self.awake = True
                    except Exception:
                        logger.exception("Failed to wake the engine after audio")

    def sleep_now(self) -> bool:
        """Park the engine and release this process's models. Returns True if parked.

        The audio model is released even when the engine cannot be parked. A
        crashed engine answers /sleep with a connection error, and the
        supervised restart only succeeds once the several GB the audio stack
        holds on the shared card are back. Skipping the release in that case
        left the engine down until the container was restarted by hand.
        """
        with self._lock:
            self._sync_power_state()
            if not self.awake:
                return False
            parked = True
            try:
                self._sleep_fn(host=self.host)
            except Exception:
                # Not parked: the engine is unreachable (or refused the call),
                # so stay "awake" and let the next idle cycle retry. Releasing
                # the audio model below is what lets the engine come back.
                parked = False
                logger.exception("Could not park the vLLM engine; released the models anyway")
            self.release_models(idle=True)
            if not parked:
                return False
            self.awake = False
            logger.info("Idle for >= %.0fs: engine asleep, audio model released", self.idle_seconds)
            return True

    def maybe_sleep(self) -> bool:
        """Park the engine and drop the idle models when nothing is in flight.

        Returns True only when this call parked the engine. An engine that is
        already parked still has its models dropped: a request can load one
        without waking the engine (object detection never needs it), and an
        idle service must not keep holding it.
        """
        with self._lock:
            if self.idle_seconds <= 0 or self._active > 0:
                return False
            if self._clock() - self.last_activity < self.idle_seconds:
                return False
            if not self.awake:
                self.release_models(idle=True)
                return False
        return self.sleep_now()

    def maybe_recycle(self) -> bool:
        """Replace a long-idle engine whose VRAM never came back. True if replaced.

        Parking returns the weights and the KV cache (~8 GB), but the EngineCore
        keeps several GB of runtime state outside those pools that ``/sleep``
        cannot touch; it grows with use and only process exit returns it (a
        freshly started engine parks down to a few hundred MB). A long idle
        period is the moment to pay for that: the entrypoint supervisor starts
        the engine again, the replacement parks to its own small footprint, and
        a request arriving while it loads waits for it instead of being sent to
        a process that is going away.
        """
        if self._recycle_fn is None or self.recycle_after_idle_seconds <= 0:
            return False
        with self._lock:
            # Only a parked engine is safe to replace: nothing is in flight and
            # the card is already released while the replacement loads.
            if self._active > 0 or self.awake:
                return False
            if self._clock() - self.last_activity < self.recycle_after_idle_seconds:
                return False
            held_mib = self.engine_vram_mib()
            if held_mib is None or held_mib < self.recycle_vram_mib:
                return False
            try:
                stopped = bool(self._recycle_fn(host=self.host))
            except Exception:
                logger.exception("Could not stop the idle vLLM engine")
                return False
            if not stopped:
                return False
            logger.info("Idle engine still held %s MiB of VRAM; restarting it", held_mib)
            self.awake = False
            self.touch()
        # Wait outside the lock: a request arriving now must wait for the
        # replacement rather than block behind it or be sent to a dead engine.
        ready = self.engine_ready()
        with self._lock:
            self.awake = bool(ready)
            self.touch()
        if ready:
            logger.info("vLLM engine restarted; it holds a fresh footprint again")
        else:
            logger.warning("The restarted vLLM engine has not answered yet")
        return True

    # -- background loop -----------------------------------------------------
    def start(self) -> None:
        """Start the idle watchdog. No-op when idling is disabled.

        Idle recycling rides on the same loop, and only a parked engine is
        recycled, so disabling idle parking disables it too.
        """
        if self.idle_seconds <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="engine-power", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        interval = max(1.0, min(5.0, self.idle_seconds / 4.0))
        while True:
            time.sleep(interval)
            try:
                self.maybe_sleep()
                self.maybe_recycle()
            except Exception:
                logger.exception("Idle power-down check failed")
