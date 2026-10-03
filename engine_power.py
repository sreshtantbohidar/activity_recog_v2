"""Idle-driven GPU power management.

While the service is busy we keep the vLLM engine and the audio model resident
for speed. When no API request has run for a while, we release the GPU so other
containers on the host can use it:

  * the vLLM engine is put to sleep (weights offloaded to CPU RAM, KV cache
    discarded, ~10 GB of VRAM returned);
  * the audio model is unloaded.

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
    """Puts the vision engine to sleep and drops the audio model when idle."""

    def __init__(self, sleep_fn: Callable[..., object], wake_fn: Callable[..., object],
                 release_audio_fn: Callable[[], object], host: Optional[str] = None,
                 idle_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self._sleep_fn = sleep_fn
        self._wake_fn = wake_fn
        self._release_audio_fn = release_audio_fn
        self._clock = clock
        self.host = host
        self.idle_seconds = float(idle_seconds or 0)
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
    def ensure_awake(self) -> None:
        """Wake the engine if it was parked. Called before any vision work."""
        with self._lock:
            self.touch()
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
                if not self.awake:
                    try:
                        self._wake_fn(host=self.host)
                        self.awake = True
                    except Exception:
                        logger.exception("Failed to wake the engine after audio")

    def sleep_now(self) -> bool:
        """Park the engine and release the audio model. Returns True if parked."""
        with self._lock:
            if not self.awake:
                return False
            self._sleep_fn(host=self.host)
            try:
                self._release_audio_fn()
            except Exception:  # a failed unload must not undo the engine sleep
                logger.exception("Could not release the audio model")
            self.awake = False
            logger.info("Idle for >= %.0fs: engine asleep, audio model released", self.idle_seconds)
            return True

    def maybe_sleep(self) -> bool:
        """Park the engine when idle and no request is in flight."""
        with self._lock:
            if self.idle_seconds <= 0 or not self.awake or self._active > 0:
                return False
            if self._clock() - self.last_activity < self.idle_seconds:
                return False
        return self.sleep_now()

    # -- background loop -----------------------------------------------------
    def start(self) -> None:
        """Start the idle watchdog. No-op when idling is disabled."""
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
            except Exception:
                logger.exception("Idle power-down check failed")
