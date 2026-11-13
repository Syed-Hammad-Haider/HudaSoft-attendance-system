"""Timing helpers: cooldowns and temporary (expiring) messages."""

import time
from typing import Optional


class Cooldown:
    """
    Simple reusable cooldown gate.

    Usage:
        cd = Cooldown(seconds=5.0)
        if cd.is_ready():
            do_the_thing()
            cd.trigger()
    """

    def __init__(self, seconds: float):
        self._duration = seconds
        self._last_triggered: Optional[float] = None

    def is_ready(self) -> bool:
        if self._last_triggered is None:
            return True
        return (time.time() - self._last_triggered) >= self._duration

    def trigger(self) -> None:
        self._last_triggered = time.time()

    def remaining(self) -> float:
        if self._last_triggered is None:
            return 0.0
        elapsed = time.time() - self._last_triggered
        return max(0.0, self._duration - elapsed)


class ExpiringMessage:
    """
    Holds a text message that is considered "active" for a fixed duration
    after being set, then automatically expires. Used to show the
    "Suggested Action" banner for ~2-3 seconds without extra bookkeeping
    in the display module.
    """

    def __init__(self, duration_seconds: float):
        self._duration = duration_seconds
        self._text: Optional[str] = None
        self._set_at: Optional[float] = None

    def set(self, text: str) -> None:
        self._text = text
        self._set_at = time.time()

    def get_active_text(self) -> Optional[str]:
        if self._text is None or self._set_at is None:
            return None
        if (time.time() - self._set_at) > self._duration:
            self._text = None
            self._set_at = None
            return None
        return self._text


class FPSCounter:
    """Rolling FPS estimate based on time between calls to tick()."""

    def __init__(self, smoothing: float = 0.9):
        self._smoothing = smoothing
        self._last_time: Optional[float] = None
        self._fps: float = 0.0

    def tick(self) -> float:
        now = time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self._fps = (self._smoothing * self._fps) + ((1 - self._smoothing) * instant_fps)
        self._last_time = now
        return self._fps
