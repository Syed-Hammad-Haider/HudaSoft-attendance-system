"""
Pure, camera-independent tracker for whether a person's overall BODY
has been relatively stationary right now.

WHY THIS EXISTS -- AN HONEST FINDING FROM TESTING, NOT A GUESS:
wave_detection/test_oscillation.py's synthetic tests found that
hand-motion kinematics ALONE (amplitude + reversal timing, however
well-designed the signal) cannot fully separate a genuine, deliberate
wave from vigorous arm-swinging while walking or jogging, when the two
happen to share a similar amplitude and cadence -- which real human
motion sometimes does. This isn't a threshold that needs more tuning;
it's a structural limit of what hand-only kinematics can distinguish,
confirmed directly by the "Jogging-speed arm pump" and "Brisk walking"
test cases both being able to pass the SAME amplitude/timing checks a
genuine wave passes.

THE FIX: use an already-available, ORTHOGONAL signal instead of
squeezing more out of hand kinematics. A person WALKING is, close to
by definition, translating their whole body across the frame; a person
PAUSING TO WAVE at an entrance typically is not. The person tracker
already computes each tracked person's bounding box every frame for
entirely different reasons (entrance-line crossing) -- reusing that
existing signal here is nearly free and directly resolves the
ambiguity hand kinematics alone cannot.

DESIGN NOTE -- WHY A SHORT WINDOW, NOT INSTANTANEOUS FRAME-TO-FRAME
SPEED: a first attempt at fixing the lag above computed speed from
consecutive-frame differences (position delta / time delta). Testing
this directly (wave_detection/test_body_motion.py) found a NEW real
bug: dividing a small, noisy frame-to-frame distance by a small time
step amplifies ordinary person-tracker bbox jitter into a large
apparent "speed" -- even a person standing perfectly still, with
just 1-2% of normal bounding-box jitter, got falsely flagged as
"moving." An empirical sweep across window lengths (documented in the
test file) found 0.5 seconds is the point where responsiveness after a
real stop is still fast (~0.3s lag, down from ~0.7-1.0s with the
original 1.0s full-window design) while zero false positives remain
even at a generous 3% jitter assumption.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class BodyMotionDebugInfo:
    speed_fraction_of_height_per_sec: Optional[float] = None
    is_stationary: bool = True
    samples: int = 0


class BodyStillnessTracker:
    def __init__(self, window_seconds: float, max_speed_fraction_of_height_per_sec: float):
        self._window_seconds = window_seconds
        self._max_speed = max_speed_fraction_of_height_per_sec
        self._history: Deque[Tuple[float, float, float, float]] = deque()  # (cx, cy, height, t)
        self.debug = BodyMotionDebugInfo()

    def set_max_speed(self, max_speed_fraction_of_height_per_sec: float) -> None:
        """
        Update the stillness-speed cap in place, for a person who is
        ALREADY being tracked (e.g. after the 'S' sensitivity hotkey is
        pressed live). Without this, a person already in frame when
        sensitivity changes would silently keep using whatever cap was
        in effect the moment they first appeared -- see
        WaveDetector._sync_config_if_changed() for the caller that
        fixes this across the whole wave-detection pipeline, not just
        this one signal.
        """
        self._max_speed = max_speed_fraction_of_height_per_sec

    def update(self, center_x: float, center_y: float, bbox_height: float, now: float) -> None:
        self._history.append((center_x, center_y, bbox_height, now))
        while self._history and (now - self._history[0][3]) > self._window_seconds:
            self._history.popleft()
        self.debug.samples = len(self._history)

    def is_relatively_stationary(self) -> bool:
        """
        True if the person's average speed over the short window
        (displacement between the oldest and newest sample still in
        the window, divided by the time between them, normalized by
        body height) is below the configured cap. With fewer than 2
        samples (e.g. right after a person first appears), returns
        True -- there isn't enough history to call someone "moving,"
        and it would be wrong to block a legitimate wave attempt during
        the first instant a person is tracked just because of missing
        data.
        """
        if len(self._history) < 2:
            self.debug.speed_fraction_of_height_per_sec = None
            self.debug.is_stationary = True
            return True

        oldest = self._history[0]
        newest = self._history[-1]
        dt = newest[3] - oldest[3]
        if dt <= 1e-3:
            self.debug.speed_fraction_of_height_per_sec = None
            self.debug.is_stationary = True
            return True

        dist = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1])
        avg_height = (oldest[2] + newest[2]) / 2.0
        if avg_height <= 1e-6:
            self.debug.speed_fraction_of_height_per_sec = None
            self.debug.is_stationary = True
            return True

        speed = (dist / avg_height) / dt
        self.debug.speed_fraction_of_height_per_sec = speed
        stationary = speed <= self._max_speed
        self.debug.is_stationary = stationary
        return stationary

