"""
Per-person state needed to detect genuine wave gestures over time.

Tracks EACH hand independently (not just "the first hand MediaPipe
reports"). A person has up to `max_num_hands` HandTrack slots. Each
frame, newly detected hands are matched to the existing slot whose
last known palm position is closest (simple nearest-neighbor
assignment) -- this keeps a hand's oscillation history intact across
frames even when MediaPipe reports hands in a different order than
the previous frame, and prevents one hand's motion from corrupting
another hand's reversal count.

REDESIGNED wave-detection core: each HandTrack now owns an
OscillationTracker (wave_detection/oscillation.py) -- a pure,
camera-independent, unit-tested 2D vector-based oscillation detector,
replacing an earlier 1D wrist-x/crop-relative design. See
oscillation.py's module docstring for the full root-cause analysis of
why the old design missed genuine waves, and
wave_detection/test_oscillation.py for the test suite that validates
this one against realistic wave patterns AND realistic confounds
(walking, jogging, random motion, static holds).

Each WaveState also owns a BodyStillnessTracker (wave_detection/
body_motion.py) -- a complementary, orthogonal signal (the person's
overall body translation) needed because testing found hand motion
ALONE cannot fully distinguish a deliberate wave from vigorous
walking/jogging arm swing at overlapping amplitude/cadence. See
body_motion.py's module docstring.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from utils.timing import Cooldown
from wave_detection.body_motion import BodyStillnessTracker
from wave_detection.oscillation import OscillationTracker


@dataclass
class HandTrack:
    """Oscillation-detection state for ONE hand."""
    oscillation: Optional[OscillationTracker] = None
    last_palm_xy: Optional[Tuple[float, float]] = None  # for nearest-neighbor slot matching, global frame px
    last_bbox_px: Optional[Tuple[int, int, int, int]] = None  # in global frame coords
    last_seen_time: float = 0.0
    # Wall-clock time until which this hand should still show as
    # "waving" in the UI after a confirmation, or None if it hasn't
    # waved recently. A plain boolean here was a real bug in an earlier
    # version of this redesign: it was set True on confirmation but
    # nothing ever set it back to False, so a hand stayed permanently
    # highlighted after its first wave for the rest of the session.
    # Computing "is currently waving" from a timestamp instead means
    # the visual state naturally expires with the wall clock, with no
    # separate cleanup step required.
    confirmed_display_until: Optional[float] = None


@dataclass
class WaveState:
    """
    All wave-related state for ONE tracked person: their hand slots, a
    shared cooldown (one wave from either hand triggers the cooldown
    for the whole person -- a person only needs one Suggested Action,
    not one per hand), and a body-stillness tracker shared across
    hands (the person only has one body, regardless of how many hand
    slots are active).
    """
    track_id: int
    hands: Dict[int, HandTrack] = field(default_factory=dict)
    cooldown: Optional[Cooldown] = None
    body_stillness: Optional[BodyStillnessTracker] = None
