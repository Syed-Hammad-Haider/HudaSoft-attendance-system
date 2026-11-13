"""
Duplicate Suggested Action guard.

PROBLEM THIS SOLVES:
If ByteTrack ever drops a person's track and reassigns a NEW ID to the
same physical person (this can happen after a brief full occlusion,
e.g. someone walks behind another person for a moment), that person's
attendance state (AttendanceStatusTracker, keyed by track ID) resets
along with it -- the "new" ID looks like a fresh person and could
trigger a duplicate Suggested Action moments after the real one, for
what is actually the same physical person.

APPROACH:
Rather than trying to solve full re-identification (out of scope for
Phase 1 -- that's what face recognition in Phase 2 is for), we use a
lightweight, practical mitigation: remember WHERE, WHEN, and WHICH
ACTION each Suggested Action was triggered. If the SAME action label
is about to fire again very close to a very recent one (in both space
and time), suppress it.

IMPORTANT: this check is label-aware, not just position-aware. An
earlier version suppressed ANY nearby recent action regardless of
label, which caused a real bug: a legitimate CHECK OUT happening near
the same spot as an earlier CHECK IN (completely normal -- people
don't teleport between entering and leaving) was incorrectly blocked.
The guard's actual job is narrower: catch the SAME action type firing
twice near the same place in a short window (which is what a churned
ID replaying the same event would look like), not flag two DIFFERENT,
legitimately sequential actions.

This is deliberately simple -- a full solution would need appearance
re-identification, which is real Phase 2/3 territory (see
"Future Improvements" in the documentation). This module exists so the
gap is closed at a practical level now, not ignored.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

from config.settings import ActionGuardConfig


@dataclass
class _RecentAction:
    position: Tuple[float, float]
    label: str
    timestamp: float


class DuplicateActionGuard:
    def __init__(self, config: ActionGuardConfig, frame_width: int):
        self._config = config
        self._frame_width = frame_width
        self._recent_actions: List[_RecentAction] = []

    def should_suppress(self, position: Tuple[float, float], label: str) -> bool:
        """
        Check whether a new action of this LABEL at this position should
        be blocked because the SAME label just fired nearby recently.
        A different label (e.g. CHECK OUT following a recent CHECK IN)
        is never suppressed by this check -- that's normal, expected
        sequencing, not a duplicate.
        """
        self._prune_expired()
        radius_px = self._config.suppression_radius_ratio * self._frame_width

        for action in self._recent_actions:
            if action.label != label:
                continue
            dx = action.position[0] - position[0]
            dy = action.position[1] - position[1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= radius_px:
                return True
        return False

    def register(self, position: Tuple[float, float], label: str) -> None:
        """Record that an action of this label just fired at this position."""
        self._recent_actions.append(_RecentAction(position=position, label=label, timestamp=time.time()))

    def _prune_expired(self) -> None:
        now = time.time()
        self._recent_actions = [
            a for a in self._recent_actions
            if (now - a.timestamp) <= self._config.suppression_seconds
        ]
