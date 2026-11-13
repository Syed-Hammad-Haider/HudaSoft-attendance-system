"""
Per-person attendance state machine.

WHY THIS EXISTS:
Without it, every confirmed wave near a recent crossing produces a
Suggested Action, with no memory of what was already suggested for
that person. That allowed CHECK IN -> CHECK IN, CHECK OUT before any
CHECK IN, and rapid repeated events from continuous waving -- all
explicitly wrong for a real attendance product, where each person
should only ever transition NOT_CHECKED_IN -> CHECKED_IN -> CHECKED_OUT
once per tracked session, however many times they physically cross the
line or wave in between.

STATE MACHINE:

    NOT_CHECKED_IN --(valid entry + wave)--> CHECKED_IN
    CHECKED_IN     --(valid exit + wave)--> CHECKED_OUT
    CHECKED_OUT    --(nothing further allowed until reset)

Only ONE transition is valid from each state. Any action that would
repeat the current state's own type, or skip ahead (CHECK OUT while
still NOT_CHECKED_IN), is rejected.

RESET: state is cleared when the person's track is removed (they
completely left camera view, or ByteTrack's own occlusion buffer
expired -- see TrackingConfig.lost_track_buffer / max_missed_frames).
This is the "configurable timeout if tracking is permanently lost"
requirement -- max_missed_frames already IS that configurable timeout,
expressed in frames rather than seconds.

KNOWN LIMITATION (documented honestly, not hidden): this state lives
only in memory, keyed by tracking ID. If the same physical person's ID
churns (see person_tracking's occlusion-related ID-switch limitation)
their attendance state resets along with it. Fully solving this needs
persistent, identity-based state (Phase 2's face recognition +
database), not something Phase 1's position-only tracking can
guarantee. The DuplicateActionGuard (position+time based) provides a
partial safety net for the common case of an ID churning right at the
moment of an action, but is not a complete fix.
"""

from enum import Enum
from typing import Dict, Optional


class AttendanceState(str, Enum):
    NOT_CHECKED_IN = "NOT_CHECKED_IN"
    CHECKED_IN = "CHECKED_IN"
    CHECKED_OUT = "CHECKED_OUT"


# The only valid (state, proposed_action_label) -> next_state transitions.
# Anything not in this table is rejected.
_VALID_TRANSITIONS = {
    (AttendanceState.NOT_CHECKED_IN, "CHECK IN"): AttendanceState.CHECKED_IN,
    (AttendanceState.CHECKED_IN, "CHECK OUT"): AttendanceState.CHECKED_OUT,
}


class AttendanceStatusTracker:
    """Tracks each person's current attendance state, keyed by track ID."""

    def __init__(self):
        self._states: Dict[int, AttendanceState] = {}

    def current_state(self, track_id: int) -> AttendanceState:
        return self._states.get(track_id, AttendanceState.NOT_CHECKED_IN)

    def is_transition_allowed(self, track_id: int, proposed_label: str) -> bool:
        """
        Whether `proposed_label` (e.g. "CHECK IN") is a valid next step
        for this person, given their current state. This is the single
        rule that prevents every case the spec called out:
        CHECK IN -> CHECK IN, CHECK OUT -> CHECK OUT, and CHECK OUT
        before any CHECK IN.
        """
        state = self.current_state(track_id)
        return (state, proposed_label) in _VALID_TRANSITIONS

    def apply_transition(self, track_id: int, proposed_label: str) -> None:
        """
        Commit the transition. Only call this after confirming
        `is_transition_allowed` -- this method does not re-check.
        """
        state = self.current_state(track_id)
        next_state = _VALID_TRANSITIONS.get((state, proposed_label))
        if next_state is not None:
            self._states[track_id] = next_state

    def status_label(self, track_id: int) -> str:
        """Short human-readable label for the live UI status block."""
        state = self.current_state(track_id)
        return {
            AttendanceState.NOT_CHECKED_IN: "NOT CHECKED IN",
            AttendanceState.CHECKED_IN: "CHECKED IN",
            AttendanceState.CHECKED_OUT: "CHECKED OUT",
        }[state]

    def remove_track(self, track_id: int) -> None:
        """Reset state for a person who has left the scene / lost tracking."""
        self._states.pop(track_id, None)
