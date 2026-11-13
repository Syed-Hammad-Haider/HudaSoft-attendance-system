"""
AttendanceDecisionEngine: the single place where "should a Suggested
Action actually fire right now" gets decided.

WHY THIS EXISTS AS ITS OWN MODULE:
Earlier, this logic was scattered: a recent-crossing check lived in
suggested_action.py, duplicate-ID-churn suppression lived in
action_guard.py, and there was NO check at all for "has this person
already been checked in." Scattering related decision logic like that
is exactly how the multiple-CHECK-IN bug happened -- no single place
was responsible for the full picture. This module fixes that by being
the one function every relevant signal passes through before a
Suggested Action is allowed to fire.

THE FULL SEQUENCE OF CHECKS (in order, cheapest/most-likely-to-reject first):

  1. Was a wave actually confirmed this frame?
  2. Did that wave happen near a RECENT line crossing (not stale
     "last crossing ever" memory)?
  3. Does the crossing direction map to a real action (ENTERING/LEAVING)?
  4. Is this a VALID next step in that person's attendance state
     machine (blocks CHECK IN->CHECK IN, CHECK OUT before CHECK IN, etc)?
  5. Is this action suspiciously close in time+position to one that
     JUST fired for a different track ID (likely the same physical
     person after an ID churn)?

Only if all five pass does an action get returned and the person's
state get updated. Every rejection reports WHY, for logging/debugging
-- "the system should think before suggesting" means the reasoning
should be inspectable, not just a black-box yes/no.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from config.settings import ActionGuardConfig, EntranceLineConfig
from direction_estimation.entrance_line import PersonGateState
from display.suggested_action import SuggestedAction, label_and_icon_for_direction
from wave_detection.action_guard import DuplicateActionGuard
from wave_detection.attendance_status import AttendanceStatusTracker


@dataclass
class DecisionResult:
    action: Optional[SuggestedAction]
    reason_if_none: Optional[str]  # human-readable reason, only set when action is None


class AttendanceDecisionEngine:
    def __init__(self, entrance_config: EntranceLineConfig, action_guard_config: ActionGuardConfig, frame_width: int):
        self._entrance_config = entrance_config
        self._status = AttendanceStatusTracker()
        self._duplicate_guard = DuplicateActionGuard(action_guard_config, frame_width)

    def evaluate(
        self,
        track_id: int,
        gate_state: PersonGateState,
        wave_confirmed: bool,
        position: Tuple[float, float],
    ) -> DecisionResult:
        # 1. No wave, nothing to decide.
        if not wave_confirmed:
            return DecisionResult(action=None, reason_if_none=None)

        # 2. Wave must be tied to a RECENT crossing, not stale history.
        if not gate_state.direction_is_recent(self._entrance_config.action_window_seconds):
            return DecisionResult(action=None, reason_if_none="wave not near a recent line crossing")

        # 3. Map crossing direction to a candidate action.
        candidate = label_and_icon_for_direction(gate_state.last_direction)
        if candidate is None:
            return DecisionResult(action=None, reason_if_none="direction unknown")

        # 4. State machine: is this a valid next step for THIS person?
        if not self._status.is_transition_allowed(track_id, candidate.label):
            current = self._status.status_label(track_id)
            return DecisionResult(
                action=None,
                reason_if_none=f"already {current} -- ignoring repeat {candidate.label}",
            )

        # 5. Duplicate-across-ID-churn safety net (only for a REPEAT of
        # this same action type nearby -- a different action type, like
        # CHECK OUT after a recent CHECK IN, is legitimate sequencing
        # and is never suppressed here).
        if self._duplicate_guard.should_suppress(position, candidate.label):
            return DecisionResult(
                action=None,
                reason_if_none="duplicate suppressed (same action nearby recently, likely same person under a churned ID)",
            )

        # All checks passed -- commit the state transition and fire.
        self._status.apply_transition(track_id, candidate.label)
        self._duplicate_guard.register(position, candidate.label)
        return DecisionResult(action=candidate, reason_if_none=None)

    def status_label(self, track_id: int) -> str:
        """For live UI display of this person's current attendance state."""
        return self._status.status_label(track_id)

    def remove_track(self, track_id: int) -> None:
        self._status.remove_track(track_id)
