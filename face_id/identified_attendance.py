"""
IdentifiedAttendanceDecisionEngine: the Phase 2 replacement for the
in-memory, track-ID-keyed attendance state machine.

WHY THIS IS A GENUINE UPGRADE, NOT JUST AN ADDITION:
Every earlier phase's attendance state lived in memory, keyed by the
ephemeral tracking ID -- explicitly documented as a limitation,
because a tracking ID could churn (occlusion, brief exit from frame)
and silently reset a person's "already checked in" memory.

Once a person is recognized by face, they have a STABLE employee_id
that survives any number of tracking ID changes. This engine queries
the DATABASE for "what did this employee last do today"
(attendance_repository.get_todays_last_action) instead of an in-memory
dict -- so the same physical person keeps their correct attendance
state regardless of how many different track IDs they're assigned
across a session, and even across app restarts within the same day.

WHAT HAPPENS FOR UNRECOGNIZED PEOPLE:
If a person hasn't been recognized (no employee_id -- either they
haven't been registered yet, or the live match score didn't clear the
threshold), NO attendance event is ever recorded for them. This is a
deliberate, conservative choice: guessing an identity is worse than
declining to record anything. They still appear fully in the live UI
(box, wave detection, crossing detection all continue to work) so
whoever is watching the screen can see "an unrecognized person just
tried to check in" and register them, or notice a matching problem --
the system just won't silently attribute their attendance to the
wrong person or to nobody by accident.
"""

import datetime
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from config.settings import ActionGuardConfig, EntranceLineConfig
from database.attendance_repository import (
    get_todays_last_action,
    get_todays_last_checkin_time,
    record_event,
)
from database.settings_repository import get_min_action_gap_hours
from direction_estimation.entrance_line import PersonGateState, Side
from display.suggested_action import SuggestedAction, label_and_icon_for_direction
from wave_detection.action_guard import DuplicateActionGuard

_VALID_NEXT_ACTION = {
    None: "CHECK IN",
    "CHECK IN": "CHECK OUT",
    "CHECK OUT": None,  # already checked out today; no further action until tomorrow
}

# Icons used when a wave fires an action WITHOUT a recent line crossing
# to borrow a direction-based icon from (see evaluate()). Kept identical
# to label_and_icon_for_direction's mapping so the on-screen banner
# looks the same whether the action came from a crossing or a wave alone.
_ICON_FOR_ACTION = {
    "CHECK IN": "✅",   # ✅
    "CHECK OUT": "\U0001F44B",  # 👋
}


def status_label_for_employee(employee_id: Optional[int]) -> str:
    """
    Human-readable current attendance status for the live UI. Separate
    free function (not a method) since it's a pure read -- no need to
    go through the decision engine instance for a status DISPLAY, only
    for actually deciding/recording a transition.
    """
    if employee_id is None:
        return "UNRECOGNIZED"
    last_action = get_todays_last_action(employee_id)
    if last_action == "CHECK IN":
        return "CHECKED IN"
    if last_action == "CHECK OUT":
        return "CHECKED OUT"
    return "NOT CHECKED IN"


@dataclass
class IdentifiedDecisionResult:
    action: Optional[SuggestedAction]
    employee_name: Optional[str]
    reason_if_none: Optional[str]
    # Distinguishes an EXPECTED, routine non-event (already checked in
    # today and waved again -- deliberately silenced on-screen, since
    # it's not a problem) from a GENUINE reason nothing happened that
    # the person actually needs to see: not recognized, or -- the most
    # common real cause of "the skeleton went green but I never got
    # checked in" -- the wave wasn't near a recent entrance-line
    # crossing. main.py / web_dashboard.py show only the non-routine
    # reasons, so a stuck first-time user gets real feedback while a
    # normal re-wave stays quiet.
    reason_is_routine: bool = False


class IdentifiedAttendanceDecisionEngine:
    def __init__(self, entrance_config: EntranceLineConfig, action_guard_config: ActionGuardConfig, frame_width: int):
        self._entrance_config = entrance_config
        # Still used as an extra safety net against near-simultaneous
        # duplicate events (e.g. a brief double-recognition glitch),
        # layered on TOP of the database state machine, not instead of
        # it -- the database check is what actually prevents CHECK
        # IN -> CHECK IN across track ID churn; this catches a much
        # narrower, faster-timescale duplicate.
        self._duplicate_guard = DuplicateActionGuard(action_guard_config, frame_width)

    def evaluate(
        self,
        employee_id: Optional[int],
        employee_name: Optional[str],
        gate_state: PersonGateState,
        wave_confirmed: bool,
        position: Tuple[float, float],
        match_confidence: float,
        source_track_id: int,
    ) -> IdentifiedDecisionResult:
        if not wave_confirmed:
            return IdentifiedDecisionResult(action=None, employee_name=employee_name, reason_if_none=None)

        if employee_id is None:
            return IdentifiedDecisionResult(
                action=None, employee_name=None,
                reason_if_none="wave confirmed but person is not recognized -- no attendance recorded",
            )

        # OUTSIDE-ONLY: the camera is mounted outside the entrance, so a
        # wave is only acted on from the OUTSIDE side of the line (see
        # EntranceLineConfig.require_wave_from_outside). Only a person
        # DEFINITIVELY on the inside side is blocked -- UNKNOWN (just
        # appeared, not yet classified) and OUTSIDE both pass, so a
        # normal outside wave is never held up. If IN/OUT is reversed
        # for a given mounting, fix it with Calibration -> Flip IN/OUT.
        if getattr(self._entrance_config, "require_wave_from_outside", False) \
                and gate_state.confirmed_side == Side.INSIDE:
            return IdentifiedDecisionResult(
                action=None, employee_name=employee_name,
                reason_if_none=f"{employee_name} waved from inside -- check in/out is only from outside the entrance",
            )

        # WAVE-DRIVEN ATTENDANCE (redesigned for reliability):
        #
        # The attendance STATE MACHINE is now the single source of truth
        # for what a recognized person's wave does:
        #     not checked in today  -> CHECK IN
        #     checked in            -> CHECK OUT
        #     already checked out   -> nothing more today
        # A recognized person simply waving is enough to fire the right
        # one. This replaces the previous design that ALSO required the
        # person to have crossed the entrance line within the last few
        # seconds -- which, in real use, was the main reason a clearly-
        # detected wave produced no check-in (someone stands at the door
        # and waves without a fresh crossing on record, so the wave had
        # nothing to pair with and was silently dropped). The line
        # crossing is now an OPTIONAL refinement, not a hard gate: if a
        # recent crossing direction is available AND it agrees with what
        # the state machine would do anyway, its icon/label is used;
        # otherwise the wave still works on its own. The crossing can
        # never FORCE an invalid transition -- the state machine decides
        # what is allowed; the crossing only labels an already-valid one.
        last_action_today = get_todays_last_action(employee_id)
        allowed_next = _VALID_NEXT_ACTION.get(last_action_today)

        if allowed_next is None:
            # Already checked in AND out today -- nothing further until
            # the next work day. Routine, so stays silent on-screen.
            return IdentifiedDecisionResult(
                action=None, employee_name=employee_name,
                reason_if_none=f"{employee_name} already completed check-in and check-out today",
                reason_is_routine=True,
            )

        # MINIMUM CHECK-IN -> CHECK-OUT GAP: a CHECK OUT is only allowed
        # once the admin-set minimum workday length has elapsed since
        # this employee's CHECK IN (default 1h). This stops an accidental second wave
        # moments after arriving from immediately checking someone back
        # out. Read live from settings each time so the admin can change
        # it from the dashboard without a restart; 0 disables the gap.
        if allowed_next == "CHECK OUT":
            gap_hours = get_min_action_gap_hours()
            if gap_hours > 0:
                checkin_time = get_todays_last_checkin_time(employee_id)
                if checkin_time is not None:
                    elapsed_h = (datetime.datetime.now() - checkin_time).total_seconds() / 3600.0
                    if elapsed_h < gap_hours:
                        remaining = gap_hours - elapsed_h
                        return IdentifiedDecisionResult(
                            action=None, employee_name=employee_name,
                            reason_if_none=(
                                f"{employee_name} checked in {elapsed_h:.1f}h ago -- "
                                f"check-out opens in {remaining:.1f}h (min {gap_hours:g}h shift)"
                            ),
                            reason_is_routine=True,
                        )

        candidate = None
        if gate_state.direction_is_recent(self._entrance_config.action_window_seconds):
            crossing_candidate = label_and_icon_for_direction(gate_state.last_direction)
            if crossing_candidate is not None and crossing_candidate.label == allowed_next:
                candidate = crossing_candidate
        if candidate is None:
            candidate = SuggestedAction(label=allowed_next, icon=_ICON_FOR_ACTION.get(allowed_next, ""))

        if self._duplicate_guard.should_suppress(position, candidate.label):
            return IdentifiedDecisionResult(
                action=None, employee_name=employee_name,
                reason_if_none="duplicate suppressed (same action nearby recently)",
                reason_is_routine=True,
            )

        # All checks passed: persist to the database and fire.
        record_event(
            employee_id=employee_id,
            action=candidate.label,
            match_confidence=match_confidence,
            source_track_id=source_track_id,
        )
        self._duplicate_guard.register(position, candidate.label)
        return IdentifiedDecisionResult(action=candidate, employee_name=employee_name, reason_if_none=None)
