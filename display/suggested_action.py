"""
SuggestedAction value object, plus the pure label/icon mapping from a
crossing direction. All GATING logic (recent-crossing check,
attendance state-machine check, duplicate-ID-churn suppression) lives
in wave_detection/attendance_decision.py -- this module only answers
"what does ENTERING/LEAVING look like on screen", nothing more.
"""

from dataclasses import dataclass
from typing import Optional

from direction_estimation.entrance_line import Direction


@dataclass
class SuggestedAction:
    label: str
    icon: str


def label_and_icon_for_direction(direction: Direction) -> Optional["SuggestedAction"]:
    """Pure mapping, no gating: ENTERING -> CHECK IN, LEAVING -> CHECK OUT."""
    if direction == Direction.LEAVING:
        return SuggestedAction(label="CHECK OUT", icon="\U0001F44B")  # 👋
    if direction == Direction.ENTERING:
        return SuggestedAction(label="CHECK IN", icon="\u2705")  # ✅
    return None
