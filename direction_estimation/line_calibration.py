"""
Persists the entrance line's calibration (position + inside/outside side)
to a small JSON file, separate from config/settings.py.

WHY THIS IS A SEPARATE FILE FROM THE MAIN CONFIG:
config/settings.py holds ENGINEERING defaults -- things a developer
decides. The entrance line's exact position is an INSTALLATION detail --
something whoever sets up the camera on-site decides, by looking at the
real doorway, not something that belongs in source code. Keeping it in
its own small JSON file means:
  - Calibrating the camera on-site never requires touching Python code.
  - The calibration survives across app restarts.
  - Multiple physical camera installs could each have their own
    calibration file later, without turning config.py into a mess.
"""

import json
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Inside Docker, ATTENDANCE_DATA_DIR points at the persistent volume so the
# saved entrance line survives container rebuilds; otherwise unchanged.
_DATA_DIR = os.environ.get("ATTENDANCE_DATA_DIR", "").strip()
CALIBRATION_PATH = (
    os.path.join(_DATA_DIR, "entrance_line_calibration.json") if _DATA_DIR
    else os.path.join(os.path.dirname(__file__), "..", "config", "entrance_line_calibration.json")
)


def load_calibration() -> Optional[dict]:
    """Return saved calibration dict, or None if no calibration file exists yet."""
    path = os.path.abspath(CALIBRATION_PATH)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info("Loaded entrance line calibration from %s", path)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read calibration file (%s); using config.py defaults instead.", e)
        return None


def save_calibration(
    line_start_ratio: Tuple[float, float],
    line_end_ratio: Tuple[float, float],
    inside_is_left_of_line: bool,
) -> None:
    """Save the current line calibration so it survives app restarts."""
    path = os.path.abspath(CALIBRATION_PATH)
    data = {
        "line_start": list(line_start_ratio),
        "line_end": list(line_end_ratio),
        "inside_is_left_of_line": inside_is_left_of_line,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved entrance line calibration to %s", path)
    except OSError as e:
        logger.error("Could not save calibration file: %s", e)
