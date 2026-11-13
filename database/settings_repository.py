"""
Repository for admin-configurable settings stored in the database
(system_settings table) rather than hardcoded in a config file --
so an admin can change them from the dashboard's Settings page
without editing code or restarting a deployment.

Currently used for: attendance_day_boundary_hour (see
attendance_repository.work_day_start()). Deliberately generic
key/value get/set so a new setting never needs a schema migration --
just a new key and a typed accessor if it needs one (see
get_day_boundary_hour() below for the pattern to follow).
"""

from pathlib import Path
from typing import Optional

from database.connection import get_connection

DEFAULT_DAY_BOUNDARY_HOUR = 6  # see attendance_repository.work_day_start()


def get_setting(key: str, default: Optional[str] = None, db_path: Optional[Path] = None) -> Optional[str]:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT value FROM system_settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str, db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO system_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )


def get_day_boundary_hour(db_path: Optional[Path] = None) -> int:
    """
    The hour (0-23) a new attendance "day" starts at, for the purpose
    of "has this employee already checked in/out TODAY". Deliberately
    NOT midnight -- see attendance_repository.work_day_start() for the
    full reasoning (a real employee checking out at 8pm shouldn't be
    allowed to check back in at 12:05am just because the calendar date
    changed).
    """
    raw = get_setting("attendance_day_boundary_hour", db_path=db_path)
    if raw is None:
        return DEFAULT_DAY_BOUNDARY_HOUR
    try:
        hour = int(raw)
    except ValueError:
        return DEFAULT_DAY_BOUNDARY_HOUR
    if not (0 <= hour <= 23):
        return DEFAULT_DAY_BOUNDARY_HOUR
    return hour


def set_day_boundary_hour(hour: int, db_path: Optional[Path] = None) -> None:
    if not (0 <= int(hour) <= 23):
        raise ValueError("day boundary hour must be between 0 and 23")
    set_setting("attendance_day_boundary_hour", str(int(hour)), db_path=db_path)


def get_camera_source(db_path: Optional[Path] = None):
    """
    The configured camera source, admin-settable from the dashboard so
    switching between the laptop webcam and a phone (DroidCam / IP
    Webcam over Wi-Fi) never needs a code edit. Returns an int webcam
    index (0, 1, ...) when the stored value is empty or numeric, else
    the URL string as-is (e.g. http://192.168.1.50:4747/video). None
    stored -> default to webcam index 0, matching CameraConfig.source.
    """
    raw = get_setting("camera_source", db_path=db_path)
    if raw is None or raw.strip() == "":
        # On a server there is no webcam 0: CAMERA_SOURCE (env var, e.g. an
        # RTSP URL set in docker-compose / .env) is used until an admin
        # saves a source in Settings, which then takes priority.
        import os
        raw = os.environ.get("CAMERA_SOURCE", "")
        if raw.strip() == "":
            return 0
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return raw


def get_camera_source_raw(db_path: Optional[Path] = None) -> str:
    """The stored camera-source string exactly as saved (for showing in
    the Settings form); empty string when unset / using the webcam."""
    raw = get_setting("camera_source", db_path=db_path)
    return "" if raw is None else raw.strip()


def set_camera_source(source: str, db_path: Optional[Path] = None) -> None:
    """Store the camera source. A blank string means 'webcam index 0'."""
    set_setting("camera_source", (source or "").strip(), db_path=db_path)


DEFAULT_MIN_ACTION_GAP_HOURS = 1.0


def get_min_action_gap_hours(db_path: Optional[Path] = None) -> float:
    """
    Minimum hours that must pass after a CHECK IN before a CHECK OUT is
    allowed for the same employee. Prevents a second wave moments after
    arriving from immediately checking someone back out, and models a
    minimum shift (default 1h). Admin-settable; 0 disables the
    gap (check out allowed any time after checking in).
    """
    raw = get_setting("min_action_gap_hours", db_path=db_path)
    if raw is None:
        return DEFAULT_MIN_ACTION_GAP_HOURS
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_MIN_ACTION_GAP_HOURS
    return val if val >= 0 else DEFAULT_MIN_ACTION_GAP_HOURS


def set_min_action_gap_hours(hours: float, db_path: Optional[Path] = None) -> None:
    if float(hours) < 0:
        raise ValueError("minimum gap hours cannot be negative")
    set_setting("min_action_gap_hours", str(float(hours)), db_path=db_path)


def get_mirror_preview(db_path: Optional[Path] = None) -> bool:
    """Whether the camera image is mirrored (flipped left-right, like a
    selfie view). Default ON: a mirrored view is what people intuitively
    expect when waving at a screen, so it's the natural default and most
    deployments never need to change it. Admin-settable for the rare
    mounting where the true (un-mirrored) view is wanted."""
    raw = get_setting("mirror_preview", db_path=db_path)
    return str(raw).lower() in ("1", "true", "yes", "on") if raw is not None else True


def set_mirror_preview(enabled: bool, db_path: Optional[Path] = None) -> None:
    set_setting("mirror_preview", "true" if enabled else "false", db_path=db_path)


DEFAULT_WORK_DAYS_PER_WEEK = 5


def get_work_days_per_week(employee_id: int, db_path: Optional[Path] = None) -> int:
    """
    How many days a week THIS employee is expected to work (default 5,
    Mon-Fri). The Monthly Report uses it to compute how many working
    days a month should have had for this person, which in turn gives
    present/absent counts and an attendance percentage. Stored
    per-employee under a 'work_days_<id>' key so different staff can
    have different schedules (e.g. a 6-day guard vs a 5-day office
    worker) without a schema change. Clamped to 1-7.
    """
    raw = get_setting(f"work_days_per_week_{int(employee_id)}", db_path=db_path)
    if raw is None:
        return DEFAULT_WORK_DAYS_PER_WEEK
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_WORK_DAYS_PER_WEEK
    return min(7, max(1, n))


def set_work_days_per_week(employee_id: int, days: int, db_path: Optional[Path] = None) -> None:
    if not (1 <= int(days) <= 7):
        raise ValueError("work days per week must be between 1 and 7")
    set_setting(f"work_days_per_week_{int(employee_id)}", str(int(days)), db_path=db_path)


def get_or_create_secret_key(db_path: Optional[Path] = None) -> str:
    """
    A stable Flask session secret, persisted in the database so that
    admin sign-ins survive a dashboard restart (a secret regenerated on
    every boot would silently invalidate everyone's session on each
    restart). Generated once with a cryptographically-secure random
    value the first time it's needed, then reused.
    """
    import secrets
    existing = get_setting("flask_secret_key", db_path=db_path)
    if existing:
        return existing
    key = secrets.token_hex(32)
    set_setting("flask_secret_key", key, db_path=db_path)
    return key
