"""
Repository functions for durable attendance events.

WHY THIS FINALLY SOLVES THE TRACK-ID-CHURN LIMITATION:
Every earlier phase's attendance state machine lived in memory, keyed
by the ephemeral tracking ID (see wave_detection/attendance_status.py).
That was always documented as a known limitation: if a person's
tracking ID changed (occlusion, brief exit from frame), their
"already checked in" memory could be lost.

Once a person has a stable, recognized employee_id, that problem goes
away structurally -- `get_todays_last_action` below queries the
DATABASE for this employee's most recent event today, not an in-memory
dict keyed by a track ID that might not exist anymore. It doesn't
matter if the same person gets tracked under three different track IDs
across a session; their employee_id and their database history stay
the same throughout.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from database.connection import get_connection
from database.settings_repository import get_day_boundary_hour


def work_day_start(reference: Optional[datetime.datetime] = None, db_path: Optional[Path] = None) -> datetime.datetime:
    """
    Real bug fix (found via direct user testing): the attendance "day"
    used to reset at plain calendar MIDNIGHT, which is wrong for a real
    deployment -- an employee who checks out at 8pm would, at 12:01am,
    already be in a "new day" with no CHECK OUT recorded yet today, and
    could immediately check back in again. That's not a flat "require a
    12-hour gap" rule either (which would incorrectly block a legitimate
    very-early or very-late shift) -- it's that the attendance day
    itself should start at a sensible WORK-DAY boundary (an hour
    reliably before any real shift starts, e.g. 6am), not at midnight.

    Returns the datetime the "current work day" started at: if
    `reference` (defaults to now) falls before today's boundary hour,
    the work day actually started YESTERDAY at that hour (e.g. it's
    2am -- still part of the work day that started at 6am yesterday).
    Otherwise it started at today's boundary hour. Configurable via the
    dashboard's Settings page (see database/settings_repository.py),
    default 6am.
    """
    now = reference or datetime.datetime.now()
    boundary_hour = get_day_boundary_hour(db_path=db_path)
    boundary_today = now.replace(hour=boundary_hour, minute=0, second=0, microsecond=0)
    if now < boundary_today:
        return boundary_today - datetime.timedelta(days=1)
    return boundary_today


@dataclass
class AttendanceEvent:
    event_id: int
    employee_id: int
    action: str
    event_timestamp: str
    match_confidence: Optional[float]


def record_event(
    employee_id: int,
    action: str,
    match_confidence: Optional[float] = None,
    source_track_id: Optional[int] = None,
    camera_id: str = "default",
    db_path: Optional[Path] = None,
) -> int:
    """Persist a confirmed CHECK IN / CHECK OUT event. Returns the new event_id."""
    now = datetime.datetime.now().isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO attendance_events "
            "(employee_id, action, event_timestamp, match_confidence, source_track_id, camera_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (employee_id, action, now, match_confidence, source_track_id, camera_id),
        )
        return cursor.lastrowid


def get_todays_last_action(employee_id: int, db_path: Optional[Path] = None) -> Optional[str]:
    """
    Return this employee's most recent action for the CURRENT WORK DAY
    ('CHECK IN', 'CHECK OUT', or None if they have no event yet this
    work day). "Work day" resets at work_day_start() above (a
    configurable boundary hour, default 6am) -- NOT plain calendar
    midnight -- so a late CHECK OUT (e.g. 8pm) correctly blocks a
    CHECK IN in the small hours right after midnight; the cycle only
    resets once the actual work day boundary (e.g. 6am) passes.
    """
    day_start = work_day_start(db_path=db_path).isoformat()
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT action FROM attendance_events "
            "WHERE employee_id = ? AND event_timestamp >= ? "
            "ORDER BY event_timestamp DESC LIMIT 1",
            (employee_id, day_start),
        ).fetchone()
    return row["action"] if row else None


def get_todays_last_checkin_time(employee_id: int, db_path: Optional[Path] = None):
    """
    Return the datetime of this employee's most recent CHECK IN in the
    current work day, or None if they haven't checked in yet. Used to
    enforce a minimum gap before a CHECK OUT is allowed, so a stray
    second wave right after arriving can't immediately check someone
    back out.
    """
    day_start = work_day_start(db_path=db_path).isoformat()
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT event_timestamp FROM attendance_events "
            "WHERE employee_id = ? AND action = 'CHECK IN' AND event_timestamp >= ? "
            "ORDER BY event_timestamp DESC LIMIT 1",
            (employee_id, day_start),
        ).fetchone()
    if not row:
        return None
    try:
        return datetime.datetime.fromisoformat(row["event_timestamp"])
    except (ValueError, TypeError):
        return None


def get_events_for_employee(employee_id: int, db_path: Optional[Path] = None, limit: int = 50):
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT event_id, employee_id, action, event_timestamp, match_confidence "
            "FROM attendance_events WHERE employee_id = ? "
            "ORDER BY event_timestamp DESC LIMIT ?",
            (employee_id, limit),
        ).fetchall()
    return [
        AttendanceEvent(
            event_id=r["event_id"], employee_id=r["employee_id"], action=r["action"],
            event_timestamp=r["event_timestamp"], match_confidence=r["match_confidence"],
        )
        for r in rows
    ]


def get_todays_events(db_path: Optional[Path] = None):
    """All attendance events across all employees for the CURRENT WORK
    DAY (see work_day_start() above, not calendar midnight) -- what a
    daily report or admin dashboard view would query."""
    day_start = work_day_start(db_path=db_path).isoformat()
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ev.event_id, ev.employee_id, e.full_name, ev.action, ev.event_timestamp
            FROM attendance_events ev
            JOIN employees e ON e.employee_id = ev.employee_id
            WHERE ev.event_timestamp >= ?
            ORDER BY ev.event_timestamp ASC
            """,
            (day_start,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_events_in_range(
    start_date: datetime.date,
    end_date: datetime.date,
    employee_id: Optional[int] = None,
    db_path: Optional[Path] = None,
):
    """
    All attendance events (joined with employee name) between
    start_date 00:00 and end_date 23:59:59 inclusive, optionally
    filtered to one employee. Backs both the CSV export and the
    in-browser Records table in the dashboard -- a real deployment
    needs to get records OUT of SQLite for payroll/HR (CSV), and also
    needs to just LOOK at them without downloading anything first
    (the Records tab).
    """
    start = datetime.datetime.combine(start_date, datetime.time.min).isoformat()
    end = datetime.datetime.combine(end_date, datetime.time.max).isoformat()
    query = """
        SELECT ev.event_id, ev.employee_id, e.full_name, e.employee_code, e.department,
               ev.action, ev.event_timestamp, ev.match_confidence
        FROM attendance_events ev
        JOIN employees e ON e.employee_id = ev.employee_id
        WHERE ev.event_timestamp >= ? AND ev.event_timestamp <= ?
    """
    params = [start, end]
    if employee_id is not None:
        query += " AND ev.employee_id = ?"
        params.append(employee_id)
    query += " ORDER BY ev.event_timestamp ASC"
    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# ADMIN CORRECTIONS -- delete today's events, delete/edit a single event
# ---------------------------------------------------------------------------
# Real attendance systems need a way to fix mistakes: a mis-detection, a
# wrong IN/OUT, or wiping a test day. These back the "Clear Today" button
# (Live Monitor) and the per-row edit/delete controls (Records tab).

def delete_todays_events(db_path: Optional[Path] = None) -> int:
    """Delete every attendance event in the CURRENT WORK DAY (all
    employees). Returns how many rows were removed. Used by the Live
    Monitor 'Clear Today's Attendance' action for a clean re-test or to
    wipe a day of bad data."""
    day_start = work_day_start(db_path=db_path).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM attendance_events WHERE event_timestamp >= ?", (day_start,)
        )
        return cur.rowcount


def delete_event(event_id: int, db_path: Optional[Path] = None) -> None:
    """Delete one attendance event by id -- used to remove a single
    mistaken record from the Records tab."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM attendance_events WHERE event_id = ?", (event_id,))


def update_event_action(event_id: int, action: str, db_path: Optional[Path] = None) -> None:
    """Change one event's action between 'CHECK IN' and 'CHECK OUT' --
    used to correct a record detected as the wrong direction."""
    if action not in ("CHECK IN", "CHECK OUT"):
        raise ValueError("action must be 'CHECK IN' or 'CHECK OUT'")
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE attendance_events SET action = ? WHERE event_id = ?",
            (action, event_id),
        )


def get_checkin_dates_in_month(employee_id: int, year: int, month: int, db_path: Optional[Path] = None):
    """
    Return the SET of calendar dates ('YYYY-MM-DD') in the given month on
    which this employee has at least one CHECK IN. This is the raw "days
    present" signal the Monthly Report tab turns into present/absent
    counts and an attendance percentage. A person counts as present on
    any day they checked in.
    """
    start = datetime.date(year, month, 1).isoformat()
    nxt = datetime.date(year + 1, 1, 1) if month == 12 else datetime.date(year, month + 1, 1)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT substr(event_timestamp, 1, 10) AS d "
            "FROM attendance_events "
            "WHERE employee_id = ? AND action = 'CHECK IN' "
            "AND substr(event_timestamp, 1, 10) >= ? AND substr(event_timestamp, 1, 10) < ?",
            (employee_id, start, nxt.isoformat()),
        ).fetchall()
    return {r["d"] for r in rows}
