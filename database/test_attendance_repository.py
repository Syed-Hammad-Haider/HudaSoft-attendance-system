"""
Standalone test suite for database/attendance_repository.py's work-day
boundary logic and the once-per-day CHECK IN/OUT cycle it backs.

WHY THIS EXISTS:
Session 15/16 found and fixed a real gap: the attendance "day" used to
reset at plain calendar midnight, so an employee who checked out at
8pm could immediately check back in at 12:01am. Fixed by introducing a
configurable work-day boundary hour (default 6am, see
database/settings_repository.py) -- work_day_start() below is what
actually implements that. These tests exist so a future change can't
silently regress that fix back to midnight-based behavior.

Run with:
    python database/test_attendance_repository.py
"""

import datetime
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from database.attendance_repository import get_todays_last_action, record_event, work_day_start
from database.connection import get_connection, initialize_database, set_db_path
from database.employee_repository import register_employee
from database.settings_repository import get_day_boundary_hour, set_day_boundary_hour

results = []


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}{': ' + detail if detail else ''}")
    results.append(passed)


def _insert_event_at(employee_id: int, action: str, when: datetime.datetime) -> None:
    """Insert an event with an EXPLICIT timestamp (bypassing record_event's
    'now') so these tests can deterministically place events at specific
    times of day without needing to mock the clock."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO attendance_events (employee_id, action, event_timestamp) VALUES (?, ?, ?)",
            (employee_id, action, when.isoformat()),
        )


def setup_fresh_db() -> Path:
    tmp_path = Path(tempfile.mktemp(suffix=".db"))
    set_db_path(tmp_path)
    initialize_database()
    return tmp_path


# --- work_day_start() boundary math ---------------------------------

db_path = setup_fresh_db()

boundary_hour = get_day_boundary_hour()
check("default day boundary hour is 6am", boundary_hour == 6, f"got {boundary_hour}")

reference_1am = datetime.datetime(2026, 3, 10, 1, 0, 0)  # before the 6am boundary
start = work_day_start(reference=reference_1am)
check(
    "1am reference belongs to the PREVIOUS calendar day's work day",
    start == datetime.datetime(2026, 3, 9, 6, 0, 0),
    f"got {start}",
)

reference_7am = datetime.datetime(2026, 3, 10, 7, 0, 0)  # after the 6am boundary
start2 = work_day_start(reference=reference_7am)
check(
    "7am reference belongs to TODAY's work day",
    start2 == datetime.datetime(2026, 3, 10, 6, 0, 0),
    f"got {start2}",
)

reference_exact = datetime.datetime(2026, 3, 10, 6, 0, 0)  # exactly at the boundary
start3 = work_day_start(reference=reference_exact)
check(
    "exactly at the boundary hour belongs to TODAY's work day (inclusive)",
    start3 == datetime.datetime(2026, 3, 10, 6, 0, 0),
    f"got {start3}",
)

# --- the actual bug this was written to fix --------------------------

emp_id = register_employee("Test Employee")
checkin_time = datetime.datetime(2026, 3, 10, 9, 0, 0)
checkout_time = datetime.datetime(2026, 3, 10, 20, 0, 0)  # 8pm, per the reported scenario
_insert_event_at(emp_id, "CHECK IN", checkin_time)
_insert_event_at(emp_id, "CHECK OUT", checkout_time)

# The real bug: querying "today's last action" a few minutes after
# midnight used to see a BRAND NEW calendar day and report no action
# yet -- silently allowing an immediate re-check-in. With the fix,
# get_todays_last_action() must be evaluated AS OF a specific moment to
# test this properly; it calls work_day_start() with no explicit
# reference (i.e. "right now"), so this test drives it indirectly via
# work_day_start() itself, which is what matters here.
just_after_midnight = datetime.datetime(2026, 3, 11, 0, 5, 0)
day_start_after_midnight = work_day_start(reference=just_after_midnight)
check(
    "8pm CHECK OUT is still within the work day at 12:05am (bug fixed: NOT a new day yet)",
    day_start_after_midnight <= checkout_time,
    f"work day start={day_start_after_midnight}, checkout={checkout_time}",
)

next_morning = datetime.datetime(2026, 3, 11, 7, 0, 0)  # after the boundary, next calendar day
day_start_next_morning = work_day_start(reference=next_morning)
check(
    "by 7am the next day, a genuinely NEW work day has started",
    day_start_next_morning > checkout_time,
    f"work day start={day_start_next_morning}, checkout={checkout_time}",
)

# --- once-per-day CHECK IN / CHECK OUT cycle, using the real DB path -

emp_id2 = register_employee("Cycle Test Employee", employee_code="CYCLE1")
check("brand new employee has no action yet today", get_todays_last_action(emp_id2) is None)

record_event(employee_id=emp_id2, action="CHECK IN")
check("after CHECK IN, last action is CHECK IN", get_todays_last_action(emp_id2) == "CHECK IN")

record_event(employee_id=emp_id2, action="CHECK OUT")
check("after CHECK OUT, last action is CHECK OUT", get_todays_last_action(emp_id2) == "CHECK OUT")

# The decision-engine layer (face_id/identified_attendance.py) is what
# actually refuses a second CHECK IN the same day using this value
# (_VALID_NEXT_ACTION maps "CHECK OUT" -> None, so no further action is
# ever allowed) -- this test just confirms the repository-level fact
# that decision depends on.
check(
    "repository correctly still reports CHECK OUT as the last action (decision engine blocks further action on this)",
    get_todays_last_action(emp_id2) == "CHECK OUT",
)

# --- settings_repository round-trip -----------------------------------

set_day_boundary_hour(7)
check("day boundary hour can be changed via settings_repository", get_day_boundary_hour() == 7)
start_after_change = work_day_start(reference=datetime.datetime(2026, 3, 10, 6, 30, 0))
check(
    "work_day_start() picks up the NEW boundary hour immediately (6:30am now before a 7am boundary)",
    start_after_change == datetime.datetime(2026, 3, 9, 7, 0, 0),
    f"got {start_after_change}",
)
try:
    set_day_boundary_hour(24)
    check("setting an out-of-range hour (24) is rejected", False, "no exception raised")
except ValueError:
    check("setting an out-of-range hour (24) is rejected", True)

os.remove(db_path)

total = len(results)
passed = sum(results)
print("=" * 70)
print(f"RESULTS: {passed}/{total} tests passed")
print("=" * 70)
sys.exit(0 if passed == total else 1)
