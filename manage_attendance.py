"""
Admin CLI for viewing/exporting attendance records.

WHY THIS EXISTS: the whole point of an attendance system is the
records it produces. Without a way to get them out of SQLite, the
project only "works" while main.py's window happens to be open showing
the live event log -- not good enough for actual payroll/HR use. This
gives a quick on-screen daily view, and a CSV export for anything more
(Excel, payroll import, etc).

USAGE:
    python manage_attendance.py today
    python manage_attendance.py export --start 2026-08-01 --end 2026-08-13 --out august.csv
    python manage_attendance.py export --out today.csv      # defaults to today only
"""

import argparse
import csv
import datetime
import sys

from database.attendance_repository import get_events_in_range, get_todays_events
from database.connection import initialize_database


def cmd_today(args) -> None:
    events = get_todays_events()
    if not events:
        print("No attendance events yet today.")
        return
    print(f"{'Time':<20} {'Name':<25} {'Action'}")
    print("-" * 55)
    for e in events:
        ts = e["event_timestamp"][11:19]  # HH:MM:SS
        print(f"{ts:<20} {e['full_name']:<25} {e['action']}")


def cmd_export(args) -> None:
    start = datetime.date.fromisoformat(args.start) if args.start else datetime.date.today()
    end = datetime.date.fromisoformat(args.end) if args.end else datetime.date.today()
    events = get_events_in_range(start, end)

    if not events:
        print(f"No attendance events between {start} and {end}. Nothing to export.")
        return

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_id", "employee_id", "full_name", "employee_code", "department",
            "action", "event_timestamp", "match_confidence",
        ])
        writer.writeheader()
        writer.writerows(events)

    print(f"Exported {len(events)} event(s) from {start} to {end} -> {args.out}")


if __name__ == "__main__":
    initialize_database()
    parser = argparse.ArgumentParser(description="View or export attendance records.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_today = sub.add_parser("today", help="Print today's attendance events.")
    p_today.set_defaults(func=cmd_today)

    p_export = sub.add_parser("export", help="Export attendance events to CSV.")
    p_export.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (default: today).")
    p_export.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: today).")
    p_export.add_argument("--out", type=str, default="attendance_export.csv", help="Output CSV path.")
    p_export.set_defaults(func=cmd_export)

    parsed = parser.parse_args()
    parsed.func(parsed)
    sys.exit(0)
