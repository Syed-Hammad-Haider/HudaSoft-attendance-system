"""
Small admin CLI for listing/removing registered employees.

register_employee.py handles ADDING someone (with the camera capture
flow); this script covers the read/manage side, which needs no camera
at all -- kept separate for the same "distinct concerns, distinct
entry points" reason documented in register_employee.py.

USAGE:
    python manage_employees.py list
    python manage_employees.py deactivate --id 3
"""

import argparse
import sys

from database.connection import initialize_database
from database.employee_repository import deactivate_employee, delete_employee, get_employee, list_employees


def cmd_list(args) -> None:
    employees = list_employees(active_only=not args.include_inactive)
    if not employees:
        print("No employees registered yet. Run 'python register_employee.py' to add one.")
        return
    print(f"{'ID':<5} {'Name':<25} {'Code':<12} {'Department':<15} {'Active'}")
    print("-" * 65)
    for e in employees:
        print(f"{e.employee_id:<5} {e.full_name:<25} {(e.employee_code or '-'):<12} "
              f"{(e.department or '-'):<15} {'Yes' if e.is_active else 'No'}")


def cmd_deactivate(args) -> None:
    deactivate_employee(args.id)
    print(f"Employee {args.id} deactivated. Their attendance history is kept, but they will no "
          f"longer be recognized or included in future matching. Press 'R' in main.py (or restart "
          f"it) to reload the known-faces list.")


def cmd_delete(args) -> None:
    employee = get_employee(args.id)
    if employee is None:
        print(f"No employee with id={args.id}. Run 'python manage_employees.py list' to see valid IDs.")
        return

    if not args.yes:
        print(f"This will PERMANENTLY delete '{employee.full_name}' (id={args.id}), including ALL "
              f"their face embeddings AND their entire attendance history. This cannot be undone.")
        print("If you just want to stop recognizing them while keeping their history, use "
              "'deactivate' instead.")
        confirm = input(f"Type the employee's exact name ('{employee.full_name}') to confirm deletion: ")
        if confirm.strip() != employee.full_name:
            print("Names did not match -- deletion cancelled, nothing was changed.")
            return

    delete_employee(args.id)
    print(f"Employee '{employee.full_name}' (id={args.id}) permanently deleted, along with their "
          f"face data and attendance history. Press 'R' in main.py (or restart it) to reload the "
          f"known-faces list.")


if __name__ == "__main__":
    initialize_database()
    parser = argparse.ArgumentParser(description="Manage registered employees.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List registered employees.")
    p_list.add_argument("--include-inactive", action="store_true", help="Also show deactivated employees.")
    p_list.set_defaults(func=cmd_list)

    p_deactivate = sub.add_parser("deactivate", help="Deactivate an employee (soft delete; keeps history).")
    p_deactivate.add_argument("--id", type=int, required=True, help="employee_id, from 'list'.")
    p_deactivate.set_defaults(func=cmd_deactivate)

    p_delete = sub.add_parser("delete", help="PERMANENTLY delete an employee, their face data, and their attendance history.")
    p_delete.add_argument("--id", type=int, required=True, help="employee_id, from 'list'.")
    p_delete.add_argument("--yes", action="store_true", help="Skip the confirmation prompt (for scripts).")
    p_delete.set_defaults(func=cmd_delete)

    parsed = parser.parse_args()
    parsed.func(parsed)
    sys.exit(0)
