"""
Repository functions for employee records and their face embeddings.

Embeddings are stored as packed float32 bytes (via numpy's .tobytes())
rather than JSON/text, which keeps storage compact and lets numpy
reconstruct the vector directly with zero parsing overhead --
meaningful when a recognition pass may compare against every stored
embedding for every stable tracked person.
"""

import datetime
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from database.connection import get_connection


class DuplicateEmployeeCodeError(Exception):
    """Raised when an employee_code collides with an already-registered one.
    Callers (register_employee.py) catch this specifically to give a clear,
    actionable message instead of a raw database traceback."""
    def __init__(self, employee_code: str):
        self.employee_code = employee_code
        super().__init__(f"Employee code '{employee_code}' is already in use by another employee.")


@dataclass
class Employee:
    employee_id: int
    full_name: str
    employee_code: Optional[str]
    department: Optional[str]
    is_active: bool


@dataclass
class StoredEmbedding:
    employee_id: int
    full_name: str
    angle_label: str
    embedding: np.ndarray  # shape (512,), float32


def employee_code_exists(employee_code: str, db_path: Optional[Path] = None) -> bool:
    """
    Check upfront whether an employee_code is already taken, so
    register_employee.py can catch this BEFORE spending several
    minutes walking someone through face capture, rather than
    discovering a UNIQUE-constraint collision only when trying to save
    at the very end. At thousands of employees, code collisions
    (typos, re-registration attempts, an HR system reusing a retired
    code) are a realistic, recurring event -- not an edge case worth
    ignoring.
    """
    if not employee_code:
        return False
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM employees WHERE employee_code = ? LIMIT 1", (employee_code,)
        ).fetchone()
    return row is not None


def find_active_employee_by_name(full_name: str, db_path: Optional[Path] = None):
    """
    Return an existing ACTIVE employee whose name matches `full_name`
    case-insensitively (ignoring case and surrounding whitespace), or
    None. Used to block duplicate registration by name -- "Hammad",
    "hammad" and " Hammad " are all the same name. Only active
    employees count: a deactivated (left-the-company) record with the
    same name shouldn't block re-registering a genuinely new hire who
    happens to share it.
    """
    if not full_name or not full_name.strip():
        return None
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT employee_id, full_name, employee_code, department, is_active "
            "FROM employees WHERE is_active = 1 AND LOWER(TRIM(full_name)) = LOWER(TRIM(?)) LIMIT 1",
            (full_name,),
        ).fetchone()
    if row is None:
        return None
    return Employee(
        employee_id=row["employee_id"], full_name=row["full_name"],
        employee_code=row["employee_code"], department=row["department"],
        is_active=bool(row["is_active"]),
    )


def register_employee(
    full_name: str,
    employee_code: Optional[str] = None,
    department: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert a new employee record. Returns the new employee_id."""
    now = datetime.datetime.now().isoformat()
    with get_connection(db_path) as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO employees (full_name, employee_code, department, is_active, registered_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (full_name, employee_code, department, now),
            )
        except sqlite3.IntegrityError as exc:
            # Defense-in-depth alongside employee_code_exists() above:
            # that upfront check has a small race window (two admins
            # registering the same code at nearly the same moment), and
            # this catches it regardless. Converted to a specific,
            # catchable exception rather than leaking a raw sqlite3
            # error up to the registration UI.
            if employee_code and "employee_code" in str(exc):
                raise DuplicateEmployeeCodeError(employee_code) from exc
            raise
        return cursor.lastrowid


def save_embedding(
    employee_id: int,
    angle_label: str,
    embedding: np.ndarray,
    db_path: Optional[Path] = None,
) -> None:
    """Store one captured-angle face embedding for an employee."""
    now = datetime.datetime.now().isoformat()
    blob = embedding.astype(np.float32).tobytes()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO face_embeddings (employee_id, angle_label, embedding_blob, captured_at) "
            "VALUES (?, ?, ?, ?)",
            (employee_id, angle_label, blob, now),
        )


def load_all_embeddings(db_path: Optional[Path] = None) -> List[StoredEmbedding]:
    """
    Load every stored embedding for every ACTIVE employee, ready for
    in-memory nearest-neighbor matching. Called once at startup (and
    whenever an admin registers someone new) -- not per-frame, so the
    cost of loading all rows here is negligible against how often
    recognition itself runs.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT e.employee_id, e.full_name, f.angle_label, f.embedding_blob
            FROM face_embeddings f
            JOIN employees e ON e.employee_id = f.employee_id
            WHERE e.is_active = 1
            """
        ).fetchall()

    return [
        StoredEmbedding(
            employee_id=row["employee_id"],
            full_name=row["full_name"],
            angle_label=row["angle_label"],
            embedding=np.frombuffer(row["embedding_blob"], dtype=np.float32),
        )
        for row in rows
    ]


def get_employee(employee_id: int, db_path: Optional[Path] = None) -> Optional[Employee]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT employee_id, full_name, employee_code, department, is_active "
            "FROM employees WHERE employee_id = ?",
            (employee_id,),
        ).fetchone()
    if row is None:
        return None
    return Employee(
        employee_id=row["employee_id"],
        full_name=row["full_name"],
        employee_code=row["employee_code"],
        department=row["department"],
        is_active=bool(row["is_active"]),
    )


def list_employees(db_path: Optional[Path] = None, active_only: bool = True) -> List[Employee]:
    query = "SELECT employee_id, full_name, employee_code, department, is_active FROM employees"
    if active_only:
        query += " WHERE is_active = 1"
    with get_connection(db_path) as conn:
        rows = conn.execute(query).fetchall()
    return [
        Employee(
            employee_id=r["employee_id"], full_name=r["full_name"],
            employee_code=r["employee_code"], department=r["department"],
            is_active=bool(r["is_active"]),
        )
        for r in rows
    ]


def deactivate_employee(employee_id: int, db_path: Optional[Path] = None) -> None:
    """Soft-delete: keeps historical attendance records intact."""
    with get_connection(db_path) as conn:
        conn.execute("UPDATE employees SET is_active = 0 WHERE employee_id = ?", (employee_id,))


def delete_employee(employee_id: int, db_path: Optional[Path] = None) -> None:
    """
    HARD delete: permanently removes the employee row, and -- via the
    `ON DELETE CASCADE` foreign keys declared in schema.py -- every one
    of their face_embeddings AND attendance_events rows too. This is
    irreversible and destroys their attendance history, unlike
    deactivate_employee() above.

    Use this for "I registered the wrong person" / test registrations.
    For someone who genuinely left the company, prefer
    deactivate_employee() instead so their historical attendance
    records are preserved (this is usually a compliance-relevant
    requirement for real attendance systems -- see schema.py).
    """
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))

