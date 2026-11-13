"""
Thin SQLite connection helper.

WHY PLAIN sqlite3 INSTEAD OF AN ORM (e.g. SQLAlchemy):
This database has exactly three tables and a handful of well-defined
queries (register employee, save embedding, find nearest embeddings,
record event, query today's last event). An ORM's value is managing
complex relationships and query composition across many tables --
neither applies here. Plain sqlite3 with clearly-named repository
functions (see employee_repository.py / attendance_repository.py) is
more transparent for a small, well-understood schema, and keeps this
Phase 2 addition from pulling in a much heavier dependency for no real
benefit.

WHY get_db_path()/set_db_path() INSTEAD OF A PLAIN MODULE CONSTANT
PASSED AS A FUNCTION DEFAULT:
Python default argument values are evaluated ONCE, at function
DEFINITION time -- so a repository function defined as
`def foo(db_path: Path = DEFAULT_DB_PATH)` permanently captures
whatever DEFAULT_DB_PATH was at import time. Reassigning the module
constant afterward (e.g. to point tests, or a configured production
path, at a different file) silently has NO effect on already-defined
functions -- a real, easy-to-miss bug, not a style nitpick. Using a
function call (`get_db_path()`) instead of a bound default means the
active path is looked up fresh every call, so it can genuinely be
reconfigured at runtime.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import os

# ATTENDANCE_DATA_DIR (optional): when set -- e.g. inside Docker, where it
# points at a mounted volume -- the database lives there so it survives
# container rebuilds. Unset (normal laptop use) -> same place as before.
_DATA_DIR = os.environ.get("ATTENDANCE_DATA_DIR", "").strip()
DEFAULT_DB_PATH = (Path(_DATA_DIR) if _DATA_DIR else Path(__file__).parent) / "attendance.db"
if _DATA_DIR:
    Path(_DATA_DIR).mkdir(parents=True, exist_ok=True)

_active_db_path: Path = DEFAULT_DB_PATH


def set_db_path(path: Path) -> None:
    """Change which database file all repository functions read/write.
    Call this once at startup (e.g. from config) before any other
    database function is used, or per-test to isolate test databases."""
    global _active_db_path
    _active_db_path = Path(path)


def get_db_path() -> Path:
    return _active_db_path


def initialize_database(db_path: Optional[Path] = None) -> None:
    """Create the database file and tables if they don't already exist."""
    from database.schema import SCHEMA_SQL
    path = db_path or get_db_path()
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        # WAL (Write-Ahead Logging) mode instead of SQLite's default
        # rollback-journal mode. This is a real production concern at
        # company scale, not a pilot-scale one: a rollback journal
        # blocks ALL readers while a write transaction is in progress,
        # so an admin running `manage_attendance.py export` or
        # `manage_employees.py list` while main.py is actively writing
        # a CHECK IN/OUT event would briefly block. WAL allows readers
        # to keep working concurrently with a writer. This is a
        # one-time, persistent setting stored in the database file
        # itself (not a per-connection pragma that needs repeating),
        # so setting it once here at initialization is sufficient.
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.executescript(SCHEMA_SQL)


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """
    Context-managed connection with foreign keys enabled and row access
    by column name (sqlite3.Row) -- makes repository code read like
    `row["full_name"]` instead of positional index guessing.
    """
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
