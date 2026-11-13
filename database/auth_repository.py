"""
Admin authentication for the dashboard.

The whole dashboard (live camera, employee data, attendance records) is
gated behind an admin sign-in. This module is the single place that
knows how admin credentials are stored and checked.

SECURITY NOTES:
- Passwords are stored ONLY as a salted one-way hash (werkzeug's
  pbkdf2:sha256), never in plaintext -- so the database file never
  contains a recoverable password, and verifying a login compares
  hashes rather than raw strings.
- First run has no admin row: `admin_exists()` returns False and the
  UI shows a one-time "create admin account" screen. `create_admin()`
  refuses to run once any admin exists, so that setup screen can't be
  used to add a second admin behind the first one's back.
- This models a single trusted admin station (the "SecOps" account in
  the dashboard). The table supports more than one row if ever needed,
  but the UI intentionally manages exactly one admin.
"""

import datetime
from pathlib import Path
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

from database.connection import get_connection

# pbkdf2:sha256 is available on every stock Python/werkzeug (scrypt can
# require OpenSSL features that aren't always present), so pin it for
# portable, dependency-free hashing across deployments.
_HASH_METHOD = "pbkdf2:sha256"

# Keep the rules light but non-trivial -- this is an internal admin
# station, not a public signup, so the aim is "no blank/trivial
# passwords", not a strict policy the sole operator would just fight.
MIN_USERNAME_LEN = 3
MIN_PASSWORD_LEN = 6


def admin_exists(db_path: Optional[Path] = None) -> bool:
    """True once at least one admin account has been created."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT 1 FROM admin_users LIMIT 1").fetchone()
    return row is not None


def get_admin_username(db_path: Optional[Path] = None) -> Optional[str]:
    """The (first) admin's username, or None on a fresh install."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT username FROM admin_users ORDER BY id LIMIT 1"
        ).fetchone()
    return row["username"] if row else None


def create_admin(username: str, password: str, db_path: Optional[Path] = None) -> None:
    """
    Create the first admin account. Raises ValueError with a clear,
    user-facing message on bad input or if an admin already exists
    (the caller surfaces the message to the setup screen).
    """
    username = (username or "").strip()
    if len(username) < MIN_USERNAME_LEN:
        raise ValueError(f"Username must be at least {MIN_USERNAME_LEN} characters.")
    if len(password or "") < MIN_PASSWORD_LEN:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    if admin_exists(db_path=db_path):
        raise ValueError("An admin account already exists.")
    now = datetime.datetime.now().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password, method=_HASH_METHOD), now),
        )


def verify_credentials(username: str, password: str, db_path: Optional[Path] = None) -> bool:
    """
    Return True only if `username` exists and `password` matches its
    stored hash. Username match is case-insensitive (one admin station;
    an operator shouldn't be locked out by the case of their own name),
    the password check is exact. Deliberately gives no hint about WHICH
    half was wrong to the caller.
    """
    username = (username or "").strip()
    if not username or not password:
        return False
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT password_hash FROM admin_users WHERE LOWER(username) = LOWER(?) LIMIT 1",
            (username,),
        ).fetchone()
    if row is None:
        return False
    return check_password_hash(row["password_hash"], password)


def touch_last_login(username: str, db_path: Optional[Path] = None) -> None:
    """Record the time of a successful sign-in (shown in Settings)."""
    now = datetime.datetime.now().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE admin_users SET last_login_at = ? WHERE LOWER(username) = LOWER(?)",
            (now, username),
        )


def get_last_login(username: str, db_path: Optional[Path] = None) -> Optional[str]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT last_login_at FROM admin_users WHERE LOWER(username) = LOWER(?) LIMIT 1",
            (username,),
        ).fetchone()
    return row["last_login_at"] if row and row["last_login_at"] else None


def change_password(
    username: str, current_password: str, new_password: str, db_path: Optional[Path] = None
) -> None:
    """
    Change the admin's password after verifying the current one. Raises
    ValueError with a clear message on any failure (wrong current
    password, too-short new password) so the Settings UI can show it.
    """
    if not verify_credentials(username, current_password, db_path=db_path):
        raise ValueError("Current password is incorrect.")
    if len(new_password or "") < MIN_PASSWORD_LEN:
        raise ValueError(f"New password must be at least {MIN_PASSWORD_LEN} characters.")
    if new_password == current_password:
        raise ValueError("New password must be different from the current one.")
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE admin_users SET password_hash = ? WHERE LOWER(username) = LOWER(?)",
            (generate_password_hash(new_password, method=_HASH_METHOD), username),
        )
