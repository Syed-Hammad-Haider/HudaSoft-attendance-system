"""
SQLite schema for Phase 2: employee records, face embeddings, and
attendance events.

WHY SQLITE:
This is a single-site deployment (one software house, one or a
handful of entrance cameras). SQLite needs no separate server process,
ships as a single file that's trivial to back up, and comfortably
handles the write volume a foot-traffic entrance produces (a handful
of events per person per day, not per frame -- attendance events are
written only on confirmed CHECK IN/CHECK OUT, not continuously). If
this ever needs multi-site deployment with concurrent writers across
locations, the schema below is deliberately plain, portable SQL (no
SQLite-specific syntax beyond AUTOINCREMENT) so migrating to Postgres
later is a mechanical export/import, not a redesign.

SCHEMA OVERVIEW:

  employees
    One row per registered person. `is_active` supports "soft delete"
    (an employee who leaves the company) without destroying their
    historical attendance records, which is generally a compliance-
    relevant requirement in real attendance systems.

  face_embeddings
    Many rows per employee (one per captured angle during
    registration: front, left, right, ...). Storing multiple angle
    embeddings (rather than averaging them into one) allows matching
    against whichever angle is closest to a person's real pose at
    recognition time, which is meaningfully more robust than a single
    frontal-only embedding.

  attendance_events
    One row per confirmed CHECK IN / CHECK OUT. This is the durable
    record an admin dashboard or report would query. `source_track_id`
    and `confidence` are kept for debugging/audit ("why did this
    event fire"), not for any live decision logic.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name       TEXT NOT NULL,
    employee_code   TEXT UNIQUE,           -- optional external HR/ID code
    department      TEXT,
    is_active       INTEGER NOT NULL DEFAULT 1,   -- 1 = active, 0 = deactivated
    registered_at   TEXT NOT NULL           -- ISO 8601 timestamp
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id     INTEGER NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
    angle_label     TEXT NOT NULL,          -- e.g. 'front', 'left', 'right'
    embedding_blob  BLOB NOT NULL,          -- 512-float32 vector, packed
    captured_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id     INTEGER NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
    action          TEXT NOT NULL CHECK (action IN ('CHECK IN', 'CHECK OUT')),
    event_timestamp TEXT NOT NULL,          -- ISO 8601 timestamp
    match_confidence REAL,                  -- face match score at time of event
    source_track_id INTEGER,                -- ephemeral tracking ID, for debugging only
    camera_id       TEXT DEFAULT 'default'  -- supports multiple cameras later
);

CREATE INDEX IF NOT EXISTS idx_embeddings_employee ON face_embeddings(employee_id);
CREATE INDEX IF NOT EXISTS idx_events_employee_time ON attendance_events(employee_id, event_timestamp);
-- Separate from the composite index above: manage_attendance.py's
-- "today" / date-range export queries filter by event_timestamp ALONE
-- across every employee, which the composite (employee_id, timestamp)
-- index above can't serve efficiently (its leading column is
-- employee_id, not timestamp). At a handful of employees a full table
-- scan is invisible; at thousands of employees with months/years of
-- daily events, it stops being invisible. This index is what keeps
-- those report queries fast as the events table grows.
CREATE INDEX IF NOT EXISTS idx_events_time ON attendance_events(event_timestamp);

-- Small key/value store for admin-configurable settings that need to
-- survive a restart WITHOUT editing code (e.g. from the dashboard's
-- Settings page) -- e.g. attendance_day_boundary_hour (see
-- database/attendance_repository.py's work_day_start()). Deliberately
-- generic (any string key/value) rather than one column per setting,
-- so adding a new admin-editable setting later never needs a schema
-- migration.
CREATE TABLE IF NOT EXISTS system_settings (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);

-- Dashboard admin accounts. The dashboard is gated behind a login; this
-- table holds the admin credential(s) that can sign in. Passwords are
-- NEVER stored in plaintext -- only a salted one-way hash (see
-- database/auth_repository.py, which uses werkzeug's password hashing).
-- On first run no row exists, and the dashboard shows a one-time
-- "create admin account" screen instead of the login form.
CREATE TABLE IF NOT EXISTS admin_users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    created_at      TEXT NOT NULL,          -- ISO 8601 timestamp
    last_login_at   TEXT                    -- ISO 8601 timestamp, updated on each sign-in
);
"""
