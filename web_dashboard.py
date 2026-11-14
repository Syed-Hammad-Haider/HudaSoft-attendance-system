"""
Local web dashboard -- a styled, browser-based alternative to main.py's
OpenCV desktop window, for a more "industry standard" look and feel.

RUN:
    python web_dashboard.py
Then open http://localhost:5000 in a browser on this same computer (or
on another device on the same network, using this computer's LAN IP
instead of "localhost").

WHAT THIS IS AND ISN'T:
This reuses the EXACT SAME detection / tracking / wave-gesture /
face-recognition / attendance-decision pipeline as main.py -- none of
that computer-vision logic is duplicated or different here. What's
different is only the DISPLAY layer: instead of drawing into an OpenCV
window, the processed video frame is streamed as an MJPEG feed into a
browser page, alongside a live attendance log, employee list, and a
manual override panel -- all served from this one local process.

CALIBRATION: this dashboard does NOT include its own line-calibration
UI (dragging the entrance line into place needs native OpenCV mouse
events, which don't translate directly to a browser page). Run
`python main.py` ONCE first, press 'C', drag the line into place
against your real doorway, and click to confirm -- that saves to
`config/entrance_line_calibration.json`, which this dashboard also
reads automatically. After that one-time step, this dashboard can be
your normal day-to-day interface.

ARCHITECTURE: a single background thread runs the same per-frame loop
as main.py (camera -> detect -> track -> wave -> recognize -> decide
-> render), except the rendered frame is JPEG-encoded into a shared,
lock-protected buffer instead of being shown in a cv2 window. Flask
serves that buffer as an MJPEG stream (`/video_feed`) and exposes small
JSON endpoints for the attendance log, employee list, live status, and
manual check-in/out actions. This keeps the actual computer-vision
code as the single source of truth -- main.py and this dashboard are
just two different front ends onto the same pipeline.
"""

import base64
import csv
import datetime
import io
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from functools import wraps

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from camera.factory import create_video_source
from camera.browser_source import (
    BROWSER_SOURCE,
    is_browser_source,
    push_frame as push_browser_frame,
    set_active_sender as set_browser_sender,
)
from config.settings import CONFIG, WAVE_SENSITIVITY_PRESETS, apply_wave_sensitivity
from database.attendance_repository import (
    delete_event,
    delete_todays_events,
    get_checkin_dates_in_month,
    get_events_in_range,
    get_todays_events,
    get_todays_last_action,
    record_event,
    update_event_action,
)
from database.auth_repository import (
    admin_exists,
    change_password,
    create_admin,
    get_admin_username,
    get_last_login,
    touch_last_login,
    verify_credentials,
)
from database.connection import initialize_database
from database.employee_repository import (
    DuplicateEmployeeCodeError,
    deactivate_employee,
    delete_employee,
    find_active_employee_by_name,
    get_employee,
    list_employees,
)
from database.settings_repository import (
    get_camera_source,
    get_camera_source_raw,
    get_day_boundary_hour,
    get_min_action_gap_hours,
    get_mirror_preview,
    get_or_create_secret_key,
    get_work_days_per_week,
    set_camera_source,
    set_day_boundary_hour,
    set_min_action_gap_hours,
    set_mirror_preview,
    set_work_days_per_week,
)
from direction_estimation.entrance_line import EntranceGate
from direction_estimation.line_calibration import load_calibration, save_calibration
from display.renderer import Renderer
from face_id.embedder import FaceEmbedder
from face_id.identified_attendance import IdentifiedAttendanceDecisionEngine, status_label_for_employee
from face_id.live_recognition import LiveFaceRecognitionCache
from face_id.recognizer import FaceRecognizer
from face_id.registration import DEFAULT_ANGLES, evaluate_capture_quality, register_new_employee
from person_detection.detector import PersonDetector
from person_detection.close_range import CloseRangeFallback
from person_tracking.tracker import PersonTracker
from utils.event_log import EventLog
from utils.geometry import bbox_center
from wave_detection.detector import WaveDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("web_dashboard")

app = Flask(__name__)

# The dashboard is gated behind an admin sign-in (see the auth routes and
# the before_request gate below). Flask signs the session cookie with this
# secret; it's generated once and persisted in the database so sessions
# survive a restart (see get_or_create_secret_key). initialize_database()
# is idempotent, so calling it here to guarantee the settings table exists
# before we read the key is safe (it also runs again at module load below).
initialize_database()
app.secret_key = get_or_create_secret_key()
app.permanent_session_lifetime = datetime.timedelta(days=7)

# Endpoints reachable WITHOUT being signed in -- the login page itself and
# the auth API it calls, plus Flask's static-file endpoint (so the logo
# loads on the login screen). Everything else requires a valid admin
# session (enforced in _require_login_gate below).
_PUBLIC_ENDPOINTS = {"login_page", "auth_status", "auth_login", "auth_setup", "static", "healthz"}


class DashboardState:
    """
    Everything the Flask routes need to read, written by the background
    detection thread and read by request handlers -- protected by one
    lock since Flask's dev server can handle requests on different
    threads.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg: bytes = b""
        self.last_event_text: str = "Waiting for activity..."
        # Rolling list of recent on-screen events (crossings "ID 5:
        # crossed -> ENTERING", check-ins, and genuine "waved but not
        # checked in" reasons) for the Live Monitor's Recent Events
        # feed. [{time, text}], newest first.
        self.recent_events: list = []
        self.people_status: list = []  # [{track_id, name, status_label, wave_confidence}]
        # Default to the snappiest tier -- the whole point of this system
        # is a quick wave checking you in; "balanced" felt laggy in live
        # use. Admins can still switch to balanced/strict in Settings.
        self.sensitivity = "relaxed"
        self.running = True
        # Live camera switching (Settings -> Camera Source, e.g. a phone
        # via DroidCam). When a new source is requested, it's parked here;
        # the detection loop picks it up at the top of its next iteration,
        # releases the current camera and opens the new one -- no restart.
        # None means "no change pending".
        self.pending_camera_source = None  # int | str | None
        self.camera_source_label = ""      # what's currently in use, for the UI
        # Mirror the camera image left-right (selfie view). Read from
        # saved settings at loop start and live-togglable from Settings.
        self.mirror_preview = False
        self.camera_error: str = ""
        # Reference to the LIVE FaceRecognizer instance the detection
        # loop is actively using, set once the loop creates it (see
        # _detection_loop() below). Needed so /api/registration/finalize
        # can call .reload_known_faces() on the SAME running instance
        # after a browser-based registration -- otherwise a newly
        # registered employee wouldn't be recognized until the whole
        # dashboard process was restarted. FaceRecognizer.reload_known_faces()
        # is internally lock-protected against concurrent recognize()
        # calls from the detection thread (see face_id/recognizer.py),
        # so calling it from this Flask request thread is safe.
        self.face_recognizer = None
        # Raw (un-annotated) copy of the most recent camera frame --
        # separate from latest_jpeg (which has boxes/skeletons/text
        # burned into it for display). REAL BUG FIX: registration used
        # to ask the BROWSER to open its own independent camera handle
        # via getUserMedia() for capturing angle photos -- but this
        # backend already holds the physical camera device open
        # exclusively for the whole detection loop (see
        # camera/factory.py -- create_video_source()). Most webcams
        # (and definitely a single USB/IP door camera) can only be
        # opened by ONE process/handle at a time, so the browser's
        # getUserMedia() call for that same physical device fails with
        # exactly the error seen in practice: "Could not access
        # camera: Device in use" -- nothing to do with browser
        # permissions at all. The fix is architectural, not a
        # permission fix: registration must reuse the frame this
        # backend is ALREADY capturing, not open a second handle to
        # the same device. This also happens to be more correct for a
        # second reason -- it guarantees the registration photos come
        # from the EXACT SAME camera (angle, lens, lighting) that will
        # later try to recognize this person, rather than whatever
        # camera the browser's OS happens to default to (which, on a
        # laptop with both a door-camera feed and a built-in webcam,
        # could easily be the wrong one).
        self.latest_raw_frame = None
        # Reference to the LIVE EntranceGate the detection loop is using,
        # set once the loop builds it. Needed so the browser-based
        # entrance-line calibration (Calibration tab -> /api/calibration/*)
        # can move the line on the SAME running gate instantly, with no
        # restart and no separate main.py step. EntranceGate.set_line()
        # / flip_inside_side() only reassign a couple of plain attributes
        # and clear per-person state, so calling them from a Flask
        # request thread while the detection loop reads them is safe in
        # practice (Python attribute assignment is atomic; worst case a
        # single frame mid-move classifies against the old line, which is
        # harmless -- the very next frame uses the new one).
        self.gate = None
        # Rolling average of how long one detect->track->recognize->render
        # pipeline iteration takes, in milliseconds. Surfaced in the
        # sidebar status block ("Camera Latency") and used to drive the
        # "NPU Inference Engine" ONLINE/OFFLINE pill, so the footer shows
        # a real, live health signal rather than a hardcoded number.
        self.frame_latency_ms = 0.0

        # --- Decoupled display pipeline ---------------------------------
        # The heavy detection loop (YOLO + wave + face) runs as fast as the
        # hardware allows; a SEPARATE display thread draws the live video at
        # a steady, high frame rate so the feed stays smooth even when
        # detection is slower. The display thread renders the most-recent
        # detection results onto the FRESHEST camera frame. None of this
        # changes detection/wave/face logic -- it only separates "show the
        # video" from "analyze the video".
        #
        # current_source: the live video source, shared so BOTH the
        # detection loop and the display thread can read the freshest frame.
        # Guarded by source_lock so a camera switch (release + reopen) can
        # never race a concurrent read.
        self.current_source = None
        self.source_lock = threading.Lock()
        # renderer + the lock protecting its internal per-track message
        # state: the detection loop MUTATES it (show_suggested_action /
        # remove_track) while the display thread READS it (render()).
        self.renderer = None
        self.renderer_lock = threading.Lock()
        # Latest detection results for the display thread to draw, as a
        # self-contained, immutable-per-frame snapshot (so the display
        # thread never iterates a structure the detection loop is mutating).
        # Tuple: (people, gate_states, wave_results, status_labels, display_names)
        self.render_snapshot = None
        self.min_history_for_stable = 1


state = DashboardState()

# --- Per-frame status-label cache ----------------------------------------
#
# status_label_for_employee() runs a small DB query (get_todays_last_action)
# every time it's called. In the detection loop it's called once PER PERSON
# PER FRAME (for the live status chips) -- so at a busy entrance with
# several people in view, that's several SQLite round-trips every single
# frame, purely to redraw a label that only actually changes when someone
# checks in or out. This caches each employee's label for a short TTL so
# the common case (same people standing in frame, nothing changing) costs
# nothing, while a real transition still shows within the TTL. The cache
# is also explicitly invalidated the instant an action is recorded (see
# the detection loop), so a check-in/out reflects immediately, not after
# the TTL. A plain dict guarded by its own lock -- read from the detection
# thread only, but kept thread-safe in case a route ever reads it too.
_status_label_cache: dict = {}  # employee_id -> (label, expiry_monotonic)
_status_label_cache_lock = threading.Lock()
_STATUS_LABEL_TTL_SECONDS = 2.0

# Valid next action for a MANUAL override, given today's last recorded
# action -- mirrors the wave flow's state machine (see
# identified_attendance._VALID_NEXT_ACTION) so a manual check-in/out can't
# create an impossible once-per-day sequence. None means "nothing more
# allowed today" (already checked in and out).
_VALID_MANUAL_NEXT_ACTION = {
    None: "CHECK IN",
    "CHECK IN": "CHECK OUT",
    "CHECK OUT": None,
}


def _cached_status_label(employee_id):
    """status_label_for_employee() with a short TTL cache -- see above."""
    if employee_id is None:
        return status_label_for_employee(None)  # "UNRECOGNIZED", no DB hit
    now = time.monotonic()
    with _status_label_cache_lock:
        hit = _status_label_cache.get(employee_id)
        if hit is not None and hit[1] > now:
            return hit[0]
    label = status_label_for_employee(employee_id)
    with _status_label_cache_lock:
        _status_label_cache[employee_id] = (label, now + _STATUS_LABEL_TTL_SECONDS)
    return label


def _invalidate_status_label(employee_id):
    """Drop a cached label so the next lookup re-reads the DB -- called the
    moment an attendance action is recorded, so the live chip updates at
    once rather than after the TTL."""
    with _status_label_cache_lock:
        _status_label_cache.pop(employee_id, None)

# --- Browser-based employee registration --------------------------------
#
# WHY THIS EXISTS: register_employee.py (the CLI tool) has real,
# valuable logic -- per-angle geometric pose validation (measuring
# actual head yaw, not just trusting a timer), an embedding-distinctness
# safety net, quality gates (detection confidence, face size) -- and
# NONE of that is reimplemented here. Every one of those checks runs
# through the exact same face_id/registration.py functions
# (evaluate_capture_quality / register_new_employee) the CLI tool
# calls. What's different is only WHO drives the loop: the CLI's `while`
# loop polling a local webcam becomes the browser polling this API with
# frames from getUserMedia, at roughly the same cadence, with the same
# "hold steady for N seconds before it auto-captures" UX (driven
# client-side in JS, but every frame is still validated authoritatively
# here on the server -- the browser can't just tell the server "trust
# me, this one was good").
#
# SESSION STATE: registration spans several HTTP requests (one per
# angle, plus finalize), so accepted captures need to persist between
# them. Kept in server memory, keyed by a random session_id the browser
# holds onto -- NOT in the database until finalize() actually commits a
# real employee, so an abandoned/cancelled registration leaves nothing
# behind. This mirrors this whole project's single-process, single-site
# deployment model (see schema.py's docstring) -- it does NOT support
# multiple admins registering different people concurrently from
# different browsers in a way that survives a process restart, which is
# an acceptable limit for what this is (an occasional admin setup task,
# not a high-concurrency user-facing flow).
_registration_sessions = {}
_registration_sessions_lock = threading.Lock()
_REGISTRATION_SESSION_TIMEOUT_SECONDS = 15 * 60  # abandoned session cleanup

_registration_embedder = None
_registration_embedder_lock = threading.Lock()


def _get_registration_embedder() -> FaceEmbedder:
    """
    A FaceEmbedder instance dedicated to registration requests, created
    lazily (only if registration is actually used) and separate from
    the detection loop's own embedder -- keeps registration's model
    instance/memory fully independent of the live monitoring pipeline,
    so nothing about registration can ever destabilize it.
    """
    global _registration_embedder
    with _registration_embedder_lock:
        if _registration_embedder is None:
            _registration_embedder = FaceEmbedder(det_size=CONFIG.face_recognition.detection_size)
        return _registration_embedder


def _decode_data_url_image(data_url: str):
    """Decode a browser canvas.toDataURL('image/jpeg') string (or plain
    base64) into a BGR numpy frame, the same format every other camera
    frame in this codebase is in. No longer used by the registration
    endpoints themselves (see _get_live_frame_for_registration below)
    -- kept only in case a future caller needs to accept an uploaded
    image some other way."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _get_live_frame_for_registration():
    """
    The frame every registration check/capture actually validates
    against: the backend's OWN live camera frame (state.latest_raw_frame),
    NOT an image uploaded from the browser's own camera. See
    DashboardState.latest_raw_frame's docstring for why -- the physical
    camera can only be held open by one process, and this backend
    already holds it for the whole detection loop. Returns None if the
    detection loop hasn't produced a frame yet (e.g. camera still
    starting up, or camera_error is set).
    """
    with state.lock:
        frame = state.latest_raw_frame
    return None if frame is None else frame.copy()


def _purge_stale_registration_sessions() -> None:
    cutoff = time.time() - _REGISTRATION_SESSION_TIMEOUT_SECONDS
    stale = [sid for sid, s in _registration_sessions.items() if s["created_at"] < cutoff]
    for sid in stale:
        del _registration_sessions[sid]


def _build_entrance_gate():
    gate = EntranceGate(CONFIG.entrance_line, CONFIG.camera.frame_width, CONFIG.camera.frame_height)
    saved = load_calibration()
    if saved:
        try:
            p1_ratio = tuple(saved["line_start"])
            p2_ratio = tuple(saved["line_end"])
            inside_left = saved["inside_is_left_of_line"]
            p1_px = (p1_ratio[0] * CONFIG.camera.frame_width, p1_ratio[1] * CONFIG.camera.frame_height)
            p2_px = (p2_ratio[0] * CONFIG.camera.frame_width, p2_ratio[1] * CONFIG.camera.frame_height)
            gate.set_line(p1_px, p2_px)
            gate._inside_is_left = inside_left
            logger.info("Using saved entrance line calibration.")
        except (KeyError, TypeError, IndexError):
            logger.warning("Saved calibration was malformed -- using the default line position. "
                            "Run 'python main.py' once and press 'C' to calibrate properly.")
    else:
        logger.warning(
            "No saved entrance line calibration found -- using the DEFAULT line position, which "
            "likely does not match your real doorway. Run 'python main.py' once, press 'C', drag "
            "the line into place, and click to confirm. This dashboard will pick up that "
            "calibration automatically on its next restart."
        )
    return gate


def _camera_label(source) -> str:
    """Human-friendly label for a camera source, for the Settings UI."""
    if is_browser_source(source):
        return "This browser's webcam"
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return f"Webcam #{source}"
    return str(source)


def _open_camera_source():
    """
    Create and open the camera for the CURRENT CONFIG.camera.source and
    return the opened source -- or None on ANY failure (with camera_error
    set for the UI). This NEVER raises. That matters: the detection thread
    has to stay alive through a bad or unreachable source (a wrong DroidCam
    IP, an offline phone, an unplugged webcam) so the admin can switch to a
    correct source -- or back to the PC webcam -- from Settings WITHOUT
    restarting. If opening ever killed the loop, nothing would be left
    running to pick up that next change, which is exactly the "I typed a
    wrong IP and now I can't connect to anything" trap.
    """
    try:
        src = create_video_source(CONFIG.camera)
        src.open()
        with state.lock:
            state.camera_error = ""
            state.camera_source_label = _camera_label(CONFIG.camera.source)
        logger.info("Camera opened: %s", _camera_label(CONFIG.camera.source))
        return src
    except Exception as exc:
        logger.warning("Could not open camera '%s': %s", _camera_label(CONFIG.camera.source), exc)
        with state.lock:
            state.camera_error = (
                f"Could not open camera '{_camera_label(CONFIG.camera.source)}': {exc}"
            )
        return None


def _detection_loop():
    """Runs forever in a background thread: the same per-frame pipeline as
    main.py, but rendering into a shared JPEG buffer instead of a cv2 window."""
    initialize_database()

    # Apply the admin-saved camera source (Settings -> Camera Source)
    # before opening the camera, so a phone/IP camera chosen in the UI
    # is used on startup without any code edit. Falls back to whatever
    # CameraConfig.source already is if nothing was saved.
    try:
        CONFIG.camera.source = get_camera_source()
    except Exception:
        logger.exception("Could not read saved camera source; using config default.")
    with state.lock:
        state.camera_source_label = _camera_label(CONFIG.camera.source)
        try:
            state.mirror_preview = get_mirror_preview()
        except Exception:
            state.mirror_preview = False

    video_source = None  # opened below, non-fatally, via _open_camera_source()
    detector = PersonDetector(CONFIG.detection)
    tracker = PersonTracker(CONFIG.tracking)
    gate = _build_entrance_gate()
    with state.lock:
        state.gate = gate  # expose to the Calibration API (see DashboardState.gate)
    wave_detector = WaveDetector(CONFIG.wave)
    apply_wave_sensitivity(CONFIG.wave, state.sensitivity)

    face_embedder = FaceEmbedder(det_size=CONFIG.face_recognition.detection_size)
    close_range = CloseRangeFallback(face_embedder)   # head-and-shoulders fallback when YOLO sees nobody
    face_recognizer = FaceRecognizer(
        min_similarity=CONFIG.face_recognition.min_match_similarity,
        min_margin_over_runner_up=CONFIG.face_recognition.min_margin_over_runner_up,
    )
    known_count = face_recognizer.reload_known_faces()
    if known_count == 0:
        logger.warning("No registered employees found -- register one from the dashboard's Register tab.")
    with state.lock:
        state.face_recognizer = face_recognizer
    recognition_cache = LiveFaceRecognitionCache(CONFIG.face_recognition, face_embedder, face_recognizer)
    decision_engine = IdentifiedAttendanceDecisionEngine(
        CONFIG.entrance_line, CONFIG.action_guard, CONFIG.camera.frame_width
    )

    renderer = Renderer(CONFIG.display)
    event_log = EventLog(CONFIG.display.event_log_max_entries)
    # Share the renderer and the "stable track" threshold with the display
    # thread (see _display_loop), which is what actually draws the live feed.
    with state.lock:
        state.renderer = renderer
        state.min_history_for_stable = CONFIG.entrance_line.min_frames_to_confirm_side

    previously_seen_ids = set()

    # Open the camera WITHOUT ever killing the loop on failure (see
    # _open_camera_source). A bad saved source leaves video_source = None
    # and the loop idles until a working one is set, rather than the thread
    # dying and leaving no way to recover from the UI.
    video_source = _open_camera_source()
    with state.source_lock:
        state.current_source = video_source  # share with the display thread

    logger.info("Web dashboard detection loop started.")
    consecutive_read_failures = 0
    max_consecutive_read_failures = 45
    last_open_attempt = time.time()
    reopen_backoff_seconds = 5.0

    while state.running:
        # 1) A pending source switch ALWAYS takes priority and is handled
        #    first -- this is what lets the admin fix a wrong IP, or fall
        #    back to the PC webcam, from Settings even while the current
        #    source is broken. Releasing the old source and opening the new
        #    one are both guarded; a failed open just leaves video_source
        #    None (handled by the idle branch below), never crashes.
        with state.lock:
            new_source = state.pending_camera_source
            state.pending_camera_source = None
        if new_source is not None:
            logger.info("Switching camera source to %r ...", new_source)
            try:
                with state.source_lock:
                    state.current_source = None  # stop the display thread reading it
                if video_source is not None:
                    video_source.release()
            except Exception:
                pass
            video_source = None
            CONFIG.camera.source = new_source
            video_source = _open_camera_source()
            with state.source_lock:
                state.current_source = video_source
            consecutive_read_failures = 0
            last_open_attempt = time.time()
            continue

        # 2) No working camera: idle, and retry the CURRENT source on a slow
        #    backoff so a transiently-down camera (phone reconnecting to
        #    Wi-Fi, webcam re-plugged) heals itself -- without hammering a
        #    permanently-bad source or freezing the UI. The pending-switch
        #    check above still runs every iteration, so a corrected IP or
        #    "Use PC Webcam" is picked up within one short sleep.
        if video_source is None:
            if time.time() - last_open_attempt >= reopen_backoff_seconds:
                last_open_attempt = time.time()
                video_source = _open_camera_source()
                with state.source_lock:
                    state.current_source = video_source
                consecutive_read_failures = 0
            time.sleep(0.2)
            continue

        frame = video_source.read_frame()
        if frame is None:
            consecutive_read_failures += 1
            if consecutive_read_failures >= max_consecutive_read_failures:
                # A previously-working source went dead mid-stream. Drop it
                # and fall into the idle/retry branch above (which reopens
                # the SAME source on the backoff, and still honors a pending
                # switch first). The loop never dies -- "Use PC Webcam" and
                # a corrected IP both always work.
                if is_browser_source(CONFIG.camera.source):
                    # Not an error: simply no browser is sending its webcam yet.
                    with state.lock:
                        state.camera_error = (
                            "Waiting for a webcam: click \"Start this browser's webcam\" "
                            "on the Live Monitor (or connect a phone/IP camera in Settings)."
                        )
                else:
                    logger.warning("Camera stopped delivering frames -- dropping source to reconnect.")
                    with state.lock:
                        state.camera_error = (
                            "Camera stopped delivering frames -- reconnecting. "
                            "Check the camera, or switch source in Settings."
                        )
                try:
                    with state.source_lock:
                        state.current_source = None
                    video_source.release()
                except Exception:
                    pass
                video_source = None
                consecutive_read_failures = 0
                last_open_attempt = time.time()
            continue
        if consecutive_read_failures:
            consecutive_read_failures = 0
            with state.lock:
                state.camera_error = ""  # frames flowing again -- clear stale error

        iteration_start = time.perf_counter()

        # Mirror (selfie view) if enabled -- flip BEFORE everything else
        # so detection, the entrance line, registration and the preview
        # all see the same image. Toggling mirror flips left/right, so
        # the entrance line's IN/OUT sides may need the Calibration tab's
        # "Flip IN/OUT" afterward (noted in the Settings hint).
        with state.lock:
            mirror_now = state.mirror_preview
        if mirror_now:
            frame = cv2.flip(frame, 1)

        # Store a RAW copy (.copy(), since renderer.render() below
        # draws boxes/text directly onto `frame` in place -- without
        # the copy, this would end up holding an ANNOTATED frame by
        # the time registration reads it, not what a face embedder
        # should see) for the Register tab to use -- see
        # DashboardState.latest_raw_frame's docstring for why
        # registration needs this instead of its own camera handle.
        with state.lock:
            state.latest_raw_frame = frame.copy()
            state.last_frame_time = time.time()  # read by /healthz

        detections = detector.detect(frame)
        detections = close_range.augment(frame, detections)
        tracked_people = tracker.update(detections)
        current_ids = {p.track_id for p in tracked_people}

        dropped_ids = previously_seen_ids - current_ids
        for track_id in dropped_ids:
            wave_detector.remove_track(track_id)
            with state.renderer_lock:  # display thread may be reading renderer state
                renderer.remove_track(track_id)
            gate.remove_track(track_id)
            recognition_cache.remove_track(track_id)
        previously_seen_ids = current_ids

        gate_states, wave_results, status_labels, display_names = {}, {}, {}, {}
        people_status_snapshot = []

        for person in tracked_people:
            center = bbox_center(person.bbox)
            gate_state = gate.update(person.track_id, center)
            gate_states[person.track_id] = gate_state

            wave_result = wave_detector.process(frame, person)
            wave_results[person.track_id] = wave_result

            recognition = recognition_cache.get_recognition(frame, person)
            employee_id = recognition.employee_id if recognition else None
            employee_name = recognition.full_name if recognition else None
            match_confidence = recognition.similarity if recognition else 0.0

            # Crossing events show the employee's NAME when the face is already
            # recognised (e.g. "Syed Hammad (ID:1): crossed -> ENTERING");
            # only people not (yet) recognised fall back to the tracker ID.
            if gate_state.just_crossed:
                who = f"{employee_name} (ID:{employee_id})" if employee_name else f"ID {person.track_id}"
                event_log.add(f"{who}: crossed -> {gate_state.last_direction.value}")

            display_names[person.track_id] = employee_name or f"ID {person.track_id}"
            status_labels[person.track_id] = _cached_status_label(employee_id)

            decision = decision_engine.evaluate(
                employee_id=employee_id, employee_name=employee_name, gate_state=gate_state,
                wave_confirmed=wave_result.wave_confirmed, position=center,
                match_confidence=match_confidence, source_track_id=person.track_id,
            )

            if decision.action is not None:
                with state.renderer_lock:  # display thread may be reading renderer state
                    renderer.show_suggested_action(person.track_id, decision.action, gate_state.last_direction, employee_name or "")
                event_log.add(f"{employee_name} (ID:{employee_id}): {decision.action.label}")
                logger.info("Attendance recorded | Employee: %s (ID:%s) | Action: %s", employee_name, employee_id, decision.action.label)
                # An action just changed this employee's state -- drop the
                # cached label so the chip reflects it immediately.
                _invalidate_status_label(employee_id)
                status_labels[person.track_id] = _cached_status_label(employee_id)
            elif wave_result.wave_confirmed and decision.reason_if_none:
                # Routine, expected non-events (already checked in today
                # and waved again) stay silent on-screen. GENUINE reasons
                # (not recognized, no recent line crossing) are shown, so
                # a stuck user can actually see WHY a green wave produced
                # no check-in instead of getting no feedback at all.
                if not decision.reason_is_routine:
                    who = f"{employee_name} (ID:{employee_id})" if employee_name else f"ID {person.track_id} (unrecognized)"
                    event_log.add(f"{who}: waved but not checked in -- {decision.reason_if_none}")
                logger.info("Wave confirmed but no action for ID %d: %s", person.track_id, decision.reason_if_none)

            people_status_snapshot.append({
                "track_id": person.track_id,
                "name": display_names[person.track_id],
                "status": status_labels[person.track_id],
                "wave_confidence": round(wave_result.wave_confidence, 2),
            })

        # Publish the detection results as a self-contained snapshot for the
        # display thread to draw (see _display_loop). Each person's fields
        # are COPIED -- crucially a list() copy of position_history -- so the
        # display thread never iterates a structure this loop is mutating.
        # This loop no longer renders or JPEG-encodes: the display thread
        # does that on the freshest frame at a steady rate, which is what
        # keeps the video smooth even when this loop runs slower. Detection,
        # wave and face logic above are entirely unchanged.
        render_people = [
            SimpleNamespace(
                track_id=p.track_id, bbox=p.bbox, confidence=p.confidence,
                position_history=list(p.position_history),
            )
            for p in tracked_people
        ]

        # Exponential moving average of the per-frame PROCESSING time, for
        # the sidebar's "Camera Latency" readout.
        iteration_ms = (time.perf_counter() - iteration_start) * 1000.0
        with state.lock:
            prev = state.frame_latency_ms
            state.frame_latency_ms = iteration_ms if prev == 0 else prev * 0.8 + iteration_ms * 0.2
            state.render_snapshot = (render_people, gate_states, wave_results, status_labels, display_names)
            state.people_status = people_status_snapshot
            recent = event_log.recent()
            if recent:
                state.last_event_text = recent[0].text
            state.recent_events = [
                {"time": time.strftime("%H:%M:%S", time.localtime(e.timestamp)), "text": e.text}
                for e in recent
            ]

    try:
        with state.source_lock:
            state.current_source = None
        if video_source is not None:
            video_source.release()
    except Exception:
        pass
    logger.info("Detection loop stopped.")


# Unused placeholder event log for the display thread's render() call -- the
# event-log panel is HTML in the dashboard, and web_mode=True skips drawing
# it, but render() still takes the argument.
_DISPLAY_EVENT_LOG = EventLog(1)

# Live-video stream tuning. Defaults (30 fps, quality 72) match the original
# laptop behaviour. On a cloud server the video has to travel over the
# internet to the viewer's browser, so docker-compose lowers these (e.g. 10
# fps, quality 60): far less bandwidth and CPU, no visible effect on
# detection, which never reads these values.
_STREAM_FPS = max(1.0, float(os.environ.get("STREAM_FPS", "30") or 30))
_STREAM_QUALITY = min(95, max(30, int(os.environ.get("STREAM_QUALITY", "72") or 72)))


def _display_loop():
    """
    Draws the live video at a steady frame rate, INDEPENDENT of how fast the
    detection loop runs -- this is what keeps the feed smooth on slower
    hardware. Each tick it grabs the freshest camera frame and paints the
    most-recent detection snapshot (boxes, entrance line, hand skeleton,
    banners) onto it. It is pure presentation: it never runs detection,
    wave, or face recognition, and it only ever READS the detection loop's
    published snapshot, so it cannot affect accuracy.
    """
    target_dt = 1.0 / _STREAM_FPS   # display rate (30 fps on a laptop, lower on a server)
    last_raw = None
    while state.running:
        tick = time.perf_counter()

        # Freshest camera frame, guarded so a camera switch (release/reopen)
        # in the detection loop can never race this read.
        with state.source_lock:
            src = state.current_source
            raw = None
            if src is not None:
                try:
                    raw = src.read_frame()
                except Exception:
                    raw = None

        if raw is not None:
            with state.lock:
                mirror_now = state.mirror_preview
            # Match the detection loop's mirroring so overlays line up. Copy
            # so we're independent of the capture thread's buffer.
            raw = cv2.flip(raw, 1) if mirror_now else raw.copy()
            last_raw = raw

        if last_raw is None:
            time.sleep(0.03)   # camera not up yet
            continue

        frame = last_raw.copy()  # draw on a copy; keep last_raw clean for reuse

        with state.lock:
            snapshot = state.render_snapshot
            gate = state.gate
            renderer = state.renderer
            sensitivity = state.sensitivity
            min_hist = state.min_history_for_stable

        if renderer is not None and gate is not None:
            people, gate_states, wave_results, status_labels, display_names = (
                snapshot if snapshot is not None else ([], {}, {}, {}, {})
            )
            try:
                with state.renderer_lock:
                    renderer.render(
                        frame, people, gate_states, wave_results, gate,
                        _DISPLAY_EVENT_LOG, min_hist, None,
                        status_labels, display_names, sensitivity, web_mode=True,
                    )
            except Exception:
                logger.exception("Display render failed for one frame; showing raw frame.")

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _STREAM_QUALITY])
        if ok:
            with state.lock:
                state.latest_jpeg = jpeg.tobytes()

        elapsed = time.perf_counter() - tick
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


def _mjpeg_generator():
    last_sent = None
    while True:
        with state.lock:
            frame_bytes = state.latest_jpeg
        # Only send a frame when it is NEW: re-sending the same picture
        # wasted bandwidth, which made the feed freeze over slow internet.
        if frame_bytes and frame_bytes is not last_sent:
            last_sent = frame_bytes
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(1.0 / _STREAM_FPS)


def _mjpeg_generator_raw():
    """
    Same MJPEG stream as _mjpeg_generator, but of the RAW (un-annotated)
    camera frame -- no detection boxes, skeletons, entrance line or
    labels drawn on. The Display (calibration) tab uses this so the
    admin drags the entrance line over a CLEAN camera image, with the
    line overlay the browser draws on top being the only line visible
    (otherwise the backend's own drawn line would double up with the
    draggable one and make placement confusing). Encodes the shared
    raw frame on demand here rather than keeping a second JPEG buffer
    in the hot detection loop -- this stream is only open while the
    Display tab is on screen, so the extra encode cost is occasional,
    not per-frame-always.
    """
    while True:
        with state.lock:
            raw = state.latest_raw_frame
        if raw is not None:
            ok, jpeg = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.05)  # ~20fps -- plenty for lining up a static doorway


# --- Authentication gate -------------------------------------------------
#
# The whole dashboard (live camera, employee data, attendance history,
# settings) sits behind an admin sign-in. Rather than decorate every
# single route, one before_request hook gates everything except the
# login page and the auth endpoints it needs. An unauthenticated request
# to an API path gets a clean 401 JSON (so the browser's fetch() calls
# fail gracefully); any other unauthenticated request is redirected to
# the login screen.

def _is_authenticated() -> bool:
    return bool(session.get("admin_user"))


@app.before_request
def _require_login_gate():
    if request.endpoint in _PUBLIC_ENDPOINTS:
        return None
    if _is_authenticated():
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not authenticated", "authenticated": False}), 401
    return redirect(url_for("login_page"))


@app.route("/login")
def login_page():
    """The sign-in screen (or first-run 'create admin account' screen --
    the page decides which to show from /api/auth/status). If already
    signed in, go straight to the dashboard."""
    if _is_authenticated():
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/healthz")
def healthz():
    """Public, no-login health check for Docker / uptime monitors. Says
    whether the app is up and whether a camera frame arrived recently --
    never any employee data."""
    last = getattr(state, "last_frame_time", None)
    camera_ok = bool(last) and (time.time() - last) < 10
    return jsonify({"status": "ok", "camera_ok": camera_ok,
                    "camera": getattr(state, "camera_source_label", "")})


@app.route("/api/auth/status")
def auth_status():
    """Tells the login page whether to show the setup or sign-in form, and
    tells the dashboard who is signed in (for the sidebar + Settings)."""
    authed = _is_authenticated()
    username = session.get("admin_user") if authed else None
    return jsonify({
        "authenticated": authed,
        "needs_setup": not admin_exists(),
        "username": username,
        "last_login": get_last_login(username) if username else None,
    })


@app.route("/api/auth/setup", methods=["POST"])
def auth_setup():
    """First-run only: create the single admin account, then sign it in.
    Refused once an admin already exists (create_admin enforces this too)."""
    if admin_exists():
        return jsonify({"error": "An admin account already exists. Please sign in."}), 409
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    try:
        create_admin(username, password)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    session.permanent = True
    session["admin_user"] = username
    touch_last_login(username)
    logger.info("Admin account created and signed in: %s", username)
    return jsonify({"ok": True, "username": username})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not admin_exists():
        return jsonify({"error": "No admin account yet -- set one up first.", "needs_setup": True}), 400
    if not verify_credentials(username, password):
        # Same message for wrong user and wrong password -- don't reveal
        # which half failed.
        return jsonify({"error": "Incorrect username or password."}), 401
    stored = get_admin_username() or username  # normalize to stored casing
    session.permanent = True
    session["admin_user"] = stored
    touch_last_login(stored)
    logger.info("Admin signed in: %s", stored)
    return jsonify({"ok": True, "username": stored})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    user = session.get("admin_user")
    session.clear()
    logger.info("Admin signed out: %s", user)
    return jsonify({"ok": True})


@app.route("/api/auth/change_password", methods=["POST"])
def auth_change_password():
    """Change the signed-in admin's password (System Settings -> Admin
    Account). Verifies the current password before changing."""
    username = session.get("admin_user")
    if not username:
        return jsonify({"error": "Not authenticated"}), 401
    data = request.get_json(force=True)
    current = data.get("current_password") or ""
    new = data.get("new_password") or ""
    try:
        change_password(username, current, new)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    logger.info("Admin password changed: %s", username)
    return jsonify({"ok": True})


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_raw")
def video_feed_raw():
    """Un-annotated camera stream for the Display/calibration tab (see
    _mjpeg_generator_raw). Referenced by dashboard.html's startCalibration()."""
    return Response(_mjpeg_generator_raw(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state.lock:
        latency = state.frame_latency_ms
        cam_err = state.camera_error
        return jsonify({
            "people": state.people_status,
            "last_event": state.last_event_text,
            "recent_events": list(state.recent_events),
            "sensitivity": state.sensitivity,
            "camera_error": cam_err,
            # Live health signals for the sidebar footer.
            "latency_ms": round(latency),
            "online": not bool(cam_err) and latency > 0,
        })


@app.route("/api/attendance/today")
def api_attendance_today():
    events = get_todays_events()
    return jsonify([
        {"event_id": e["event_id"], "employee_id": e["employee_id"],
         "time": e["event_timestamp"][11:19],
         "name": e["full_name"], "action": e["action"]}
        for e in reversed(events)
    ])


@app.route("/api/attendance/records")
def api_attendance_records():
    """
    JSON version of the CSV export below -- backs the in-browser
    Records tab, so an admin can actually LOOK AT records (filtered by
    date range and optionally one employee) without downloading a file
    first. Newest-first, since that's what an admin scanning recent
    history wants (the CSV export below stays chronological/oldest-
    first, unchanged, since that's the more natural order for a
    payroll import and nothing about that was broken).
    """
    start = request.args.get("start")
    end = request.args.get("end")
    employee_id = request.args.get("employee_id")
    # A malformed date or employee_id from a hand-edited URL used to raise
    # and surface as an opaque 500; return a clear 400 instead.
    try:
        start_date = datetime.date.fromisoformat(start) if start else datetime.date.today() - datetime.timedelta(days=6)
        end_date = datetime.date.fromisoformat(end) if end else datetime.date.today()
        emp_filter = int(employee_id) if employee_id else None
    except (ValueError, TypeError):
        return jsonify({"error": "start and end must be YYYY-MM-DD dates and employee_id a number"}), 400
    events = get_events_in_range(start_date, end_date, employee_id=emp_filter)
    return jsonify(list(reversed(events)))


@app.route("/api/attendance/clear_today", methods=["POST"])
def api_clear_today():
    """Delete all of the current work day's attendance events -- the
    Live Monitor 'Clear Today's Attendance' button. For a clean re-test
    or to wipe a day of bad/test data. Confirmed in the UI first."""
    removed = delete_todays_events()
    logger.info("Cleared today's attendance via dashboard: %d event(s) removed", removed)
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/attendance/event/<int:event_id>/delete", methods=["POST"])
def api_delete_event(event_id):
    """Delete one attendance record (Records tab) -- removing a single
    mistaken/mis-detected entry."""
    delete_event(event_id)
    logger.info("Attendance event %s deleted via dashboard", event_id)
    return jsonify({"ok": True})


@app.route("/api/attendance/event/<int:event_id>/action", methods=["POST"])
def api_update_event(event_id):
    """Correct one record's direction (CHECK IN <-> CHECK OUT) from the
    Records tab, for when detection logged the wrong one."""
    data = request.get_json(force=True)
    action = data.get("action")
    if action not in ("CHECK IN", "CHECK OUT"):
        return jsonify({"error": "action must be 'CHECK IN' or 'CHECK OUT'"}), 400
    update_event_action(event_id, action)
    logger.info("Attendance event %s corrected to %s via dashboard", event_id, action)
    return jsonify({"ok": True, "action": action})


@app.route("/api/monthly")
def api_monthly():
    """
    Per-employee monthly attendance summary for the Monthly Report tab.
    For the chosen year/month, for each ACTIVE employee:
      - working_days  : how many of this employee's scheduled work days
                        fall in the month UP TO today (so a mid-month
                        report doesn't count days that haven't happened
                        yet). "Scheduled work days" = the first N weekdays
                        of each week, where N is their work-days-per-week
                        (5 -> Mon-Fri, 6 -> Mon-Sat, 7 -> every day).
      - present       : how many of those days they actually checked in.
      - absent        : working_days - present.
      - percentage    : present / working_days * 100 (0 when no work days).
    """
    today = datetime.date.today()
    try:
        year = int(request.args.get("year", today.year))
        month = int(request.args.get("month", today.month))
    except (TypeError, ValueError):
        return jsonify({"error": "year and month must be integers"}), 400
    if not (1 <= month <= 12):
        return jsonify({"error": "month must be 1-12"}), 400

    # All calendar dates in the month, bounded at today for the current
    # month (don't count the future as absent).
    import calendar as _cal
    days_in_month = _cal.monthrange(year, month)[1]
    last_day = days_in_month
    if year == today.year and month == today.month:
        last_day = today.day

    rows = []
    for emp in list_employees(active_only=True):
        dpw = get_work_days_per_week(emp.employee_id)
        # A day is a "scheduled work day" if its weekday index (Mon=0)
        # is < dpw -- so 5 => Mon-Fri, 6 => Mon-Sat, 7 => all week.
        scheduled_dates = []
        for d in range(1, last_day + 1):
            dt = datetime.date(year, month, d)
            if dt.weekday() < dpw:
                scheduled_dates.append(dt.isoformat())
        present_dates = get_checkin_dates_in_month(emp.employee_id, year, month)
        present = sum(1 for d in scheduled_dates if d in present_dates)
        working = len(scheduled_dates)
        absent = working - present
        pct = round(present / working * 100, 1) if working else 0.0
        rows.append({
            "employee_id": emp.employee_id, "name": emp.full_name,
            "department": emp.department, "work_days_per_week": dpw,
            "working_days": working, "present": present, "absent": absent,
            "percentage": pct,
        })
    rows.sort(key=lambda r: r["name"].lower())
    return jsonify({"year": year, "month": month, "employees": rows})


@app.route("/api/employees/<int:employee_id>/work_days", methods=["POST"])
def api_set_work_days(employee_id):
    """Set an employee's expected work-days-per-week (1-7), used by the
    Monthly Report's absent/percentage maths."""
    data = request.get_json(force=True)
    try:
        days = int(data.get("days"))
        set_work_days_per_week(employee_id, days)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "days must be an integer 1-7"}), 400
    return jsonify({"ok": True, "days": days})


@app.route("/api/attendance/export")
def api_attendance_export():
    start = request.args.get("start")
    end = request.args.get("end")
    try:
        start_date = datetime.date.fromisoformat(start) if start else datetime.date.today()
        end_date = datetime.date.fromisoformat(end) if end else datetime.date.today()
    except (ValueError, TypeError):
        return jsonify({"error": "start and end must be YYYY-MM-DD dates"}), 400
    events = get_events_in_range(start_date, end_date)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=[
        "event_id", "employee_id", "full_name", "employee_code", "department",
        "action", "event_timestamp", "match_confidence",
    ])
    writer.writeheader()
    writer.writerows(events)
    mem = io.BytesIO(buffer.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True,
                      download_name=f"attendance_{start_date}_{end_date}.csv")


@app.route("/api/employees")
def api_employees():
    employees = list_employees()
    return jsonify([
        {
            "id": e.employee_id, "name": e.full_name, "code": e.employee_code,
            "department": e.department, "status": status_label_for_employee(e.employee_id),
        }
        for e in employees
    ])


@app.route("/api/employees/<int:employee_id>/deactivate", methods=["POST"])
def api_deactivate_employee(employee_id):
    """
    Soft-delete (keeps historical attendance records intact -- see
    database/employee_repository.deactivate_employee's docstring).
    Also drops them from live recognition immediately, same reasoning
    as the registration-finalize reload below: a deactivated employee
    (e.g. someone who left the company) shouldn't still be recognized
    and able to check in/out on the live camera until a restart.
    """
    deactivate_employee(employee_id)
    with state.lock:
        recognizer = state.face_recognizer
    if recognizer is not None:
        recognizer.reload_known_faces()
    logger.info("Employee deactivated via dashboard: employee_id=%s", employee_id)
    return jsonify({"ok": True})


@app.route("/api/employees/<int:employee_id>/delete", methods=["POST"])
def api_delete_employee(employee_id):
    """
    HARD delete: permanently removes the employee AND -- via the
    ON DELETE CASCADE foreign keys (see database/schema.py) -- all of
    their face embeddings and attendance history. Irreversible, unlike
    deactivate above. The dashboard asks the admin to confirm before
    calling this. The live recognizer is reloaded afterward so the
    removed person stops being recognized immediately, no restart.
    """
    emp = get_employee(employee_id)
    name = emp.full_name if emp else f"ID {employee_id}"
    delete_employee(employee_id)
    with state.lock:
        recognizer = state.face_recognizer
    if recognizer is not None:
        recognizer.reload_known_faces()
    logger.info("Employee PERMANENTLY DELETED via dashboard: %s (id=%s)", name, employee_id)
    return jsonify({"ok": True, "name": name})


@app.route("/api/settings")
def api_get_settings():
    with state.lock:
        cam_label = state.camera_source_label
        cam_error = state.camera_error
    return jsonify({
        "day_boundary_hour": get_day_boundary_hour(),
        "sensitivity": state.sensitivity,
        "camera_source": get_camera_source_raw(),   # "" means the default (browser webcam)
        "camera_is_browser": is_browser_source(get_camera_source()),
        "camera_source_label": cam_label,
        "camera_error": cam_error,
        "min_action_gap_hours": get_min_action_gap_hours(),
        "mirror_preview": get_mirror_preview(),
    })


@app.route("/api/settings/action_gap", methods=["POST"])
def api_set_action_gap():
    """Minimum hours after a CHECK IN before a CHECK OUT is allowed
    (default 8). 0 disables it."""
    data = request.get_json(force=True)
    try:
        hours = float(data.get("hours"))
        set_min_action_gap_hours(hours)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "hours must be a non-negative number"}), 400
    logger.info("Min check-in/out gap changed via dashboard to %.2fh", hours)
    return jsonify({"ok": True, "hours": hours})


@app.route("/api/settings/mirror", methods=["POST"])
def api_set_mirror():
    """Toggle selfie-mirror of the camera image (applies live)."""
    data = request.get_json(force=True)
    enabled = bool(data.get("enabled"))
    set_mirror_preview(enabled)
    with state.lock:
        state.mirror_preview = enabled
    logger.info("Mirror preview set to %s via dashboard", enabled)
    return jsonify({"ok": True, "mirror_preview": enabled})


@app.route("/api/settings/camera", methods=["POST"])
def api_set_camera():
    """
    Point the system at a different camera -- most usefully a phone over
    Wi-Fi via the DroidCam / IP Webcam apps -- by entering its URL, with
    no restart. Blank means the built-in webcam (index 0). The new source
    is persisted AND handed to the live detection loop, which swaps to it
    between frames (see the live-switch block in _detection_loop).
    """
    data = request.get_json(force=True)
    raw = (data.get("source") or "").strip()
    # Accept a plain IP or host for DroidCam and expand it to DroidCam's
    # default MJPEG endpoint, so the admin can just type "192.168.1.7".
    expanded = raw
    if is_browser_source(raw):
        # "Use this browser's webcam": this page becomes the camera; any
        # other page that was sending gets refused and stops.
        expanded = BROWSER_SOURCE
        set_browser_sender(str(data.get("sender_id") or ""))
    else:
        set_browser_sender("")   # switching away: stop accepting browser frames
        if raw and not raw.lower().startswith(("http://", "https://", "rtsp://")) and not raw.isdigit():
            expanded = f"http://{raw}:4747/video"
    set_camera_source(expanded)
    new_source = get_camera_source()  # normalized (int or url)
    with state.lock:
        state.pending_camera_source = new_source
    logger.info("Camera source change requested via dashboard: %r", new_source)
    return jsonify({"ok": True, "camera_source": expanded, "label": _camera_label(new_source)})


@app.route("/api/browser_camera/frame", methods=["POST"])
def api_browser_camera_frame():
    """
    One JPEG frame from the browser that chose "Use this browser's webcam"
    (Settings). Login-protected like every /api/ route. Answers
    {"active": false} when this page is no longer the camera (another
    source or another browser was chosen) so the page stops sending.
    """
    sender_id = (request.headers.get("X-Sender-Id") or "").strip()
    # Saved setting, not CONFIG: it changes the instant Settings is saved,
    # while the detection loop applies the switch a moment later.
    if not is_browser_source(get_camera_source()):
        return jsonify({"active": False})
    body = request.get_data(cache=False)
    if not body or len(body) > 3 * 1024 * 1024:
        return jsonify({"error": "frame missing or too large"}), 400
    try:
        seq = int(request.headers.get("X-Frame-Seq") or -1)
    except ValueError:
        seq = -1
    try:
        accepted = push_browser_frame(sender_id, body, seq)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"active": accepted})


@app.route("/api/settings/day_boundary", methods=["POST"])
def api_set_day_boundary():
    data = request.get_json(force=True)
    try:
        hour = int(data.get("hour"))
        set_day_boundary_hour(hour)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "hour must be an integer between 0 and 23"}), 400
    logger.info("Attendance day-boundary hour changed via dashboard to %d:00", hour)
    return jsonify({"ok": True, "hour": hour})


# --- Browser-based entrance-line calibration -----------------------------
#
# WHY THIS EXISTS: calibration (placing the virtual entrance line onto
# the real doorway in the camera view) was previously ONLY possible via
# main.py's OpenCV window ('C' to drag the line, click to confirm) --
# the one remaining reason anyone had to touch a terminal/second program
# at all. These endpoints move it into the dashboard so the whole system
# is one browser app with no main.py step. The actual crossing math,
# the saved-calibration file format, and the live EntranceGate are all
# REUSED unchanged (EntranceGate.set_line/flip_inside_side/get_calibration
# and line_calibration.save_calibration) -- this only adds a browser
# front door to logic that already existed and was already proven.

@app.route("/api/calibration")
def api_get_calibration():
    """Current entrance line as frame-ratio points (0-1), plus which side
    is 'inside', for the Calibration tab to draw its overlay. Reads the
    live gate if the detection loop has built it yet, else the saved
    file, else the config default."""
    with state.lock:
        gate = state.gate
    if gate is not None:
        p1, p2, inside_is_left = gate.get_calibration()
        return jsonify({"line_start": list(p1), "line_end": list(p2),
                        "inside_is_left_of_line": inside_is_left, "source": "live"})
    saved = load_calibration()
    if saved:
        return jsonify({**saved, "source": "saved"})
    return jsonify({
        "line_start": list(CONFIG.entrance_line.line_start),
        "line_end": list(CONFIG.entrance_line.line_end),
        "inside_is_left_of_line": CONFIG.entrance_line.inside_is_left_of_line,
        "source": "default",
    })


@app.route("/api/calibration/set_line", methods=["POST"])
def api_set_calibration_line():
    """
    Place the entrance line. The browser sends the two endpoints as
    frame RATIOS (0-1), so it never needs to know the actual capture
    resolution -- the same unit the saved file already uses. Applied to
    the LIVE gate immediately (no restart) AND persisted so it survives
    one. Clearing per-person crossing state is EntranceGate.set_line's
    own job, so a moved line can't leave someone stuck mid-crossing.
    """
    data = request.get_json(force=True)
    try:
        sx, sy = float(data["line_start"][0]), float(data["line_start"][1])
        ex, ey = float(data["line_end"][0]), float(data["line_end"][1])
    except (KeyError, TypeError, ValueError, IndexError):
        return jsonify({"error": "line_start and line_end must each be [x, y] ratios between 0 and 1"}), 400
    if not all(0.0 <= v <= 1.0 for v in (sx, sy, ex, ey)):
        return jsonify({"error": "coordinates must be ratios between 0 and 1"}), 400

    with state.lock:
        gate = state.gate
    if gate is None:
        return jsonify({"error": "Camera/detection not ready yet -- try again in a moment."}), 503

    w, h = CONFIG.camera.frame_width, CONFIG.camera.frame_height
    gate.set_line((sx * w, sy * h), (ex * w, ey * h))
    p1, p2, inside_is_left = gate.get_calibration()
    save_calibration(p1, p2, inside_is_left)
    logger.info("Entrance line calibrated via dashboard: start=%s end=%s", p1, p2)
    return jsonify({"ok": True, "line_start": list(p1), "line_end": list(p2),
                    "inside_is_left_of_line": inside_is_left})


@app.route("/api/calibration/flip", methods=["POST"])
def api_flip_calibration_side():
    """Swap which side of the line counts as INSIDE -- used when CHECK IN
    and CHECK OUT come out backwards for a given camera mounting. Live +
    persisted, same as set_line above."""
    with state.lock:
        gate = state.gate
    if gate is None:
        return jsonify({"error": "Camera/detection not ready yet -- try again in a moment."}), 503
    gate.flip_inside_side()
    p1, p2, inside_is_left = gate.get_calibration()
    save_calibration(p1, p2, inside_is_left)
    logger.info("Entrance IN/OUT side flipped via dashboard: inside_is_left=%s", inside_is_left)
    return jsonify({"ok": True, "inside_is_left_of_line": inside_is_left})


# --- Browser-based employee registration ---------------------------------

@app.route("/api/registration/start", methods=["POST"])
def api_registration_start():
    with _registration_sessions_lock:
        _purge_stale_registration_sessions()
        session_id = uuid.uuid4().hex
        _registration_sessions[session_id] = {
            "captures": [],       # list of (angle_label, DetectedFace)
            "embeddings": [],     # parallel list of just the embeddings, for the similarity check
            "turn_signs": [],     # yaw signs already used by non-frontal angles
            "created_at": time.time(),
        }
    angles = list(CONFIG.face_recognition.registration_angles) or DEFAULT_ANGLES
    return jsonify({"session_id": session_id, "angles": angles})


def _evaluate_registration_frame(session: dict, angle_label: str, frame):
    angles = list(CONFIG.face_recognition.registration_angles) or DEFAULT_ANGLES
    frontal_angle = angles[0] if angles else None
    is_frontal = angle_label == frontal_angle
    return evaluate_capture_quality(
        frame, _get_registration_embedder(),
        min_detection_confidence=CONFIG.face_recognition.registration_min_detection_confidence,
        min_face_size_px=CONFIG.face_recognition.registration_min_face_size_px,
        previous_embeddings=session["embeddings"],
        max_similarity_to_previous=CONFIG.face_recognition.registration_max_similarity_to_previous_angle,
        require_frontal=is_frontal,
        require_turned=not is_frontal,
        min_turn_yaw_ratio=CONFIG.face_recognition.registration_min_turn_yaw_ratio,
        max_frontal_yaw_ratio=CONFIG.face_recognition.registration_max_frontal_yaw_ratio,
        forbidden_yaw_signs=session["turn_signs"],
    )


@app.route("/api/registration/check_frame", methods=["POST"])
def api_registration_check_frame():
    """
    Live, non-committing quality check -- called repeatedly (every few
    hundred ms) while the Register tab is open, so the person
    registering gets the same real-time "adjust: move closer" / "hold
    steady" feedback the CLI tool's overlay gives, driven by the exact
    same evaluate_capture_quality() logic. Reads the backend's own live
    camera frame (see _get_live_frame_for_registration) rather than an
    uploaded image -- does NOT accept the capture; see /capture below
    for that.
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    angle_label = data.get("angle_label")
    with _registration_sessions_lock:
        session = _registration_sessions.get(session_id)
    if session is None:
        return jsonify({"error": "Registration session expired or not found -- start again."}), 404

    frame = _get_live_frame_for_registration()
    if frame is None:
        return jsonify({"accepted": False, "reason": "Camera not ready yet -- check the Live Monitor tab for camera errors."})

    result = _evaluate_registration_frame(session, angle_label, frame)
    return jsonify({
        "accepted": result.accepted,
        "reason": result.reason,
        "confidence": result.face.detection_confidence if result.face else None,
        "yaw": result.measured_yaw_ratio,
        "similarity_to_previous": result.similarity_to_previous,
    })


@app.route("/api/registration/capture", methods=["POST"])
def api_registration_capture():
    """
    Commits ONE angle's capture -- re-validates the CURRENT live camera
    frame server-side (never trusts a client-reported "it was ready"),
    exactly mirroring the CLI tool's SPACE-to-capture behavior, which
    also only accepts if evaluate_capture_quality() passes at the
    moment of capture, not at whatever moment the UI last said "ready".
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    angle_label = data.get("angle_label")
    with _registration_sessions_lock:
        session = _registration_sessions.get(session_id)
    if session is None:
        return jsonify({"error": "Registration session expired or not found -- start again."}), 404

    frame = _get_live_frame_for_registration()
    if frame is None:
        return jsonify({"ok": False, "reason": "Camera not ready yet -- check the Live Monitor tab for camera errors."})

    result = _evaluate_registration_frame(session, angle_label, frame)
    if not result.accepted:
        return jsonify({"ok": False, "reason": result.reason})

    angles = list(CONFIG.face_recognition.registration_angles) or DEFAULT_ANGLES
    frontal_angle = angles[0] if angles else None
    with _registration_sessions_lock:
        session["captures"].append((angle_label, result.face))
        session["embeddings"].append(result.face.embedding)
        if angle_label != frontal_angle and result.measured_yaw_ratio is not None:
            session["turn_signs"].append(1 if result.measured_yaw_ratio > 0 else -1)
        captured_angles = [a for a, _ in session["captures"]]

    logger.info("Registration: captured angle '%s' (session %s)", angle_label, session_id[:8])
    return jsonify({"ok": True, "captured_angles": captured_angles})


@app.route("/api/registration/cancel", methods=["POST"])
def api_registration_cancel():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    with _registration_sessions_lock:
        _registration_sessions.pop(session_id, None)
    return jsonify({"ok": True})


@app.route("/api/registration/finalize", methods=["POST"])
def api_registration_finalize():
    """
    Commits a fully-captured session as a real employee in the
    database -- the browser equivalent of register_employee.py reaching
    the end of its angle loop and calling register_new_employee().
    Reloads the LIVE recognizer afterward so the new employee is
    recognized on the live camera immediately, with no restart needed
    (see DashboardState.face_recognizer's docstring for why this is
    safe to do from this request thread).
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    full_name = (data.get("full_name") or "").strip()
    employee_code = (data.get("employee_code") or "").strip() or None
    department = (data.get("department") or "").strip() or None

    with _registration_sessions_lock:
        session = _registration_sessions.get(session_id)
    if session is None:
        return jsonify({"error": "Registration session expired or not found -- start again."}), 404
    if not full_name:
        return jsonify({"error": "Full name is required."}), 400
    if not session["captures"]:
        return jsonify({"error": "No angles captured yet."}), 400

    # DUPLICATE GUARD 1 -- same NAME (case-insensitive). Blocks
    # registering "hammad" when "Hammad" already exists.
    existing_by_name = find_active_employee_by_name(full_name)
    if existing_by_name is not None:
        return jsonify({
            "error": f"Employee already registered: '{existing_by_name.full_name}' "
                     f"(ID {existing_by_name.employee_id}) is already enrolled with that name."
        }), 409

    # DUPLICATE GUARD 2 -- same FACE under a different name. Run each
    # captured angle through the LIVE recognizer (which holds every
    # already-enrolled employee's embeddings; the person being
    # registered is NOT in it yet, so a match means a genuinely
    # different, already-enrolled person). If any angle clearly matches
    # an existing employee, block -- this is what stops the same person
    # enrolling twice under two names. Skipped only if the recognizer
    # isn't up yet (camera/detection still starting), since then there's
    # nothing to compare against.
    with state.lock:
        recognizer = state.face_recognizer
    if recognizer is not None:
        for emb in session["embeddings"]:
            match = recognizer.recognize(emb)
            if match.is_recognized:
                return jsonify({
                    "error": f"Employee already registered: this face is already enrolled as "
                             f"'{match.full_name}' (ID {match.employee_id}). The same person "
                             f"cannot be registered twice under a different name."
                }), 409

    try:
        employee_id = register_new_employee(
            full_name, session["captures"], employee_code=employee_code, department=department,
        )
    except DuplicateEmployeeCodeError as exc:
        return jsonify({"error": str(exc)}), 409

    with _registration_sessions_lock:
        _registration_sessions.pop(session_id, None)

    with state.lock:
        recognizer = state.face_recognizer
    if recognizer is not None:
        recognizer.reload_known_faces()

    logger.info("Employee registered via dashboard: '%s' (id=%d)", full_name, employee_id)
    return jsonify({"ok": True, "employee_id": employee_id})


@app.route("/api/manual_event", methods=["POST"])
def api_manual_event():
    """
    Manual override: log a CHECK IN / CHECK OUT for a selected employee
    without needing a wave -- a real, practical fallback for a case
    wave/face detection doesn't handle (e.g. someone's hands are full,
    or their face isn't recognized well from this angle). Matches the
    "Manual PIN / Visitor Entry / Check In" row in the reference UI
    design -- this is the "Check In"-style manual action; a distinct
    PIN or visitor-badge flow would be a separate, larger feature.
    """
    data = request.get_json(force=True)
    action = data.get("action")
    if action not in ("CHECK IN", "CHECK OUT"):
        return jsonify({"error": "action must be 'CHECK IN' or 'CHECK OUT'"}), 400

    # Validate the employee_id is a real, parseable id (a malformed value
    # used to reach record_event and either fail obscurely or write a
    # dangling row). Cast explicitly rather than trusting the client.
    raw_id = data.get("employee_id")
    try:
        employee_id = int(raw_id)
    except (TypeError, ValueError):
        return jsonify({"error": "employee_id is required and must be a number"}), 400

    # The employee must exist AND still be active -- a manual action for a
    # deleted or deactivated person would create an orphaned/meaningless
    # record (and a HARD-deleted id would violate the foreign key).
    employee = get_employee(employee_id)
    if employee is None:
        return jsonify({"error": f"No employee with id {employee_id}."}), 404
    if not employee.is_active:
        return jsonify({"error": f"{employee.full_name} is deactivated -- reactivate or re-register before logging attendance."}), 409

    # Enforce the SAME once-per-day CHECK IN -> CHECK OUT ordering the
    # automatic wave flow uses, so a manual override can't create an
    # impossible sequence (two check-ins in a row, a check-out before any
    # check-in, or a third action after both are done today). The admin
    # still has full freedom to fix any mistake afterwards from the
    # Records tab (Switch / Delete). The minimum-shift-length gap is
    # deliberately NOT enforced here: a manual override is a considered
    # admin action, not an accidental double-wave, so blocking it on the
    # workday-length timer would defeat the point of having an override.
    last_action = get_todays_last_action(employee_id)
    allowed_next = _VALID_MANUAL_NEXT_ACTION.get(last_action)
    if action != allowed_next:
        if allowed_next is None:
            reason = f"{employee.full_name} has already checked in and out today."
        else:
            reason = f"{employee.full_name}'s next valid action today is {allowed_next}, not {action}."
        return jsonify({"error": reason}), 409

    record_event(employee_id=employee_id, action=action, match_confidence=None, source_track_id=None)
    _invalidate_status_label(employee_id)  # live chip reflects it at once
    logger.info("Manual event recorded via dashboard: employee_id=%s action=%s", employee_id, action)
    return jsonify({"ok": True})


@app.route("/api/sensitivity", methods=["POST"])
def api_set_sensitivity():
    data = request.get_json(force=True)
    level = data.get("level")
    if level not in WAVE_SENSITIVITY_PRESETS:
        return jsonify({"error": f"level must be one of {list(WAVE_SENSITIVITY_PRESETS)}"}), 400
    apply_wave_sensitivity(CONFIG.wave, level)
    with state.lock:
        state.sensitivity = level
    return jsonify({"ok": True, "level": level})


# Runs on import (module load), NOT just "if __name__ == '__main__'" --
# a REAL bug caught and fixed this session (Session 16). A production
# WSGI server (waitress, gunicorn) IMPORTS this module and serves the
# `app` object directly; it never executes the file as __main__. With
# this startup code previously living only inside that guard, running
# `waitress-serve web_dashboard:app` (exactly the command this
# project's own DEPLOYMENT.md recommends for a real, always-on
# deployment) would have started a dashboard with the database never
# initialized and the detection loop thread never started -- no live
# camera, no wave detection, nothing -- while `python web_dashboard.py`
# directly (the dev-server path) would have looked completely fine,
# masking the problem until the exact moment someone actually deployed
# this for real. Moving it here means both paths behave identically.
initialize_database()
_detection_thread = threading.Thread(target=_detection_loop, daemon=True)
_detection_thread.start()
# Separate thread that paints the live video at a steady frame rate from the
# detection loop's latest results -- this is what keeps the feed smooth even
# when detection is slower than the camera (see _display_loop).
_display_thread = threading.Thread(target=_display_loop, daemon=True)
_display_thread.start()

if __name__ == "__main__":
    # threaded=True lets Flask handle the MJPEG stream and JSON API
    # requests concurrently -- without it, one open video stream
    # connection would block every other request. Only used for local
    # development (`python web_dashboard.py` directly) -- a real
    # deployment uses waitress instead (see DEPLOYMENT.md), which
    # serves the `app` object above without ever reaching this block.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
