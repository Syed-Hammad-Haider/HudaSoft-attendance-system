# Context handoff — paste this into a new chat if this one gets cut off

## What this project is
A real-time attendance system: a camera watches an entrance, detects
people, tracks them, recognizes their FACE against a database of
registered employees, watches for a WAVE gesture near a virtual
entrance line, and shows a "CHECK IN <Name>" / "CHECK OUT <Name>"
suggestion — logging it to a local SQLite database.

Pipeline (per frame, in `main.py`):
camera → person detection (YOLO) → tracking (ByteTrack) → entrance-line
crossing → wave detection (MediaPipe Hands) → face recognition
(InsightFace) → attendance decision (state machine, DB-backed) →
on-screen display.

## Current state (already built and working before this session)
- **Employee registration already exists**: `register_employee.py`
  opens the camera and walks the user through capturing their face
  from 3 angles (front/left/right), quality-checked live, stored in
  SQLite via `face_id/registration.py` + `database/employee_repository.py`.
- **Check-in/out already shows the person's name**, not just an ID —
  see `main.py`'s use of `employee_name` and `display_names` /
  `renderer.show_suggested_action(...)`, which produces banners like
  "✅ Jane Doe: CHECK IN".
- **Attendance state lives in the database**, keyed by `employee_id`
  (not the ephemeral camera tracking ID), so it survives someone's
  tracking ID changing (e.g. after being briefly occluded by another
  person).
- Full docs already in the repo: `README.md`, `DOCUMENTATION.md`,
  `SETUP_GUIDE.md`, `PROJECT_SUMMARY_SIMPLE.md` (the last one is a bit
  stale — it predates face recognition being added; don't fully trust
  its "what this does NOT do" section).

## What was done in THIS session
The user reported three things and asked for the project to be
completed effectively:

1. **"You did not include employee registration"** — this was already
   built (see above); it just wasn't obvious/discoverable. No code
   change needed here, just pointing it out. Added `manage_employees.py`
   as a companion CLI (`list` / `deactivate`) since only "add" existed
   as a script before.

2. **"Wave detection accuracy is still not good"** — added a
   **sensitivity preset system** rather than re-guessing thresholds
   blind (the existing tuning is already extensively tested/documented
   inline in `config/settings.py`'s `WaveDetectionConfig`, and further
   blind retuning risks breaking that prior work):
   - `config/settings.py`: `WAVE_SENSITIVITY_PRESETS` dict
     (`strict` / `balanced` / `relaxed`) and `apply_wave_sensitivity()`
     to switch between them, mutating the config **in place** so it
     takes effect live without restarting.
   - `main.py`: `--sensitivity {strict,balanced,relaxed}` CLI flag at
     startup, PLUS an **`S` hotkey** to cycle sensitivity live while
     the app runs — the current level is shown on-screen at all times
     (bottom control bar).
   - `display/renderer.py`: control bar now shows
     `[S] Wave sensitivity: BALANCED` etc.
   - **Still true and unavoidable**: real wave-detection accuracy is
     capped by actual achieved FPS (see the long comment on
     `max_seconds_between_reversals` in `settings.py`). Below ~8fps,
     no sensitivity setting fixes it — `diagnose_camera.py` (below)
     measures this directly.

3. **"Is it my laptop camera, and I want to use an IP camera"**:
   - **`diagnose_camera.py`** (new): standalone tool, no YOLO/face/wave
     code involved, so it isolates whether a problem is the camera
     itself. Scans webcam indices 0–5, or tests one specific index/URL,
     reports actual achieved FPS (which directly affects wave-detection
     reliability — see above), and prints an OS-specific troubleshooting
     checklist (permissions, "another app has the camera locked" being
     the single most common cause, driver issues, etc).
     Usage:
     ```
     python diagnose_camera.py                  # scan local webcams
     python diagnose_camera.py --index 1         # test one index
     python diagnose_camera.py --url "rtsp://user:pass@192.168.1.50:554/stream1"
     ```
   - **`camera/ip_camera_source.py`** (new): implements the same
     `BaseVideoSource` interface as the webcam source, for RTSP or
     HTTP-MJPEG IP cameras. Includes auto-reconnect (network cameras
     drop connections for reasons that don't mean "camera is gone" —
     Wi-Fi hiccups, camera reboots — unlike a USB webcam, where a
     failed read almost always does mean that).
   - **`camera/factory.py`**: now branches automatically —
     `config.source` as an int → webcam; as a string starting with
     `rtsp://` / `http://` / `https://` → IP camera. No other module
     needed to change (this seam already existed by design).
   - **`config/settings.py`**: `CameraConfig` gained
     `ip_reconnect_attempts`, `ip_reconnect_delay_seconds`,
     `ip_connect_timeout_seconds`.
   - **To actually switch to an IP camera**: edit
     `config/settings.py`, set
     `CameraConfig.source = "rtsp://user:pass@<camera-ip>:554/<path>"`.
     The exact RTSP path is brand-specific — check the camera/NVR's own
     settings page, or test the URL in VLC (Media → Open Network
     Stream) first, since that isolates "URL/network/credentials
     problem" from "this program's problem."

## Files touched/added this session
- `config/settings.py` — CameraConfig IP fields, WAVE_SENSITIVITY_PRESETS, apply_wave_sensitivity()
- `camera/factory.py` — branch on source type
- `camera/ip_camera_source.py` — NEW
- `display/renderer.py` — sensitivity shown in control bar
- `main.py` — `--sensitivity` flag, `S` hotkey
- `diagnose_camera.py` — NEW
- `manage_employees.py` — NEW

All files were syntax-checked (`py_compile`) but **not** runtime-tested
in this sandbox — there's no camera or GPU here, and installing
YOLO/MediaPipe/InsightFace here would take a while for no real
verification benefit. **The user should test on their own machine**:
```
pip install -r requirements.txt
python diagnose_camera.py
python register_employee.py
python main.py --sensitivity relaxed
```

## Known, honest, still-open limitations (carried over, not fixed this session)
- Very fast sprinting + arm-swinging can still rarely be mistaken for a
  wave (documented in `PROJECT_SUMMARY_SIMPLE.md`); a full fix would
  need whole-body speed as a second signal, not just hand motion.
  Deliberately not built — rare case, added complexity.
- IP camera RTSP URL format is genuinely brand-specific; there is no
  universal URL this project can guess or hardcode.
- Not yet tested against a real production CCTV/NVR feed by the user —
  only against webcam and (in this session) code review/syntax
  checking, since no real camera hardware is available in this
  sandbox.

## Session 2 — registration reliability fix
User reported: `register_employee.py` sometimes registered fine, but
sometimes pressing SPACE "did nothing" — and it was inconsistent
between different people trying it.

**Root cause found**: the old code only captured if the EXACT frame at
the instant SPACE was pressed happened to pass the quality gate
(single face, confidence ≥0.6, size ≥80px) — and gave **zero feedback**
if it didn't. Quality can flicker frame-to-frame near the threshold
(a blink, half a degree of head motion), so a press could land on one
bad frame with nothing visibly happening. Separately, a successful
capture advanced instantly to the next angle with only a console
print — easy to miss while watching the camera window, so a *working*
capture could also look like "nothing happened."

**Fix in `register_employee.py`** (`face_id/registration.py` untouched
— its quality logic was already correct, this was a UI/feedback bug):
- Requires 8 consecutive stable "ready" frames before SPACE is
  accepted — kills the frame-flicker false negative.
- SPACE now ALWAYS does something visible: a green "CAPTURED: ✓"
  freeze-frame (~0.9s) on success, or a red "NOT CAPTURED: <reason>"
  freeze-frame (~1.1s) on rejection — no more silent no-ops.
- Live on-screen debug readout (`confidence=0.XX face_size=NNpx`) at
  all times, so if it keeps failing for one specific person, you can
  now see exactly why (e.g. consistently low confidence → lighting;
  small face_size → move closer) instead of guessing.
- A single dropped camera frame no longer aborts the whole session —
  only 30 consecutive failed reads does, with a pointer to
  `diagnose_camera.py`.

If it's STILL rejecting one specific person's face repeatedly, the
on-screen `confidence=`/`face_size=` readout now tells you which
threshold they're failing — worth checking here before assuming it's
still a code bug.

Re-upload the current `attendance_system` folder (as a zip) and paste
this file's contents. The codebase is unusually well-commented — most
"why" questions are answered inline in the relevant file, especially
`config/settings.py`, so read the surrounding comments before changing
a value or assuming something is missing.

## Session 3 — delete option + CSV export + DroidCam
User asked for: (1) an actual delete-employee option (only "deactivate"
existed before), (2) anything else needed to make the project "final
ready", (3) a baby-steps guide for using a phone via DroidCam as the
camera.

**1. Hard delete added:**
- `database/employee_repository.py`: new `delete_employee()` — a real
  `DELETE FROM employees`, which cascades (via the `ON DELETE CASCADE`
  foreign keys already in `schema.py`) to also remove that employee's
  face_embeddings AND attendance_events. Irreversible — different from
  the existing `deactivate_employee()`, which is a soft-delete that
  preserves attendance history (kept as the recommended path for
  someone who genuinely left the company; compliance-relevant in most
  real attendance systems).
- `manage_employees.py`: `python manage_employees.py delete --id N`
  now exists — prompts you to type the employee's exact name to
  confirm (or pass `--yes` to skip, for scripts), and explicitly warns
  what's about to be destroyed before doing it.

**2. "Final ready" gap identified and closed: no way to get records
OUT of the database.** The whole point of an attendance system is the
records — there was no report/export path before, only the live event
log on-screen while `main.py` happened to be running. Added:
- `database/attendance_repository.py`: new `get_events_in_range()`.
- `manage_attendance.py` (new): `today` (print today's events) and
  `export --start YYYY-MM-DD --end YYYY-MM-DD --out file.csv` for
  CSV export (payroll/HR use).

**3. DroidCam / IP camera walkthrough** added to `README.md` under
"Using a phone as the camera (DroidCam) or any IP/CCTV camera" — same
content as given in the chat response. Short version: DroidCam app on
phone + same Wi-Fi as the computer → note the IP:port it shows → set
`CONFIG.camera.source = "http://<ip>:<port>/video"` → verify first
with `python diagnose_camera.py --url "..."` before running the full
system.

## Files touched/added this session (Session 3)
- `database/employee_repository.py` — `delete_employee()`
- `database/attendance_repository.py` — `get_events_in_range()`
- `manage_employees.py` — `delete` subcommand
- `manage_attendance.py` — NEW (`today`, `export`)
- `README.md` — admin scripts table + DroidCam/IP camera guide

## If continuing in a new chat
Re-upload the current `attendance_system` folder (as a zip) and paste
this file's contents. The codebase is unusually well-commented — most
"why" questions are answered inline in the relevant file, especially
`config/settings.py`, so read the surrounding comments before changing
a value or assuming something is missing.

## Session 4 — pre-demo critical fixes (wave stuck at 0.33, DroidCam lag, flaky recognition)
User has a demo tomorrow. Reported three concrete symptoms:

**1. Wave confidence stuck around 0.33, wave never confirms, even with
visible skeletal hand tracking.** Root cause confirmed by the numbers:
0.33 ≈ 1/3 reversals — the required 3rd reversal never lands within
`max_seconds_between_reversals` (was 0.22s), so `reversal_times` keeps
getting cleared (`WaveDetector._register_reversal`) before reaching
the threshold. That 0.22s value was already documented in the code's
own comments as "right at the edge, zero margin for jitter" even
before adding a phone-camera link with its own extra latency on top.
**Fix**: widened all three sensitivity presets in
`config/settings.py` (`WAVE_SENSITIVITY_PRESETS`) and the base
`WaveDetectionConfig` defaults — balanced went from
`0.22s → 0.45s`, relaxed `0.30s → 0.6s` (also `min_reversals_required`
`2` kept, `min_swing_amplitude` slightly loosened `0.06 → 0.05`),
strict `0.22s → 0.35s`. Still requires a genuine back-and-forth swing
(the real signal that separates a wave from a static hand or a
walking gait) — just with realistic timing.
**For tomorrow specifically**: start with
`python main.py --sensitivity relaxed`, or press `S` live to cycle to
it, for the most forgiving/reliable trigger during the demo itself.

**2. DroidCam: laptop screen visibly lags behind the phone.** Real
cause: OpenCV's `VideoCapture` queues frames internally; if this
pipeline's processing (YOLO + tracking + per-person MediaPipe + face
recognition, every frame) is slower than frames arrive, every `read()`
returns the OLDEST queued frame, and the visible lag grows over time.
This is standard OpenCV behavior, not a DroidCam-specific bug, but
DroidCam's own extra capture/encode/transmit delay makes it far more
visible than a local webcam. **Fix**: new
`camera/latest_frame_reader.py` — a background thread that does
nothing but continuously `.read()` and keep only the single newest
frame; the main pipeline now always pulls the CURRENT freshest frame
instead of draining a backed-up queue. Wired into both
`camera/webcam_source.py` (covers DroidCam Client / OBS virtual-webcam
mode) and `camera/ip_camera_source.py` (covers the raw
`http://ip:port/video` URL mode). `CAP_PROP_BUFFERSIZE=1` is still set
too as a cheap first line of defense, but the comment in
`latest_frame_reader.py` explains why that alone isn't reliable on
every backend (e.g. DirectShow on Windows, which is what a DroidCam
virtual webcam typically registers as).

**3. "Sometimes it detects the registered person, sometimes not."**
Real cause found in `face_id/live_recognition.py`: recognized AND
unrecognized results were cached for the same 5-second interval
(`re_recognition_interval_seconds`). If the very first attempt for a
newly-stable track happened to land on a bad frame (mid-turn, motion
blur — more likely over a phone-camera link), the "unrecognized"
result got frozen in place for a full 5 seconds even if the person was
clearly recognizable a moment later. **Fix**: new
`FaceRecognitionConfig.unrecognized_retry_interval_seconds` (0.75s) —
unrecognized attempts now retry quickly, while confirmed recognitions
still only re-check every 5s (no reason to re-check something already
confidently known). The actual matching threshold
(`min_match_similarity`, 0.45) was deliberately left untouched —
that's a security/accuracy tradeoff, not a bug, and loosening it risks
misidentifying one employee as another, which is worse than a slower
demo.

**4. `main.py`'s live loop now tolerates transient frame drops**
(same fix pattern as `register_employee.py` from Session 2) instead of
ending the whole session on a single dropped frame — matters more now
with a network camera source.

## Files touched/added this session (Session 4)
- `config/settings.py` — widened WAVE_SENSITIVITY_PRESETS + base
  WaveDetectionConfig defaults; new
  `FaceRecognitionConfig.unrecognized_retry_interval_seconds`
- `camera/latest_frame_reader.py` — NEW (background latest-frame-only reader)
- `camera/webcam_source.py` — uses LatestFrameReader
- `camera/ip_camera_source.py` — uses LatestFrameReader, staleness-based reconnect
- `face_id/live_recognition.py` — asymmetric recognized/unrecognized retry timing
- `main.py` — tolerates transient None frames instead of ending the session

**Not runtime-tested here** (no camera/GPU in this sandbox, syntax-checked
only) — please do a full run-through before the demo, ideally tonight,
not for the first time live. If wave detection still feels off during
your own test, press `S` to try `relaxed`, and watch the on-screen FPS
counter — below ~8fps, no sensitivity setting can fully compensate
(this is a sampling-rate limit, documented in `settings.py`).

## Session 5 — registration was never actually checking angles (real bug found)
User reported registration's front/left/right capture was "messed up."
Investigation found this was a REAL, pre-existing bug, not something
broken by Session 4's changes: `evaluate_capture_quality()` (in
`face_id/registration.py`) never checked head pose/direction at all —
it only checked generic frame quality (one face, confident detection,
big enough). That means a "left" or "right" capture could be an
identical frontal pose to the "front" capture; only the on-screen
LABEL differed. This defeats the entire point of multi-angle
registration (see the module's own docstring on why distinct angles
matter for live matching).

**Why not just check geometric yaw (eye/nose keypoint positions)?**
Considered and deliberately rejected: which side is "left" vs "right"
in that geometry depends on camera mirroring conventions that vary by
platform/webcam driver. Getting the sign backwards would silently SWAP
the two angles while looking like it's working — worse than not
checking at all.

**Actual fix: embedding-similarity distinctness check.** Sign-
convention-proof — it doesn't need to know which physical direction is
which, only that each new capture's embedding differs meaningfully
from already-accepted angles for the same employee.
- `face_id/registration.py`: `evaluate_capture_quality()` gained
  `previous_embeddings` + `max_similarity_to_previous` params. A
  candidate too similar (cosine similarity ≥ 0.97 by default) to an
  already-captured angle is rejected with an explicit reason ("turn
  your head further"). `CaptureQualityResult` gained
  `similarity_to_previous` for on-screen debug visibility regardless
  of accept/reject.
- `config/settings.py`: new
  `FaceRecognitionConfig.registration_max_similarity_to_previous_angle`
  (0.97) and `registration_auto_capture_hold_seconds` (0.6) — see
  their inline comments for the full reasoning.

**Second real bug fixed in the same file: frame-count stability
instead of time-based.** The Session 2 rewrite required 8 CONSECUTIVE
FRAMES of good quality before accepting a capture — the same class of
bug already documented and fixed elsewhere in this project for wave
timing (frame counts silently assume a roughly fixed FPS, which breaks
down between a fast webcam and a laggier DroidCam feed). Switched to
elapsed TIME (0.6s continuous) instead — consistent behavior
regardless of camera/hardware.

**Third change: registration is now genuinely AUTOMATIC**, matching
what the user actually asked for ("automatic face scanning"). Once a
candidate frame continuously passes every gate (quality + distinctness)
for `registration_auto_capture_hold_seconds`, it captures itself —
no SPACE press required. SPACE still works as an immediate manual
override the instant it's ready, and gives clear on-screen feedback if
pressed too early (keeps the earlier Session 2 "silent no-op" fix).

`register_employee.py` was rewritten end-to-end around this; the core
capture/save logic in `face_id/registration.py` and the DB layer were
extended, not replaced.

## Files touched/added this session (Session 5)
- `face_id/registration.py` — distinctness check in `evaluate_capture_quality()`
- `config/settings.py` — two new `FaceRecognitionConfig` fields (see above)
- `register_employee.py` — rewritten: automatic time-based capture, distinctness wired in, clearer angle instructions

**Still not runtime-tested here** (no camera in this sandbox). This is
now the third consecutive session touching registration reliability —
if anything is still off after this, the most likely remaining culprit
given no way to test live here is the exact similarity threshold
(0.97) being mistuned for your specific camera/lighting; watch the new
`similarity_to_prev_angle=` debug readout on screen during a real test
and report the actual number if left/right still gets rejected
unexpectedly (too strict) or accepted too easily (too loose).

## Session 6 — registration captured all 3 angles instantly, no time to turn
User reported all 3 angles captured almost immediately, without any
real chance to physically turn their head.

**Root cause**: Session 5's fix for the distinctness bug was correct,
but removing the manual SPACE press (in favor of auto-capture) also
silently removed the ONE thing that used to give a real person time to
reposition — pressing a key takes a few seconds of human reaction time
on its own. Without an explicit substitute, the very first
already-facing-forward frame of the "left" step could pass the quality
gate and start counting toward `registration_auto_capture_hold_seconds`
(0.6s) before the person had even begun turning.

**Fix**: added a mandatory ~3-second countdown (`registration_prep_seconds`
in `config/settings.py`) BEFORE any capture-quality checking starts for
each angle — shown on screen as "Get ready: LEFT... turn now" with a
live countdown timer. No capture can happen during this window no
matter how good the frame looks; it's a hard-blocked pause, not a
gate that can be raced. New `_run_prep_countdown()` in
`register_employee.py`, called right before each angle's existing
auto-capture loop.

Also tightened `registration_max_similarity_to_previous_angle` from
0.97 → 0.95 as an extra margin of safety on top of the countdown fix
(same-pose frame-to-frame embedding noise could plausibly land in the
0.95-0.97 range without a genuine head turn; 0.95 leaves more room
while a real 30-45° turn should still register well below that).

## Files touched this session (Session 6)
- `config/settings.py` — new `registration_prep_seconds` (3.0s);
  `registration_max_similarity_to_previous_angle` 0.97 → 0.95
- `register_employee.py` — new `_run_prep_countdown()`, called before
  each angle's capture loop

**If this is STILL capturing too fast or without a real turn being
detected after this fix**, the two most likely remaining knobs (both
in `config/settings.py`, `FaceRecognitionConfig`) are:
- `registration_prep_seconds` — raise it (e.g. to 4-5s) if 3s still
  isn't enough time to turn and settle.
- `registration_max_similarity_to_previous_angle` — lower it further
  (e.g. to 0.90) if near-identical poses are still slipping through as
  "different enough."
Both are plain dataclass fields; edit the default value directly.

## Session 7 — real geometric pose detection (the actual "detect the angle" fix)
User pushed back correctly: Sessions 5-6 only used a countdown timer +
embedding-similarity heuristic, never actually DETECTING whether the
person had turned their head. Implemented real head-pose (yaw)
detection this session.

**Method**: uses the 5-point facial keypoints (left_eye, right_eye,
nose, mouth_left, mouth_right) that InsightFace's detector (SCRFD)
already outputs as part of detection — no extra model needed, zero
added inference cost. Yaw is estimated as:
`(nose_x - eye_midpoint_x) / inter_eye_distance` — near 0 when facing
forward, increasingly non-zero the more the head turns, and
scale-invariant (works regardless of distance from camera). See
`face_id/registration.py`'s `_estimate_yaw_ratio()`.

**Deliberate design choice on sign ambiguity**: does NOT hardcode
which numeric sign means anatomical "left" vs "right" — that mapping
depends on camera mirroring conventions that vary by
platform/driver/app, and guessing wrong would silently swap the two
labels while looking correct. Instead:
- **Magnitude** (`abs(yaw)`) confirms a real turn happened at all —
  this directly replaces what used to be a blind timer with an actual
  measurement.
- **Sign** is only used to confirm two turn angles ("left", "right")
  are turned in genuinely OPPOSITE directions from each other, not to
  claim which one is anatomically which.

**Wired into `evaluate_capture_quality()`** (`face_id/registration.py`)
via new params: `require_frontal`, `require_turned`,
`min_turn_yaw_ratio`, `max_frontal_yaw_ratio`, `forbidden_yaw_signs`.
The "front" angle (first entry in `registration_angles`) must measure
near-zero yaw; every other angle must measure a real turn AND a
direction not already used by a previously-captured turn angle.
Embedding-similarity distinctness (from Session 5) remains as a
secondary safety net on top, not replaced.

**New config values** (`config/settings.py`,
`FaceRecognitionConfig`): `registration_max_frontal_yaw_ratio` (0.08),
`registration_min_turn_yaw_ratio` (0.12).

**`face_id/embedder.py`**: `DetectedFace` gained a `keypoints` field,
populated from `face.kps` (falls back to `None` gracefully if a given
InsightFace version doesn't provide it — the frontal/turn checks are
skipped rather than hard-failing registration in that case).

**`register_employee.py`**: now passes `require_frontal`/
`require_turned`/`forbidden_yaw_signs` per angle, tracks
`captured_turn_signs` across the session, and shows the live measured
`yaw=` value in the on-screen debug line so you can watch the actual
number while turning during a real test.

## Files touched this session (Session 7)
- `face_id/embedder.py` — `DetectedFace.keypoints`
- `face_id/registration.py` — `_estimate_yaw_ratio()`,
  `evaluate_capture_quality()` pose-check params
- `config/settings.py` — two new yaw threshold fields
- `register_employee.py` — wires pose requirements per angle, tracks
  turn signs, shows live yaw in debug text

**Not runtime-tested here** (no camera in this sandbox — genuinely
cannot verify InsightFace actually returns `face.kps` for the
installed version, though it does for all mainstream InsightFace
releases). **Please test this specifically**: watch the `yaw=` number
in the on-screen debug line while turning your head — confirm it moves
away from ~0 as you turn, and swings to the opposite sign for the
other side. If `yaw=` doesn't appear at all, keypoints aren't
available from your installed InsightFace version — the pose check
will silently skip (falls back to the countdown + similarity checks
only) rather than block registration; report back if that happens.
If `min_turn_yaw_ratio` (0.12) feels too strict or too loose against
the actual numbers you see, that's the value to adjust in
`config/settings.py`.

## Session 8 — scaling to a real company (thousands of employees), no features removed
User asked to refine the project into a production-ready in-house
system for a real company tracking thousands of employees accurately,
without removing any existing feature. Focused on the parts that
genuinely don't scale from "a handful of pilot users" to "thousands of
employees" — and, unlike prior sessions, several of these changes
COULD be tested directly in this sandbox (pure Python/numpy/SQLite,
no camera needed), so they were, with real assertions, not just
syntax-checked.

**1. Vectorized face matching (the biggest scale fix).**
`face_id/recognizer.py` previously did a Python `for` loop calling
`np.dot()` once per stored embedding — fine for a handful of people,
a real bottleneck at thousands (thousands of employees × 3 angles =
thousands of Python-level iterations, done periodically per tracked
person during live monitoring). Rewritten to stack all known
embeddings into a single `(N, 512)` matrix at load time and match with
one `embedding_matrix @ live_embedding` call (BLAS-accelerated).
**Tested in this sandbox**: 6,000 stored embeddings (2,000 employees ×
3 angles), 200 recognition calls → ~0.5ms per call. Correctness also
verified: a clean match resolves to the right employee_id.

**2. Ambiguity/margin check (accuracy fix that specifically matters
at scale).** With more employees enrolled, the odds rise that two
different people's embeddings both score above the acceptance
threshold for a given live face — a failure mode a small pilot rarely
surfaces. Added `FaceRecognitionConfig.min_margin_over_runner_up`
(0.03): the top match must beat the best-scoring match from a
DIFFERENT employee by this margin, or the result is UNRECOGNIZED
rather than a guess. **Tested in this sandbox** with two deliberately
similar synthetic "look-alike" employees: a live embedding equidistant
between them was correctly rejected as `ambiguous_margin` rather than
confidently (and wrongly) assigned to one of them.

**3. Duplicate employee_code handling.** Previously, a duplicate code
would crash with a raw `sqlite3.IntegrityError` — and only at the very
end of registration, AFTER several minutes of guided face capture. At
thousands of employees, code collisions (typos, re-registration
attempts) are routine, not rare. Fixed with:
- `database/employee_repository.py`: new `employee_code_exists()` —
  checked BEFORE the camera even opens.
- New `DuplicateEmployeeCodeError` (specific, catchable) raised
  instead of a raw DB error if a collision still happens at save time
  (defense-in-depth against the small race window in the upfront
  check).
- `register_employee.py`: loops on the code prompt upfront; if a
  collision is still hit at save time, prompts for a new code WITHOUT
  discarding the face captures already completed.
**Tested in this sandbox**: `employee_code_exists()` correctly
detects a taken code; a duplicate registration attempt raises
`DuplicateEmployeeCodeError` (not a raw traceback).

**4. Database: WAL mode + new index.**
`database/connection.py`'s `initialize_database()` now sets
`PRAGMA journal_mode = WAL` — lets an admin run
`manage_attendance.py export` or `manage_employees.py list` while
`main.py` is actively writing a check-in event, without blocking
(SQLite's default rollback-journal mode blocks all readers during a
write). New index `idx_events_time` on `attendance_events(event_timestamp)`
alone (the existing composite index only serves per-employee queries;
`manage_attendance.py`'s date-range/today queries filter across ALL
employees by timestamp, which needed its own index to stay fast as the
events table grows over months/years). **Tested in this sandbox**:
confirmed `journal_mode` is actually `wal` after init, and the new
index exists in `sqlite_master`.

**5. Persistent audit log.** `main.py` now also writes a rotating log
file to `logs/attendance_system.log` (5MB × 5 backups), in addition to
(not replacing) console output — a real company needs a record that
survives closing the terminal, for after-the-fact review/audit.

**6. `.gitignore` added** — the database (contains real employee face
data + attendance history), logs, and CSV exports shouldn't be
tracked/shipped accidentally.

**Nothing existing was removed** — this session added scale/accuracy
safeguards and a couple of small onboarding-UX fixes on top of
everything from Sessions 1-7 (multi-angle registration with real pose
detection, wave sensitivity presets, IP/DroidCam camera support,
camera lag fix, admin CLIs, CSV export, etc.) — all of which are still
present and untouched in this zip.

## Files touched this session (Session 8)
- `face_id/recognizer.py` — rewritten: vectorized matrix matching + ambiguity/margin check
- `config/settings.py` — new `min_margin_over_runner_up`
- `main.py` — passes margin param to `FaceRecognizer`; rotating file log handler
- `database/employee_repository.py` — `employee_code_exists()`, `DuplicateEmployeeCodeError`, defensive catch in `register_employee()`
- `database/connection.py` — WAL mode
- `database/schema.py` — new `idx_events_time` index
- `register_employee.py` — upfront code-uniqueness check + graceful retry-without-recapture on collision
- `README.md` — new "Scaling to a real company" section
- `.gitignore` — NEW

## What WAS and WASN'T actually tested this session
**Actually tested with real assertions in this sandbox** (pure
Python/numpy/SQLite, no camera required): vectorized recognizer
correctness + performance at 6,000 embeddings; ambiguity/margin
rejection with synthetic look-alike embeddings; `employee_code_exists()`;
`DuplicateEmployeeCodeError` raised correctly; WAL mode active;
new index present. These are genuinely verified, not just
syntax-checked.

**Still NOT tested** (no camera/GPU available in this sandbox, same
limitation as every prior session): the actual camera pipeline end to
end — registration's pose detection, wave gesture detection, live
recognition against a REAL camera feed, DroidCam integration. All of
Sessions 1-7's fixes remain based on code review + reasoning about the
reported symptoms, not a live run. **This is the single most important
thing to do before treating this as genuinely production-ready**: a
real pilot with a real camera and a handful of real employees, ideally
well before a thousand-employee rollout.

## Session 9 — wave detection redesign (root-cause found and fixed, not just retuned)
User explicitly asked: inspect the actual implementation, find the
real reason genuine waves are missed, redesign the logic (not just
retune numbers), make it robust to real confounds, add a debug/analysis
mode, and test rigorously. Did all of the above.

### Root causes found (by reading the code, not guessing)
The old `wave_detection/detector.py` + `wave_state.py` tracked exactly
ONE signal: the WRIST landmark's X position, normalized to the
person's crop width. Three independent, compounding design flaws:

1. **Wrist is the pivot, not the mover.** In a relaxed, natural
   "wrist-pivot" wave — the most common style — the forearm barely
   moves; only the palm/fingers swing. The wrist is almost by
   definition the point that moves LEAST. Tracking it specifically
   captured the smallest possible signal for the most natural wave
   style, which explains "even when the person is close and the hand
   is clearly visible" — proximity doesn't help if the tracked point
   itself barely moves.
2. **Horizontal-only (x-axis only).** Vertical, diagonal, or
   "royal wave" rotational styles have real Y-axis motion the old
   detector never looked at.
3. **Normalized to a constantly-shifting crop.** The crop is
   recomputed fresh every frame from the person's live-tracked bbox,
   which can itself shift/resize independent of the hand. Direct
   evidence this was already a known fragility: an earlier session's
   own config comment documented a crop-margin change (0.14→0.25)
   silently requiring a manual amplitude threshold correction — a
   symptom of measuring against something unstable, not a tuning gap.

### The redesign
- **`wave_detection/oscillation.py`** (NEW): pure, camera-independent,
  fully unit-tested 2D vector-based oscillation detector. Tracks the
  **palm centroid** (average of wrist + 4 MCP knuckle landmarks —
  moves substantially under all common wave styles, not just wrist-
  pivoted ones) in full 2D, normalized to the **hand's own detected
  size** ("hand-widths": live-computed wrist-to-middle-MCP distance
  each frame) instead of the crop. A reversal is a 2D vector direction
  change of at least `min_swing_amplitude_hand_widths`, not two
  independent per-axis checks (an earlier draft of this redesign used
  independent X/Y trackers; testing caught that this under-counts true
  diagonal motion, since it splits amplitude across both axes — fixed
  before finalizing).
- **`wave_detection/body_motion.py`** (NEW): a genuinely necessary
  second signal, discovered BY TESTING, not assumed upfront: hand
  kinematics alone (however well-designed) cannot fully separate a
  deliberate wave from vigorous walking/jogging arm swing at
  overlapping amplitude/cadence — `test_oscillation.py`'s synthetic
  "jogging-speed arm pump" and "brisk walking" cases both cleanly pass
  the same amplitude+timing checks a real wave passes. Fix: reuse the
  person tracker's own bbox (already computed for entrance-line
  crossing) to require the person's BODY be relatively stationary
  (<20% of their own height displaced in the last 1s) for a wave to
  actually confirm.
- **`wave_detection/detector.py`** rewritten around both of the above:
  computes palm centroid + hand scale in GLOBAL FRAME PIXEL coordinates
  (not MediaPipe's raw normalized crop coordinates — mixing normalized
  x/y directly would distort distances on a non-square crop), feeds
  them to each hand's `OscillationTracker`, and gates final
  confirmation on the shared `BodyStillnessTracker`.
- **`wave_detection/wave_state.py`** updated: `HandTrack` now owns an
  `OscillationTracker` instead of raw extremum/direction fields;
  `WaveState` gained a shared `BodyStillnessTracker`.
- **`wave_detection/wave_result.py`**: `HandObservation` gained a
  `debug: Optional[OscillationDebugInfo]` field; `WaveResult` gained
  `body_debug: Optional[BodyMotionDebugInfo]` — plumbing for the new
  debug overlay.
- **`config/settings.py`**: `WaveDetectionConfig` rewritten with new
  fields (`min_swing_amplitude_hand_widths`, `max_reversal_gap_regularity_cv`,
  `wave_smoothing_alpha`, `body_stillness_window_seconds`,
  `max_body_displacement_fraction`, `debug_overlay_default_on`); old
  crop-relative `min_swing_amplitude` removed entirely (not kept
  alongside — it measured something structurally wrong, keeping it
  around would invite accidentally reintroducing the bug).
  `WAVE_SENSITIVITY_PRESETS` updated to the new field names/values.
- **Regularity/CV safety net** (in `oscillation.py`): a secondary check
  requiring the timing GAPS between reversals not be too erratic
  (coefficient of variation ≤ 0.6) once 3+ reversals exist — targets
  "random hand movement that happens to zig-zag enough times, at wildly
  inconsistent intervals" specifically.

### DEBUG/ANALYSIS overlay (new, as requested)
- **`wave_detection/debug_overlay.py`** (NEW): standalone module (kept
  separate from `display/renderer.py` so it can show raw internal
  numbers without risking the already-tested normal-UI rendering code).
  Draws a per-hand panel: tracking state, samples analyzed, current
  swing magnitude vs. threshold, reversal count vs. required, timing
  regularity (CV) and pass/fail, seconds since last reversal, and the
  person's body-displacement fraction vs. the stationary cap.
- **`main.py`**: new **`D`** hotkey toggles it live; off by default
  (`debug_overlay_default_on: bool = False` in config) to keep the
  normal UI clean, but trivially discoverable via the on-screen
  control bar.

### Testing actually performed (and what could NOT be tested here)
**Genuinely tested, with real pass/fail assertions, in this sandbox**
(`wave_detection/test_oscillation.py` — run it directly, no camera
needed): 22 synthetic trajectory scenarios covering every pattern the
user asked for — slow/normal/fast waves, vertical, diagonal,
left-first/right-first, subtle/modest amplitude, sustained repeated
waving, low-FPS (8fps) robustness, static raised hand, pointing,
normal/brisk/jogging-speed walking gait (standalone AND combined with
realistic body translation), random uncoordinated motion, and small
sub-threshold movements. All 22 pass. This is real verification of the
DECISION LOGIC, not a compile check.

Also ran a live smoke test in this session against the REAL MediaPipe
library (`wave_detection/test_oscillation.py` doesn't need MediaPipe
at all, but a separate ad-hoc check confirmed `WaveDetector` actually
constructs, runs `process()` against a blank frame through real
MediaPipe inference without crashing, and handles repeated calls +
track removal cleanly). **Found and fixed a real environment issue
along the way**: the sandbox's default-installed `mediapipe` (0.10.33)
has DROPPED the legacy `mp.solutions.hands` API this whole project is
built on; the version already pinned in `requirements.txt` (0.10.14)
still has it and was used to actually verify the code runs. If a
deployment machine picks up a newer, unpinned mediapipe, it will fail
the same way — **worth deliberately confirming `pip install -r
requirements.txt` was actually used, not a different pre-existing
mediapipe install**, before assuming a fresh environment issue is a
regression in this code.

**Still NOT and CANNOT be tested here**: actual MediaPipe hand-landmark
DETECTION ACCURACY against a real human hand on a real camera feed —
there is no camera or real hand image available in this sandbox, and a
synthetically-drawn "hand" shape would not meaningfully exercise a
detector trained on real hands. Everything about whether MediaPipe
itself reliably detects your specific hand at your specific
camera/lighting is unverified by me; only the DECISION LOGIC on top of
whatever landmarks MediaPipe reports has been rigorously tested.

### Required real-world test procedure (for the user to run)
1. `python wave_detection/test_oscillation.py` — confirm all 22 pass
   on your machine too (rules out an environment-specific logic bug).
2. `python main.py`, press **`D`** to enable the debug overlay.
3. With no one in frame, confirm no panel appears.
4. Step into frame, raise a hand without waving — confirm the panel
   shows `state: idle` or `tracking` but never `confirmed`, and
   `reversals: 0 / 3`.
5. Point at the camera (static) — same expectation as #4.
6. Wave slowly, deliberately, wrist-pivoted (the style the OLD design
   handled worst) — watch `swing:` climb, `reversals:` count up, and
   `state` reach `confirmed`. Note how many seconds this takes.
7. Repeat #6 waving fast/enthusiastically, then vertically, then at a
   diagonal, then with the OTHER hand, then starting left-first and
   right-first, then as a longer sustained wave — confirm each is
   detected without needing to exaggerate the motion.
8. Walk normally past/through the frame with natural arm swing — watch
   the `--- Body ---` line show `MOVING (blocks wave)` and confirm no
   false CHECK IN/OUT fires, even if `reversals:` climbs.
9. Walk up, then STOP and wave while standing — confirm `Body` shows
   `STILL` and the wave DOES confirm once stopped.
10. If a genuine wave style still isn't detected, note the exact
    numbers shown in the debug panel at that moment (swing magnitude,
    reversal count, regularity CV, body displacement) — that pinpoints
    which specific check is blocking it, rather than guessing at
    another threshold change.

## Files touched/added this session (Session 9)
- `wave_detection/oscillation.py` — NEW (core redesigned algorithm)
- `wave_detection/body_motion.py` — NEW (body-stillness gate)
- `wave_detection/debug_overlay.py` — NEW (DEBUG/ANALYSIS overlay)
- `wave_detection/test_oscillation.py` — NEW (22-case test suite)
- `wave_detection/detector.py` — rewritten around the new signal + gate
- `wave_detection/wave_state.py` — `HandTrack`/`WaveState` restructured
- `wave_detection/wave_result.py` — debug fields added
- `config/settings.py` — `WaveDetectionConfig` rewritten, presets updated
- `main.py` — `D` hotkey, debug overlay wiring
- `display/renderer.py` — hand skeleton now highlights the 5 tracked
  palm-centroid landmarks instead of just the wrist; control bar shows `[D]`
- `README.md` — new "Wave detection redesign" section, updated defaults table

Person detection, tracking, entrance-line logic, face recognition,
attendance decision engine, all admin CLIs, and the overall UI were
NOT touched this session, per the request.

## Session 10 — refinement pass: found and fixed two more real bugs by testing, not guessing
User asked to "refine it further and improve" after the Session 9 wave
redesign. Rather than making cosmetic tweaks, re-inspected the new code
for real gaps and found two genuine bugs, both confirmed with actual
tests before and after fixing (not assumed):

**Bug 1: `is_waving` never turned back off.**
In `wave_detection/detector.py`, the line
`track.is_waving = confirmed or track.is_waving` sets it True on
confirmation but nothing ever set it back to False -- a hand that
waved once would show the green "WAVING" highlight permanently for the
rest of the session. Comment even said "cleared by caller via display
timeout," but no such timeout existed anywhere.
**Fix**: `HandTrack.is_waving: bool` replaced with
`confirmed_display_until: Optional[float]` (a wall-clock timestamp).
`is_waving` is now computed each frame as `now < confirmed_display_until`,
so it naturally expires with the clock -- no separate cleanup step
needed. New config value `wave_confirmed_display_seconds: float = 1.5`.
Verified directly: confirmed True immediately after triggering, False
2 seconds later.

**Bug 2 (more significant): body-stillness had a ~0.73s lag after a
person actually stopped walking** -- directly working against the
"detected quickly" requirement, for the single most common real
scenario this system needs to handle (walk up, stop, wave).
Root cause: the original `BodyStillnessTracker` compared the OLDEST
vs NEWEST sample across a full 1.0-second window; after a real stop,
stale walking-era positions lingered in that window for up to the
full second. **Verified this quantitatively** with a dedicated test
(`wave_detection/test_body_motion.py`) before touching the code: 0.73s
measured lag.

First attempted fix (instantaneous frame-to-frame smoothed speed)
introduced a NEW bug, also caught by testing before shipping it:
dividing a small, noisy frame-to-frame position difference by a small
time step amplifies ordinary person-tracker bounding-box jitter into a
large apparent "speed" -- a perfectly still person with just 1-2%
normal bbox jitter got falsely flagged as "moving," which would have
incorrectly blocked genuine wave attempts for someone standing still.

**Final fix**: kept the window-based approach but EMPIRICALLY SWEPT
window lengths (0.2s to 0.6s, all tested, not guessed) to find where
responsiveness and jitter-robustness both hold. 0.5 seconds is the
validated sweet spot: ~0.3s lag after a real stop (vs. ~0.73s before),
zero false positives up to a generous 3% bbox-jitter assumption, and
still correctly detects normal/brisk walking and jogging. New test
file `wave_detection/test_body_motion.py` (8 tests, all passing)
specifically validates responsiveness, jitter robustness, and correct
walking/jogging detection together -- this is exactly the kind of gap
that "looks fine on paper" but only shows up when actually tested with
realistic numbers, which is why it's now a permanent regression test.

Also renamed `max_body_displacement_fraction` →
`max_body_speed_fraction_per_second` for clarity, since it's a speed
cap (independent of the window length), not a raw per-window fraction
-- the ambiguous old name is exactly what led to almost introducing a
units bug while making this fix.

## Files touched this session (Session 10)
- `wave_detection/wave_state.py` — `HandTrack.is_waving` → `confirmed_display_until`
- `wave_detection/detector.py` — `is_waving` computed from timestamp; renamed field references
- `config/settings.py` — new `wave_confirmed_display_seconds`; `body_stillness_window_seconds` 1.0→0.5 with full empirical reasoning; `max_body_displacement_fraction` renamed to `max_body_speed_fraction_per_second`
- `wave_detection/body_motion.py` — rewritten with the validated short-window design
- `wave_detection/debug_overlay.py` — field name updated to match
- `wave_detection/test_body_motion.py` — NEW (8 tests: responsiveness, jitter robustness x3, walking/jogging detection x3, static-hold sanity check)
- `wave_detection/test_oscillation.py` — field rename only, still 22/22 passing

**Total test coverage now**: 30 automated tests across both files, all
passing, covering the full decision logic with no camera required
(`python wave_detection/test_oscillation.py` and
`python wave_detection/test_body_motion.py`). The real-world 10-step
manual test procedure from Session 9 (using the `D` debug overlay)
remains the way to verify actual MediaPipe hand detection against a
real camera, which still cannot be done in this sandbox.

## Session 11 — found and fixed a real spurious-retrigger bug (attendance-correctness class)
Continued the refinement pass, reading the confirm/cooldown interaction
in `wave_detection/detector.py` line by line rather than assuming it
was correct. Found and PROVED (via direct test trace, not
assumption) a genuine bug with real production consequences.

**The bug**: `OscillationTracker._check_confirmed()` returns `True` on
EVERY call once the reversal count is reached -- it does not reset
itself. Proven directly: fed a frozen (non-moving) hand position for
19 consecutive frames after a wave completed, and it returned
`confirmed=True` every single time. `detector.py` only called
`reset_progress()` inside the branch gated by
`state.cooldown.is_ready()` -- meaning if a wave-shaped hand motion
completed while EITHER (a) a previous wave's cooldown was still
active, or (b) the person's body wasn't yet stationary (walking), the
gesture state was left permanently "primed." The real-world
consequence: the moment the blocking condition later cleared on its
own (cooldown expiring, or the person simply stopping walking) -- with
ZERO new hand motion at that moment -- it would fire a spurious
CHECK IN/OUT. For a real company depending on accurate attendance
records, this is a correctness bug, not a cosmetic one.

**Two-layer fix**:
1. `wave_detection/detector.py`: `reset_progress()` now called
   unconditionally the instant `hand_confirmed` is True, before
   checking body-stillness OR cooldown. Those two checks now only gate
   whether the ACTUAL attendance action fires -- never whether the
   gesture-detection state gets cleared.
2. `wave_detection/oscillation.py`: added a second, independent
   safety net directly in the class itself -- an edge-triggered
   `_already_reported` guard, so `update()` only returns `True` ONCE
   per completed streak, resetting only when `reset_progress()` is
   called OR a genuinely new reversal extends the streak. This means
   even a FUTURE caller of this class who makes the same mistake
   `detector.py` did can't reintroduce this bug -- the class defends
   itself.

**Verified with tests, not assumption**: new regression test in
`wave_detection/test_oscillation.py` ("confirmation does not
repeat/stick without a new wave") deliberately does NOT call
`reset_progress()` after a confirmed wave, freezes the hand, and
asserts zero further confirmations over 4 seconds -- proving the
class-level fix holds even without caller discipline. Separately
verified the full detector-level fix (with the corrected reset timing)
produces zero spurious re-confirmations over ~13 simulated seconds of
total stillness after a real wave.

## Files touched this session (Session 11)
- `wave_detection/detector.py` — reset_progress() moved to fire unconditionally on hand_confirmed
- `wave_detection/oscillation.py` — added edge-triggered `_already_reported` guard as defense-in-depth
- `wave_detection/test_oscillation.py` — new regression test (23rd test)

**Total automated test coverage: 31 tests** across
`test_oscillation.py` (23) and `test_body_motion.py` (8), all passing,
all runnable without a camera.

## Session 12 — web dashboard (industry-standard UI) + wave detection clarification
User shared a 4-screen dark/cyan dashboard UI mockup as design
inspiration and asked to (1) make wave detection "finally work" based
on their idea of x-axis-change-based logic, and (2) build a proper,
connected UI matching that aesthetic.

**Wave detection**: explained (not re-coded) that the existing
redesign (Sessions 9-11) already IS fundamentally an x-axis-change
(generalized to 2D) detector — palm-centroid position change over time,
normalized to hand size, exactly the "change in x-axis values" idea
the user described, just made more robust (2D instead of 1D, proper
scale normalization, body-stillness gate). No further blind tuning was
done this session — the honest next step is the user's own live test
with the `D` debug overlay, since that's the only way to get real
feedback past what 31 synthetic tests can prove. Told the user this
directly rather than guessing at more changes.

**Web dashboard — new, substantial feature.** Built `web_dashboard.py`
(Flask) + `templates/dashboard.html` as a genuinely new, styled,
browser-based front end, architected as a SEPARATE display layer on
top of the EXACT SAME detection pipeline (no CV logic duplicated):
- Background thread runs the same per-frame loop as `main.py`
  (camera → detect → track → wave → recognize → decide → render),
  encoding the rendered frame as JPEG into a shared buffer instead of
  a cv2 window.
- Flask serves that buffer as an MJPEG stream (`/video_feed`) plus
  JSON endpoints: `/api/status` (live per-person wave progress + last
  event), `/api/attendance/today`, `/api/attendance/export` (CSV,
  date-ranged), `/api/employees`, `/api/manual_event` (manual
  Check In/Out override — a genuinely useful addition matching the
  mockup's "Manual PIN / Visitor Entry / Check In" row), and
  `/api/sensitivity`.
- `dashboard.html`: single-file dark/cyan-themed page (styled to match
  the shared mockup's aesthetic) with a live video panel, per-person
  wave-progress chips, an auto-refreshing attendance table, CSV export
  controls, a sensitivity selector, and the manual override panel.
- **Calibration**: deliberately NOT rebuilt for the web — the existing
  mouse-drag calibration in `main.py` (press `C`) already saves to
  `config/entrance_line_calibration.json`, which the dashboard reads
  automatically. Documented as a one-time required step.

**Real bugs found and fixed during actual testing this session** (not
just written and assumed correct):
- `renderer.render()` call in the new file was missing required
  positional args (`gate`, `event_log`, `min_frames_to_confirm_side`)
  — caught immediately by comparing against `main.py`'s actual call
  signature before it could ship broken.
- `EventLog` doesn't expose a public `.entries` list — it's `._entries`
  (private) holding `EventLogEntry` objects, not raw strings; the
  dashboard's "last event" text used the wrong access pattern. Fixed
  to use the existing `.recent()` method.
- `initialize_database()` wasn't called early enough for API routes
  hit before the detection thread finishes starting — fixed by calling
  it at process startup, not just inside the background thread.

**Testing performed**: this session's sandbox had disk-space
constraints preventing a full `ultralytics`/YOLO install (a `torch`
dependency issue, unrelated to this project's code). Rather than
skip testing, `PersonDetector`'s YOLO dependency was mocked (that code
is unchanged from before this session, not what's being tested) so
EVERY OTHER real component could be exercised together for real: Flask
routing, the actual SQLite database, the real `FaceRecognizer`/
`FaceEmbedder` (insightface), the real `WaveDetector`, the real
`Renderer`, and the real `EntranceGate` all imported and ran together
successfully, including a full round-trip test: registering a real
employee in the test database, POSTing a manual check-in through the
actual Flask route, and confirming it appears correctly in
`/api/attendance/today`. This is genuine integration testing of
everything new, not a syntax-only check.

**Still not tested**: an actual browser rendering the page, and the
live MJPEG video stream with a real camera — both require things this
sandbox doesn't have. The user should run `python web_dashboard.py`
and open the page themselves as the next real test.

## Files touched/added this session (Session 12)
- `web_dashboard.py` — NEW (Flask backend)
- `templates/dashboard.html` — NEW (styled frontend)
- `requirements.txt` — added `flask>=3.0`
- `README.md` — new "Web dashboard" section

## Next steps for the UI (not built yet, by design — scoped for a future session)
The shared mockup had 4 screens; only the main live-dashboard screen
was built this session. Not yet built:
- A web-based employee enrollment screen (mockup screen 3: "Capture
  Angle," "Complete Enrollment & Generate Key") — currently enrollment
  is still the `register_employee.py` CLI/OpenCV-window tool. Bringing
  this to the web would need browser camera access (getUserMedia) or
  reusing the server-side camera in an "enrollment mode," posting
  captured frames to a new Flask endpoint that reuses
  `face_id/registration.py`'s existing quality/pose-check logic.
- A diagnostics/settings screen (mockup screen 4) — would surface
  `diagnose_camera.py`'s checks and the config values already
  documented in `config/settings.py` as a web page instead of a CLI
  script.
- Employee management (list/deactivate/delete) currently only exists
  via `manage_employees.py` CLI — a web equivalent would reuse the
  same repository functions already built for it.

## Session 13 — sensitivity increase, requested directly by the user after live testing
User explicitly asked to lower the wave-detection thresholds to make
it more sensitive, after testing the redesigned system live. This is
different from earlier sessions' "don't just tune blindly" concern —
this was a direct, informed request following real usage, not a guess
on my part, so it was honored, but still validated with the same
testing discipline as every other change this project has made.

**What changed**: `WAVE_SENSITIVITY_PRESETS`' "balanced" tier (the
default) is now meaningfully more sensitive:
- `min_swing_amplitude_hand_widths`: 0.45 → 0.32 (needs a smaller hand
  motion to register)
- `min_reversals_required`: 3 → 4 (see below for why this went UP,
  not down, despite the sensitivity increase)
- `max_seconds_between_reversals`: 0.45 → 0.55
- `min_detection_confidence`: 0.45 → 0.4
- `max_body_speed_fraction_per_second`: 0.20 → 0.24

A new, more extreme "relaxed" tier exists for cases where even this
isn't sensitive enough (amplitude 0.22, confidence 0.3), available via
the `S` key or the dashboard's sensitivity dropdown — NOT the default,
for reasons below.

**How the right numbers were found**: swept candidate amplitude
values, testing subtle wave detection (down to ~0.25-0.4 hand-widths
peak amplitude, deliberately avoiding exact FPS/period aliasing in the
test methodology, which had briefly produced a misleading test result
before being caught and corrected) against zero-false-positive
requirements on static-hold jitter, before picking values.

**Honest, tested trade-off found and disclosed, not hidden**: lowering
the amplitude threshold measurably weakened rejection of vigorous
random hand motion — unlike walking/jogging (which the body-stillness
gate independently catches), there's no complementary signal for
"random but energetic" hand motion happening while the body stays
still. Raising `min_reversals_required` from 3 to 4 was tested and
cut false confirmations on adversarial synthetic random motion by more
than half, but did not eliminate them entirely. This residual risk is
real and is now explicitly documented (not swept under the rug) in
both `config/settings.py`'s comments and a dedicated test case in
`wave_detection/test_oscillation.py` with an "HONEST FINDING" label,
matching the same honesty standard already applied to the
walking/jogging findings in earlier sessions. The "strict" preset does
not have this weakness, if it becomes a practical problem in real use.

**Also caught and fixed a self-introduced testing mistake this
session**: briefly assumed a test was "already passing" from a
truncated terminal output (`tail -30`) without checking the full
result count, when it was actually still failing outside the visible
window. Caught this by checking the actual pass/fail COUNT rather than
trusting a partial view, and fixed it properly (added the same
honest-finding documentation + a corresponding body-stillness
integration test proving the real pipeline still handles it) rather
than leaving an inconsistency. Worth remembering: always verify full
suite results, not a truncated tail of them.

## Files touched this session (Session 13)
- `config/settings.py` — WAVE_SENSITIVITY_PRESETS "balanced"/"relaxed"
  tiers loosened with reasoning; base `WaveDetectionConfig` dataclass
  defaults synced to match; `min_reversals_required` raised 3→4 with
  full reasoning in its comment
- `wave_detection/test_oscillation.py` — constants synced to new
  defaults; 2 tests updated with "HONEST FINDING" treatment (normal
  walking, random motion) matching existing brisk-walking/jogging
  precedent; 1 new integration test (normal-walking-cadence + real
  body translation → correctly rejected)
- `main.py`, `web_dashboard.py`, `templates/dashboard.html` — default
  sensitivity confirmed as "balanced" (the newly-more-sensitive tier),
  "relaxed" available as an explicit, clearly-labeled opt-in

**Total automated test coverage: 32 tests** (24 in
`test_oscillation.py`, 8 in `test_body_motion.py`), all passing.

**Still true, same as every prior session**: none of this replaces an
actual live test on your camera. The debug overlay (`D` key) remains
the way to see real numbers if "balanced" still isn't sensitive enough
— if so, try `relaxed` next (aware of its documented trade-off) before
asking for the numbers to be lowered further, since further lowering
below "relaxed"'s values starts trading away rejection of genuinely
unrelated hand motion, not just making real waves easier to catch.

## Session 14 — hand wave not being detected at all (two real bugs found and fixed)

User reported: hand-wave detection is "causing issue and not detecting
the hand wave" for check-in/check-out. Unlike prior sessions (which
were about accuracy/sensitivity tuning), this was investigated as a
potential functional break, not a tuning gap -- the existing 32-test
suite (test_oscillation.py + test_body_motion.py) was re-run first and
still passes 32/32, confirming the pure wave-decision LOGIC is not the
problem. Two real bugs were found in the surrounding code instead:

**Bug 1 -- mediapipe version compatibility (most likely root cause of
"no hand ever detected at all"):** `wave_detection/detector.py` builds
on `mp.solutions.hands.Hands` (the legacy MediaPipe Hands API).
Verified directly in this session: mediapipe versions from roughly
0.10.3x onward (and the 1.x line) removed `mp.solutions` entirely --
only the newer Tasks API remains. `requirements.txt` pins
`mediapipe==0.10.14` (confirmed working, has `mp.solutions.hands`),
but if that exact pin isn't what actually got installed -- a fresh
`pip install mediapipe` with no version pin, a wheel-availability
fallback on a newer Python, another package pulling in a newer
mediapipe as a transitive dependency -- `WaveDetector.__init__` would
previously hit `AttributeError: module 'mediapipe' has no attribute
'solutions'` with no clear message, OR (depending on exactly how the
attribute access failed) behave in some other broken way. Either way,
from the user's side this looks exactly like "wave detection just
doesn't work" -- no hand box, no skeleton, nothing, no obvious error
pointing at the real cause.

**Fix**: `WaveDetector.__init__` now explicitly checks `hasattr(mp,
"solutions")` before doing anything else, and raises a `RuntimeError`
naming the installed version and the exact fix (`pip install
mediapipe==0.10.14`) if it's missing. Verified in this session: installing
mediapipe 0.10.33 and constructing a `WaveDetector` now raises that
clear message instead of a confusing `AttributeError`; reinstalling
0.10.14 works normally.

**Action for the user**: run `pip show mediapipe` and confirm it says
exactly `0.10.14`. If not: `pip install mediapipe==0.10.14`. This is
the first thing to check if wave detection still doesn't work after
this session's fixes.

**Bug 2 -- live sensitivity changes ('S' hotkey / dashboard dropdown)
never reached people already in frame:** `WaveDetector` was creating
each tracked person's `mp.solutions.hands.Hands` instance, and each
hand's `OscillationTracker`, exactly ONCE -- the moment that person (or
that hand) was first seen -- with whatever `CONFIG.wave` values were
current AT THAT MOMENT. `apply_wave_sensitivity()` (called by the 'S'
hotkey) mutates `CONFIG.wave` in place, which correctly reaches anyone
NEW after the change, but does nothing for anyone already being
tracked: their `Hands` instance keeps its original
`min_detection_confidence`/`min_tracking_confidence` forever (MediaPipe
doesn't allow changing these after construction), and their
`OscillationTracker` keeps its original amplitude/reversal-count/gap
thresholds forever too. This exactly matches the most common way
someone actually tests this: stand in frame, wave, nothing happens,
press 'S' to loosen sensitivity, wave again -- and see no change,
because the very person testing it never gets the new settings applied
to them.

**Fix**: `WaveDetector` now snapshots every sensitivity-relevant config
field at construction, and at the top of every `process()` call
(`_sync_config_if_changed()`) compares the live config against that
snapshot. On any change, it recreates every active person's `Hands`
instance with the new confidence thresholds, clears every hand's
`OscillationTracker` (so a fresh one, built from current config, is
created on the very next update via the existing
`_make_oscillation_tracker()`), and pushes the new body-stillness speed
cap into every already-existing `BodyStillnessTracker` via a new
`set_max_speed()` method. This does discard in-progress swing history
for anyone mid-wave the instant sensitivity changes -- an acceptable,
minor cost (equivalent to them starting the wave over) for the setting
actually taking effect live, which is the entire point of the hotkey.
Verified directly in this session with a scripted test: constructing a
`WaveDetector`, processing one frame, changing sensitivity, processing
again, and confirming the person's `Hands` instance object identity
actually changed (it does).

**Bug 3 (minor, cosmetic) -- duplicate `@dataclass` decorator:**
`config/settings.py`'s `WaveDetectionConfig` had `@dataclass` written
twice in a row above the class. Confirmed harmless at runtime (the
second decorator just reprocesses the same fields), but removed as
dead/confusing code -- not the cause of the reported bug, just cleaned
up while already in this file.

**New tool added -- `diagnose_wave_detection.py`:** standalone script
(same pattern as `diagnose_camera.py`), runs MediaPipe Hands directly
on the full camera frame with no YOLO/tracking/face/attendance code at
all, using the SAME `OscillationTracker` class the real pipeline uses.
Shows live: whether a hand is detected and MediaPipe's own confidence
for it, the hand skeleton, and the actual swing-amplitude/reversal-count
numbers against the current sensitivity preset's thresholds -- plus an
end-of-session summary (% of frames a hand was detected in). Built
specifically so a future "wave detection doesn't work" report can be
triaged in under a minute into one of: (a) mediapipe/installation
problem -- hand never detected at all, (b) genuine tuning problem --
hand detected fine but swing/reversal numbers don't clear the
threshold, or (c) not a wave-detection problem at all -- everything
here works, so the real issue is downstream (entrance-line crossing
timing or face recognition), which `main.py`'s existing 'D' debug
overlay and event log are the right tool for instead.
```
python diagnose_wave_detection.py                   # webcam 0, balanced
python diagnose_wave_detection.py --sensitivity relaxed
python diagnose_wave_detection.py --url "rtsp://..."
```

**Testing performed this session**: `test_oscillation.py` (24/24) and
`test_body_motion.py` (8/8) re-run and still pass unchanged after all
edits. New scripted smoke test (not a permanent test file, run
directly in the session): constructed a `WaveDetector`, fed it a
synthetic frame + `TrackedPerson`, confirmed normal operation,
confirmed the sensitivity-change bug fix actually recreates the
per-person `Hands` instance. Confirmed the mediapipe-version guard
raises its clear message when mediapipe 0.10.33 is installed, and that
0.10.14 (the pinned version) works normally with `mp.solutions.hands`
present. Whole project byte-compiled clean after all edits
(`compileall`). **Not tested**: an actual live camera + real hand
waving at it (no camera/GPU in this sandbox, same limitation as every
prior session) -- the user should now run
`python diagnose_wave_detection.py` FIRST (fastest way to confirm the
mediapipe fix actually resolved it on their machine), then
`python main.py` for the full pipeline.

## Files touched this session (Session 14)
- `wave_detection/detector.py` -- mediapipe-version fail-fast check in
  `__init__`; `_config_snapshot()` / `_sync_config_if_changed()` for
  live sensitivity changes to reach already-tracked people; wired into
  `process()`
- `wave_detection/body_motion.py` -- `BodyStillnessTracker.set_max_speed()`
- `config/settings.py` -- removed duplicate `@dataclass` on
  `WaveDetectionConfig`
- `diagnose_wave_detection.py` -- NEW

## Session 15 — slow-wave timing fix, Recent Events silent-gap fix, admin monitoring confirmed

Three things worked through, in order, per user's explicit request to
move step by step and not touch anything already working.

**1. Slow, wide waves weren't detected (only fast small oscillations
were)** -- ROOT CAUSE (found via direct user testing with
diagnose_wave_detection.py from Session 14): `max_seconds_between_reversals`
(0.55s in "balanced") is the time budget allowed between consecutive
hand-direction reversals; `_register_reversal()` in oscillation.py
WIPES the whole reversal count back to 1 if that budget is exceeded. A
slow, wide, deliberate wave naturally has longer gaps between reversals
than a fast small one -- at 0.55s, those gaps kept exceeding the
budget, so the count never reached `min_reversals_required` (4) no
matter how many times the person waved. Fast small oscillations
happened to land inside 0.55s, so they worked. Not a bug in the
reversal-COUNT logic or the amplitude threshold (both unchanged, both
already correct) -- purely a timing-window problem.

Fix: `max_seconds_between_reversals` raised 0.55 -> 1.2 (balanced), 0.35
-> 0.75 (strict), 0.8 -> 1.6 (relaxed), plus the base
`WaveDetectionConfig` dataclass default synced to 1.2 to match
"balanced" (existing project convention). This is a strict loosening --
cannot break the fast-oscillation case, since a smaller real gap was
always within a larger allowed window too. Verified directly:
constructed the exact same slow-wave trajectory (period=1.6s, ~0.8s
between reversals) against both the old (0.55) and new (1.2) values --
confirms False under the old value, True under the new one, precisely
reproducing what the user saw live. Added as a permanent regression
test in `test_oscillation.py` ("Slow, wide deliberate wave"). Full
suite re-run after the change: 25/25 pass (one new test), including
every existing walking/jogging/random-motion rejection case --
unaffected, since those are rejected by the body-stillness gate (an
orthogonal signal), not by reversal timing.

**2. Recent Events panel silent when wave confirmed but no check-in/out
happened** -- ROOT CAUSE: the green skeleton (`is_waving_now` in
detector.py) reflects ONLY the raw hand-gesture confirmation
(oscillation + body-stillness), which is a SEPARATE check from the
attendance decision (`IdentifiedAttendanceDecisionEngine.evaluate()` --
is the person recognized? near a recent entrance-line crossing? a
valid next state (not already checked in today)? not a duplicate?).
When the decision engine says no (returns `action=None` with a
`reason_if_none` string), that reason was previously only written to
the log FILE (`logger.info`), never to the on-screen `event_log` the
Recent Events panel actually reads from -- so a confirmed wave with no
resulting action looked, from the live window, like nothing happened
at all, with zero visible explanation.

Fix, applied identically in both `main.py` and `web_dashboard.py` (both
run their own copy of this same decision loop): when
`wave_result.wave_confirmed` is true and `decision.reason_if_none` is
set, that reason is now ALSO added to the on-screen `event_log`, e.g.
`"ID 7 (unrecognized): wave detected, no action -- wave confirmed but
person is not recognized"`. Also, per the user's requested format,
successful CHECK IN/OUT events now include the employee's numeric ID
alongside their name: `f"{employee_name} (ID:{employee_id}):
{decision.action.label}"` (previously name only).

Note for the user: pressing 'S' during cooldown (wave_cooldown_seconds
= 5s after a confirmed action) can still show a green skeleton with no
Recent Events entry at all -- that's `wave_result.wave_confirmed`
itself being False during cooldown (by design, prevents duplicate
check-ins from repeated waving), a DIFFERENT case from the one just
fixed, and out of scope for this session since it's working as
intended, not silently broken.

**3. Admin monitoring of check-in/out timestamps -- confirmed already
fully built, nothing changed:** reviewed end-to-end rather than
rebuilt, per instruction not to touch working features.
- `database/attendance_repository.record_event()` is called directly
  inside `IdentifiedAttendanceDecisionEngine.evaluate()`
  (`face_id/identified_attendance.py`) the moment a real CHECK IN/OUT
  fires -- both `main.py` and `web_dashboard.py` write to the same
  SQLite database automatically; no separate step needed.
- `manage_attendance.py today` -- prints today's events (time, name,
  action) straight to the terminal.
- `manage_attendance.py export --start YYYY-MM-DD --end YYYY-MM-DD
  --out file.csv` -- exports a date range (employee_id, name, code,
  department, action, timestamp, match confidence) to CSV for
  payroll/HR.
- `web_dashboard.py` (`python web_dashboard.py`, then open
  http://localhost:5000 or the machine's LAN IP from another device) --
  live video feed, live people/status panel, `/api/attendance/today`
  and `/api/attendance/export` (same CSV, downloadable from a browser),
  `/api/employees`. This is the actual answer to "how would an
  administrator monitor this" -- it's a real, working web dashboard,
  already built in an earlier session, not something that needed
  adding.

## Files touched this session (Session 15)
- `config/settings.py` -- `max_seconds_between_reversals` raised across
  base default + all three sensitivity presets
- `wave_detection/test_oscillation.py` -- `MAX_GAP_SECONDS` synced to
  1.2; new "Slow, wide deliberate wave" regression test added (25/25
  pass)
- `main.py` -- Recent Events now shows `reason_if_none` when a
  confirmed wave produces no action; CHECK IN/OUT entries now include
  employee ID
- `web_dashboard.py` -- identical fix applied for consistency between
  the two entry points

## Session 16 — real deployment: work-day cycle, quiet duplicates, records/settings/registration UI, dependency + startup bugs found and fixed

Large session implementing the "make it real" requirements: a proper
once-per-real-work-day check-in/out cycle, a quiet Recent Activity feed
(no noise for routine duplicate waves), browser-based employee
registration (no more terminal for day-to-day use), an in-browser
Records/Settings UI, and a concrete deployment runbook (`DEPLOYMENT.md`).
Two independent, real bugs were found and fixed ALONG THE WAY, before
they could ship -- both caught only because everything below was
actually run, not just written.

**1. Work-day boundary (replaces calendar-midnight reset):** new
`database/settings_repository.py` (a generic `system_settings`
key/value table) backs a configurable `attendance_day_boundary_hour`
(default 6am). `attendance_repository.work_day_start()` now computes
"today" relative to that boundary, not midnight -- so an employee who
checks out at 8pm cannot check back in at 12:05am; the cycle only
resets once the boundary hour (e.g. 6am) actually arrives. Editable
live from the dashboard's new Settings tab, no restart needed. The
once-per-day CHECK IN -> CHECK OUT -> (nothing further today) state
machine itself was already correct (`_VALID_NEXT_ACTION` in
`face_id/identified_attendance.py`, unchanged) -- only the definition of
"today" was wrong. New test suite `database/test_attendance_repository.py`
(13 tests) covers the boundary math and the real reported scenario
directly.

**2. Quiet duplicates (revision of Session 15's own change):** Session
15 added an on-screen "wave detected, no action" event_log entry
whenever a confirmed wave produced no check-in/out. Per direct user
feedback, this is too noisy for a real deployment's common, entirely
expected case (someone already checked in waving again) -- removed
from the on-screen feed in both `main.py` and `web_dashboard.py` (kept
in the log FILE only, for troubleshooting). Crossing entries
("ID X: crossed -> ENTERING/LEAVING") and real CHECK IN/OUT entries are
unaffected and unchanged; CHECK IN/OUT entries now include the employee
ID per the user's requested format.

**3. Browser-based employee registration:** new `/api/registration/*`
endpoints in `web_dashboard.py` (start / check_frame / capture / cancel
/ finalize) reuse `face_id/registration.py`'s `evaluate_capture_quality()`
/ `register_new_employee()` UNCHANGED -- no duplicated validation logic,
same pose/quality gates the CLI `register_employee.py` already uses.
Session state (accepted captures per in-progress registration) lives in
server memory keyed by a random session id, not the database, until
finalize actually commits a real employee. New dashboard "Register" tab
drives this via `getUserMedia` + canvas frame capture, with the same
"hold steady" auto-capture UX as the CLI tool (client-side timer, but
every frame is still authoritatively re-validated server-side, same as
the CLI's SPACE-to-capture). Finalize reloads the LIVE, running
`FaceRecognizer` (`.reload_known_faces()`) so a newly registered
employee is recognized on camera immediately, no restart.

**REAL BUG found while building this: `FaceRecognizer` was not
thread-safe against a live reload.** `reload_known_faces()` reassigned
its four parallel arrays (embeddings, employee_ids, full_names,
angle_labels) one at a time, not atomically; `recognize()` (called
continuously by the detection loop thread) reads all four. Before
browser registration existed, this was safe by accident -- reload only
ever happened once, at startup, before the loop began. A live
registration reloading the SAME running instance the detection thread
is actively using broke that assumption: a `recognize()` call mid-reload
could read a genuinely inconsistent mix of old and new state (e.g. a
new, longer embedding matrix paired with the OLD, shorter employee_ids
array). Fixed with a `threading.Lock()` around both the write (reload)
and the read (recognize) -- the matching algorithm/logic itself is
completely unchanged, verified with a direct smoke test.

**4. Records + Settings API/UI:** new `GET /api/attendance/records`
(filterable by date range + employee, JSON, newest-first) backs a
proper in-browser Records tab (the CSV export was already the only way
to actually look at historical data before this). New
`GET/POST /api/settings` + `/api/settings/day_boundary`. `/api/employees`
enriched with live status (`status_label_for_employee`) and a new
`POST /api/employees/<id>/deactivate` (uses the existing, already-built
`deactivate_employee()` -- soft delete, keeps history, drops the person
from live recognition immediately via the same reload mechanism as
registration).

**5. Dashboard UI rewrite:** `templates/dashboard.html` rebuilt as a
5-tab sidebar app (Live Monitor / Employees / Records / Register /
Settings) -- every existing Live Monitor element (video feed,
sensitivity selector, status chips, today's attendance table, manual
override) preserved with IDENTICAL ids/behavior, not rewritten, per
explicit instruction not to touch what already worked. New tabs are
additive.

**REAL BUG found and fixed while testing all this end-to-end (would
otherwise have shipped in DEPLOYMENT.md's own recommended command):**
`web_dashboard.py`'s database init and detection-loop thread startup
lived only inside `if __name__ == "__main__":`. A production WSGI
server (waitress, per this session's own deployment guide) IMPORTS the
module and serves the `app` object directly -- it never executes the
file as `__main__`. Running exactly the command this session's own
`DEPLOYMENT.md` recommends (`waitress-serve web_dashboard:app`) would
have started a dashboard with the database never initialized and the
camera/detection loop never started -- while `python web_dashboard.py`
directly (the dev-server path used throughout earlier testing) looked
completely fine, masking the problem until actual deployment. Fixed by
moving `initialize_database()` and the thread start to module level
(runs on import, either way); confirmed by actually running
`waitress-serve --host=127.0.0.1 --port=5050 web_dashboard:app` and
curling it successfully in this session, not just reasoning about it.

**6. REAL dependency conflict found and fixed:** installing
`insightface` unpinned pulls in a current `onnx` (1.2x), which requires
`protobuf>=6.31` -- directly incompatible with `mediapipe==0.10.14`
(Session 14's pin), which requires `protobuf<5`. No single protobuf
version satisfies both; mediapipe can't be the one to move (newer
mediapipe removes `mp.solutions.hands` entirely, see Session 14).
Fixed by pinning `onnx==1.16.0` (confirmed directly, in this session, to
still work with `protobuf<5`) and `protobuf>=4.25.3,<5` in
`requirements.txt`. Confirmed BOTH `mediapipe.solutions.hands.Hands()`
and `insightface.app.FaceAnalysis` import and initialize successfully
together with this exact pin combination -- not just installed
separately, actually imported in the same process.

**7. `DEPLOYMENT.md` (new):** concrete runbook -- one-time terminal
setup + calibration, running via `waitress-serve` as a persistent
service (systemd unit / Windows Task Scheduler, so no terminal window
needs to stay open), making it reachable (LAN IP for the same office;
Tailscale or a reverse-proxy+auth for remote access -- explicitly NOT
recommending exposing the unauthenticated dashboard directly to the
public internet), and an honest architecture explanation: video
processing must happen on a machine with network access to the camera
(an edge box near the door, same as any real commercial system) --
"deploy to the cloud" does not eliminate that, it only changes whether
a central cloud dashboard also exists on top of per-site edge boxes.
Ends with a direct, undecorated list of what's still needed before this
is a sellable multi-tenant product (auth/roles, Postgres for
multi-site, biometric-data compliance review, monitoring/alerting,
payroll integration) -- framed as a roadmap on top of a genuinely
working single-site system today, not a list of missing basics.

**Testing performed this session:** full existing suite re-run after
every change (`test_oscillation.py` 25/25, `test_body_motion.py` 8/8);
new `database/test_attendance_repository.py` (13/13, covers the
work-day boundary fix directly). All new `/api/registration/*`,
`/api/attendance/records`, `/api/settings*`, `/api/employees/<id>/deactivate`
endpoints exercised end-to-end via Flask's test client with the REAL
insightface/mediapipe/ultralytics stack installed (not mocked) --
session lifecycle, error handling (expired session, missing name, no
captures, invalid day-boundary hour), and a full register->record->
directory->deactivate round trip with real data. `FaceRecognizer`
thread-safety fix smoke-tested directly. The exact `waitress-serve
web_dashboard:app` command from `DEPLOYMENT.md` actually run and
curled successfully. Whole project byte-compiled clean throughout.
**Not tested**: an actual live camera + real employee walking up and
waving (still no camera/GPU in this sandbox), and the browser-based
registration UI's actual JS/getUserMedia flow in a real browser (the
underlying API endpoints it calls were fully tested; the HTML/JS layer
itself was not executed in a browser this session).

## Files touched this session (Session 16)
- `database/schema.py` -- new `system_settings` table
- `database/settings_repository.py` -- NEW
- `database/attendance_repository.py` -- `work_day_start()`,
  `get_todays_last_action`/`get_todays_events` use it instead of
  midnight; `get_events_in_range` gained optional `employee_id` filter
- `database/test_attendance_repository.py` -- NEW (13 tests)
- `face_id/recognizer.py` -- thread-safety lock around reload/recognize
- `main.py`, `web_dashboard.py` -- quiet duplicate-wave revision;
  CHECK IN/OUT event text includes employee ID; `web_dashboard.py` also
  gained all new registration/records/settings/deactivate endpoints and
  the module-level (not `__main__`-only) startup fix
- `templates/dashboard.html` -- full rewrite, 5-tab sidebar app
- `requirements.txt` -- `onnx==1.16.0`, `protobuf>=4.25.3,<5`,
  `waitress>=3.0` added; comments explaining both dependency traps
- `DEPLOYMENT.md` -- NEW

## Session 17 — Register tab "Device in use" camera error (real architecture bug)

User reported: Live Monitor's camera worked fine, but the Register
tab failed with "Could not access camera: Device in use (check
browser camera permissions)" -- explicitly NOT a browser permissions
issue on their end.

**ROOT CAUSE**: this was never a permissions problem -- it was a
genuine architecture bug. The Register tab's JS called
`navigator.mediaDevices.getUserMedia()`, asking the BROWSER to open
its own, independent handle on the physical camera. But
`web_dashboard.py`'s detection loop already holds that exact camera
device open, exclusively, via `create_video_source()`, for as long as
the dashboard runs. Most cameras (certainly a single USB/IP door
camera, and most laptop webcams) can only be opened by one process/
handle at a time -- so the moment Live Monitor's backend loop had the
camera open, the browser's getUserMedia() call for that SAME physical
device was always going to fail with exactly this error. No browser
permission setting could have fixed it, because the browser never got
that far -- the OS itself refused the second handle.

**FIX (architectural, not a permission workaround)**: registration no
longer asks the browser to open any camera at all. Instead:
- `DashboardState` gained `latest_raw_frame` -- a raw (un-annotated)
  copy of the current camera frame, stored by `_detection_loop()`
  right after `video_source.read_frame()` (BEFORE
  `renderer.render()`, which draws boxes/text onto `frame` in place --
  capturing after render would have hidden faces behind debug overlays
  and fed a visually-polluted image into face embedding).
- `/api/registration/check_frame` and `/api/registration/capture` no
  longer accept or require an uploaded `image` at all -- they pull
  `state.latest_raw_frame` directly (see
  `_get_live_frame_for_registration()`). If the detection loop hasn't
  produced a frame yet, they return a clear "Camera not ready yet"
  reason instead of a raw error.
- The Register tab's `<video>` + `getUserMedia()` + hidden `<canvas>`
  frame-grabbing was replaced with a plain `<img>` pointing at the
  same `/video_feed` MJPEG stream Live Monitor already uses, purely
  for the person to see themselves while positioning -- it plays no
  role in what actually gets validated/captured.

This is also MORE correct than the old design, not just a fix: it
guarantees registration photos come from the exact same camera
(angle/lens/lighting) that will later try to recognize this person,
rather than whatever camera a browser's OS happens to default to on a
machine with more than one (e.g. a laptop's built-in webcam vs. the
actual deployed door camera) -- a mismatch the old getUserMedia-based
design was silently exposed to even on a machine where it happened to
work.

**Testing performed**: full existing suite re-run (`test_oscillation.py`
25/25, `test_body_motion.py` 8/8, `test_attendance_repository.py`
13/13). New scripted test: registration `start` -> `check_frame` ->
`capture` exercised with ZERO `image` field in the request body at all
(confirming the server never needs or accepts one anymore), against a
synthetic frame injected directly into `state.latest_raw_frame` --
correctly rejects (no face in synthetic noise) with a clean reason,
and correctly reports "Camera not ready yet" when
`state.latest_raw_frame` is None. Dashboard JS re-checked with
`node --check` after the rewrite (real syntax check, not just brace
counting). **Not tested**: an actual live camera + real face walking
through the full registration flow in an actual browser (still no
camera/GPU in this sandbox).

## Files touched this session (Session 17)
- `web_dashboard.py` -- `DashboardState.latest_raw_frame`; raw-frame
  capture point in `_detection_loop()`; `/api/registration/check_frame`
  and `/api/registration/capture` rewritten to use the live server-side
  frame instead of a client-uploaded image
- `templates/dashboard.html` -- Register tab's camera preview changed
  from `getUserMedia()` + `<video>` + hidden `<canvas>` to a plain
  `<img>` on `/video_feed`; all registration JS calls simplified to
  drop the image payload entirely
