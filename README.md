# Real-Time Attendance Detection System — Phase 1

Real-time person detection, multi-person tracking, wave-gesture detection,
and entry/exit direction estimation from a webcam feed. This phase only
detects and suggests — it does not identify anyone, store anything, or
connect to any database or API.

## What this does

For every person visible in the camera, the system:
1. Detects them and draws a bounding box.
2. Assigns them a stable tracking ID that persists across frames.
3. Estimates whether they are ENTERING, LEAVING, or UNKNOWN direction.
4. Watches for a genuine hand-wave gesture (not just a raised hand).
5. When a wave is confirmed, shows a temporary "Suggested Action"
   banner (CHECK IN / CHECK OUT) for ~2.5 seconds, then clears it.

Everything is scoped **per tracking ID**, so multiple people can be
detected, tracked, and waving at the same time with zero cross-talk
between them.

## Project structure

```
config/                 All tunable settings in one place (settings.py)
camera/                 Video source abstraction (webcam today, CCTV/RTSP-ready)
person_detection/       YOLOv8 wrapper -> raw person detections
person_tracking/        ByteTrack wrapper -> stable per-person IDs + position history
direction_estimation/   Entrance Line (Virtual Gate) crossing detection
wave_detection/         MediaPipe Hands wrapper -> per-hand genuine wave detection,
                         plus duplicate-action guard (survives track ID churn)
face_id/                Phase 2: face detection/embedding (InsightFace), matching
                         against known employees, and the database-backed
                         attendance decision engine
database/               SQLite schema + repository functions for employees,
                         face embeddings, and attendance events
display/                Overlay rendering (boxes, hands, trail, arrow, line, event log)
utils/                  Shared helpers (geometry, cooldown timers, FPS counter, event log)
main.py                 Wires everything together and runs the live loop
register_employee.py    Standalone guided face-registration tool (run this first)
requirements.txt        Pinned dependencies
```

See `DOCUMENTATION.md` (and `DOCUMENTATION.pdf`) for the complete
module-by-module technical writeup, including Computer Vision concepts
used, alternatives considered, and a full summary of every round of
improvements.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run will auto-download the YOLOv8n model weights (~6 MB) from
Ultralytics — an internet connection is needed the first time only.

## Run

```bash
python main.py
```

A window will open showing your webcam feed with live overlays.

**Live controls (shown at the bottom of the window):**
- **`C`** — calibrate the entrance line: click two points directly on
  the video to trace where the real doorway is. This is the main
  workflow for a CCTV install — walk up to the camera's live feed,
  click along the actual entrance threshold, done. No code editing.
- **`F`** — flip which side is labeled IN vs OUT, if they come out
  backwards for your camera's angle.
- **`Q`** — quit.

Calibration is saved automatically to
`config/entrance_line_calibration.json` and reloaded next time you run
the app — you only need to calibrate once per camera position.

## Phase 2: Employee Face Recognition + Database

The system now recognizes registered employees by face and records
real CHECK IN / CHECK OUT events to a local database — the "Suggested
Action" from Phase 1 has become an actual recorded attendance system.

**First, register each employee's face:**

```bash
python register_employee.py
```

This opens a guided capture flow: enter the employee's name, then the
camera opens and **automatically** captures their face from a few
angles (front, left, right) — hold the pose shown on screen and it
captures itself the moment it's confident, no button needed. Each
capture is automatically quality-checked (must be a single, clear,
well-lit face, AND genuinely different from angles already captured
for this person) before being accepted.

**Then run the live system as usual:**

```bash
python main.py
```

Recognized employees now show their real name on screen instead of
`ID N`, and a genuine `CHECK IN` / `CHECK OUT` record is written to
`database/attendance.db` (a plain SQLite file) the first time each of
them waves near the entrance line. Unrecognized people (not yet
registered, or the match wasn't confident enough) show as `UNKNOWN`
and never get an attendance record — the system never guesses.

**New live control:** press **`R`** to reload known faces from the
database without restarting `main.py` — useful right after registering
someone new.

**The bigger upgrade this unlocks:** in Phase 1, attendance state
lived only in memory, tied to a temporary tracking ID that could reset
if tracking briefly lost someone. Now that a person has a stable,
recognized identity, their attendance state lives in the database
instead — so it survives tracking hiccups, and even survives
restarting the app, for the whole day.

**Honest scope note:** face matching uses a similarity threshold
(`FaceRecognitionConfig.min_match_similarity`, default `0.45`) tuned
and verified against real distinct faces during development — but the
right value for your specific camera and lighting should be validated
with your actual employees during a pilot, not just trusted blindly.
See `DOCUMENTATION.md` Section 21 for the full reasoning and how to
adjust it.

## Web dashboard (styled browser UI, alternative to the OpenCV window)

`python main.py` shows a plain OpenCV window — functional, but not a
polished front end. `web_dashboard.py` is a styled, browser-based
alternative that reuses the exact same detection/tracking/wave/face-
recognition/attendance pipeline underneath; only the *display* layer
differs.

**One-time setup**: run `python main.py` once and press `C` to
calibrate the entrance line against your real doorway — this saves to
`config/entrance_line_calibration.json`, which the web dashboard also
reads automatically. (The web dashboard doesn't have its own
mouse-drag calibration UI, since that needs a native OpenCV window.)

**Run it**:
```
python web_dashboard.py
```
Then open **http://localhost:5000** in a browser on the same computer
(or another device on the same network, using this computer's LAN IP
instead of `localhost`).

**What it shows**: a live video panel (the same detection/tracking/
wave overlays as the OpenCV window, just streamed into the browser),
a live "who's currently being seen and their wave progress" list,
today's attendance table (auto-refreshing), a CSV export with a date
range, a sensitivity selector, and a manual Check In / Check Out panel
for a selected employee — a practical fallback for a case wave/face
detection doesn't handle well (hands full, bad angle, etc).

This is a genuinely separate front end, not a replacement — `main.py`
still works exactly as before if you prefer the desktop window.

## Admin & utility scripts

| Script | Purpose |
|---|---|
| `python register_employee.py` | Add a new employee (guided face capture). |
| `python manage_employees.py list` | List registered employees. Add `--include-inactive` to also show deactivated ones. |
| `python manage_employees.py deactivate --id N` | Soft-delete: stops recognizing them, **keeps** their attendance history. Use this when someone leaves the company. |
| `python manage_employees.py delete --id N` | **Permanently** deletes the employee, their face data, AND their entire attendance history. Asks you to type their name to confirm. Use this for mistaken/test registrations only. |
| `python manage_attendance.py today` | Print today's CHECK IN / CHECK OUT events. |
| `python manage_attendance.py export --start 2026-08-01 --end 2026-08-13 --out report.csv` | Export attendance events to CSV (e.g. for payroll). Both `--start`/`--end` default to today if omitted. |
| `python diagnose_camera.py` | Isolate camera problems (webcam or IP camera) from the rest of the pipeline. Scans webcam indices, or test one index/URL directly. |

After `deactivate` or `delete`, press **`R`** in `main.py` (or restart
it) so the change takes effect — face embeddings are loaded into
memory at startup and on each `R` press, not read live per-frame.

## Using a phone as the camera (DroidCam) or any IP/CCTV camera

`config/settings.py` → `CameraConfig.source` accepts either a webcam
index (`0`, `1`, ...) or a URL string, and `camera/factory.py` picks
the right implementation automatically. For an Android phone via
**DroidCam** specifically:

1. Install the **DroidCam** app on your phone (Play Store) and, on the
   same computer that will run this project, install the DroidCam
   client **or** just use its plain network stream — no special
   client is required for OpenCV to read it.
2. Connect the phone to the **same Wi-Fi network** as the computer.
   This step trips people up more than anything else — DroidCam over
   Wi-Fi will not work if the phone is on mobile data or a different
   network.
3. Open the DroidCam app on the phone. It shows an IP address and port
   on screen, e.g. `192.168.1.42:4747`.
4. In `config/settings.py`, set:
   ```python
   CONFIG.camera.source = "http://192.168.1.42:4747/video"
   ```
   (use the IP/port your app actually shows — this is just an example).
5. Verify it BEFORE running the full system:
   ```bash
   python diagnose_camera.py --url "http://192.168.1.42:4747/video"
   ```
   If this fails, test the exact same URL in VLC (Media → Open Network
   Stream) first — if VLC can't play it either, it's a phone/network
   issue, not this project.
6. Once `diagnose_camera.py` reports success, just run
   `python register_employee.py` / `python main.py` as normal — they
   use the same `CONFIG.camera.source`, so nothing else changes.

The same steps work for any other IP camera or CCTV/NVR, just with an
`rtsp://user:pass@ip:554/path` URL instead — see the comments in
`camera/ip_camera_source.py` for finding your camera's specific RTSP
path.

## Scaling to a real company (thousands of employees)

The core pipeline (detection → tracking → face recognition → wave
confirmation) doesn't change with headcount — the only thing that
scales with employee count is how many stored face embeddings a live
match gets compared against. What changed to make that scale well:

- **Vectorized matching**: `face_id/recognizer.py` compares a live
  face against every stored embedding with a single matrix-vector
  multiply (numpy/BLAS), not a Python loop. Verified in this repo's
  test run: ~0.5ms per recognition against 6,000 stored embeddings
  (2,000 employees × 3 angles) — effectively free even at real company
  scale, on CPU.
- **Ambiguity/margin check**: with more people enrolled, the odds rise
  that two different employees' embeddings both look plausible for a
  given live face. `FaceRecognitionConfig.min_margin_over_runner_up`
  requires the top match to clearly beat the best-scoring *different*
  employee, or the result is UNRECOGNIZED rather than a risky guess.
- **Duplicate employee code handling**: `register_employee.py` checks
  code uniqueness *before* opening the camera (not after several
  minutes of face capture), and if a code collides at save time
  anyway, prompts for a new one without discarding the captures
  already made.
- **Database**: WAL mode is enabled (concurrent report/export reads
  don't block a live check-in write), and there's a dedicated index
  for date-range queries (`manage_attendance.py export`) so reports
  stay fast as the events table grows over months/years, not just at
  small scale.
- **Audit log file**: `main.py` now writes a rotating log file to
  `logs/attendance_system.log` (in addition to the console), so there's
  a persistent record to check after the fact — not just whatever was
  visible in a terminal that's since been closed.

None of this changes what a single running instance covers: it's still
one entrance / one camera per `main.py` process. Multiple entrances
would mean running one instance per camera against the same shared
`attendance.db` (the schema's `camera_id` column already exists for
this) — a reasonable next step, not implemented here since it wasn't
asked for and would need its own testing.

## Key engineering decisions (defaults chosen for you)

All of these live in `config/settings.py` and can be changed without
touching any other file:

| Decision | Default | Why |
|---|---|---|
| Detection model | YOLOv8n | Fastest variant, real-time on CPU; upgrade to `yolov8s.pt`/`yolov8m.pt` if you have a GPU and want higher accuracy on crowded/occluded scenes |
| Detection confidence | 0.45 | Standard balance point between missed detections and false positives for person detection |
| Tracker | ByteTrack | Lighter than DeepSORT (no re-ID network), handles brief occlusion well, good fit for real-time |
| Track history length | 30 frames | ~1 second at 30 FPS — enough to smooth out jitter for direction estimation without lagging too far behind |
| Wave: reversals required | 3 | A real wave has several quick back-and-forth swings; a raised static hand has 0, and walking arm-swing rarely produces this many quick reversals |
| Wave: min swing amplitude | 0.45 hand-widths | Measured relative to the hand's OWN detected size, not the crop — see "Wave detection redesign" below |
| Wave: body-stillness gate | 20% of body height / 1s | Requires the person to be relatively still for a wave to confirm — resolves cases hand motion alone can't (see below) |
| Wave: cooldown | 5 seconds | One wave = one Suggested Action; prevents the same wave spamming repeated triggers |
| Suggested Action display time | 2.5 seconds | Matches the "~2-3 seconds" requirement |
| Entrance line: confirm delay | 5 frames | Debounces jitter right at the line so side changes don't flicker |

## Wave detection redesign: why genuine waves were being missed, and the fix

An earlier version tracked one landmark (the wrist), on one axis
(horizontal), normalized to the person's crop width. Three compounding
problems, found by actually inspecting the algorithm rather than just
retuning numbers:

1. **The wrist is the pivot point in a relaxed, natural wave** — it's
   almost by definition the part of the hand that moves LEAST. Tracking
   it specifically captured the smallest possible signal for the most
   common, natural wave style.
2. **Horizontal-only tracking missed vertical/diagonal waves** entirely,
   however clear and close the hand was.
3. **Normalizing by the crop width tied the required hand motion to
   something that isn't the hand** — the crop is recomputed from the
   person's live-tracked bbox every frame, which can shift and resize
   independent of actual hand movement.

**The redesign** (`wave_detection/oscillation.py`, `wave_detection/body_motion.py`):
tracks the **palm centroid** (wrist + four knuckle landmarks averaged)
in full **2D**, normalized to the **hand's own detected size**
("hand-widths") instead of the crop. A genuine, testing-derived finding
along the way: hand motion *alone* — however well-designed — cannot
fully separate a deliberate wave from vigorous walking/jogging arm
swing at overlapping amplitude and cadence. The fix is a second,
orthogonal signal already available for free: the person's own tracked
bbox is used to require their **body** be relatively stationary
(<20% of their own height per second) for a wave to confirm.

This is backed by two test suites — 30 synthetic test cases total,
both runnable directly with no camera needed:
```
python wave_detection/test_oscillation.py   # 22 cases: hand-motion patterns + confirmed integration with body-stillness
python wave_detection/test_body_motion.py   # 8 cases: body-stillness responsiveness and jitter robustness
```
`test_body_motion.py` exists because the body-stillness gate went
through two real, tested design iterations before shipping: a
1.0-second window had a measured ~0.73s lag after a person actually
stopped walking (bad for "detected quickly"); an instantaneous-speed
attempt fixed that but introduced false positives from ordinary
tracking jitter; a 0.5-second window (found by empirically sweeping
window lengths, not guessed) is the validated point where both
responsiveness (~0.3s lag) and jitter-robustness (zero false positives
up to 3% bbox jitter) hold. This history is documented inline in
`wave_detection/body_motion.py`.

Press **`D`** while `main.py` is running for a live DEBUG/ANALYSIS
overlay showing exactly why a wave is or isn't being detected in real
time (swing magnitude, reversal count, regularity, body speed).

Where the entrance line goes, and which side is IN vs OUT, depends
entirely on your physical camera position — that's a per-installation
decision, not something I can pick generically. **You no longer need
to edit any file for this** — press `C` in the live video window and
click the line onto the real doorway; press `F` if IN/OUT come out
backwards. See the "Live controls" note above. (The old approach of
editing `line_start`/`line_end` ratios in `config/settings.py` still
works as a fallback default, but the on-screen calibration is the
intended workflow now.)

## Designed for painless future extension

- **CCTV instead of webcam**: change `CameraConfig.source` to an RTSP
  URL and add one branch in `camera/factory.py`. Nothing else changes.
- **Face recognition**: `TrackedPerson` already carries a stable ID
  per frame sequence; an `identity` field can be attached to it without
  touching detection, tracking, wave, or direction logic.
- **Attendance database / API**: `main.py`'s "wave confirmed" block is
  the single hook point — that's where a call to a future
  `attendance_service` module would go, instead of (or alongside) the
  on-screen banner.
