# Hudasoft Attendance System — Deep Technical Guide (Line‑by‑Line)

**Purpose:** teach you *exactly how every feature is built*, which file and function implements it, and what each important line does — in plain words. Read this top‑to‑bottom once; after that you can jump to any feature.

> How to read a reference like `web_dashboard.py → _detection_loop()`: open that file, find that function (Ctrl+F the name). Line numbers shift as code changes, so I reference **file + function name** (stable) and paste the real lines that matter.

---

## 0. The 30‑second mental model

A single Python program does everything:

```
Camera → Detect people (YOLO) → Track them (ByteTrack)
      → Recognize faces (InsightFace) + Detect waves (MediaPipe)
      → Which side of the door line? (Entrance gate)
      → Decide CHECK IN / CHECK OUT (state machine) → Save to database (SQLite)
      → Show it all in the browser dashboard (Flask web server)
```

It runs as **one Flask process** with **three threads**:
1. **Detection thread** (`_detection_loop`) — the brain: runs the vision pipeline.
2. **Display thread** (`_display_loop`) — the eyes: draws the smooth video.
3. **Flask/Waitress** — the web server answering the browser.

Everything the threads share lives in one object, `DashboardState` (in `web_dashboard.py`), protected by locks so two threads never corrupt each other's data.

---

## 1. Camera input

**Files:** `camera/factory.py`, `camera/webcam_source.py`, `camera/ip_camera_source.py`, `camera/latest_frame_reader.py`

### 1a. Picking the right camera class
`camera/factory.py → create_video_source()`:
```python
if isinstance(source, str) and source.lower().startswith(("rtsp://","http://","https://")):
    return IPCameraSource(config)   # phone/CCTV over the network
return WebcamSource(config)         # a number like 0 = local USB/built-in webcam
```
- **Plain words:** if the camera "address" is a web URL (your phone via DroidCam), use the network‑camera class; if it's a number, use the plain webcam class. Nothing else in the app needs to know the difference.

### 1b. Always showing the *freshest* frame (no lag build‑up)
`camera/webcam_source.py → open()` starts a `LatestFrameReader` thread. The key idea:
```python
self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ask the driver not to queue old frames
self._reader = LatestFrameReader(self._capture)  # a thread that constantly grabs the newest frame
```
- **Why:** a camera produces frames faster than we can process them. If we read them in order, we'd fall behind and show old frames (lag). The reader thread keeps **only the latest** frame, so `read_frame()` always returns "now," not a backlog.

### 1c. Wrong‑IP can't freeze the app
`camera/ip_camera_source.py` (top of file):
```python
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000")  # 5 seconds, in microseconds
```
- **Plain words:** by default, OpenCV will sit for ~30+ seconds trying to reach a dead IP, freezing everything. This caps that wait at 5 seconds so a typo'd DroidCam IP fails fast and you can correct it.

---

## 2. Person detection (YOLOv8 + OpenVINO)

**File:** `person_detection/detector.py` → class `PersonDetector`

### 2a. Loading the fastest available engine
`_load_model()`:
```python
ov_dir = _openvino_dir_for(self._config.model_path)   # "yolov8n.pt" -> "yolov8n_openvino_model"
if os.path.isdir(ov_dir):
    for dev in ("intel:gpu", "intel:npu", "intel:cpu"):
        try:
            model = YOLO(ov_dir, task="detect")
            model.predict(source=np.zeros((64,64,3), np.uint8), device=dev, ...)  # tiny warm-up test
            return model, dev
        except Exception:
            ...  # try the next device
```
- **Line by line:**
  - `ov_dir = ...` → work out the name of the OpenVINO model folder that sits next to the normal weights.
  - `if os.path.isdir(ov_dir)` → *only* if you actually exported one (you did: `yolov8n_openvino_model/`).
  - the `for dev in (...)` loop → try the Intel **GPU** first, then NPU, then Intel CPU. OpenVINO runs the **same** YOLO weights, just on faster Intel hardware.
  - `model.predict(... np.zeros ...)` → a **warm‑up** on a blank image. If this device can't actually run, it throws, and we try the next one. This is why a broken GPU can never crash the app.
  - If no OpenVINO device works (or the folder doesn't exist), it falls through to the normal PyTorch model below — identical detection, just slower.
- **Why it matters for you:** this is the "it uses your Intel GPU automatically and never breaks" feature. Same accuracy, much faster.

### 2b. Actually detecting
`detect()`:
```python
results = self._model.predict(source=frame, classes=[0], conf=self._config.confidence_threshold,
                              device=self._device, imgsz=self._config.inference_size, verbose=False)
```
- `classes=[0]` → class 0 in the COCO dataset is **"person"**; we ignore everything else (chairs, bags…).
- `conf=0.35` → keep boxes the model is at least 35% sure about. Lower = detect people in harder light (but more false boxes). This was lowered from 0.45 to fix "doesn't detect in bright light."
- `imgsz=384` → shrink the image to 384px before the neural net looks at it. Smaller = faster. YOLO cost grows with this number *squared*, so it's the biggest speed dial.
- The loop after it reads each box's corners (`xyxy`) and confidence into our own simple `Detection` objects.

---

## 3. Tracking people across frames (ByteTrack)

**File:** `person_tracking/tracker.py` → `PersonTracker.update()`, and `person_tracking/tracked_person.py`

Detection only answers "where are the people in *this* frame." Tracking answers "which box is the *same person* as last frame" and gives each a stable **track_id**.

```python
tracked_sv = self._tracker.update_with_detections(sv_detections)  # ByteTrack does the matching
...
if track_id not in self._tracks:
    self._tracks[track_id] = TrackedPerson(track_id=..., bbox=..., history_length=30)
self._tracks[track_id].update(bbox, confidence)   # append this frame's position to their history
```
- **Why ByteTrack:** it's light on CPU (no extra neural network) and it even uses *low‑confidence* boxes to keep a track alive through brief occlusion (someone walks in front of someone else).
- `TrackedPerson.position_history` is a `deque(maxlen=30)` — a rolling list of the last 30 centre points. **This one list is shared** by wave detection (to see hand motion relative to the body) and the entrance line (to see which way they moved). That's why tracking "owns" the per‑person memory.
- **Stale cleanup:** a track not seen for `max_missed_frames` (45) is deleted so memory doesn't grow forever.

---

## 4. Wave detection (the trigger) — the heart of the system

Three cooperating files:
- `wave_detection/detector.py` — runs MediaPipe on each person, feeds the signal to the maths.
- `wave_detection/oscillation.py` — the pure "is this motion a wave?" maths.
- `wave_detection/body_motion.py` — the "is the person standing still?" safety check.

### 4a. One hand‑tracker per person, on a *crop* not the whole frame
`detector.py → process()`:
```python
crop_bbox = expand_bbox(tracked_person.bbox, frame_w, frame_h, margin_ratio=...)
crop = frame[cy1:cy2, cx1:cx2]
...
results = self._hands_instances[track_id].process(crop_rgb)
```
- **Plain words:** instead of searching the whole picture for hands, we cut out just this person's box and look there. That's (1) faster and (2) automatically ties the hand to the right person — no "whose hand is this?" problem when several people wave at once.
- Each `track_id` gets its **own** MediaPipe `Hands` instance (`self._hands_instances[track_id]`), so two people waving at the same time are analysed completely independently.

### 4b. Which point on the hand we follow
`detector.py`:
```python
_PALM_CENTROID_INDICES = (0, 5, 9, 13, 17)   # wrist + 4 knuckle bases
palm_x = average of those landmarks' x ; palm_y = average of their y
hand_scale_px = distance(landmark 0, landmark 9)   # wrist → middle knuckle
```
- **Why not the wrist:** in a normal relaxed wave you pivot at the wrist, so the wrist barely moves — tracking it misses real waves. The **palm centroid** (middle of the hand) sweeps a big arc, so it's a strong signal.
- `hand_scale_px` is the hand's own size, used to measure motion in **"hand‑widths"** instead of pixels — so a wave counts the same whether you're close to or far from the camera.

### 4c. The wave maths
`oscillation.py → OscillationTracker.update()`. It watches the palm centroid move and counts **direction reversals** (left‑right‑left…):
```python
dist = math.hypot(dx, dy)                    # how far the hand moved since the last turning point
if dist >= min_swing_units:                  # far enough to count as a real swing (not jitter)
    dot = (dx/dist)*dirx + (dy/dist)*diry    # is this swing OPPOSITE the previous one?
    if dot < 0:                              # yes -> that's a reversal
        reversed_now = True
```
- **Line by line:**
  - `dist` → distance the hand travelled from the last "extreme" point.
  - `min_swing_units` → the minimum swing size (in hand‑widths) to count; rejects tiny jitter.
  - `dot < 0` → a **dot product** test: if the new movement points roughly *opposite* to the last movement, the hand changed direction — that's one reversal. A wave = several reversals in a row.
- **Confirmation** (`_check_confirmed`): you need `min_reversals_required` reversals, each within `max_seconds_between_reversals` of the last, and with reasonably steady rhythm (the "regularity CV" check rejects random flailing). These three numbers are the **sensitivity presets** (Strict/Balanced/Relaxed).

### 4d. The anti‑false‑alarm check: body must be still
`body_motion.py → BodyStillnessTracker.is_relatively_stationary()`:
```python
speed = (distance_body_moved / body_height) / time_elapsed   # over a 0.5s window
return speed <= self._max_speed
```
- **Why:** a brisk walking arm‑swing can look like a wave to the hand‑maths alone. But a **walking person's whole body slides across the frame**, while someone pausing to wave doesn't. So a wave only counts if the body has been roughly still. This is measured over a 0.5‑second window (instant frame‑to‑frame was too noisy).
- In `detector.py`: `confirmed = hand_confirmed and body_ok` — both must be true.

### 4e. Stopping one wave firing twice
`wave_detection/action_guard.py → DuplicateActionGuard` remembers the position+label+time of recent actions and suppresses the **same** action firing again nearby within a few seconds. (It's label‑aware: a CHECK OUT near an earlier CHECK IN is *not* blocked — that's normal.)

---

## 5. Entrance line (the virtual door) + "outside only"

**File:** `direction_estimation/entrance_line.py` → class `EntranceGate`

### 5a. Which side of the line are you on?
`_signed_distance()`:
```python
cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
return cross / length
```
- **Plain words:** this is a 2D **cross product**. Its **sign** (+/−) tells you which side of the line the point is on; its **size** tells you how far. One cheap calculation gives both. This works for a line at *any* angle, so the camera can be mounted however suits the door.

### 5b. Confirming a crossing (no flicker)
`update()`:
```python
raw_side = INSIDE if ((distance > 0) == self._inside_is_left) else OUTSIDE
if raw_side == state.pending_side: state.pending_count += 1
...
if (pending_count >= min_frames_to_confirm_side        # seen the new side for N frames
    and confirmed_side != raw_side                     # it's actually a change
    and abs(distance) >= commit_distance_px):          # they went far enough past, not just leaned
    ... set last_direction = ENTERING or LEAVING ; just_crossed = True
```
- **Why each guard:** `min_frames_to_confirm_side` stops box‑jitter at the line flipping the result; `commit_distance_px` stops a half‑step over the threshold from counting as a full entry.
- The direction is **sticky** — it remembers your last crossing instead of flickering back to "UNKNOWN."

### 5c. "Check in/out only from outside"
Used by the decision engine (next section): a wave is ignored if you're confirmed **INSIDE**, because the camera is mounted outside the door. If IN/OUT comes out backwards for a mounting, the **Display tab → Flip IN/OUT** button swaps it (`flip_inside_side()`), stored as a 0–1 ratio so it survives restarts and any resolution.

---

## 6. Face recognition (who is it?)

Three files: `face_id/embedder.py` (make the face "fingerprint"), `face_id/recognizer.py` (match it), `face_id/live_recognition.py` (decide *when* to run it).

### 6a. Turning a face into numbers
`embedder.py → FaceEmbedder.detect_and_embed()`:
```python
self._app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"],
                         allowed_modules=["detection","recognition"])
...
embedding = face.embedding.astype(np.float32)
embedding = embedding / np.linalg.norm(embedding)   # L2-normalize
```
- **Plain words:** InsightFace's "buffalo_s" model turns a face into a **512‑number vector** (an "embedding"). Same person → similar numbers; different people → very different numbers.
- `allowed_modules=["detection","recognition"]` → don't load age/gender/landmark models we don't use (faster start).
- **L2‑normalize** → scale the vector to length 1, so comparing two faces becomes a simple multiply (cosine similarity = dot product).

### 6b. Matching — fast even with thousands of staff
`recognizer.py → recognize()`:
```python
similarities = embedding_matrix @ live_embedding     # one matrix multiply vs ALL stored faces
best_idx = np.argmax(similarities)                   # the closest stored face
...
if best_similarity < self._min_similarity: return "not recognized"
if (best_similarity - runner_up_similarity) < self._min_margin_over_runner_up: return "not recognized"
```
- **Line by line:**
  - `embedding_matrix @ live_embedding` → compare the live face to **every** stored face at once (one fast math operation, not a slow Python loop).
  - `argmax` → pick the best match.
  - **Threshold check** → if even the best match isn't close enough, say "UNRECOGNIZED" rather than guess. For attendance, a wrong name is worse than "I don't know."
  - **Margin check** → the winner must beat the best *different* person by a margin; otherwise it's too ambiguous (two look‑alikes) and we refuse. This matters as headcount grows.
- Thread‑safe: a `threading.Lock` guards the stored arrays so re‑loading after a new registration can't collide with a live match in progress.

### 6c. Running recognition only when needed
`live_recognition.py → LiveFaceRecognitionCache.get_recognition()`:
```python
if len(person.position_history) < min_track_stability_frames: return None   # too new, wait
if recognized: recheck every re_recognition_interval_seconds (5s)
else:          recheck every unrecognized_retry_interval_seconds (1s)
```
- **Why:** face recognition is the expensive step. Someone's identity doesn't change frame to frame, so once recognized we only re‑check every 5s. If *not* recognized yet, we retry quickly (1s) so a bad first frame doesn't lock them out. This caching is a big reason multiple people don't tank performance.

---

## 7. The attendance decision (CHECK IN vs CHECK OUT vs nothing)

**File:** `face_id/identified_attendance.py` → `IdentifiedAttendanceDecisionEngine.evaluate()`. This is the rulebook; it runs only when a recognized person waves.

In order, it checks:
```python
if not wave_confirmed: return nothing
if employee_id is None: return "not recognized"                         # unknown person -> never recorded
if require_wave_from_outside and confirmed_side == INSIDE: return blocked # must be outside
last_action = get_todays_last_action(employee_id)
allowed_next = _VALID_NEXT_ACTION.get(last_action)   # None->CHECK IN, CHECK IN->CHECK OUT, CHECK OUT->None
if allowed_next is None: return "already done today"
if allowed_next == "CHECK OUT": enforce minimum-gap since check-in
... duplicate_guard check ...
record_event(...)   # save it
```
- **The state machine** (`_VALID_NEXT_ACTION`) is the core rule: first wave of the day = **CHECK IN**, next = **CHECK OUT**, after that nothing until tomorrow. It reads the **database**, not memory, so it survives a person's track id changing or even an app restart mid‑day.
- **Minimum gap:** a CHECK OUT is refused until N hours after CHECK IN (default 1), so an accidental second wave right after arriving can't check you out. Read live from settings each time.
- **Why the database, not the track id:** earlier versions kept "already checked in" in memory keyed by track id — if the id churned, the memory reset. Using the stable `employee_id` + database fixes that permanently.

`status_label_for_employee()` in the same file is a pure read that turns "last action today" into the words shown on screen (CHECKED IN / CHECKED OUT / NOT CHECKED IN). In `web_dashboard.py` it's wrapped by `_cached_status_label()` (a 2‑second cache) so the live screen doesn't hit the database every frame per person.

---

## 8. The database (durable memory)

**Files:** `database/schema.py`, `database/connection.py`, and the `*_repository.py` files.

### 8a. Tables
`schema.py` defines four tables: `employees`, `face_embeddings` (many per employee — one per angle), `attendance_events` (every CHECK IN/OUT), `system_settings` (key/value), and `admin_users` (login).
- `ON DELETE CASCADE` on the foreign keys means deleting an employee also deletes their faces and attendance automatically.

### 8b. Connection settings
`connection.py → initialize_database()`:
```python
conn.execute("PRAGMA foreign_keys = ON;")   # enforce the links between tables
conn.execute("PRAGMA journal_mode = WAL;")  # readers don't block the writer
```
- **WAL (Write‑Ahead Logging):** lets the dashboard read records while the detection loop writes a new event, without them blocking each other. Important for a live system.

### 8c. Repositories = plain functions, no ORM
`attendance_repository.py` has the queries. The clever one is `work_day_start()`:
```python
boundary = now.replace(hour=boundary_hour, ...)   # e.g. 6am
if now < boundary: return boundary - 1 day        # 2am still belongs to yesterday's work day
return boundary
```
- **Why:** the attendance "day" resets at **6am**, not midnight — so someone who checks out at 8pm can't check back in at 12:05am; the cycle only resets at 6am. Every "today" query uses this boundary.

---

## 9. Admin login & security

**Files:** `database/auth_repository.py`, routes in `web_dashboard.py`.

### 9a. Passwords are never stored as text
`auth_repository.py`:
```python
generate_password_hash(password, method="pbkdf2:sha256")   # store a salted one-way hash
check_password_hash(stored_hash, password)                 # compare without ever un-hashing
```
- **Plain words:** we store a scrambled, one‑way version of the password. Even someone with the database file can't read it. Login compares scrambles.

### 9b. The whole site is gated
`web_dashboard.py → _require_login_gate()` (runs before every request):
```python
if request.endpoint in _PUBLIC_ENDPOINTS: return   # login page + auth APIs are open
if _is_authenticated(): return                      # logged in -> allow
if request.path.startswith("/api/"): return 401     # APIs: clean "not logged in"
return redirect("/login")                            # pages: send to login screen
```
- **First run:** if no admin exists, `/api/auth/status` reports `needs_setup`, and the login page shows a "create admin account" form instead. `create_admin()` refuses to run twice.
- **Session:** on login we set `session["admin_user"]`; Flask signs the cookie with a secret key that's generated once and saved in the database (`get_or_create_secret_key`) so logins survive restarts.
- **Change password:** `/api/auth/change_password` verifies the current password, then stores a new hash.

---

## 10. The web dashboard & its APIs

**Files:** `web_dashboard.py` (server + routes), `templates/dashboard.html` (the whole UI), `templates/login.html`, `static/` (your logo assets).

- The page is one HTML file with a left **sidebar** (Hudasoft branding, nav, live status, Admin card) and tabs: **Live Monitor, Employees, Records & Logs, Monthly Reports, Register Biometrics, Display/Kiosk Mode, System Settings**.
- The browser **polls** small JSON endpoints every 1–4 seconds (`/api/status`, `/api/attendance/today`) and shows the camera via an **MJPEG stream** (`<img src="/video_feed">`).
- **Security detail (XSS):** every value that comes from the database (names, departments, event text) is passed through an `esc()` helper before being put on the page, so a malicious name like `<img onerror=…>` renders as text instead of running.
- **Key API routes:** `/api/attendance/records` + `/export` (browse/CSV, with 400 errors on bad dates), `/api/monthly` (the report maths), `/api/employees/...` (deactivate/delete), `/api/settings/...` (day boundary, min gap, camera source, mirror), `/api/calibration/...` (move/flip the line), `/api/registration/...` (enrolment), `/api/manual_event` (manual override, validated against the same state machine).

---

## 11. Registration (enrolling a face from the browser)

**Files:** `web_dashboard.py → /api/registration/*`, `face_id/registration.py`.

- The browser shows the **backend's own camera feed** (never opens a second camera — a webcam can only be opened by one program, which is why the old "Device in use" error happened).
- For each angle (front/left/right), `evaluate_capture_quality()` checks: exactly one face, good confidence, big enough, and — importantly — **real head pose** via `_estimate_yaw_ratio()` (compares nose position to the eye midpoint). "Left" and "right" must be genuinely opposite turns, not a timer guess.
- **Duplicate guards at save:** block the same **name** (case‑insensitive) and the same **face** under a new name (it runs the captured faces through the live recognizer — a match means that person already exists).
- On finalize it saves the employee + embeddings and calls `recognizer.reload_known_faces()` so they're recognized immediately, no restart.

---

## 12. Smooth video: the threaded display (the big performance feature)

**File:** `web_dashboard.py` → `_detection_loop()` (brain) and `_display_loop()` (eyes).

**The problem:** if one loop both analyses *and* draws, the video can only refresh as fast as the slow analysis → choppy.

**The fix — split them:**
- `_detection_loop` runs the pipeline and, instead of drawing, **publishes a snapshot** of results:
```python
render_people = [SimpleNamespace(track_id=p.track_id, bbox=p.bbox, confidence=p.confidence,
                                 position_history=list(p.position_history)) for p in tracked_people]
state.render_snapshot = (render_people, gate_states, wave_results, status_labels, display_names)
```
  - `list(p.position_history)` → a **copy**, so the display thread never reads a list while this thread is changing it.
- `_display_loop` runs ~30 times a second:
```python
with state.source_lock:
    raw = state.current_source.read_frame()     # freshest camera frame
if mirror: raw = cv2.flip(raw, 1)
frame = raw.copy()
with state.renderer_lock:
    renderer.render(frame, *snapshot, web_mode=True)   # draw the LAST known boxes on the FRESH frame
state.latest_jpeg = cv2.imencode(".jpg", frame, [IMWRITE_JPEG_QUALITY, 72])
```
- **Plain words:** the eyes grab the newest camera frame at full speed and paint the most recent boxes on it. The brain updates those boxes whenever it can. Result: **smooth video even when detection is slow.** The only visible trade‑off is boxes trailing a hair behind very fast motion.
- **Locks keep it safe:** `source_lock` stops a camera switch from yanking the camera out while the display reads it; `renderer_lock` stops the two threads from touching the renderer's internal state at once. **No detection/wave/face code was changed** — only where drawing happens.

### The camera loop never dies
`_detection_loop` + `_open_camera_source()`: a failed camera open just sets the source to `None` and the loop **idles and retries** instead of the thread crashing. A pending source switch is always handled first, so "Use PC Webcam" or a corrected IP always recovers. This is the fix for "a wrong IP froze everything."

---

## 13. Every performance improvement, in one place

| Improvement | Where | What it does | Accuracy impact |
|---|---|---|---|
| OpenVINO on Intel GPU | `person_detection/detector.py` | Same YOLO weights, runs on the iGPU | None (identical weights) |
| GPU auto‑detect + CPU fallback | `config/settings.py _auto_detect_device`, `detector.py` | Uses CUDA/MPS/Intel if present, never breaks | None |
| Inference size 416→384 | `config/settings.py inference_size` | Fewer pixels for YOLO | Tiny; margin kept via 0.35 conf |
| Confidence 0.45→0.35 | `config/settings.py` | Detects people in harsh light | Slightly more recall |
| MediaPipe crop 260→224 | `config/settings.py hand_crop_max_dimension_px` | Matches model's native size | None |
| Unknown‑retry 0.75→1.0s | `config/settings.py` | Fewer face runs in a crowd | None |
| JPEG quality 80→72 | `web_dashboard.py` encodes | Lighter encode + browser decode | Display only |
| Threaded display | `web_dashboard.py _display_loop` | Smooth video independent of detection speed | None (plumbing) |
| Latest‑frame reader | `camera/latest_frame_reader.py` | No lag build‑up from buffered frames | None |

---

## 14. Fast "where is X" index

- **Start the app / routes / threads:** `web_dashboard.py`
- **Detect people:** `person_detection/detector.py`
- **Track people:** `person_tracking/tracker.py` (+ `tracked_person.py`)
- **Wave logic:** `wave_detection/detector.py` → `oscillation.py` + `body_motion.py` (+ `action_guard.py`)
- **Door line / direction:** `direction_estimation/entrance_line.py`
- **Face:** `face_id/embedder.py` (make), `recognizer.py` (match), `live_recognition.py` (when), `registration.py` (enrol)
- **Decide check in/out:** `face_id/identified_attendance.py`
- **Database:** `database/schema.py`, `connection.py`, `attendance_repository.py`, `employee_repository.py`, `settings_repository.py`, `auth_repository.py`
- **Config (all the tunable numbers):** `config/settings.py`
- **Drawing overlays:** `display/renderer.py`
- **UI:** `templates/dashboard.html`, `templates/login.html`

---

*Tip for studying: pick one feature, open the file, and trace it with this guide beside you. If you can explain sections 2, 4, 6, 7 and 12 in your own words, you can answer almost any question about this project.*
