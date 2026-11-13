# Hudasoft Attendance System — Master Project Document

**Prepared for:** study + live demonstration to the CTO
**Project:** Face‑recognition + hand‑wave contactless attendance system ("Hudasoft")
**Author:** Syed Hammad Haider — AI Engineer Intern, Hudasoft

> This is the **single reference** for the whole project: what it does, how it works, every library/tool and its **version**, which feature each one powers, the design decisions, how to run it, and the questions you'll likely be asked. If you can explain Sections 2, 3, 5, and 10, you can defend this project from any angle.

---

## 1. What it is (one paragraph)

Hudasoft is a **contactless attendance system** for an office entrance. A camera mounted **outside** the door watches people arrive and leave. A **registered employee simply waves** — the system recognizes their face, confirms the wave gesture, checks they're on the outside of the door, and automatically logs a **CHECK IN** (or **CHECK OUT** later). Everything — live monitoring, enrolling faces, viewing/correcting records, monthly reports, and settings — is managed from **one web dashboard** in the browser, secured behind an admin login. No cards, no fingerprints, no physical contact.

---

## 2. How it works — end to end

For every camera frame, this pipeline runs:

```
Camera ─► Person Detection (YOLOv8n) ─► Tracking (ByteTrack)
      ├─► Face Recognition (InsightFace)  ┐
      ├─► Wave Detection (MediaPipe Hands)┤ (run per person)
      └─► Entrance Line: inside/outside?  ┘
                         │
                         ▼
        Attendance Decision (state machine) ─► Database (SQLite)
                         │
                         ▼
             Web Dashboard (Flask) shows it live
```

1. **Detect** every person (YOLOv8n).
2. **Track** each person so they keep a stable ID across frames (ByteTrack).
3. **Recognize** who they are by face (InsightFace) **and** detect a wave (MediaPipe) — per person, in parallel.
4. **Locate** them relative to a virtual **entrance line** (inside vs outside).
5. **Decide**: a recognized person's wave, from outside, triggers the correct once‑per‑day action, saved to the database.
6. **Display** it all live in the browser dashboard.

---

## 3. Technology stack — versions + what each is used for

**Runtime environment**
| Thing | Version | Notes |
|---|---|---|
| OS (deployment) | Windows (tested on a Lenovo IdeaPad, Intel i5‑1235U, integrated graphics) | runs on a normal laptop |
| Python | **3.11.9** | |
| PyTorch | **2.14.0+cpu** | pulled in by Ultralytics; CPU build |

**Core libraries (from `requirements.txt`, with the installed runtime versions where known)**
| Library | Version (pin / installed) | Powers which feature |
|---|---|---|
| **ultralytics** (YOLOv8) | `>=8.2.0` (installed **8.4.165**) | Person detection — finds people in each frame (`yolov8n` model) |
| **openvino** | **2026.4.1** (installed separately) | Speed‑up — runs the same YOLO on the Intel iGPU/NPU (FP16) |
| **supervision** | `==0.24.0` | Multi‑object tracking (its **ByteTrack** implementation) — stable per‑person IDs |
| **mediapipe** | `==0.10.14` (pinned exactly) | Hand‑wave detection — 21‑point hand landmarks (uses legacy `mp.solutions.hands`) |
| **insightface** | `>=0.7.3` | Face recognition — "buffalo_s" model makes 512‑d face embeddings |
| **onnxruntime** | `>=1.16.0` | Runs the InsightFace ONNX models |
| **onnx** | `==1.16.0` (pinned) | Model format for InsightFace (pinned for protobuf compatibility) |
| **protobuf** | `>=4.25.3,<5` (pinned range) | Shared dependency — the one range mediapipe **and** onnx both accept |
| **opencv-python** (`cv2`) | `>=4.9.0` | Camera capture, image ops, MJPEG encoding, entrance‑line math, drawing overlays |
| **numpy** | `>=1.26.0` | All array/vector math (embeddings, geometry) |
| **flask** | `>=3.0` | Web dashboard server + JSON APIs |
| **waitress** | `>=3.0` | Production WSGI server (runs Flask as a real, always‑on service) |
| **werkzeug** | bundled with Flask | Admin password hashing (`pbkdf2:sha256`) |
| **sqlite3** | Python standard library | The database — no separate DB server to install |

**Models used**
| Model | Where from | Role |
|---|---|---|
| **YOLOv8n** ("nano") | Ultralytics (`yolov8n.pt` + exported `yolov8n_openvino_model/`) | Person detection |
| **ByteTrack** | supervision | Tracking (no neural net of its own) |
| **MediaPipe Hands** | mediapipe | Hand landmark detection for waves |
| **buffalo_s** (SCRFD detector + recognition) | insightface | Face detection + 512‑d embeddings |

**Key pin "traps" worth knowing (good to mention to the CTO):**
- **mediapipe is pinned to exactly `0.10.14`** — newer releases removed the `mp.solutions.hands` API the wave detector uses; an unpinned install silently breaks wave detection.
- **onnx `1.16.0` + protobuf `<5`** — insightface (via onnx) normally wants protobuf ≥6, but mediapipe needs protobuf <5. `onnx==1.16.0` is the newest version that works with protobuf <5, so both run in the same process.

---

## 4. Why these choices — what's used and what's deliberately NOT

| Area | Chosen | Why | Rejected alternative (why not) |
|---|---|---|---|
| Detection | YOLOv8n | Smallest/fastest; enough to find "a person" on CPU | Larger YOLO (needs a GPU) |
| Speed | OpenVINO FP16 | Same weights on Intel iGPU, several× faster | CUDA (needs NVIDIA GPU most laptops lack) |
| Tracking | ByteTrack | Light on CPU, survives brief occlusion, no extra network | DeepSORT (runs a re‑ID net every frame) |
| Waves | MediaPipe Hands | Accurate hand landmarks, CPU‑friendly | Training a custom gesture model (data‑hungry, unnecessary) |
| Face | InsightFace (buffalo_s) | Prebuilt wheels, accurate, CPU‑friendly | dlib / `face_recognition` (needs compiling C++, install blocker) |
| Web | Flask + Waitress | One simple process; Waitress is production‑grade | Django (too heavy), Flask dev server (not for production) |
| Database | SQLite (WAL mode) | Single file, zero‑install, perfect for one site | Postgres/MySQL (separate server, overkill) |
| UI | Plain HTML/CSS/JS | No build step, instant load | React/Vue (build pipeline not needed) |
| Auth | Werkzeug hashing + Flask sessions | Salted one‑way hashing, already bundled | Storing plain passwords (insecure) |

---

## 5. Features & how each is implemented (feature → file → library)

1. **Person detection** — `person_detection/detector.py` using **Ultralytics YOLOv8n**. Auto‑uses the **OpenVINO** model on the Intel GPU if present, with a safe fallback to CPU PyTorch. Confidence threshold `0.35`, inference size `384`.
2. **Tracking** — `person_tracking/tracker.py` using **supervision ByteTrack**. Gives each person a stable `track_id` and a rolling history of positions (shared by wave + entrance line).
3. **Wave detection** — `wave_detection/` using **MediaPipe Hands**. Runs on each person's cropped box; tracks the **palm centroid**, counts **direction reversals** in "hand‑widths" (`oscillation.py`), and requires the **body to be still** (`body_motion.py`) so walking isn't mistaken for a wave. Sensitivity presets: Strict / Balanced / Relaxed.
4. **Face recognition** — `face_id/` using **InsightFace**. `embedder.py` makes a 512‑number embedding; `recognizer.py` matches it against all stored faces with one vectorized matrix multiply, with a similarity threshold + ambiguity margin (refuses to guess); `live_recognition.py` runs recognition only periodically (caching) so it's cheap.
5. **Entrance line / direction** — `direction_estimation/entrance_line.py`. A virtual line + **signed distance (2D cross product)** tells inside vs outside and ENTERING vs LEAVING; works at any camera angle; debounced to avoid flicker. "Check in/out only from outside" is enforced here.
6. **Attendance decision** — `face_id/identified_attendance.py`. A **once‑per‑day state machine** (CHECK IN → CHECK OUT → done) read from the **database** (not memory), with a **minimum shift gap** (default 1h) and a **6am work‑day boundary**.
7. **Database** — `database/` (**SQLite**, WAL mode). Tables: `employees`, `face_embeddings`, `attendance_events`, `system_settings`, `admin_users`. Plain repository functions, no ORM.
8. **Admin login & security** — `database/auth_repository.py` + routes in `web_dashboard.py`. **Werkzeug** salted password hashing, Flask signed sessions, whole dashboard gated, first‑run "create admin" screen, change‑password in Settings. Database text is HTML‑escaped (XSS protection).
9. **Web dashboard** — `web_dashboard.py` + `templates/dashboard.html`. Tabs: Live Monitor, Employees, Records & Logs, Monthly Reports, Register Biometrics, Display/Kiosk, System Settings. Live video via **MJPEG**; data via small JSON APIs polled every 1–4s. Branded with the Hudasoft logo (`static/`).
10. **Registration (enrol a face)** — `web_dashboard.py /api/registration/*` + `face_id/registration.py`. Captures front/left/right with **real head‑pose checks**; blocks duplicate name or duplicate face; recognized immediately, no restart.
11. **Records & Monthly reports** — browse/filter/CSV export; correct (Switch) or delete entries; per‑employee present/absent/percentage with a donut chart, based on each person's work‑days‑per‑week.
12. **Phone‑as‑camera** — enter a DroidCam IP in Settings to use a phone over Wi‑Fi; resilient to a wrong IP (bounded timeout, loop never dies).

---

## 6. Architecture (threads & safety)

One Flask process, **three threads**:
1. **Detection thread** (`_detection_loop`) — runs the full vision pipeline and publishes a results snapshot.
2. **Display thread** (`_display_loop`) — paints the live video at ~30 fps from the latest snapshot on the freshest camera frame, so the feed is smooth **independent of detection speed**.
3. **Flask/Waitress** — serves the browser.

Shared state lives in one `DashboardState` object, protected by locks (`source_lock`, `renderer_lock`) so a camera switch or the renderer can't be corrupted by two threads at once. The camera itself has a background **latest‑frame reader** so lag never builds up.

---

## 7. Project structure (where everything lives)

```
attendance_system/
├─ web_dashboard.py          # the app: server, routes, detection + display threads
├─ main.py                   # legacy OpenCV desktop entry (debugging only)
├─ register_employee.py      # legacy CLI enrolment (dashboard replaces it)
├─ config/settings.py        # ALL tunable numbers (models, thresholds, presets)
├─ camera/                   # webcam_source, ip_camera_source (DroidCam), factory, latest_frame_reader
├─ person_detection/         # detector.py (YOLO + OpenVINO)
├─ person_tracking/          # tracker.py (ByteTrack), tracked_person.py
├─ wave_detection/           # detector.py, oscillation.py, body_motion.py, action_guard.py
├─ direction_estimation/     # entrance_line.py (virtual gate), line_calibration.py
├─ face_id/                  # embedder, recognizer, live_recognition, registration, identified_attendance
├─ database/                 # schema, connection, *_repository.py (employee/attendance/settings/auth)
├─ display/                  # renderer.py (overlays), suggested_action.py
├─ templates/                # dashboard.html, login.html
├─ static/                   # Hudasoft logo assets
├─ yolov8n.pt + yolov8n_openvino_model/   # detection model (+ OpenVINO version)
└─ requirements.txt
```

---

## 8. Performance engineering (how it was made smooth on a no‑GPU laptop)

| Lever | What it does | Accuracy impact |
|---|---|---|
| OpenVINO on Intel GPU | Same YOLO weights, runs on the iGPU | None (identical weights) |
| GPU auto‑detect + CPU fallback | Uses CUDA/Intel if present, never breaks | None |
| Inference size 416→384 | Fewer pixels for YOLO | Tiny (margin kept via 0.35 conf) |
| Confidence 0.45→0.35 | Detects people in harsh/bright light | Slightly higher recall |
| MediaPipe crop cap →224px | Matches the model's native size | None |
| Recognition caching | Face check runs periodically, not every frame | None |
| JPEG quality 80→72 | Lighter encode + browser decode | Display only |
| **Threaded display** | Smooth video independent of detection speed | None (pure plumbing) |

---

## 9. Security & data handling (important for a biometric system)

- Dashboard is **gated behind an admin login**; passwords stored only as **salted hashes** (never plaintext).
- Flask **signed session** cookie; secret key generated once and persisted.
- All database‑sourced text is **HTML‑escaped** before display (prevents stored XSS).
- **For a real product** (roadmap): encryption at rest, HTTPS, a data‑retention policy, and **written biometric consent** — face data is sensitive personal data.

---

## 10. How to run & demo (script for the CTO meeting)

**Run it (no terminal needed day‑to‑day):**
1. Start the dashboard: `python web_dashboard.py` (or `waitress-serve web_dashboard:app` for an always‑on service — see `DEPLOYMENT.md`).
2. Open `http://localhost:5000`. First run shows **Create admin account**; after that, log in.
3. (One‑time GPU speed‑up, already done on this laptop) `pip install openvino` + `yolo export model=yolov8n.pt format=openvino half=True imgsz=384`.

**2‑minute demo flow:**
1. **Display tab** — show the entrance line sitting on the doorway.
2. **Register tab** — enrol yourself (front/left/right) → "recognized instantly."
3. Try to register again → **"already registered"** (duplicate guard).
4. **Live Monitor** — stand outside the line, **wave** → **CHECK IN** appears; KPI tiles + Today's Attendance update; Recent Events shows the crossing + check‑in.
5. **Records tab** — show the logged row; demo **Switch / Delete / Export CSV**.
6. **Monthly Reports** — present/absent/% and change someone's work‑days/week.
7. **Settings** — typo‑tolerant search, phone‑camera option, change admin password.
8. Mention the **startup log line** `Detection running via OpenVINO on 'intel:gpu'` to prove GPU acceleration.

---

## 11. Likely CTO questions — with answers

**Q: Walk me through one wave.** YOLO detects the person → ByteTrack gives a stable ID → InsightFace matches the face → MediaPipe + the oscillation maths confirm a wave while the body‑stillness check rules out walking → the entrance gate confirms they're outside → the decision engine looks up their last action today in the database and fires the valid next one → saved to SQLite → shown on the dashboard.

**Q: How do you avoid checking in the wrong person?** A match must clear a similarity threshold **and** beat the best different‑person match by a margin; otherwise it's "UNRECOGNIZED" and nothing is recorded. A miss is better than a wrong attribution.

**Q: Real wave vs walking arm‑swing?** Three layers: minimum swing amplitude (in hand‑widths), required reversals within a time window with steady rhythm, and a body‑stillness gate (a walker's whole body moves; a waver's doesn't).

**Q: Why SQLite, not a server DB?** Single‑site, zero‑install, one backup‑able file; WAL mode lets the dashboard read while the detector writes. Schema is plain SQL, so a Postgres migration later is mechanical.

**Q: Why was it slow, and how is it fixed for good?** It was CPU‑bound and the display refreshed only as fast as detection. Fixes: **OpenVINO** (same model on the Intel GPU) + a **threaded display** (smooth video independent of detection speed), plus smaller inference size and lighter streaming. None of it changed detection accuracy.

**Q: Is it secure? It's biometric.** Admin login with hashed passwords, signed sessions, XSS‑escaped output. For production I'd add encryption at rest, HTTPS, retention policy, and consent.

**Q: How would you productize it?** One‑click Windows installer (PyInstaller) with everything bundled, auto‑start on boot, a stated minimum spec, per‑client licensing, code signing, and the security/compliance items above.

---

## 12. Limitations & roadmap (raise these yourself — it reads as maturity)

- **Performance depends on the host machine** → ship a minimum spec + the installer.
- **Accuracy depends on camera quality, lighting and distance** → good placement; optional image normalization.
- **Relaxed sensitivity is eager in a crowd** → switch to Balanced/Strict.
- **Biometric compliance not fully productized** → consent + encryption + retention.
- **Single admin role, single site** → roles (viewer/HR/admin), audit logs, and SQLite→Postgres + an API for multi‑site.
- **Track‑ID churn** is handled by a lightweight duplicate‑action guard; full appearance re‑identification would be the proper long‑term fix.

---

*Study tip: master the end‑to‑end flow (Section 2), the versioned stack and why (Sections 3–4), and the demo + Q&A (Sections 10–11). That covers 90% of anything you'll be asked.*
