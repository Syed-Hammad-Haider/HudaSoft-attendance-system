# Hudasoft Attendance System — Overview & Viva Preparation

**Purpose:** get you confident to explain and defend the whole project from any angle — what it does, how it's built, *what technology is used and why*, what is deliberately **not** used, the trade‑offs, and the questions an examiner is likely to ask (with answers). Pair this with the deep guide (`HUDASOFT_TECHNICAL_GUIDE.md`) for line‑by‑line detail.

---

## 1. One‑paragraph pitch

> **Hudasoft** is a contactless, face‑recognition attendance system for an office entrance. A camera mounted outside the door watches people arrive and leave. A registered employee simply **waves** — the system recognizes their face, confirms the wave gesture, checks they're on the outside of the door, and automatically logs a **CHECK IN** (or **CHECK OUT** later). Everything — live monitoring, enrolling faces, viewing and correcting records, monthly attendance reports, and settings — is managed from a single **web dashboard** in the browser, secured behind an admin login. No cards, no fingerprints, no touching anything.

---

## 2. How it works, end to end

For every camera frame:
1. **Detect** all people (YOLOv8n).
2. **Track** each person so they keep a stable ID across frames (ByteTrack).
3. **Recognize** who they are by face (InsightFace), and **detect a wave** (MediaPipe Hands) — in parallel, per person.
4. **Locate** them relative to a virtual **entrance line** (inside vs outside).
5. **Decide**: a recognized person's wave, from outside, fires the correct action from a once‑per‑day state machine, and it's **saved** to the database.
6. **Display** everything live in the dashboard.

---

## 3. Technology stack — what's used and WHY

| Layer | Technology used | Why this one | What we deliberately did NOT use (and why) |
|---|---|---|---|
| Person detection | **YOLOv8n** (Ultralytics) | Smallest/fastest YOLO; enough to find "a person" on CPU | Bigger YOLO (s/m/l) — needed a GPU; overkill for "is there a person" |
| Speed‑up runtime | **OpenVINO** (FP16) | Runs the *same* YOLO on Intel integrated GPU, several× faster | Plain PyTorch‑on‑CPU only (too slow); CUDA (needs an NVIDIA GPU most laptops lack) |
| Tracking | **ByteTrack** (via `supervision`) | Light on CPU, survives brief occlusion, no extra network | DeepSORT — needs a re‑ID neural net every frame, heavier |
| Wave gesture | **MediaPipe Hands** | Accurate 21‑point hand landmarks, runs on CPU | Training a custom gesture model — unnecessary and data‑hungry |
| Face recognition | **InsightFace** "buffalo_s" (ONNX) | Installs cleanly (prebuilt), accurate, CPU‑friendly | dlib/`face_recognition` — needs compiling C++, install blocker |
| Web server | **Flask** + **Waitress** | Simple, one process; Waitress is a real production server for Windows | Django (too heavy for this); Flask dev server in production (not robust) |
| Database | **SQLite** (WAL mode) | Single file, no server to install, perfect for one site | Postgres/MySQL — needs a separate server; overkill for one entrance |
| UI | Plain **HTML/CSS/JS** | No build step, loads instantly, easy to ship | React/Vue — a build pipeline this doesn't need |
| Auth | **Werkzeug** password hashing + Flask sessions | Salted one‑way hashing, already bundled with Flask | Storing plain passwords (insecure); a full auth framework (overkill) |

**Languages/skills shown:** Python (vision + backend), SQL (SQLite), HTML/CSS/JS (frontend), multithreading, computer vision, basic security.

---

## 4. The features (each in a few sentences)

- **Wave‑to‑check‑in/out:** a recognized person waves; the system confirms it's a real wave (palm motion + body stillness) and logs the action. One wave = check in; next = check out.
- **Face recognition:** identifies the employee so attendance is tied to a real person, never a guess. Below a confidence/margin threshold it says "UNRECOGNIZED" and records nothing.
- **Entrance line + outside‑only:** a virtual line marks the doorway; the system knows inside vs outside and only acts on waves from outside (camera is mounted outside). Direction (ENTERING/LEAVING) is shown live.
- **Once‑per‑day state machine with 6am boundary:** check in → check out → done for the day; the "day" resets at 6am, not midnight, so a late‑night checkout isn't undone after midnight.
- **Minimum shift length:** a checkout is blocked until N hours after check‑in (default 1), preventing accidental instant checkout.
- **Browser enrolment (Register Biometrics):** capture front/left/right angles with real head‑pose checks; blocks duplicate name or duplicate face; recognized immediately, no restart.
- **Employees tab:** directory with Deactivate (keeps history) and Remove (permanent, with cascade delete + double confirm).
- **Records & Logs:** browse/filter history, **Switch** a wrong IN/OUT, **Delete** a bad entry, **Export CSV** for payroll.
- **Monthly Reports:** per‑employee present/absent/percentage based on each person's work‑days‑per‑week, with a donut chart and per‑person % bars.
- **Display/Kiosk (calibration):** drag the entrance line onto the real doorway, flip IN/OUT, toggle mirror — all live, saved as resolution‑independent ratios.
- **Settings:** wave sensitivity, camera source (webcam or phone via DroidCam), min shift length, day boundary, typo‑tolerant search, and **Admin Account** (change password, sign out).
- **Admin auth:** first‑run setup screen, login/logout, hashed passwords, the whole dashboard gated.
- **Smooth live video:** a separate display thread paints the feed at ~30fps independent of detection speed.
- **Phone‑as‑camera:** enter a DroidCam IP to use a phone over Wi‑Fi; resilient to a wrong IP.

---

## 5. Key design decisions & trade‑offs (great viva material)

1. **One process, three threads (detection, display, web).** Simple to deploy; the display thread is what keeps video smooth on weak hardware. Trade‑off: shared state needs locks — handled with a few targeted locks.
2. **Database‑backed attendance state, not memory.** The once‑per‑day rule reads the database keyed by the stable `employee_id`, so it survives the tracker losing/renumbering a person, and even an app restart mid‑day.
3. **Palm‑centroid, hand‑width‑normalized wave signal.** Tracks the middle of the hand (not the wrist, which barely moves in a natural wave) and measures motion in "hand‑widths" so distance to camera doesn't matter.
4. **Body‑stillness gate for waves.** Hand motion alone can't tell a wave from a walking arm‑swing; a walking body slides across the frame, a waving one doesn't — an orthogonal, nearly‑free signal.
5. **Recognition caching.** Identity doesn't change frame‑to‑frame, so recognition runs only periodically — this is what lets several people be in frame without tanking FPS.
6. **Refuse ambiguous face matches (margin check).** Better to say "unknown" than attribute attendance to the wrong person.
7. **Multi‑angle embeddings stored separately, not averaged.** Averaging blurs the information; matching against each angle and taking the best is more robust (and made cheap by a vectorized matrix multiply).
8. **OpenVINO with full fallback.** Big speed‑up on Intel hardware, but if it's missing/broken it silently falls back — so it's safe to ship.
9. **Calibration as 0–1 ratios.** The entrance line is stored as fractions of the frame, so it's independent of resolution and survives restarts.
10. **SQLite + WAL.** Zero‑install, single‑file, and readers don't block the writer.

---

## 6. Likely viva questions — with answers

**Q: Walk me through what happens when I wave at the camera.**
A: YOLO detects me as a person; ByteTrack gives me a stable ID and remembers my recent positions; InsightFace matches my face to an employee; MediaPipe tracks my hand and the oscillation maths counts direction reversals while the body‑stillness check confirms I'm not just walking; the entrance gate confirms I'm outside the door line; the decision engine looks up my last action today in the database and fires the valid next one (check in or out); it's saved to SQLite and appears on the dashboard.

**Q: How does it tell a real wave from random hand movement or walking?**
A: Three layers — a minimum swing amplitude (in hand‑widths) rejects jitter; a required number of direction reversals within a time window with steady rhythm rejects random motion; and a body‑stillness gate rejects walking (a walker's whole body translates across the frame). All three must pass.

**Q: Why face recognition AND a wave — isn't one enough?**
A: The wave is the deliberate *trigger* (intent to check in/out); the face is the *identity*. A wave alone couldn't say who you are; a face alone would check people in just for walking past. Together they give an intentional, correctly‑attributed action.

**Q: How do you avoid checking in the wrong person?**
A: A match must clear a similarity threshold **and** beat the best different‑person match by a margin; otherwise it reports UNRECOGNIZED and records nothing. We prefer a miss over a wrong attribution.

**Q: What stops a double check‑in, or a check‑out one second after arriving?**
A: A once‑per‑day state machine (check in → check out → done) read from the database, plus a minimum‑shift gap (default 1h) before a checkout is allowed, plus a short duplicate‑action guard for track‑ID churn.

**Q: Why SQLite and not a real database server?**
A: It's a single‑site deployment (one entrance). SQLite needs no server, is one backup‑able file, and WAL mode lets the dashboard read while the detector writes. The schema is plain SQL so moving to Postgres later would be mechanical if it ever went multi‑site.

**Q: Why was it laggy, and how did you fix it permanently?**
A: It was CPU‑bound (no dedicated GPU) and the display refreshed only as fast as detection. Two fixes: (1) **OpenVINO** runs the same YOLO on the Intel GPU, several× faster; (2) a **threaded display** paints the video at ~30fps independent of detection speed. Plus smaller inference size, lighter JPEG, and a latest‑frame reader to avoid buffer lag.

**Q: Did the speed‑ups change detection accuracy?**
A: No. OpenVINO runs the identical weights (FP16 is numerically ~identical); the threaded display is pure presentation; the wave, face, and decision logic were untouched.

**Q: How does the entrance direction work at any camera angle?**
A: A virtual line is drawn on the doorway; a signed distance (2D cross product) says which side a person is on and how far. Because the line is configured to match the real door, it works for any camera angle. Crossings are debounced to avoid flicker.

**Q: Is it secure? It's biometric data.**
A: The dashboard is behind an admin login with salted, hashed passwords and signed sessions. Database‑sourced text is HTML‑escaped to prevent stored XSS. For a real product I'd add encryption at rest, HTTPS, a data‑retention policy, and written biometric consent.

**Q: What are the limitations?**
A: Performance depends on the host machine (mitigated, not eliminated); recognition/wave accuracy depends on camera quality, lighting and distance; the very relaxed sensitivity can be eager in a crowd; biometric‑data compliance isn't fully productized yet.

**Q: How would you productize it to sell to offices?**
A: Package it as a one‑click Windows installer (PyInstaller) with everything bundled; auto‑start on boot; a stated minimum hardware spec; per‑client licensing; code signing; and the security/compliance items above.

---

## 7. Glossary (say these correctly)

- **Embedding:** a face turned into a 512‑number vector; same person → similar numbers.
- **Cosine similarity:** how aligned two vectors are; here a simple dot product because vectors are normalized.
- **Reversal (wave):** a change of hand direction; several in rhythm = a wave.
- **ByteTrack:** a tracker that links detections across frames into stable IDs.
- **Signed distance / cross product:** the maths giving side‑of‑line and nearness in one step.
- **WAL:** SQLite mode where reads don't block the writer.
- **OpenVINO:** Intel's runtime that runs the same model faster on Intel GPU/NPU.
- **MJPEG:** a video stream that's just a rapid sequence of JPEG images — how the browser shows the live feed.
- **Threaded/decoupled display:** drawing the video in a separate thread so it's smooth regardless of detection speed.

---

## 8. Honest limitations & future work (good to raise yourself)

- **Hardware‑dependent speed** → ship a minimum spec + the one‑click installer.
- **Biometric compliance** → consent, encryption at rest, retention policy.
- **No multi‑site/cloud sync** → would move SQLite→Postgres and add an API.
- **Single admin role** → add roles (viewer/HR/admin) and audit logs.
- **Lighting/angle sensitivity** → better camera placement + optional image normalization.
- **Re‑identification** → the duplicate‑action guard is a lightweight stand‑in; full appearance re‑ID would be the proper fix.

---

*If you can confidently deliver §2 (the end‑to‑end flow), §3 (stack + why), and the first five answers in §6, you're ready for the viva.*
