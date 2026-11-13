# Smart Attendance System — Feature Guide & Demo Notes

A face-recognition + hand-wave attendance system for a company entrance.
A camera mounted **outside** the door watches people arrive and leave; an
employee simply **waves** to check in or out. Everything is run and
managed from one **web dashboard** in the browser — no terminal needed
for day-to-day use.

This document lists every feature, how to use it in the demo, and a
short explanation of how it works under the hood (semi-technical →
technical).

---

## 1. The big picture (say this first in the demo)

**One-line pitch:** "A contactless attendance system — you walk up to the
door and wave, the camera recognises your face and logs your check-in or
check-out automatically. Admins manage everything from a web dashboard."

**How it works end to end (per camera frame):**

```
Camera → Person detection (YOLOv8) → Tracking (ByteTrack)
      → Face recognition (InsightFace) → Wave detection (MediaPipe Hands)
      → Entrance-line / side check → Attendance decision → Database → Dashboard
```

1. **Detect** every person in the frame (YOLOv8n).
2. **Track** each person across frames so they keep a stable ID (ByteTrack).
3. **Recognise** who they are by face (InsightFace embeddings matched
   against registered employees).
4. **Detect a wave** — a deliberate hand oscillation (MediaPipe Hands).
5. **Decide** whether that wave is a valid CHECK IN or CHECK OUT and
   **record** it to the database.
6. **Show** it live on the dashboard.

**Tech stack:** Python, OpenCV, YOLOv8 (Ultralytics), ByteTrack
(supervision), MediaPipe Hands, InsightFace (ONNX), SQLite, Flask +
Waitress (web server), plain HTML/CSS/JS dashboard.

**Why this split of models:** each model does one job it's best at —
YOLO finds *where* people are, InsightFace says *who* they are, MediaPipe
reads *hand gestures*. Keeping them separate makes each piece swappable
and debuggable.

---

## 2. The dashboard tabs (the demo walkthrough)

Open `http://127.0.0.1:5000`. Six tabs down the left:

### Live Monitor
- **What it shows:** the live camera with coloured boxes around people,
  their names, the entrance line, the hand skeleton while waving, and a
  big CHECK IN / CHECK OUT banner when it fires.
- **KPI tiles** across the top: **Present now**, **Check-ins today**,
  **Check-outs today**, **Enrolled**.
- **Recent Events** feed (left): a live log — `ID 5: crossed → ENTERING`,
  `Hammad (ID:2): CHECK IN`, etc. Colour-coded green (in) / red (out).
- **Today's Attendance** table (right): every check-in/out today, newest
  first, with a **Clear Today** button to wipe the day for a clean re-test.
- **Manual Override:** pick an employee and Check In / Check Out by hand
  (for when someone's hands are full or the camera missed them).
- **Expand** button: full-screen the camera.
- **Live camera view** is 16:9, un-cropped, and resizable.

*How it works:* a background thread runs the full detection pipeline and
JPEG-encodes each annotated frame into a shared buffer; Flask streams
that to the browser as MJPEG (`/video_feed`). The tiles and tables poll
small JSON endpoints (`/api/status`, `/api/attendance/today`) every 1–4
seconds.

### Employees
- Directory of everyone enrolled: name, code, department, today's status.
- **Deactivate** — stops recognising them but **keeps** their history
  (use when someone leaves the company).
- **Remove** — permanently deletes the person **and all their records**
  (two-step confirmation; irreversible).

*How it works:* `GET /api/employees` lists active employees;
deactivate/remove hit per-employee endpoints that update the database and
immediately reload the live face recogniser so the change takes effect
with no restart.

### Records
- Browse full attendance history, filter by **date range** and
  **employee**, and **Export CSV** for payroll/HR.
- **Correct mistakes:** each row has **Switch** (flip a wrong
  CHECK IN ↔ CHECK OUT) and **Delete** (remove a mis-detected entry).

*How it works:* `GET /api/attendance/records` runs a date-range SQL query
joining events to employees; Switch/Delete call
`/api/attendance/event/<id>/action` and `/delete`, which `UPDATE` or
`DELETE` that one row.

### Monthly Report
- Per-employee month view: **Working days, Present, Absent, Attendance %**.
- **Work days/week** is editable per person (5 = Mon–Fri, 6 = Mon–Sat,
  7 = every day) — so a 6-day guard and a 5-day office worker are scored
  correctly.
- Pick any month; attendance % is colour-coded (green ≥90%, amber ≥75%,
  red below).

*How it works:* for the chosen month, the server builds each employee's
**scheduled work days** (the first N weekdays of each week, N = their
work-days/week), counts how many of those days they actually checked in
(`present`), and the rest are `absent`. For the current month it only
counts days **up to today**, so the future isn't counted as absent.
Percentage = present ÷ working days. Endpoint: `GET /api/monthly`.

### Register
- Enrol a new employee from the browser: type their name (code and
  department optional), capture **front / left / right** angles, Save.
- Recognised on the live camera **immediately** — no restart.
- **Duplicate protection:** blocks registering the **same name** (case-
  insensitive) or the **same face under a different name**, with an
  "Employee already registered" message.

*How it works:* the browser shows the backend's own camera feed (it never
opens a second camera — the backend already holds the device). Each angle
is validated **server-side** using real head-pose geometry (it checks you
actually turned your head, not just that time passed). At save, the name
is checked against the database and each captured face is run through the
recogniser; a match means that person already exists, so it's rejected.
The accepted faces are stored as 512-number "embeddings" (a mathematical
fingerprint of the face), and the live recogniser is reloaded.

### Display
- **Entrance line calibration:** drag two dots to place the line across
  the real doorway, mark which side is IN, **Flip IN/OUT** if reversed,
  **Save** (instant, no restart).
- **Mirror** toggle (selfie view) — **on by default**.

*How it works:* the line is stored as ratio coordinates (0–1) so it's
resolution-independent; the browser overlays a draggable SVG on the live
feed and posts the new line to `/api/calibration/set_line`, which moves
it on the running detector and saves it to a JSON file that survives
restarts.

### Settings
- **Wave Sensitivity:** Strict / Balanced / Relaxed (default Relaxed).
- **Camera Source:** use the PC webcam, or connect your **phone over
  Wi-Fi** (DroidCam / IP Webcam) by typing its IP — no restart.
- **Minimum Shift Length:** hours that must pass between check-in and
  check-out (default 1).
- **Attendance Day Boundary:** the hour a new work day starts (default
  6am, not midnight).
- **Search box:** find any setting by typing — tolerant of typos
  (`camra` → Camera, `mirorr` → Mirror).

---

## 3. Feature-by-feature (with the "how")

### Face recognition
- **What:** identifies which registered employee a detected face belongs
  to, so attendance is tied to a real person, not a guess.
- **How:** InsightFace converts a face into a 512-number vector
  ("embedding"). Each employee has several stored (front/left/right).
  A live face is matched by cosine similarity against all stored vectors;
  it's only accepted above a confidence threshold **and** when the best
  match clearly beats the runner-up (so it won't guess between two
  look-alikes). If nothing clears the bar, the person is "UNRECOGNISED"
  and no attendance is recorded — deliberately, guessing is worse than
  declining.

### Hand-wave detection
- **What:** the trigger. A person waves; the system confirms it's a
  genuine wave (not walking arm-swing or random motion) and fires the
  check-in/out.
- **How:** MediaPipe Hands tracks 21 hand landmarks. The system follows
  the **palm centre** and counts **direction reversals** (left-right-left…)
  of a big-enough swing within a time window. A configurable number of
  reversals confirms a wave. A **body-stillness gate** rejects walking
  (a walking person's whole body moves across the frame; someone pausing
  to wave doesn't).
- **Sensitivity tiers** (monotonic — each looser than the last):
  - **Strict** — 4 reversals, large swing. Fewest false triggers.
  - **Balanced** — 3 reversals, medium swing.
  - **Relaxed** (default) — 2 reversals, small swing. Fastest; "waving
    twice" checks you in almost instantly.

### Entrance line + "outside only"
- **What:** the camera is outside the door; a wave only counts when the
  person is on the **outside** side of the line. Crossings are shown as
  `ENTERING` / `LEAVING` in Recent Events.
- **How:** a virtual line is drawn across the doorway. For each person the
  system computes which side they're on (a signed-distance calculation)
  and remembers crossings. The attendance decision ignores a wave from
  someone firmly **inside**. The line's direction also labels the action
  (entering → check in) when available.

### Attendance decision (the rules)
- A wave checks a **recognised** person in or out based on their current
  state: not-in-today → **CHECK IN**; checked-in → **CHECK OUT**; after
  both → nothing more today.
- **Minimum shift length** (default 1h): a CHECK OUT won't fire until
  that long after the CHECK IN — this both models a real workday and stops
  an accidental double-wave from instantly checking someone back out.
- **Work-day boundary** (default 6am): the "day" resets at 6am, not
  midnight — so an 8pm check-out can't be undone by a 12:05am check-in.
- A short **cooldown** and a **duplicate guard** stop one wave firing
  twice.

### Recording & corrections
- Every confirmed action is written to **SQLite** with employee, action,
  timestamp and match confidence.
- Admins can **Clear Today** (Live Monitor), **Switch/Delete** individual
  records (Records), **Clear/keep** employees (Employees), and **Export
  CSV** (Records).

### Camera flexibility (phone as camera)
- **What:** use the laptop webcam **or** a phone over Wi-Fi.
- **How:** DroidCam/IP Webcam on the phone exposes a video URL; entering
  the phone's IP in Settings points the system at that stream. The running
  detection loop releases the old camera and opens the new one between
  frames — live, no restart.

### Reliability / deployment notes (good to mention)
- Runs as one process; start with `python web_dashboard.py` (or
  `waitress-serve` for an always-on service — see `DEPLOYMENT.md`).
- SQLite in WAL mode so the dashboard can read while the detector writes.
- The database is keyed on **employee identity**, not the camera's
  temporary tracking ID, so attendance survives someone being briefly
  occluded or re-tracked.

---

## 4. Honest limitations (good to say proactively in a demo)

- **No login yet** on the dashboard — fine for a single trusted machine;
  a real multi-site product would add authentication and roles.
- **Recognition/▸wave accuracy depends on camera quality, lighting and
  distance** — a decent, well-lit, reasonably close camera performs best.
- **CPU speed matters:** the heaviest cost is YOLO person detection; on a
  slow CPU the frame rate drops. (`diagnose_performance.py` measures it.)
- **Relaxed sensitivity** is deliberately eager; at a very busy entrance
  with lots of hand movement, switch to Balanced/Strict.
- **Phone camera** needs the phone and PC on the **same Wi-Fi**.

---

## 5. Quick demo script (2 minutes)

1. **Display tab** → show the entrance line is set on the doorway.
2. **Register tab** → enrol yourself (front/left/right) → "recognised
   instantly."
3. Try to register again → **"already registered"** (show the duplicate
   guard).
4. **Live Monitor** → stand outside the line, **wave** → **CHECK IN**
   appears; KPI tiles and Today's Attendance update; Recent Events shows
   the crossing + check-in.
5. **Records tab** → show the logged row; demo **Switch**/**Delete** and
   **Export CSV**.
6. **Monthly Report** → show Present/Absent/% and change someone's
   work-days/week.
7. **Settings** → show the typo-tolerant search and the **phone camera**
   option.

*(Tip: set **Minimum Shift Length = 0** in Settings while demoing so you
can show check-in and check-out back-to-back; put it back to 8 for real
use.)*
