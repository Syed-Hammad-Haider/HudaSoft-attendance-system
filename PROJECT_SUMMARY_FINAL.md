# Project Summary — Current Features & Full Change History

## What this system does, end to end

A real-time attendance system: a camera (laptop webcam, IP camera, or
phone via DroidCam) watches an entrance. For each person in frame, it:

1. **Detects and tracks** them (YOLO + ByteTrack-style tracking).
2. **Recognizes their face** against a database of registered
   employees (InsightFace), once they've stood still long enough for a
   confident match.
3. **Watches for a genuine wave gesture** near a virtual entrance line.
4. **Decides CHECK IN vs CHECK OUT** based on which direction they
   crossed the line and their last recorded state (a database-backed
   state machine, not just "guess from the current frame").
5. **Displays** their name and the suggested action live, and logs it
   to a local SQLite database.

## Current feature list

**Core pipeline**
- Person detection & multi-person tracking, entrance-line crossing
  with direction detection and jitter debouncing.
- Face recognition against a registered-employee database, with
  vectorized matching (fast even at thousands of employees) and an
  ambiguity/margin safeguard against misidentifying similar-looking
  people.
- Wave-gesture detection with real 2D hand-motion analysis, normalized
  to the hand's own size, plus a body-stillness gate that distinguishes
  a deliberate wave from walking/jogging arm swing.
- Attendance decision engine: per-employee state persisted in the
  database (survives tracking hiccups and app restarts), a
  configurable action-repeat cooldown, and a live on-screen "Suggested
  Action" banner + event log.

**Employee registration** (`register_employee.py`)
- Guided, fully automatic multi-angle face capture (front/left/right)
  — no button press needed, it captures itself once your pose holds.
- Real geometric head-pose (yaw) detection from facial landmarks —
  verifies you've actually turned for left/right, not just waited out
  a timer.
- A mandatory countdown before each angle starts, giving real time to
  physically reposition.
- Upfront employee-code uniqueness check (before the camera even
  opens), with a graceful retry that doesn't discard already-captured
  face data if a collision still occurs.

**Admin & reporting CLIs**
- `manage_employees.py` — `list`, `deactivate` (soft, keeps history),
  `delete` (hard, with a confirmation step).
- `manage_attendance.py` — `today` (on-screen), `export` (CSV, for
  payroll/HR, with a date range).
- `diagnose_camera.py` — isolates webcam/IP-camera problems from the
  rest of the pipeline, reports achieved FPS.

**Camera flexibility**
- Local webcam, IP camera (RTSP/HTTP URL), or a phone via DroidCam/OBS
  virtual webcam — auto-detected from how `camera.source` is set.
- A background thread keeps the video feed from lagging behind the
  live camera under CPU load (a real, tested fix, not just a buffer
  flag).

**Wave detection (rebuilt this session block, see below)**
- 2D, hand-size-normalized motion tracking, robust to different wave
  styles, speeds, and directions.
- A body-stillness gate that resolves cases hand motion alone can't.
- A live `D`-key DEBUG/ANALYSIS overlay showing exactly why a wave is
  or isn't being detected in real time.
- 31 automated, camera-free tests validating the decision logic
  against realistic wave patterns and confounds (walking, jogging,
  static holds, pointing, random motion).

**Production/scale readiness**
- SQLite in WAL mode (concurrent report reads don't block live
  writes), indexed for fast date-range queries even after years of
  data.
- A rotating audit log file (`logs/attendance_system.log`).
- Sensible defaults chosen and explained everywhere, with a runtime
  `S`-key sensitivity toggle (strict/balanced/relaxed) if a specific
  deployment needs adjusting.

## What was worked on, session by session

**Sessions 1 (initial build)**: base pipeline — person detection/
tracking, entrance-line crossing, wave detection v1, face recognition,
attendance state machine, live display. Already a mature, working
system before any of the following sessions.

**Session 2 — registration reliability**: found that pressing SPACE to
capture a face gave zero feedback when rejected (looked like the app
"did nothing"). Added always-visible capture/rejection feedback and a
live debug readout (confidence, face size).

**Session 3 — admin completeness**: added `manage_employees.py delete`
(hard delete had no path before, only soft-deactivate) and
`manage_attendance.py` (there was no way to get attendance records out
of the database at all before this — CSV export added). Documented
DroidCam/IP-camera setup.

**Session 4 — pre-demo fixes**: fixed wave confidence getting
permanently stuck at 0.33 (a too-tight timing window that real-world
camera latency made worse), fixed visible camera lag when using
DroidCam (a background-thread frame reader), and fixed inconsistent
face recognition (a failed match was cached for 5 full seconds before
retrying, even after a bad single frame).

**Session 5 — registration was never checking angles**: found that
"front/left/right" capture never actually verified the person had
turned — any pose could be captured under any label. Added an
embedding-distinctness check so each angle must be measurably
different from the others.

**Session 6 — registration captured too fast**: removing the manual
SPACE press (for automatic capture) had also removed the only thing
giving a person time to physically turn. Added a mandatory countdown
before each angle.

**Session 7 — real pose detection**: replaced the timer + similarity
heuristic with genuine geometric head-yaw measurement from facial
landmarks, so the system actually detects a real head turn instead of
inferring one indirectly.

**Session 8 — scaling to thousands of employees**: found and fixed a
Python-loop face-matching bottleneck (rewritten as one vectorized
matrix operation, tested at ~0.5ms per lookup against 6,000 stored
embeddings), added an ambiguity/margin safeguard against
misidentification at scale, fixed a crash-at-the-last-second bug for
duplicate employee codes, enabled database WAL mode, and added a
persistent audit log.

**Session 9 — wave detection redesign**: found the root cause of
missed waves — the system tracked only the wrist (the least-moving
point in a natural wave), only horizontally, normalized against an
unstable reference (the person's shifting crop). Rebuilt around
palm-centroid tracking, full 2D motion, and hand-size normalization,
plus discovered (by testing) that hand motion alone can't separate a
wave from walking/jogging arm swing, and added a body-stillness gate
to resolve it. Added the debug overlay and a 22-case test suite.

**Session 10 — refinement pass**: found the wave-confirmed visual
indicator never turned back off (a real bug from the Session 9
rewrite). Found and fixed a ~0.73-second lag in recognizing that
someone had stopped walking, then found and fixed a *second* bug
introduced by the first fix (naive instantaneous speed calculation
amplified ordinary tracking jitter into false "still moving" flags) —
resolved by empirically sweeping window sizes rather than guessing,
landing on a validated 0.5-second window. Added an 8-case test suite
specifically for this.

**Session 11 — spurious re-trigger bug**: found, by tracing the
confirm/cooldown interaction directly, that a completed wave-shaped
gesture could stay permanently "primed" if a cooldown was active or
the person was still walking at that exact moment — meaning it could
fire later with zero new hand motion, purely because a cooldown timer
expired or someone happened to stop walking. Fixed at two levels: the
integration layer (detector.py) now resets gesture state
unconditionally, and the core class itself (oscillation.py) now
self-limits to one confirmation per completed gesture as a second,
independent safeguard.

## Honest, stated limitations (not hidden)

- **No real camera or human hand was available in this working
  environment at any point.** Every fix to the actual MediaPipe
  hand-detection pipeline is based on code review, reasoning about
  reported symptoms, and testing the surrounding DECISION LOGIC with
  synthetic data — not a live human test. The 31 automated tests prove
  the algorithm is internally sound against realistic *simulated*
  motion; they cannot prove MediaPipe itself reliably detects your
  specific hand, in your specific lighting, on your specific camera.
- Someone waving *while still visibly walking* toward the camera will
  not confirm until they slow down or stop — this is an intentional,
  documented trade-off (a wave mid-stride is genuinely ambiguous
  against arm-swing), not an oversight.
- The wave/body-stillness thresholds have sensible, reasoned defaults,
  but — like the face-match similarity threshold — real per-site
  camera angle, lighting, and distance can shift what "works best."
  The `S` sensitivity toggle and debug overlay exist specifically so
  this can be validated and adjusted in a real pilot without needing
  code changes.
- **A real pilot with a real camera and real employees, before a full
  rollout, remains the single most important next step** — this has
  been said in every session's handoff notes and remains true.

## Where to look for more detail
`CONTEXT_HANDOFF.md` in this same folder has the full technical
writeup for every session above, including exact root causes, code
locations, and reasoning for every default value chosen — useful if
picking this up in a new chat.
