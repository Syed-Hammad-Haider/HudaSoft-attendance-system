# Attendance Detection System — Phase 1
## Complete Technical Documentation

**Version:** 2.2 (Final correctness pass: cross-validated interaction bugs fixed)
**Phase:** 1 (detection/tracking/wave/entrance-line) + 2 (identity + persistence)
**Status:** No authentication/admin UI yet (see Section 21.6 for honest scope boundaries)

---

## Table of Contents

1. Project Overview
2. Architecture
3. Folder Structure
4. Person Detection
5. Person Tracking
6. Wave Detection
7. Entrance Line (Virtual Gate) Logic
8. UI System
9. Configuration Reference
10. Testing Guide
11. Deployment Guide
12. Troubleshooting Guide
13. Future Phase: Face Recognition Integration
14. Future Phase: Attendance Database Integration
15. Future Improvements & Known Limitations
16. Summary of This Round of Improvements
17. Round 3: Fixes From Live Two-Person Testing
18. Round 4: Attendance State Machine, Entrance-Line Precision, and Calibration/UI Overhaul
19. Final Project Review (End of Phase 1 Refinement)
20. Round 5: Frame-Count Timing Bug, Crop Margin, and the Mirroring Question
21. Phase 2: Employee Face Recognition + Database Integration
22. Round 6: Performance Investigation — What Actually Helped, and What Didn't
23. Round 7: Final Correctness Pass — Two Interaction Bugs Found and Fixed

---

## 1. Project Overview

This system watches a camera feed in real time and, for every person visible:

- Detects them and assigns a stable tracking ID
- Watches for a genuine hand-wave gesture
- Determines whether they crossed a configurable virtual "entrance line"
  (Outside → Inside, or Inside → Outside)
- When a genuine wave is confirmed, shows a temporary **Suggested Action**
  (CHECK IN / CHECK OUT) on screen

It does **not** identify who the person is, does not store any record
anywhere, and does not connect to any database. This is intentional —
Phase 1 is the "eyes and brain" of the system; identity and record-keeping
are deliberately deferred to a later phase (see Sections 13–14), so that
this detection core can be built, tested, and hardened on its own first.

### What I Learned
A commercial product isn't built by cramming in every feature at once.
It's built by getting one layer really solid (in this case: "can I
reliably see and understand people"), before adding the next layer on
top ("can I identify them"). Trying to do face recognition and a
database in the same pass as getting detection accuracy right would
have made it much harder to tell *which* part of the system was
actually broken when something didn't work.

---

## 2. Architecture

### High-level data flow

```
 +-----------+     +--------------+     +----------------+
 |  Camera   |---->|   Person     |---->|  Person         |
 | (webcam / |     |  Detection   |     |  Tracking       |
 |  CCTV)    |     |  (YOLOv8n)   |     |  (ByteTrack)    |
 +-----------+     +--------------+     +--------+--------+
                                                   |
                     +-----------------------------+
                     v                             v
           +-------------------+        +--------------------+
           |  Entrance Line     |        |  Wave Detection     |
           |  (Virtual Gate)    |        |  (MediaPipe Hands)  |
           |  -> side + crossing|        |  -> per-hand zigzag |
           +---------+----------+        +----------+----------+
                     |                               |
                     +---------------+---------------+
                                     v
                        +-------------------------+
                        |  Duplicate Action Guard  |
                        |  (position + time based) |
                        +------------+-------------+
                                     v
                        +-------------------------+
                        |   Display / Renderer     |
                        | (boxes, hands, trail,    |
                        |  arrow, line, log, FPS)  |
                        +-------------------------+
```

### Why this shape?

Each stage only depends on the stage before it, and nothing reaches
backward. This means:
- Swapping YOLOv8 for a different detector never touches tracking code.
- Swapping ByteTrack for a different tracker never touches wave/direction code.
- Adding face recognition later is a matter of attaching an `identity`
  field to the object that already flows through this pipeline
  (`TrackedPerson`) — not restructuring the pipeline itself.

### What I Learned
This is called a **pipeline architecture** — data flows one direction
through independent stages, like a factory assembly line. The benefit:
each stage can be tested, replaced, or upgraded on its own without
understanding or breaking the others. The cost: if a later stage needs
information only an earlier stage has, that information has to be
deliberately passed forward (this is why `TrackedPerson` carries a
`position_history` buffer — both direction and wave logic need it, so
tracking computes it once and both later stages just read it).

---

## 3. Folder Structure

```
attendance_system/
+-- config/                  Central settings (the ONLY file to tune values in)
|   +-- settings.py
+-- camera/                  Video source abstraction (webcam today, CCTV-ready)
|   +-- base_source.py       Interface every camera type must implement
|   +-- webcam_source.py     Local webcam implementation
|   +-- factory.py           Picks the right source based on config
+-- person_detection/        YOLOv8 wrapper
|   +-- detection.py         Raw Detection data structure
|   +-- detector.py          PersonDetector class
+-- person_tracking/         ByteTrack wrapper
|   +-- tracked_person.py    TrackedPerson data structure (ID + history)
|   +-- tracker.py           PersonTracker class
+-- direction_estimation/    Entrance Line / Virtual Gate logic
|   +-- entrance_line.py     EntranceGate class, Direction/Side enums
|   +-- line_calibrator.py   Interactive click-to-draw line calibration (mouse input)
|   +-- line_calibration.py  Saves/loads calibration to config/entrance_line_calibration.json
+-- wave_detection/           MediaPipe Hands wrapper
|   +-- wave_state.py         Per-hand zigzag tracking state
|   +-- wave_result.py        Output objects for the UI
|   +-- detector.py           WaveDetector class
|   +-- action_guard.py       Duplicate-action suppression (ID-churn safe)
+-- display/                  All rendering
|   +-- renderer.py           Draws everything onto the frame
|   +-- suggested_action.py   Maps direction -> CHECK IN / CHECK OUT
+-- utils/                    Shared helpers
|   +-- geometry.py            bbox math
|   +-- timing.py              Cooldown, ExpiringMessage, FPSCounter
|   +-- event_log.py           Recent-events panel data
+-- main.py                   Wires everything together, runs the live loop
+-- requirements.txt           Pinned dependencies
```

---

## 4. Person Detection

**What it does:** finds every person in a video frame and returns a
bounding box + confidence score for each.

**Why it exists:** everything downstream (tracking, wave detection,
direction) needs to know *where the people are* first.

**How it works internally:** we use YOLOv8n ("nano"), a convolutional
neural network pretrained on the COCO dataset (which includes a
`person` class). The model looks at the whole frame once and predicts
bounding boxes + class + confidence for everything it recognizes; we
filter that down to just `person` detections above a confidence
threshold (0.45 by default).

**Computer Vision concepts used:** object detection, convolutional
neural networks, non-max suppression (built into YOLO's output stage,
removes duplicate overlapping boxes for the same object).

**Why this approach was chosen:** YOLOv8n is the fastest variant in
the YOLOv8 family — critical for real-time processing on a CPU, which
is what most laptops (and many budget CCTV setups) run on. Accuracy is
more than sufficient for "is a person standing here," which is all we
need in Phase 1.

**Alternative approaches:** larger YOLO variants (s/m/l) for higher
accuracy at the cost of speed; classical methods like HOG+SVM
(much less accurate, mostly obsolete now); cloud-based detection APIs
(adds network latency and ongoing cost, unsuitable for real-time local
video).

**Common beginner mistakes:** forgetting to filter by class ID (YOLO
detects 80 different object types by default, not just people);
setting the confidence threshold too low (floods you with false
positives) or too high (misses real people, especially partially
occluded ones).

**How to improve later:** swap to `yolov8s.pt` or `yolov8m.pt` if a
GPU is available and higher accuracy is needed in dense crowds; consider
fine-tuning on your own office's camera footage if accuracy in your
specific environment isn't good enough out of the box.

---

## 5. Person Tracking

**What it does:** takes this frame's detections and last frame's
tracked people, and figures out which detection belongs to which
already-known person (assigning a new ID only to genuinely new
people). Maintains a short position history per person.

**Why it exists:** detection alone has no memory — every frame is
treated independently, so "Person 1" in frame 10 and "Person 1" in
frame 11 would otherwise be unrelated boxes with no connection. Without
tracking, direction estimation and wave detection (both of which need
several frames of history) would be impossible.

**How it works internally:** we use **ByteTrack**, via the
`supervision` library. ByteTrack matches detections to existing tracks
frame-to-frame primarily using bounding-box overlap (IoU) and a Kalman
filter that predicts where each tracked person should be next, given
their recent motion. A key strength: it also considers *low-confidence*
detections (that a naive tracker would throw away) as candidates for
matching existing tracks — this is what makes it more robust through
brief occlusion (e.g. one person briefly blocking another).

**Computer Vision concepts used:** multi-object tracking,
Intersection-over-Union (IoU) matching, Kalman filtering (motion
prediction), the Hungarian algorithm (optimal assignment between
detections and existing tracks).

**Why this approach was chosen:** ByteTrack has no appearance
re-identification network to run (unlike DeepSORT), making it
significantly lighter on CPU — important for real-time performance,
especially as person count grows in a crowded entrance.

**Alternative approaches:** DeepSORT (heavier, uses an appearance
embedding network for more robust re-identification through longer
occlusions, at real CPU cost); SORT (ByteTrack's simpler predecessor,
less robust to occlusion); centroid-distance-only trackers (very
simple, but easily confused when people cross paths).

**Common beginner mistakes:** assuming a tracking ID is a permanent
identity (it isn't — see the ID-churn discussion in Section 15); not
handling the "detections require a `class_id` field" quirk of the
`supervision` library (this bit us during development — worth knowing
if you extend this code).

**How to improve later:** if occlusion-related ID switches become a
real problem in testing, consider adding a lightweight appearance
descriptor (e.g. color histogram) as a tie-breaker — short of full
re-identification, cheaper than DeepSORT, more robust than pure motion
matching.

---

## 6. Wave Detection

**What it does:** for each tracked person, looks at their hand(s) and
determines whether they are performing a **genuine, intentional wave**
— not just holding a hand up, not scratching, not a natural walking
arm-swing.

**Why it exists:** a "wave" is the one deliberate signal this phase
uses to decide when to suggest an attendance action. Getting this
right (not missing real waves, not falsely triggering on other
movement) is the single most important accuracy question in the whole
system.

**How it works internally, step by step:**

1. For each tracked person, we crop the frame to their bounding box
   (with a small margin) and run **MediaPipe Hands** on just that
   crop — not the whole frame. This is both cheaper (small image) and
   automatically associates any hand found with the correct person (no
   separate hand-to-person matching needed).
2. MediaPipe returns 21 landmark points per detected hand, including
   the wrist. We track the wrist's horizontal (x) position frame to
   frame.
3. We look for a **zig-zag pattern**: the wrist must swing past a
   minimum amplitude threshold, reverse direction, swing back, and
   reverse again — repeated at least `min_reversals_required` (3)
   times, with each reversal happening close together in time
   (`max_frames_between_reversals`). This is the core insight: a
   raised static hand produces zero reversals; a real wave produces
   several quick ones; walking arm-swing produces at most one slow
   reversal per gait cycle — too slow and too infrequent to satisfy
   the pattern.
4. Every detected hand (not just the first) is tracked independently
   and matched frame-to-frame by nearest previous wrist position, so
   two hands (or two different people's hands near a crop boundary)
   never get their motion histories mixed together.
5. Each tracked person gets ONE MediaPipe `Hands` instance, created
   only when needed and capped at a configurable maximum
   (`max_concurrent_wave_trackers`, default 8) so a crowded entrance
   can't spike CPU/memory unboundedly.

**Computer Vision concepts used:** hand landmark detection (a
specialized keypoint-detection neural network), time-series motion
analysis, peak/trough (zig-zag) detection — a classic signal-processing
technique.

**Why this approach was chosen over simpler alternatives:**
"hand raised = wave" is the naive approach and produces constant false
positives (anyone reaching for something, stretching, adjusting hair).
Requiring *oscillation* is what actually captures the intent behind a
wave — it's a deliberate, repeated gesture, and that repetition is a
detectable, measurable signal.

**Alternative approaches:** a trained gesture-classification neural
network (e.g. a small CNN/LSTM fed sequences of hand landmarks,
classifying "wave" vs "not wave") — more robust in principle, but
needs a labeled training dataset and is real over-engineering for
Phase 1's needs; optical-flow-based motion detection on the hand
region (cheaper to compute but noisier and harder to tune reliably).

**Common beginner mistakes:** triggering on a single raised hand;
not requiring *repeated* reversals (a single reversal happens
constantly from ordinary movement); forgetting a cooldown (so one
continuous wave doesn't spam dozens of triggers); assuming MediaPipe
only ever returns one hand per frame (index-0-only logic silently
drops a second hand — this was an actual bug fixed in this round of
improvements).

**How to improve later:** if scratching/pointing/other repetitive
gestures start producing false positives in real testing, consider
adding a vertical-motion check too (genuine waves are usually mostly
horizontal) or requiring the hand to be above a certain height
relative to the shoulder.

**Tuning story — why the current numbers are what they are:**
Early user testing found wave detection felt slow to respond. The
first attempt at fixing this lowered `min_reversals_required` from 3
to 2 (fewer swings needed = faster trigger) and widened the reversal
timing window. Before shipping that change, it was checked against a
systematic test: synthetic data generating a realistic slow walking
arm-swing pattern, run through the exact same detection code. **That
test caught a real problem** — at 2 required reversals, a slow walking
gait could occasionally accumulate enough reversals to falsely trigger
a wave. The fix was reverted back to 3 reversals (verified safe across
every tested parameter combination, with zero false positives on both
walking-gait and static-hand test patterns), while keeping the parts of
the speed improvement that testing showed were actually safe: a
slightly lower amplitude threshold (0.09 vs 0.12) and a slightly wider
hand-detection crop margin, both of which improve responsiveness
without reintroducing the false-positive risk. This is worth recording
plainly: **for a commercial attendance product, a false "CHECK IN" for
someone who was just walking past is a worse failure than a genuine
wave taking a fraction of a second longer to confirm** — so when those
two goals conflicted, correctness won.

### What I Learned
This is a good example of why you test *before* you ship a "faster"
change, not just trust that faster feels better. The instinct
"less strict = faster = better" is often wrong in detection systems,
because strictness is usually what's preventing false positives.
Writing a quick synthetic test (a fake walking-arm-swing pattern) took
a few minutes and caught a real bug before it reached a live camera --
that's a much cheaper way to find this kind of problem than discovering
it during a demo.

---

## 7. Entrance Line (Virtual Gate) Logic

**What it does:** determines whether a person is **ENTERING**,
**LEAVING**, or direction is still **UNKNOWN**, based on which side of
a configurable virtual line they're currently on, and whether they've
recently switched sides.

**Why it exists — and why it replaced the earlier approach:**
The original version simply looked at whether a person's net
horizontal movement was left-to-right or right-to-left. That has two
real problems: it only works if the camera happens to be angled so
"left/right" maps cleanly onto "in/out" of the building, and if a
person paused mid-movement, the label could flicker between a
direction and UNKNOWN, which looks unreliable in a live product.

**How it works internally:**

1. A line is defined by two points (as fractions of frame
   width/height, so it scales correctly regardless of camera
   resolution).
2. Every frame, for each tracked person, we compute the **signed
   perpendicular distance** from their center point to that line,
   using a 2D cross product. The *sign* tells us which side they're
   on; the *magnitude* tells us how close they are to it.
3. To avoid flicker from detection jitter right at the line, a side is
   only considered "confirmed" after being observed for
   `min_frames_to_confirm_side` (5) consecutive frames.
4. When a person's confirmed side switches — Outside → Inside or
   Inside → Outside — that's a genuine **crossing event**, and the
   resulting direction (ENTERING or LEAVING) is remembered ("sticky")
   until their next crossing, rather than flickering back to UNKNOWN.

**Interactive calibration (click-to-draw, added for CCTV deployment):**
A hardcoded line position in a config file was never going to be
realistic for a real installation — every CCTV camera is mounted at a
different height, angle, and distance from the actual doorway. So the
line is now calibrated by clicking directly on the live video:

- Press **C** to enter calibration mode. The next two left-clicks on
  the video window define the line's two endpoints — you trace it
  along wherever the real entrance appears in that specific camera's
  view.
- Press **F** to flip which side is labeled IN vs OUT, if the camera's
  physical orientation makes them come out backwards.
- Both settings are written to `config/entrance_line_calibration.json`
  immediately, and reloaded automatically next time the app starts —
  so a camera only needs to be calibrated once, on-site, after
  installation.

This is implemented via OpenCV's mouse callback mechanism
(`cv2.setMouseCallback`) — a function that gets invoked with pixel
coordinates every time the mouse clicks inside the video window, while
calibration mode is active.

**IN / OUT on-screen labels:** rather than relying on someone
remembering an abstract "inside_is_left_of_line" setting, the live
video now draws literal "IN" (green) and "OUT" (orange) pill-shaped
labels directly next to the line, on their respective sides. These are
computed geometrically from the exact same signed-distance formula
used for the real crossing classification (`EntranceGate.label_points_px`),
so the labels can never mismatch what the system is actually deciding
— what you see on screen is guaranteed to match the underlying logic.

**Why this works regardless of camera angle:** the line is just two
points in the frame — it can be drawn vertically, horizontally, or at
any diagonal to match wherever the real doorway appears in that
specific camera's field of view. The math (cross product distance)
doesn't care about the line's orientation. This was specifically
tested with a diagonal line (not just horizontal/vertical) to confirm
the IN/OUT labels repositioned correctly.

**Computer Vision / math concepts used:** 2D geometry (point-to-line
signed distance via cross product, perpendicular vector construction
for label placement), temporal debouncing (a common technique any time
a boolean/categorical signal needs to be stable against per-frame
noise), mouse-event-driven UI interaction.

**Alternative approaches:** a "region of interest" polygon instead of
a line (more expressive — could handle an L-shaped doorway — but
substantially more configuration complexity for Phase 1's needs);
depth-based proximity to a real-world door zone using a depth camera
(much higher hardware cost, out of scope here).

**Common beginner mistakes:** trying to infer direction from raw
movement direction alone (breaks for angled cameras); not debouncing
side changes (causes rapid-fire false crossing events for anyone
standing near the line); forgetting the line's placement is a
**per-installation, physical decision** — it must be recalibrated for
every different camera position, it isn't a fixed constant; assuming
the on-screen IN/OUT labels are just decoration rather than mathematically
tied to the actual classification logic.

**How to improve later:** support saving multiple named calibration
profiles (one per physical camera) rather than a single calibration
file, for sites with more than one entrance camera; allow calibrating
a curved/multi-segment entrance boundary instead of a single straight
line, for irregularly-shaped doorways.

---

## 8. UI System

**What it does:** draws every live overlay onto the video frame —
person boxes, hand boxes, movement trail, direction arrow, entrance
line, status text, Suggested Action banners, and a recent-events log.

**Why it exists:** the whole point of this product, in this phase, is
to be watched and understood by a human in real time (later phases add
automation on top). A cluttered or confusing overlay undermines trust
even if the underlying detection logic is solid — this is genuinely
part of the product, not just debug output.

**How it works internally:** the `Renderer` class receives, every
frame: the list of tracked people, each person's entrance-gate state,
each person's wave-detection result, the entrance line's pixel
coordinates, and the recent event log. It draws layers in this order:
entrance line → movement trails → person boxes (colored by state) →
direction arrows → hand boxes (colored by state) → per-person status
text → Suggested Action banners → event log panel → FPS counter.

**Color language** (from `ColorScheme` in config):
| Element | Meaning |
|---|---|
| Gray person box | Tracked, no direction established yet |
| Green person box | Last confirmed direction: ENTERING |
| Orange person box | Last confirmed direction: LEAVING |
| Yellow person box | Currently near/crossing the entrance line |
| Gray hand box | Hand visible, not currently gesturing |
| Amber hand box | Hand actively mid-gesture (building toward a wave) |
| Green hand box | Confirmed wave in progress |
| Magenta line | The entrance line itself |
| Green "IN" pill | Marks which side of the line is INSIDE |
| Orange "OUT" pill | Marks which side of the line is OUTSIDE |
| Dark green banner | Suggested Action: CHECK IN |
| Dark blue banner | Suggested Action: CHECK OUT |
| Dimmed frame + yellow instructions | Calibration mode active (click 2 points) |

**Computer Vision concepts used:** this is mostly classical image
annotation (OpenCV drawing primitives) rather than CV inference, but
it does use one CV-adjacent idea worth naming: alpha blending
(`cv2.addWeighted`) for the semi-transparent event log panel, so text
underneath remains visible.

**Common beginner mistakes:** drawing everything the same color
(forces the viewer to read text for every piece of state instead of
seeing it at a glance); not handling text running off the edge of the
frame (fixed in this round for the per-person status block — see
Section 16); overloading the frame with too much text for dense
scenes (a real, currently-open trade-off — see Section 15).

**How to improve later:** an optional "compact mode" that only shows
boxes + IDs (for a cleaner demo) versus "debug mode" showing every
confidence value (for development) — currently everything always
shows, which is right for now but wouldn't scale well to a polished
customer-facing kiosk view later.

---

## 9. Configuration Reference

All values live in `config/settings.py`, grouped by module. Selected
highlights (see the file itself for the complete, commented list):

| Setting | Default | Why |
|---|---|---|
| `DetectionConfig.confidence_threshold` | 0.45 | Standard balance point for person detection |
| `TrackingConfig.history_length` | 30 frames | ~1s at 30fps; enough for stable direction/wave analysis |
| `WaveDetectionConfig.min_reversals_required` | 3 | Verified (via synthetic gait testing) as the minimum that stays false-positive-free |
| `WaveDetectionConfig.min_swing_amplitude` | 0.09 | Tuned for faster response while remaining safe at 3 reversals |
| `WaveDetectionConfig.max_frames_between_reversals` | 10 | Timing window a genuine wave fits, walking gait does not |
| `WaveDetectionConfig.hand_crop_margin_ratio` | 0.14 | Balances catching full wave motion vs. neighbor bleed-through |
| `WaveDetectionConfig.max_concurrent_wave_trackers` | 8 | Caps CPU/memory cost in crowded scenes |
| `WaveDetectionConfig.wave_cooldown_seconds` | 5.0 | One wave = one Suggested Action |
| `EntranceLineConfig.min_frames_to_confirm_side` | 5 | Debounces jitter right at the line |
| `ActionGuardConfig.suppression_seconds` | 4.0 | Survives track-ID churn shortly after a real trigger |
| `DisplayConfig.message_display_seconds` | 2.5 | Matches the "~2-3 second" banner requirement |

**Business-dependent settings (not engineering decisions):**
- Entrance line position and IN/OUT side — set interactively (press
  `C` / `F` in the live video), saved to
  `config/entrance_line_calibration.json`. The `line_start` / `line_end`
  / `inside_is_left_of_line` values in `config/settings.py` are only
  the fallback default used before a camera has ever been calibrated.
- `CameraConfig.source` — webcam index vs. RTSP URL, depends on deployment.

---

## 10. Testing Guide

See `SETUP_GUIDE.md` Section 6 for the full 7-point manual test
checklist (detection, multi-person, direction, genuine wave, false-
positive rejection, cooldown, simultaneous multi-person waves). With
this round's changes, additionally verify:

**Entrance line test:** walk across the configured line in both
directions. Confirm the box color changes to yellow near the line, and
the `Dir:` label updates to ENTERING/LEAVING only after fully crossing
— not before.

**Duplicate-action guard test:** wave once, briefly step fully out of
frame (or behind an obstruction) right after, then quickly re-enter
roughly the same spot. Confirm you do NOT get a second Suggested Action
banner within a few seconds, even though your tracking ID likely
changed.

**Hand box test:** raise a hand — confirm a separate box appears
specifically around the hand (not just the whole body box), with a
percentage label that climbs as you begin waving and briefly shows
higher values right as a wave confirms.

**Capacity test (optional, needs 8+ people or a recorded crowd video):**
confirm the system stays responsive rather than freezing when many
people are in frame simultaneously.

---

## 11. Deployment Guide

See `SETUP_GUIDE.md` Sections 8–11 for the complete walkthrough:
USB webcam swap, IP camera (RTSP) connection, NVR/DVR connection,
at-home testing strategy (including using a phone as a fake RTSP
source), and on-site deployment steps. Nothing in this round of
improvements changes those steps — the entrance-line and UI
improvements are all "inside" the pipeline and don't affect how the
camera source is configured.

**New deployment consideration from this round:** after installing
the real camera, you must calibrate `EntranceLineConfig.line_start` /
`line_end` to match where the physical doorway actually appears in
that camera's frame — this is a one-time, per-installation step.

---

## 12. Troubleshooting Guide

See `SETUP_GUIDE.md` Section 7 for the general setup/runtime error
table. Issues specific to this round's changes:

| Symptom | Cause | Fix |
|---|---|---|
| Direction always shows UNKNOWN even after walking across the room | The entrance line isn't positioned where people actually walk | Adjust `line_start`/`line_end` in config to match your camera's view |
| CHECK IN / CHECK OUT seem reversed | `inside_is_left_of_line` doesn't match your camera's physical orientation | Flip the boolean in config |
| Status text overlapping for closely-spaced people | Known cosmetic limitation for people standing very close together (see Section 15) | No config fix currently; noted for a future layout-collision improvement |
| Some people in a large group never get wave analysis | `max_concurrent_wave_trackers` cap reached | Raise the value in config if your hardware can handle more concurrent MediaPipe instances, or accept the trade-off for very large groups |

---

## 13. Future Phase: Face Recognition Integration

This system was deliberately built so that adding face recognition
later does **not** require restructuring the pipeline:

- `TrackedPerson` already carries a stable identity-shaped slot
  conceptually (its `track_id`); a face-recognition module would run
  once per newly-confirmed-stable track, compute a face embedding, and
  attach a resolved `employee_id` alongside the existing `track_id` —
  additively, not replacing anything.
- The entrance-line crossing event and wave-confirmation event are
  already discrete, loggable moments in `main.py` — exactly where a
  future `attendance_service.record(employee_id, action, timestamp)`
  call would be inserted.
- No existing module (detection, tracking, wave, direction, display)
  would need to change shape to support this — face recognition is a
  new module that reads from the same `TrackedPerson` stream.

---

## 14. Future Phase: Attendance Database Integration

Similarly additive:
- A new `attendance_service` module would own writing to a real
  database (SQLite for a single-site deployment, or Postgres/MySQL for
  multi-site).
- The natural schema: an `attendance_events` table with
  `employee_id, action (IN/OUT), timestamp, camera_id, confidence`.
- The `DuplicateActionGuard` concept built in this round becomes even
  more valuable here — the same position+time suppression logic
  directly prevents duplicate *database rows*, not just duplicate UI
  banners.
- Admin dashboard / analytics / reports are then just read-queries
  against that table — no changes needed to the detection pipeline.

---

## 15. Future Improvements & Known Limitations

Being transparent about what's *not* solved yet, in priority order:

1. **Track ID churn** is mitigated (via the position-based duplicate
   action guard) but not eliminated. True robustness would need a
   lightweight appearance re-identification signal.
2. **Status text overlap** for people standing very close together is
   a known cosmetic issue — a proper fix needs a text-layout collision
   system, which is more complexity than justified for Phase 1.
3. **MediaPipe runs sequentially per person, every frame** — this
   doesn't scale indefinitely; past the `max_concurrent_wave_trackers`
   cap, excess people simply don't get wave analysis. A production
   system at very high foot-traffic volume would eventually want a
   frame-skipping or motion-triggered pre-filter before running
   MediaPipe at all.
4. **No automated unit tests** — the wave-detection reversal logic in
   particular is exactly the kind of pure-logic code that should have
   tests independent of the camera; this is worth adding before
   calling the system "production-hardened."
5. **No persistent structured logging** — console logs exist, but
   nothing writes a durable, queryable trail of decisions, which would
   help diagnose a real customer's "why didn't it trigger" complaint
   after the fact.

---

## 16. Summary of This Round of Improvements

| # | Problem | Why it mattered | What was done |
|---|---|---|---|
| 1 | Direction was pure movement-guessing, camera-angle-dependent, and flickered | Core spec requirement; also a trust/UX issue in a live product | Replaced with a configurable Entrance Line (Virtual Gate) using signed-distance crossing detection with debouncing |
| 2 | Wave detector only looked at hand index 0 | Silently dropped a second hand — real accuracy bug | Rewrote to track every detected hand independently via nearest-neighbor slot assignment |
| 3 | Hand-crop margin (15%) risked pulling in a neighbor's hand in crowds | Directly undermines "every hand belongs to the correct person" | Reduced default margin to 10%, documented the trade-off |
| 4 | No cap on concurrent MediaPipe instances | Real scalability risk at a crowded entrance | Added `max_concurrent_wave_trackers` cap with graceful degradation |
| 5 | Wave cooldown defeated by track-ID churn | Could cause duplicate Suggested Actions for the same physical person | Added `DuplicateActionGuard` — position+time based suppression, independent of track ID |
| 6 | No confidence values shown anywhere | Explicit spec requirement; also a debugging/demo need | Detection confidence, wave confidence (per-hand progress + overall) now rendered live |
| 7 | No hand visualization, movement trail, direction arrow, entrance line, or event log | Explicit spec requirements; also just makes the product look and feel premium | All added to the renderer, with a documented state-driven color scheme |
| 8 | Status text could overlap or run off-frame in dense scenes | Found via testing on a real multi-person photo | Compacted to 2 lines, added frame-boundary-aware placement |
| 9 | All colors were one fixed green | Spec asked for state-driven color coding | New `ColorScheme` config section drives every element's color by current state |

### This round's improvements (CCTV-readiness + accuracy feedback)

| # | Problem | Why it mattered | What was done |
|---|---|---|---|
| 10 | Entrance line was only adjustable by hand-editing config file ratios | Unrealistic for a real CCTV install — every camera is mounted differently | Added click-to-draw interactive calibration (press `C`, click two points on the live video); saved automatically to a calibration file, reloaded on next run |
| 11 | No visual indication of which side of the line meant IN vs OUT | User explicitly reported this was unclear | Added live "IN" (green) / "OUT" (orange) pill labels, computed from the same signed-distance math as the actual classification logic, so they can never mismatch |
| 12 | Wave detection felt slow/unresponsive | Direct user feedback — a core product-feel issue | Retuned thresholds for faster response; caught and reverted an unsafe change (2 reversals) via systematic synthetic-gait testing before shipping it |
| 13 | No way to fix a backwards IN/OUT reading without editing code | Real friction for a non-technical on-site setup | Added `F` key to flip sides instantly, saved automatically |
| 14 | No on-screen reminder of what the controls do | Needed for demoing to other people without a manual in hand | Added a persistent controls hint bar at the bottom of the video |

**What should happen in the next phase:** face recognition (Section
13) and attendance database integration (Section 14) are the two big
additions; both were explicitly designed for in this round so they can
be added without restructuring what already works. Before that, the
most valuable next step is genuinely testing this on the real CCTV
camera at the software house — the entrance-line math is designed to
work at any angle and was tested with a synthetic diagonal line, but
real-world confirmation on the actual hardware is the real proof.

---

## 17. Round 3: Fixes From Live Two-Person Testing

This round came directly from testing the system live with two people
walking together, and it surfaced one genuine design bug plus several
tuning/performance issues. Documented plainly, including what was
wrong, not just what was fixed.

### 17.1 The CHECK IN / CHECK OUT bug (real design flaw, now fixed)

**What was wrong:** a confirmed wave produced a Suggested Action based
on `last_direction` — the direction of that person's MOST RECENT
crossing ever, with no time limit. Because this value was deliberately
made "sticky" (Section 7) to avoid UI flicker, it never expired. The
practical effect: someone could cross in, walk deep into the office,
and wave to a colleague minutes later — and the system would still
suggest CHECK IN, because that's the last thing it remembered them
doing at the line. **The answer to "does this happen only outside or
also inside" was, honestly: also inside, using stale data. That was a
bug, not intended behavior.**

**The fix:** `PersonGateState` now records `last_crossing_time`
whenever a crossing is confirmed. `get_suggested_action()` checks
`direction_is_recent(action_window_seconds)` (default 6 seconds)
before producing any action at all. If the most recent crossing is
older than that window — or there's no crossing on record yet — a
confirmed wave produces **no Suggested Action**, rather than guessing.

**What I Learned:** "sticky" state (kept on purpose to avoid flicker)
and "always valid" are not the same thing. A value can be intentionally
persistent for display purposes while still needing a separate
freshness check before it's used to *trigger* something. Conflating
"what should I show" with "what should I act on" was the actual root
cause here.

### 17.2 The entrance line "felt fixed" (mostly a UX/focus issue, plus one real gap)

**Two separate things were going on here:**

1. **A real gap:** on the very first run, the line sits at a default
   position and nothing forces you to notice you can change it. Fixed:
   calibration mode now starts **automatically** on first run (when no
   saved calibration file exists yet) — you're required to click the
   line into place before normal detection begins, so it's no longer
   possible to miss that it's adjustable.

2. **A likely UX gotcha, worth knowing regardless:** OpenCV video
   windows only receive keyboard input when they have OS focus. If the
   terminal running `python main.py` still has focus (common if you
   launched it and didn't click the video window), pressing `C` does
   nothing — not because calibration is broken, but because the
   keypress went to the terminal instead. **Click directly on the
   video window first, then press the key.** This is now called out
   explicitly at the top of `main.py` and in the Troubleshooting
   section below.

### 17.3 Two people together: inconsistent entering/leaving detection, occasional missed detections

Three compounding causes, addressed with real (not cosmetic) fixes:

**Cause A — ByteTrack was never actually tuned.** The tracker was
being constructed with zero custom parameters (`sv.ByteTrack()`),
silently using library defaults tuned for a generic 30fps use case.
Fixed: `TrackingConfig` now exposes and wires through
`lost_track_buffer` (raised 30→60, more tolerance for brief occlusion),
`minimum_matching_threshold` (lowered 0.8→0.65, tolerates bigger
frame-to-frame position jumps), and `assumed_frame_rate` (lowered
30→15 to match this pipeline's real-world throughput, keeping its
motion prediction realistic).

**Cause B — the entrance line's per-person confirmation counter reset
completely on ID churn.** When ByteTrack drops and reassigns an ID
during close-proximity occlusion (extremely common with two people
walking together), the new ID gets a brand-new `PersonGateState` with
zero confirmed-side history — so a crossing that was "in progress"
under the old ID has to restart from scratch under the new one, and
can be missed entirely if the occlusion happens right as they cross.
Partially mitigated: `min_frames_to_confirm_side` lowered from 5 to 3,
so confirmation completes faster and is less likely to be interrupted
by a brief ID hiccup. **Full elimination of this class of problem
needs real re-identification (matching by appearance, not just
position) — that's realistically Phase 2 territory alongside face
recognition, not a Phase 1 fix.**

**Cause C — low real-world FPS makes both A and B worse.** See 17.4 —
this is the connecting thread between "tracking feels unreliable" and
"the video is laggy": they're the same root cause.

**What I Learned:** tracker defaults are tuned for a generic case, not
your specific pipeline's real frame rate and its combination of fast
motion / close proximity. "It's using a well-known tracking algorithm"
does not mean "it's tuned correctly for this application" — those are
different questions.

### 17.4 Camera/video lag: software, not (primarily) your PC

**Honest diagnosis:** this pipeline runs one YOLO inference **plus one
full MediaPipe Hands inference per visible person**, fully sequentially,
every single frame. For 2 people in frame, that's 3 separate neural
network passes per frame, on the CPU, one after another. On a typical
laptop CPU this is genuinely heavy, and the resulting lower FPS is what
was making tracking feel unreliable (Section 17.3, Cause C) — lower FPS
means bigger position jumps between the frames that actually get
processed, which is exactly what breaks IOU-based tracking matching.

This is a software architecture cost, not primarily a hardware
verdict — though a faster CPU will still help.

**What was done:** added `wave_processing_frame_interval` (default 2)
— MediaPipe Hands now only actually runs every other frame per person;
skipped frames reuse the last real result. A wave is a motion lasting
several hundred milliseconds, so sampling it at half rate has
negligible impact on detection reliability while meaningfully cutting
per-frame CPU cost. Raise this to 3 on slower hardware if lag persists;
drop to 1 for maximum responsiveness on faster machines.

**What I Learned:** not every frame needs every kind of analysis at
full rate. Detection and tracking need to run every frame (position
accuracy matters continuously), but wave detection is inherently a
"slow enough to sample" signal — recognizing which parts of a pipeline
can tolerate a lower processing rate, without silently breaking
correctness, is a real, practical way to buy back performance instead
of just accepting the cost or throwing more hardware at it.

### 17.5 How tracking IDs actually work (and why two people can get swapped)

Every frame, the tracker (ByteTrack) does three things: (1) predicts
where each existing tracked person should now be, based on their
recent motion (a Kalman filter); (2) compares those predictions against
this frame's actual detected boxes, using bounding-box overlap (IoU);
(3) uses an optimal-assignment algorithm (the Hungarian algorithm) to
decide the best one-to-one pairing between predictions and detections.
A detection that doesn't match any existing prediction well enough
becomes a brand new ID.

**Why two people close together are the hardest case for this:** when
two people are near each other, their predicted positions and actual
boxes can be close enough that the matching step picks the wrong
pairing — or, if one briefly and fully blocks the other, the blocked
person has no detection at all that frame, so their track has to
survive purely on motion prediction until they reappear (governed by
`lost_track_buffer`, Section 17.3). Fast movement makes this worse
because the predicted-vs-actual gap grows every frame that's "missed"
due to processing lag (Section 17.4).

**Direct answer:** getting different IDs frequently when close
together is a known, expected limitation of frame-to-frame
appearance-blind tracking, not a bug specific to this implementation
— every system built this way (ByteTrack, SORT, DeepSORT without its
appearance model) has this weakness to some degree. This round's
tuning (17.3) reduces how often it happens; it does not eliminate it.
The real fix is appearance-based re-identification, which is
explicitly deferred to the face-recognition phase (Section 13) rather
than solved here, because building a parallel, weaker re-identification
system just for Phase 1 would be throwaway work once real face
recognition arrives.

### 17.6 Updated Troubleshooting Entries

| Symptom | Cause | Fix |
|---|---|---|
| Pressing C or F does nothing | Video window doesn't have keyboard focus | Click directly on the video window first, then press the key |
| CHECK IN/OUT stopped appearing at all for waves | Working as intended now — a wave far from a recent crossing correctly shows nothing | Wave near the entrance line, shortly after crossing it, to get a suggestion |
| Two people together sometimes don't both register entering/leaving | Tracker ID churn during occlusion resets crossing-confirmation progress for whichever ID re-emerges | Partially mitigated (Section 17.3); full fix is Phase 2 re-identification |
| Video feels laggy with 2+ people | Software: one YOLO + one MediaPipe pass per person, every frame, sequentially on CPU | `wave_processing_frame_interval` now reduces this; raise it further (e.g. to 3) if still slow |
| IDs swap between two people who walk close together or cross paths | Inherent limitation of IoU/motion-based tracking without appearance matching | Same as above — expected, tuned to be less frequent, not eliminated in Phase 1 |

---

## 18. Round 4: Attendance State Machine, Entrance-Line Precision, and Calibration/UI Overhaul

This round came from a formal spec plus direct testing feedback focused
on one central bug (multiple CHECK IN/OUT for the same person) and
several precision/usability issues. Documented plainly, including one
real integration bug this round's own testing caught and fixed before
shipping.

### 18.1 The multiple-CHECK-IN bug -- now fixed with a real state machine

**What was wrong:** every confirmed wave near a recent line crossing
produced a Suggested Action, with no memory of what had already been
suggested for that person. A person walking in, waving, pausing, and
waving again could trigger CHECK IN twice. Worse, someone could
trigger CHECK OUT without ever having checked in.

**The fix:** `wave_detection/attendance_status.py` implements an
explicit state machine per tracked person:

```
NOT_CHECKED_IN --(valid entry + wave)--> CHECKED_IN
CHECKED_IN     --(valid exit + wave)--> CHECKED_OUT
CHECKED_OUT    --(nothing further allowed until the person's
                   track is reset -- they leave camera view)
```

Only the transitions explicitly listed above are valid. Everything
else -- CHECK IN while already CHECKED_IN, CHECK OUT while
NOT_CHECKED_IN, any repeat of the current state's own action -- is
rejected outright, structurally, not by a timing coincidence.

**Why a state machine instead of a longer cooldown:** a cooldown
(e.g. "block repeats for 30 seconds") was the obvious simpler
alternative, but it only delays the bug -- wait long enough and it
fires again. A state machine encodes the actual business rule ("you
cannot check in twice") as a structural fact.

**A real bug this round's own testing caught:** the position-based
`DuplicateActionGuard` (built in an earlier round to catch duplicate
actions across a churned tracking ID) was blind to action type -- it
suppressed ANY action firing near a recent one, regardless of label.
Integration testing found this incorrectly blocked a legitimate CHECK
OUT that happened to occur near the same spot as an earlier CHECK IN
(completely normal -- people don't teleport). Fixed by making the
guard label-aware: it now only suppresses a REPEAT of the same label
nearby, never a different, legitimately sequential one. This was
caught by a test that deliberately checked "same person, same spot,
CHECK IN then CHECK OUT" -- exactly the kind of test worth writing
before shipping a safety mechanism, not after.

### What I Learned
A "safety net" mechanism (like the duplicate-action guard) can itself
introduce a new bug if it isn't aware of the same context the primary
logic uses. Position and time alone weren't enough information to
safely suppress duplicates -- the mechanism needed to know WHAT was
being suppressed, not just WHERE and WHEN. This is a good example of
why layered safety checks need to be tested together, not just each in
isolation.

### 18.2 Entrance-line precision: rejecting hesitation and partial crossings

**What was wrong:** the line-crossing debounce only required a side to
be observed for a few consecutive frames. That's enough to filter
single-frame jitter, but not enough to distinguish a genuine walk-
through from someone leaning across the line, hesitating, or turning
back right at the threshold.

**The fix:** `EntranceLineConfig.min_crossing_commit_distance_ratio`
adds a second requirement alongside the existing frame-count debounce:
the person must have moved at least this far PAST the line (not just
barely tipped over it) before a crossing is confirmed. Tested against
three scenarios: repeated small hesitation/leaning near the line
(correctly produces no crossing), a genuine full walkthrough (still
crosses correctly), and tipping slightly over the line then turning
back (correctly produces no crossing).

**Why this approach:** the alternative would be a longer time-based
debounce, but that penalizes everyone equally, including people
walking through normally. A DISTANCE-based requirement specifically
targets the failure mode (staying near the line = suspicious;
committing away from it = genuine), without slowing down real
walkthroughs at all.

### 18.3 Wave detection: faster, with an honestly-documented limit

**What was wrong:** wave confirmation took longer than the target
~0.5-1 second, and required more swings than felt natural.

**The engineering process (not just the answer):** a systematic sweep
tested every combination of amplitude, reversal count, and timing
window against synthetic wave and gait patterns at multiple speeds --
including normal walking, brisk walking, jogging, and (as a stress
test) running-speed arm pumping. Lowering the required reversal count
below 3 was unsafe at every setting tested. Tightening the timing
window instead (from 10 frames down to 5) safely bought back real
speed -- confirmed roughly frame 6-10 instead of frame 12, while
remaining false-positive-free against normal walking, brisk walking,
and moderate jogging gait.

**Honest limitation found and kept, not hidden:** the same sweep found
that genuinely fast jogging/running-speed arm pumping (roughly 2+ full
swings per second) CAN still occasionally false-trigger, and no
amplitude/timing-only combination fully eliminates this without also
making the system too slow or insensitive to catch real waves. At that
speed, the arm motion is kinematically almost identical to a fast
deliberate wave -- there is no timing/amplitude signal left to tell
them apart. This is a genuine signal limitation, not a tuning miss.
The real fix would check the person's own forward walking SPEED
(already available from `position_history`) and suppress wave
confirmation if they're moving unusually fast -- deliberately not
added in this round to avoid over-engineering for a narrow edge case
(someone literally running through the entrance while also pumping
their arm at wave-like frequency is rare in practice).

**Additional responsiveness work:** `max_num_hands` defaults to 1
(MediaPipe's own per-call cost scales with how many hands it searches
for -- most attendance waves use one hand), and each hand crop is
downscaled to a capped pixel size before being handed to MediaPipe
(`hand_crop_max_dimension_px`), since MediaPipe's landmark model works
at a small fixed internal resolution regardless of input size, so
feeding it an unnecessarily large crop only added our own overhead for
no accuracy benefit. Together, these bought back enough performance to
restore full every-frame wave processing (`wave_processing_frame_interval`
back to 1), directly addressing feedback that the hand indicator felt
slow to update.

### What I Learned
"Can't be solved" and "wasn't solved because it wasn't worth the
complexity yet" are different statements, and it matters which one you
make. The sprint-speed edge case genuinely can't be solved with the
signal this module has access to (hand position over time alone) --
solving it needs a DIFFERENT signal (body speed) that the system
already computes elsewhere but doesn't yet cross-reference here.
Naming that clearly is more useful than either overclaiming safety or
vaguely gesturing at "future work."

### 18.4 Hand visualization: skeleton instead of a plain box

**What changed:** `HandObservation` now carries the full 21-point
MediaPipe landmark skeleton (wrist, palm, and each finger joint),
mapped to global frame coordinates. The renderer draws this as an
actual connected skeleton (dots + lines following real hand topology)
colored by state (gray = visible, amber = building toward a
confirmed wave, green = confirmed), instead of a plain bounding
rectangle.

**Why this addresses "hand detection doesn't look accurate":** a
bounding box only communicates "MediaPipe found something in this
area" -- it doesn't show what it actually understood about the hand's
shape or position. A skeleton visibly tracks each finger and the
wrist, which is both a clearer confirmation that detection is precise
and a substantially more polished, "premium product" visual than a
rectangle.

### 18.5 Calibration: vertical slider instead of two arbitrary clicks

**What changed:** `direction_estimation/line_calibrator.py` was
rewritten. Instead of clicking two arbitrary points (which could
produce any angled line), calibration mode now shows a vertical line
that follows the mouse cursor's horizontal position in real time,
spanning the full camera frame height automatically. A single click
confirms its position.

**Why this is the better default:** most camera installs (a webcam or
CCTV facing an entrance roughly head-on) only need to answer one
question -- "where left-to-right is the doorway" -- not an arbitrary
angle. A slider answers exactly that question, with less room for
producing an accidentally crooked line, and is a much faster, more
intuitive interaction than placing two separate points. (The
underlying `EntranceGate.set_line()` still accepts any two arbitrary
points, so nothing about the entrance-line math itself changed --
only how the two points get chosen during calibration.)

### 18.6 Window sizing: why the video felt "stuck small"

**Root cause found:** OpenCV's `cv2.imshow` creates a window with
`WINDOW_AUTOSIZE` by default, which locks the window to the image's
exact pixel size and silently ignores both manual resizing and the OS
maximize button. This -- not the actual camera resolution -- was why
the window couldn't be made fullscreen no matter what was tried.

**The fix:** the window is now created explicitly with
`cv2.namedWindow(..., cv2.WINDOW_NORMAL)`, which makes it freely
resizable and maximizable through the normal OS controls. A new `M`
key additionally toggles true fullscreen
(`cv2.WND_PROP_FULLSCREEN`) for a clean kiosk-style demo presentation.
Camera capture/processing resolution is unchanged (still 960x540 for
performance) -- the display window can now be freely scaled up for
viewing without any change to how much work the AI models actually do
per frame, since scaling the displayed image is just image stretching,
not re-processing.

### 18.7 Updated Testing Checklist for This Round

- **Attendance state machine:** wave + cross in, confirm CHECK IN
  appears once even if you wave again while still inside. Then cross
  out + wave, confirm CHECK OUT appears. Wave again while still
  outside -- confirm nothing further appears until you fully leave
  camera view and return.
- **Hesitation rejection:** lean across the entrance line and step
  back several times without committing -- confirm no crossing/action
  is registered. Then walk through fully -- confirm it registers
  normally.
- **Vertical slider calibration:** press `C`, move the mouse left/
  right, confirm the dashed preview line follows it and spans the
  full frame height automatically, click to confirm.
- **Fullscreen:** press `M`, confirm the window goes fullscreen; press
  again to restore. Try manually resizing/maximizing the window via
  normal OS controls even without pressing `M`.
- **Hand skeleton:** raise a hand, confirm you see connected dots
  tracing finger/wrist positions (not just a rectangle), changing
  color as a wave builds toward confirmation.

---

## 19. Final Project Review (End of Phase 1 Refinement)

**Remaining known weaknesses, stated plainly:**

1. Attendance state resets when a track is dropped (person leaves
   camera view, or their ID churns badly enough that `lost_track_buffer`
   expires). This is correct behavior for "person completely leaves the
   scene," but an ID churning while the person NEVER actually left can
   still cause a reset that shouldn't happen. Full fix needs identity
   (Phase 2 face recognition), not further Phase 1 tuning.
2. Sprint/running-speed arm-pumping can still rarely false-trigger a
   wave (Section 18.3) -- a narrow, honestly-documented edge case, not
   silently accepted.
3. This has been tested on webcam + real photos, not yet on the actual
   CCTV hardware at the software house -- that remains the real proof
   of the entrance-line-at-any-angle and calibration-workflow claims.
4. No automated unit test suite ships with the project -- all testing
   in this document was done ad hoc during development. Formalizing
   these into a real `tests/` directory (pytest) would be the single
   highest-value next step for long-term maintainability.

**Suggested next steps, in priority order:** (1) real on-site CCTV
testing, (2) a formal automated test suite covering the scenarios
already validated ad hoc in this document, (3) Phase 2 face recognition
integration, which this architecture was deliberately built to support
without requiring changes to detection, tracking, wave, or entrance-
line logic.

---

## 20. Round 5: Fixing the Frame-Count Timing Bug, Crop Margin, and the Mirroring Question

This round came directly from real hands-on testing feedback: waves
still not being detected reliably despite the skeleton correctly
drawing, a question about connecting to a real CCTV camera, and a
report that the video "flips" left/right.

### 20.1 The real root cause: frame-count timing instead of real time

**What was wrong:** the timing window that separates a genuine wave
from arm-swing (`max_frames_between_reversals`, as it was called) was
measured in **frame count**, not real elapsed time. Every earlier round
of tuning validated this against a *synthetic* frame-count simulation
that implicitly assumed a roughly fixed frame rate (around 10-15fps).

**Why this actually broke real-world detection:** this pipeline's real
achieved FPS depends heavily on hardware and how many people are in
frame -- YOLO detection, tracking, and per-person MediaPipe hand
analysis all run sequentially, every frame. On real hardware this can
easily run slower and less consistently than the 10-15fps assumed
during tuning. A "5 frame" window could represent anywhere from a
fraction of a second to multiple seconds of real time depending on
actual throughput at any given moment -- which silently broke the
wave-vs-not-a-wave timing logic in exactly the way that could make a
real, deliberate wave fail to register, even though the identical logic
tested perfectly in a synthetic frame-count simulation.

**The fix:** `HandTrack.reversal_frames` (a deque of frame indices)
became `HandTrack.reversal_times` (a deque of `time.time()`
timestamps). The config field `max_frames_between_reversals: int`
became `max_seconds_between_reversals: float = 0.20`. This makes the
system's timing behavior correct regardless of the pipeline's actual
achieved FPS at any moment -- a fundamental correctness fix, not a
tuning tweak.

**How 0.20 seconds was chosen:** re-ran the safety validation as a
genuine time-based simulation (not frame-count), testing across a
realistic range of *actual achieved FPS* (5, 8, 12, 20fps) rather than
assuming one value. Result: 0.20s is safe (zero false positives) against
walking, brisk walking, and jogging-speed gait across that whole range.

**An honest, new limitation this surfaced:** at very low achieved FPS
(around 5fps or below), the pipeline is not *sampling* fast enough to
clearly see a real wave's motion at all -- this is a fundamentally
different problem from timing-window tuning (a Nyquist-type sampling
limit, not a threshold problem), and no amount of retuning
`max_seconds_between_reversals` fixes it. If wave detection still feels
unreliable, the on-screen FPS counter is the first thing to check --
below roughly 8fps, real degradation should be expected, and the fix is
improving pipeline throughput (fewer concurrent hand trackers, lower
camera resolution, faster hardware), not further threshold tuning.

### What I Learned
A value that looks "the same" (a count) can secretly mean two very
different things depending on what it's counting against. Frame count
is only meaningful when frame rate is stable and known -- treating it
as if it always represents a fixed slice of real time is an easy,
easy-to-miss assumption to bake into timing logic. Wall-clock time
(`time.time()`) is the more honest unit whenever the thing you actually
care about is real-world duration, which a "how fast did the hand
move" question always is.

### 20.2 Hand crop margin widened again (0.14 -> 0.25)

**What was wrong:** a natural, full-arc wave -- especially with a bent
or extended elbow -- swings the hand well outside a person's tight body
bounding box. When the hand left the 0.14-margin crop mid-swing,
MediaPipe simply had nothing to detect for those frames, silently
breaking the reversal sequence exactly during the part of the motion
that mattered most.

**The fix:** widened to 0.25. This is an explicit trade-off, stated
honestly: a wider crop margin does increase the chance of a
neighboring person's hand bleeding into this crop in a genuinely
crowded, shoulder-to-shoulder queue. If that becomes a real problem
during CCTV testing, the fix is lowering this value specifically (a
targeted trade-off), not raising detection confidence (which doesn't
address attribution at all).

### 20.3 The "camera flips left/right" report -- not actually a bug

**Investigation finding:** there was no mirroring anywhere in the
codebase. The video shows the camera's true, unmirrored view.

**Why this is the physically correct default, not a bug:** if you face
a camera and raise your right hand, it correctly appears on the LEFT
side of an unmirrored image -- that's genuinely where it is from the
camera's point of view. Only "selfie mode" interfaces (video calls, a
phone's front camera) mirror the image, purely so it feels like looking
in a mirror. That convention would actively be WRONG for an entrance
camera: the entrance line and IN/OUT sides need to match the TRUE
physical layout of the room for the geometry to mean anything correct.
A real CCTV camera is never mirrored, and this is exactly the correct
default for one.

**What was added anyway, for personal comfort while self-testing:** an
optional `mirror_preview` config flag (default `False`), toggled live
with the `V` key. The flip is applied immediately after reading each
frame, BEFORE any detection/tracking/entrance-line processing runs --
so everything downstream stays internally consistent with what's shown
on screen, rather than the display and the underlying logic
disagreeing about which side is which. Toggling it automatically
triggers recalibration mode, since flipping changes what "left" and
"right" mean in pixel coordinates, silently invalidating any saved
entrance line calibration otherwise.

### What I Learned
Not everything that feels surprising is a bug. The instinct "this feels
wrong, fix it" needs a follow-up question: wrong *relative to what
expectation*? Here, the expectation (mirrored, like a selfie camera)
was the mismatched one, not the code -- and blindly "fixing" it to
match that expectation would have introduced a real bug (wrong-side
entrance line geometry) to satisfy a false alarm. Understanding *why*
something looks wrong before changing it matters as much as being
responsive to the report itself.

### 20.4 CCTV connection -- practical next step

Not a code change, but worth recording here: the concrete steps to
connect this to a real CCTV camera are (1) obtain the camera's RTSP URL
from its settings page or manual, (2) set `CameraConfig.source` in
`config/settings.py` to that URL string instead of `0`, (3) ensure the
testing laptop is on the same network as the camera, (4) run
`python main.py` completely unchanged otherwise, (5) press `C` to
calibrate the line against the real doorway. See `SETUP_GUIDE.md`
Section 9 for the full walkthrough including USB, RTSP, and NVR/DVR
variants.

### 20.5 Updated Troubleshooting Entries

| Symptom | Cause | Fix |
|---|---|---|
| Wave still not detected, even after this round's fix | Achieved FPS may be very low (sampling-rate limit, not a logic bug) | Check the on-screen FPS counter; below ~8fps expect real degradation -- reduce concurrent hand trackers or camera resolution |
| Video looks "flipped" compared to a normal webcam app | Expected -- this shows the camera's TRUE view, which is correct for a real entrance camera, not mirrored like a selfie app | Press 'V' if you want a mirrored view for personal comfort while self-testing; recalibrate the line afterward |
| IN/OUT feel backwards after pressing 'V' | Mirroring changes which pixels mean left/right | This is expected -- press 'C' to recalibrate immediately after toggling mirror mode |

---

## 21. Phase 2: Employee Face Recognition + Database Integration

This is the transition this project was explicitly built for: Phase 1
was documented from the start as "detection and suggestion only,"
deliberately deferring identity and persistence. This phase adds both,
and in doing so, **structurally fixes** a limitation that every
earlier phase's documentation flagged but couldn't solve: attendance
state living only in memory, tied to a tracking ID that could churn.

### 21.1 What this module does, in plain terms

For every stable tracked person, the system now periodically checks
"who is this" by comparing their face against a database of registered
employees. If confidently recognized, their name replaces the generic
`ID N` label on screen, and a genuine `CHECK IN` / `CHECK OUT` database
record is written the first time they wave near the entrance line each
day -- not just a temporary on-screen suggestion.

### 21.2 Why it exists

Phase 1 could tell you "someone waved while entering" but not "who."
For a real commercial attendance product, identity is the entire point
-- an attendance system that can't say whose attendance it recorded
isn't one yet. This phase closes that gap.

### 21.3 How it works internally

**Face detection + embedding (`face_id/embedder.py`):** wraps
InsightFace, an ONNX-based face analysis library. For each detected
face, it produces a 512-dimensional vector (an "embedding") --
mathematically, two embeddings from the SAME person's face end up
close together in that 512-dimensional space, and embeddings from
DIFFERENT people end up far apart. Recognition is then just "find the
closest known embedding, and check if it's close enough to trust."

**Why InsightFace instead of the library most face-recognition
tutorials use (dlib-based `face_recognition`):** this was a genuine,
practical constraint, not a style choice. `dlib` has no prebuilt
Python wheel for the environment this was developed in and requires
compiling a C++ library from source, which was impractical within
reasonable setup time. InsightFace ships prebuilt ONNX models via
`onnxruntime` and installed cleanly in under a minute. This ended up
being a genuine upgrade, not just a workaround: InsightFace's
recognition model is generally regarded as more accurate than the
older dlib ResNet embeddings it would have replaced.

**Why the "buffalo_s" (small) model pack specifically:** InsightFace
offers several model packs trading size/speed for accuracy. Matching
this project's established CPU-first philosophy (YOLOv8n over larger
variants, MediaPipe over heavier alternatives), buffalo_s is the right
default -- comfortably fast on CPU for a handful of people in frame.
Only the detection and recognition sub-models are loaded
(`allowed_modules=['detection', 'recognition']`); the pack's
landmark/age/gender models are skipped entirely since nothing here
uses them -- faster startup, lower memory, zero functional loss.

**Matching (`face_id/recognizer.py`):** because embeddings are
L2-normalized, cosine similarity between two of them reduces to a
plain dot product -- cheap to compute against every stored embedding.
A match is only accepted above `min_match_similarity` (default 0.45);
below that, the person is reported as unrecognized rather than
guessing. **Verified empirically** during development: comparing two
genuinely different real people's faces (from a real test photo)
scored around **-0.02** similarity, and a random noise vector scored
around **0.05** -- both dramatically below the 0.45 threshold,
confirming real headroom against false matches. **Honest limitation
of that test:** it only validates against an identical crop compared
to itself and against different people; it does not (and, without
multiple real photos of the same actual person, cannot) validate the
harder, more realistic case of the SAME person photographed under
different lighting/angle/expression on a different day, where
similarity would be meaningfully below 1.0 even for a correct match.
This is why the config comment and README both recommend validating
and adjusting this threshold with real repeated photos of actual
employees during a pilot, rather than trusting the default blindly.

**Why compare against every stored ANGLE separately, not one averaged
embedding per employee:** averaging an employee's front/left/right
embeddings into a single vector is tempting (fewer comparisons), but
it blurs the very information multi-angle capture is meant to
provide -- the averaged vector doesn't closely resemble any real angle
well. Comparing against each stored angle separately and taking the
best match is more compute (trivial at this scale) but meaningfully
more robust to whatever angle the camera happens to catch someone at
live.

**Registration workflow (`face_id/registration.py`,
`register_employee.py`):** a deliberately separate script from
`main.py`, since enrollment is an admin/setup task with a different
workflow (guided, per-angle prompts with live pass/fail feedback) than
continuous monitoring. Each capture is quality-gated before being
accepted: exactly one face in frame (not a bystander), detection
confidence above a minimum, and a minimum face size in pixels (not too
far from the camera). These gates exist because a registration capture
is deliberate and controlled -- there's no excuse for accepting a poor
one when the person can simply be asked to hold still and try again,
unlike live recognition which has to work with whatever motion/lighting
reality provides.

**Live recognition caching (`face_id/live_recognition.py`):** running
face detection on every stable tracked person, every single frame,
would add real avoidable cost on top of YOLO detection, tracking, and
per-person MediaPipe hand analysis -- consistent with this project's
established pattern (see `wave_processing_frame_interval`) of only
re-running an expensive model as often as the underlying reality can
actually change. A person's identity doesn't change frame to frame, so
recognition runs once when a track becomes stable, then periodically
afterward (`re_recognition_interval_seconds`, default 5s) rather than
continuously.

**The database (`database/`):** SQLite, chosen because this is a
single-site deployment (one software house, one or a handful of
entrance cameras) -- SQLite needs no separate server process, ships as
one file trivial to back up, and comfortably handles this write volume
(a handful of events per person per day, written only on confirmed
CHECK IN/OUT, not continuously). The schema (`employees`,
`face_embeddings`, `attendance_events`) is deliberately plain, portable
SQL, so migrating to Postgres later for a multi-site deployment would
be a mechanical export/import, not a redesign. `employees.is_active`
supports soft-deleting someone who leaves the company without
destroying their historical attendance records.

### 21.4 The architectural upgrade this unlocks: solving the ID-churn limitation

Every earlier phase's attendance state machine
(`wave_detection/attendance_status.py`) lived in memory, keyed by the
ephemeral ByteTrack tracking ID -- explicitly documented in Section 15
as a known limitation, because a tracking ID could change (occlusion,
brief exit from frame) and silently reset a person's "already checked
in" memory.

`face_id/identified_attendance.py`'s `IdentifiedAttendanceDecisionEngine`
replaces that in-memory lookup with a database query
(`get_todays_last_action(employee_id)`). Because a recognized person's
`employee_id` is stable regardless of how many different tracking IDs
they're assigned across a session, **this genuinely eliminates the
ID-churn problem for recognized employees** -- not a mitigation, a
structural fix. This was verified directly: a test simulated the same
employee being assigned a completely different, unrelated `track_id`
(as if occlusion had caused a new track) and confirmed the database
correctly still knew they'd already checked out, blocking a duplicate
event. Unrecognized people don't benefit from this (there's no stable
identity to key on for them), which is the honest remaining boundary
of the fix -- see 21.6.

### 21.5 What happens for unrecognized people

No attendance event is ever recorded for someone who hasn't been
recognized -- either they haven't been registered yet, or the live
match score didn't clear the threshold. This is deliberate and
conservative: guessing an identity is a worse failure than declining
to record anything. They still appear fully in the live UI (box, wave
detection, crossing detection all continue to work, labeled `ID N`
with status `UNRECOGNIZED`) so whoever is watching the screen can spot
"someone unregistered just tried to check in" and register them, or
notice a matching problem -- the system just never silently attributes
their attendance to the wrong person or invents a record for nobody.

### What I Learned
A recurring theme across this entire project's build has been: a
feature is not really done until it's been tested against the
*specific* failure mode it's meant to prevent, not just tested to
confirm it runs without crashing. "Employee A gets a new track ID
mid-session and shouldn't lose their checked-in status" is the ACTUAL
promise Phase 2 makes over Phase 1 -- so that's exactly what got
tested directly, not just "does registration work" and "does
recognition work" in isolation.

### 21.6 Honest limitations of this phase

- **The match threshold is verified safe against impostors, not yet
  validated for real same-person recall** across different days/
  lighting -- see 21.3's honest caveat. Pilot testing with real
  employees is the recommended next step before trusting this in
  production.
- **Unrecognized-person ID churn is not solved** -- only recognized
  employees get the database-backed fix in 21.4. Someone who's never
  registered still relies on the older, weaker position-based
  duplicate guard from Phase 1.
- **The face crop region assumes an upright, roughly-forward-facing
  person** (top ~45% of their body bounding box). This is realistic for
  someone walking through an entrance, but was specifically found (via
  testing) to fail for unusual poses like crouching -- worth knowing if
  a real deployment's camera angle produces unusual body poses at the
  point of recognition.
- **No admin UI yet** for viewing attendance history, deactivating an
  employee, or re-registering a face -- these exist as repository
  functions (`database/employee_repository.py`,
  `database/attendance_repository.py`) ready to be wrapped in a UI, but
  the UI itself is a reasonable next-phase addition, not built here to
  avoid scope creep beyond what was asked.

### 21.7 A real bug this phase caught, worth recording

While wiring the database layer, testing surfaced a genuine Python
bug, not just a design decision: `db_path: Path = DEFAULT_DB_PATH`
used as a function default parameter is evaluated ONCE, at function
*definition* time -- so reassigning the module-level `DEFAULT_DB_PATH`
constant afterward (e.g. to point at a test database) silently had NO
effect on already-defined repository functions. The fix was a
`get_db_path()` / `set_db_path()` pair in `database/connection.py`,
looked up fresh on every call instead of bound once at import time.
This is worth understanding for anyone extending this codebase: any
function default that's meant to be "the current configured X" should
be resolved inside the function body, not captured as a default
parameter value.

### 21.8 Testing Guide Addendum

**Registration test:** run `register_employee.py`, register yourself,
confirm each angle capture shows a green "Ready" status before
accepting, and confirm the final success message reports the correct
number of angles captured.

**Recognition test:** run `main.py` afterward -- your box should show
your real name instead of `ID N` within a few seconds of being tracked
stably. Have someone unregistered stand in frame too -- they should
show `ID N` / `UNRECOGNIZED`, never a name.

**Attendance persistence test:** wave near the entrance line while
crossing it (as a recognized employee) -- confirm a `CHECK IN` (or
`CHECK OUT`, depending on direction) appears in the event log AND
persists. To directly verify persistence, inspect
`database/attendance.db` with any SQLite browser, or run:
```python
from database.attendance_repository import get_todays_events
print(get_todays_events())
```

**ID-churn survival test (the key one):** after checking in, briefly
have another person fully block you from the camera for a couple of
seconds (forcing a likely track ID change), then step back into view
and wave again in the same direction. Confirm NO duplicate CHECK IN
fires -- the system should recognize you again (likely under a new
track ID) but correctly refuse a repeat action, because the database
already has today's CHECK IN for your employee_id regardless of which
track ID is asking.

### 21.9 Deployment Guide Addendum

`database/attendance.db` is a plain file -- back it up like any other
important file (copy it, or set up a scheduled copy job). It is
deliberately excluded from the project's distributed zip (a fresh
installation should start with a clean, empty database, not carry
over test data). Run `register_employee.py` once per real employee
after deploying to the actual CCTV camera location, since face capture
quality benefits from being captured on the actual camera that will
later recognize them, not necessarily a different laptop webcam.

---

## 22. Round 6: Performance Investigation — What Actually Helped, and What Didn't

Following through on a direct request to consider swapping underlying
technology (specifically "something other than YOLO") for better
speed/accuracy, this round did real, measured investigation rather
than guessing -- and the honest result includes one genuine win and
one deliberately-declined change, both worth recording plainly.

### 22.1 PyTorch vs ONNX Runtime for YOLO inference: investigated, NOT adopted

**The idea:** Ultralytics' `YOLO()` class can load either a `.pt`
(PyTorch) or `.onnx` model file with an identical `.predict()`
interface -- a backend swap requiring zero code changes elsewhere,
only a config value. Since `onnxruntime` was already a project
dependency (used by face recognition, Section 21), trying this
backend for person detection too was a natural, low-risk experiment.

**What was found:** benchmarked directly in the development
environment -- PyTorch: ~118-150ms/frame. ONNX Runtime (both via the
Ultralytics wrapper and a hand-rolled raw `onnxruntime` session,
bypassing wrapper overhead): consistently the SAME OR SLOWER, not
faster.

**Why this was NOT adopted despite being investigated:** the
development/testing environment used for this benchmark has only a
single CPU core, which is very unlikely to match the actual deployment
laptop and makes any backend-speed comparison unreliable and
potentially misleading -- backend performance differences (PyTorch's
internal CPU kernels vs ONNX Runtime's execution provider) can go
either way depending on CPU architecture, core count, and library
versions, in a way that image-resolution-based savings (see 22.2)
fundamentally don't. Shipping a "the ONNX version" default based on a
measurement that contradicted itself between runs would have been
guessing dressed up as an optimization.

**What was shipped instead:** the ONNX export path remains fully
available and documented (`person_detection/detector.py`'s docstring
includes the exact export command) as an easy, zero-code-change
experiment for anyone deploying on their own hardware to try and
benchmark for themselves -- the honest, practical resolution to "I
couldn't reliably determine the answer here" is to hand over a clean
way to answer it on the hardware that actually matters, not to guess.

### What I Learned
Not every reasonable-sounding optimization is real. "ONNX Runtime is
usually faster than PyTorch for CPU inference" is broadly true as a
generalization, but broadly true is not the same as verified true in
a SPECIFIC environment -- and shipping an unverified change because it
sounds right, especially one framed as a deliberate accuracy/speed
improvement, would have been a worse outcome than admitting the
benchmark was inconclusive here and providing the tool to check
properly elsewhere.

### 22.2 Inference resolution: investigated, ADOPTED (genuine, verified win)

**The idea:** Ultralytics resizes every frame to a square of
`imgsz` pixels before running the model, regardless of the camera's
actual captured resolution -- defaulting to 640x640. A CNN's compute
cost scales roughly with pixel count, so a smaller `imgsz` is a real,
architecture-independent reduction in work, unlike a backend swap.

**What was found, with numbers:** benchmarked 640, 480, 416, and 384
directly against a real multi-person test photo. 640: ~149ms/frame.
416: ~58ms/frame (~2.6x faster). 384: ~47ms/frame (~3.2x faster) but
the least-confident detection's score dropped to 0.456 -- uncomfortably
close to the 0.45 confidence threshold, risking flaky detection right
at the boundary. 416 kept all 4 people in the test photo detected
consistently across 8 repeated runs with a healthier minimum confidence
of 0.627.

**What was shipped:** `DetectionConfig.inference_size = 416` (was
implicitly 640, the Ultralytics default). This is a genuine,
verified-safe speedup that should hold on essentially any hardware,
because it's a real reduction in the amount of computation the model
has to do, not a claim about which software library happens to run
that computation faster.

**Why this matters beyond a bigger FPS number:** as established at
length in Section 20, overall pipeline throughput is the direct
limiting factor on wave detection actually working -- a faster
detection stage means more frames processed per second, which means
the system samples fast hand motion often enough to see it clearly in
the first place. This is a real contributor to "wave detection feels
more responsive," not a cosmetic speed number.

### What I Learned
The most trustworthy kind of optimization is one whose mechanism of
action you can state in one sentence without hand-waving: "fewer
pixels means less computation" holds regardless of what chip it runs
on. "This library is probably faster" needs to actually be measured on
the hardware that matters, and even then the measurement needs to be
consistent, not just the first number that came out favorable.

### 22.3 Summary of this round's changes

| # | Investigated | Adopted? | Why |
|---|---|---|---|
| 1 | ONNX Runtime backend for YOLO detection | No | Benchmark inconclusive/contradictory on available (single-core) hardware; backend speed genuinely depends on target hardware. Export path documented for the user to try on their real machine. |
| 2 | Reduced YOLO inference resolution (640 -> 416) | Yes | Verified ~2.6x speedup, hardware-independent mechanism, detection accuracy confirmed stable across repeated runs |

**Practical guidance for the person deploying this:** if `main.py`'s
on-screen FPS counter is still lower than desired on your actual
hardware after this change, the ONNX export path documented in
`person_detection/detector.py` is worth trying next -- on a multi-core
machine, unlike this development environment, it may well provide a
further real speedup. Benchmark it yourself with the exact command
provided in that file's docstring before committing to it.

---

## 23. Round 7: Final Correctness Pass — Two Interaction Bugs Found and Fixed

Before calling the project complete, a final self-review deliberately
went looking for interaction effects between settings changed in
*different* rounds, rather than assuming each round's isolated testing
was still valid after later changes. It found two real, related bugs.

### 23.1 The crop-margin / amplitude-threshold interaction

`min_swing_amplitude` (Section 6) was originally tuned against
`hand_crop_margin_ratio = 0.14`. A later round (Section 20) widened
that margin to 0.25 to stop clipping wide waves -- but amplitude is
measured as a FRACTION of crop width, not absolute pixels, so widening
the crop silently made the same physical hand swing produce a smaller
normalized amplitude (~15% smaller, by direct calculation). A full,
enthusiastic wave still cleared the old 0.09 threshold either way, but
a more modest, still-genuine wave that used to just clear it could now
fall short. Corrected to 0.08 -- a proportional correction to preserve
the originally-tuned real-world sensitivity, not a fresh loosening --
and re-verified safe against the full gait test suite.

### 23.2 The reversal-timing-gap boundary problem

Re-testing with genuinely realistic real-time sampling (actual
`time.sleep()` delays between samples, not simulated timestamps)
surfaced something the earlier synthetic sweep hadn't caught: a
relaxed-but-still-clearly-intentional wave paced right around the
0.20-second gap threshold could miss detection, because real timing
has jitter (Python scheduling, minor variance between frames) that the
earlier idealized simulation didn't model, leaving zero margin at
exactly the boundary. Corrected to 0.22 seconds -- re-verified via the
same full gait safety sweep (walking, brisk walking, jogging-speed
motion, static hand, scratching) that every relevant round has used,
confirming this stayed a deliberately narrow, safe correction rather
than reopening the false-positive risk that wider values (0.28+) were
directly tested and found to reintroduce.

### What I Learned
Every individual round of tuning in this project was tested carefully
in isolation -- but settings that interact (a crop size and an
amplitude measured relative to it; a timing threshold and the
real-world jitter around it) can each pass their own test suite while
still being subtly wrong together. The discipline that caught both of
these bugs was the same in each case: go back and re-run the FULL
safety validation using the MOST realistic test methodology available
(genuine wall-clock delays, not simulated ones) after any change,
rather than trusting that an earlier round's synthetic validation
still perfectly describes the system's current behavior. "It passed
its test when I built it" is not the same claim as "it still passes
every test after everything that came after it" -- only the second one
is actually true completion.

### 23.3 Final Configuration Summary

| Setting | Final value | Verified against |
|---|---|---|
| `WaveDetectionConfig.min_swing_amplitude` | 0.08 | Fast/relaxed/modest genuine waves (all detected); static jitter, scratching (correctly rejected) |
| `WaveDetectionConfig.min_reversals_required` | 3 | Walking, brisk walking, jogging-speed gait (all correctly rejected at every amplitude/gap combination tested) |
| `WaveDetectionConfig.max_seconds_between_reversals` | 0.22 | Same full gait suite, re-verified after the correction; relaxed-pace genuine waves now reliably detected |
| `WaveDetectionConfig.hand_crop_margin_ratio` | 0.25 | Real full-arc wave motion staying within the crop through the whole swing |
| `DetectionConfig.inference_size` | 416 | Detection count and confidence stability across repeated runs on a real multi-person photo |
| `FaceRecognitionConfig.min_match_similarity` | 0.45 | Real distinct people scoring ~-0.02, random noise ~0.05 -- both far below threshold |

This table represents the fully cross-validated final state, not just
each individual value's own isolated justification -- every row was
re-confirmed safe in the presence of every other row's current value.
