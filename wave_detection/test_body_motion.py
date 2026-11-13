"""
Standalone test suite for wave_detection/body_motion.py.

Validates two properties that matter in practice:
1. RESPONSIVENESS: after a person actually stops walking, how quickly
   does the tracker recognize they're stationary? (An earlier
   window-based design had ~0.7s of lag here -- see body_motion.py's
   module docstring.)
2. JITTER ROBUSTNESS: does ordinary person-tracker bbox jitter (the
   detected bounding box center wobbles a little frame to frame even
   for someone standing still, since the underlying detector isn't
   pixel-perfect) ever get mistaken for "moving"?

Run with:
    python wave_detection/test_body_motion.py
"""

import random
import sys
import time

sys.path.insert(0, ".")

from wave_detection.body_motion import BodyStillnessTracker

FPS = 15
HEIGHT = 400.0

# These must match config/settings.py's WaveDetectionConfig defaults --
# kept in sync manually here since this test predates/validates that
# config.
SPEED_SMOOTHING_ALPHA = None  # unused now -- see WINDOW_SECONDS
WINDOW_SECONDS = 0.5
MAX_SPEED_FRACTION = 0.20

results = []


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}{': ' + detail if detail else ''}")
    results.append(passed)


def make_tracker():
    return BodyStillnessTracker(WINDOW_SECONDS, MAX_SPEED_FRACTION)


print("=" * 70)
print("TEST 1: responsiveness after actually stopping")
print("=" * 70)

tracker = make_tracker()
t0 = time.time()
walk_duration, still_duration = 2.0, 2.0
speed_fraction_per_sec = 0.7
became_stationary_at = None

for i in range(int((walk_duration + still_duration) * FPS)):
    t = t0 + i / FPS
    if i < walk_duration * FPS:
        cx = (i / FPS) * speed_fraction_per_sec * HEIGHT
    else:
        cx = walk_duration * speed_fraction_per_sec * HEIGHT
    tracker.update(cx, 0.0, HEIGHT, t)
    time_since_stop = (i / FPS) - walk_duration
    if time_since_stop >= 0 and tracker.is_relatively_stationary() and became_stationary_at is None:
        became_stationary_at = time_since_stop

print(f"  Became stationary {became_stationary_at:.2f}s after actually stopping.")
check("Responds within 0.5s of a real stop (was ~0.73s with the old 1.0s-window design)",
      became_stationary_at is not None and became_stationary_at <= 0.5,
      f"took {became_stationary_at:.2f}s")

print()
print("=" * 70)
print("TEST 2: jitter robustness (standing still, realistic bbox noise)")
print("=" * 70)

random.seed(11)
for jitter_pct in [0.01, 0.02, 0.03]:
    tracker = make_tracker()
    t0 = time.time()
    false_positive_frames = 0
    total_frames = int(3.0 * FPS)
    for i in range(total_frames):
        t = t0 + i / FPS
        cx = random.uniform(-jitter_pct, jitter_pct) * HEIGHT
        cy = random.uniform(-jitter_pct, jitter_pct) * HEIGHT
        h = HEIGHT * (1 + random.uniform(-jitter_pct, jitter_pct))
        tracker.update(cx, cy, h, t)
        if i > 3 and not tracker.is_relatively_stationary():  # skip warm-up frames
            false_positive_frames += 1
    false_positive_rate = false_positive_frames / total_frames
    check(f"Standing still with {jitter_pct*100:.0f}% bbox jitter never falsely flags 'moving'",
          false_positive_frames == 0,
          f"{false_positive_frames}/{total_frames} frames incorrectly flagged as moving")

print()
print("=" * 70)
print("TEST 3: still correctly detects actual walking/jogging")
print("=" * 70)

for label, speed in [("normal walking (0.7 height/s)", 0.7), ("brisk walking (1.0 height/s)", 1.0),
                      ("jogging (1.5 height/s)", 1.5)]:
    tracker = make_tracker()
    t0 = time.time()
    n = int(2.0 * FPS)
    detected_moving = False
    for i in range(n):
        t = t0 + i / FPS
        cx = (i / FPS) * speed * HEIGHT
        tracker.update(cx, 0.0, HEIGHT, t)
        if i > 3 and not tracker.is_relatively_stationary():
            detected_moving = True
    check(f"Correctly detects {label} as moving", detected_moving)

print()
print("=" * 70)
print("TEST 4: standing still (no walking at all) is always seen as stationary")
print("=" * 70)

tracker = make_tracker()
t0 = time.time()
always_stationary = True
for i in range(int(3.0 * FPS)):
    t = t0 + i / FPS
    tracker.update(0.0, 0.0, HEIGHT, t)
    if i > 1 and not tracker.is_relatively_stationary():
        always_stationary = False
check("Perfectly still person never flagged as moving", always_stationary)

print()
print("=" * 70)
total = len(results)
passed = sum(results)
print(f"RESULTS: {passed}/{total} tests passed")
print("=" * 70)

if passed != total:
    sys.exit(1)
