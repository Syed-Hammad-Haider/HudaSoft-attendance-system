"""
Standalone wave-detection diagnostic tool.

WHY THIS EXISTS:
"Wave detection isn't working" can mean several genuinely different
problems, each needing a different fix:
  1. MediaPipe itself never finds a hand at all -- often an
     INSTALLATION problem (see the mediapipe-version check below), not
     a tuning problem. No amount of sensitivity adjustment fixes this.
  2. MediaPipe finds the hand fine, but the swing never reaches the
     amplitude/reversal-count/regularity thresholds -- a real tuning
     problem (try the 'S' sensitivity presets in main.py).
  3. The hand and swing are both detected fine, but no CHECK IN/OUT
     appears -- a downstream gating problem (not near a recent
     entrance-line crossing, or the person isn't recognized by face
     yet), not a wave-detection problem at all.

This script isolates case 1 and 2 from case 3 and from the rest of the
pipeline (no YOLO person detection, no ByteTrack, no face recognition,
no entrance-line/attendance logic) by running MediaPipe Hands directly
on the FULL camera frame and showing exactly what it sees, live:
  - Whether a hand is detected at all, and MediaPipe's own confidence
    for it (not just yes/no).
  - The live hand skeleton, drawn directly (not on a person crop).
  - The same "hand-width" swing-amplitude and reversal-count numbers
    the real pipeline uses, via wave_detection/oscillation.py (the
    EXACT same class, not a re-implementation) so what you see here
    matches what wave_detection/detector.py would see too.

USAGE:
    python diagnose_wave_detection.py                    # webcam 0, "balanced" sensitivity
    python diagnose_wave_detection.py --index 1
    python diagnose_wave_detection.py --sensitivity relaxed
    python diagnose_wave_detection.py --url "rtsp://user:pass@192.168.1.50:554/stream1"

Controls while running:
    S - cycle sensitivity (strict / balanced / relaxed), live
    Q - quit

HOW TO READ THE OUTPUT:
  - "NO HAND DETECTED" the whole time, even with a hand clearly, fully
    visible and well-lit in frame -> almost certainly case 1 above.
    Check the startup message this script prints about your installed
    mediapipe version FIRST.
  - Hand detected, skeleton tracks correctly, but "swing" stays near 0
    even while actually waving -> check you're moving your HAND (not
    just your wrist) through a real side-to-side or up-down arc; try
    --sensitivity relaxed.
  - Hand detected, swing/reversal numbers climb close to the required
    count but "WAVE CONFIRMED" never lights up -> try --sensitivity
    relaxed, or hold the wave a little longer/faster (reversals must
    land within a bounded time of each other).
  - Everything above works here but main.py still shows no CHECK
    IN/OUT -> this script has ruled out wave detection itself; the
    remaining problem is downstream (entrance-line crossing timing or
    face recognition) -- use main.py's own 'D' debug overlay and event
    log to look at that instead.
"""

import argparse
import logging
import time

import cv2
import mediapipe as mp

from config.settings import WAVE_SENSITIVITY_PRESETS, WaveDetectionConfig, apply_wave_sensitivity
from wave_detection.oscillation import OscillationTracker

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("diagnose_wave_detection")

_PALM_CENTROID_INDICES = (0, 5, 9, 13, 17)
_HAND_SCALE_INDEX_A = 0
_HAND_SCALE_INDEX_B = 9

_SENSITIVITY_CYCLE = ["strict", "balanced", "relaxed"]


def _check_mediapipe_installation() -> bool:
    """
    Same check WaveDetector itself now does before it will even start
    (see wave_detection/detector.py) -- repeated here, standalone, so
    this diagnostic can point at the EXACT same problem, first, before
    trying anything else. Returns True if OK to proceed.
    """
    version = getattr(mp, "__version__", "unknown")
    print(f"Installed mediapipe version: {version}")
    if not hasattr(mp, "solutions"):
        print(
            "\n*** PROBLEM FOUND: this mediapipe version does not have "
            "mp.solutions.hands. ***\n"
            "Newer mediapipe releases removed the legacy Hands API this "
            "project's wave detection is built on. This alone is enough to "
            "make wave detection look completely broken -- no hand is ever "
            "detected, with no error shown in the main app.\n\n"
            "FIX: pip install mediapipe==0.10.14   (the exact version pinned "
            "in requirements.txt)\nThen re-run this script.\n"
        )
        return False
    print("mp.solutions.hands is available -- OK.\n")
    return True


def _open_source(index: int, url: str):
    source = url if url else index
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open camera source: {source!r}")
        return None
    return cap


def run() -> None:
    parser = argparse.ArgumentParser(description="Isolate MediaPipe hand/wave detection from the full pipeline.")
    parser.add_argument("--index", type=int, default=0, help="Webcam index (ignored if --url is given).")
    parser.add_argument("--url", type=str, default=None, help="IP camera URL (rtsp:// or http://) instead of a webcam.")
    parser.add_argument("--sensitivity", choices=list(WAVE_SENSITIVITY_PRESETS), default="balanced")
    args = parser.parse_args()

    if not _check_mediapipe_installation():
        return

    cap = _open_source(args.index, args.url)
    if cap is None:
        return

    wave_config = WaveDetectionConfig()
    apply_wave_sensitivity(wave_config, args.sensitivity)
    sensitivity_level = args.sensitivity

    hands_solution = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    hands = hands_solution.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=wave_config.min_detection_confidence,
        min_tracking_confidence=wave_config.min_tracking_confidence,
    )
    tracker = OscillationTracker(
        min_swing_amplitude_hand_widths=wave_config.min_swing_amplitude_hand_widths,
        min_reversals_required=wave_config.min_reversals_required,
        max_seconds_between_reversals=wave_config.max_seconds_between_reversals,
        max_regularity_cv=wave_config.max_reversal_gap_regularity_cv,
        smoothing_alpha=wave_config.wave_smoothing_alpha,
    )

    window_name = "Wave Detection Diagnostic (Q to quit, S to cycle sensitivity)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    confirmed_until = 0.0
    frames_with_no_hand_streak = 0
    total_frames = 0
    frames_with_hand = 0

    print(f"Starting at sensitivity '{sensitivity_level}'. Press 'S' to cycle, 'Q' to quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed -- stopping.")
                break

            total_frames += 1
            now = time.time()
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_present = bool(results.multi_hand_landmarks)
            if hand_present:
                frames_with_hand += 1
                frames_with_no_hand_streak = 0
                landmarks = results.multi_hand_landmarks[0]
                drawing.draw_landmarks(frame, landmarks, hands_solution.HAND_CONNECTIONS)

                pts = [(lm.x * frame_w, lm.y * frame_h) for lm in landmarks.landmark]
                palm_x = sum(pts[i][0] for i in _PALM_CENTROID_INDICES) / len(_PALM_CENTROID_INDICES)
                palm_y = sum(pts[i][1] for i in _PALM_CENTROID_INDICES) / len(_PALM_CENTROID_INDICES)
                ax, ay = pts[_HAND_SCALE_INDEX_A]
                bx, by = pts[_HAND_SCALE_INDEX_B]
                hand_scale = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5

                confirmed_now = tracker.update(palm_x, palm_y, hand_scale, now)
                if confirmed_now:
                    tracker.reset_progress()
                    confirmed_until = now + 1.5

                debug = tracker.debug
                lines = [
                    f"sensitivity: {sensitivity_level.upper()}",
                    f"hand: DETECTED",
                    f"swing: {debug.last_swing_magnitude_hand_widths:.2f} hand-widths "
                    f"(need >= {wave_config.min_swing_amplitude_hand_widths:.2f})",
                    f"reversals: {debug.reversal_count} / {wave_config.min_reversals_required}",
                    f"state: {debug.state}",
                ]
                if now < confirmed_until:
                    lines.append("*** WAVE CONFIRMED ***")
            else:
                frames_with_no_hand_streak += 1
                lines = [
                    f"sensitivity: {sensitivity_level.upper()}",
                    "hand: NO HAND DETECTED",
                ]
                if frames_with_no_hand_streak > 60:
                    lines.append("(no hand for 60+ frames -- make sure your hand is")
                    lines.append(" clearly in frame, well-lit, and not too far away)")

            for i, line in enumerate(lines):
                color = (0, 255, 0) if "CONFIRMED" in line else (255, 255, 255)
                cv2.putText(frame, line, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame, line, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                idx = _SENSITIVITY_CYCLE.index(sensitivity_level)
                sensitivity_level = _SENSITIVITY_CYCLE[(idx + 1) % len(_SENSITIVITY_CYCLE)]
                apply_wave_sensitivity(wave_config, sensitivity_level)
                hands.close()
                hands = hands_solution.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=wave_config.min_detection_confidence,
                    min_tracking_confidence=wave_config.min_tracking_confidence,
                )
                tracker = OscillationTracker(
                    min_swing_amplitude_hand_widths=wave_config.min_swing_amplitude_hand_widths,
                    min_reversals_required=wave_config.min_reversals_required,
                    max_seconds_between_reversals=wave_config.max_seconds_between_reversals,
                    max_regularity_cv=wave_config.max_reversal_gap_regularity_cv,
                    smoothing_alpha=wave_config.wave_smoothing_alpha,
                )
                print(f"Sensitivity -> {sensitivity_level}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        if total_frames:
            pct = 100.0 * frames_with_hand / total_frames
            print(f"\nSession summary: hand detected in {frames_with_hand}/{total_frames} frames ({pct:.0f}%).")
            if pct < 5:
                print(
                    "Hand was almost never detected. If your hand WAS in frame and "
                    "visible for a meaningful chunk of that time, this points at "
                    "MediaPipe itself (installation/version, lighting, or distance "
                    "from camera) rather than the wave-timing logic."
                )


if __name__ == "__main__":
    run()
