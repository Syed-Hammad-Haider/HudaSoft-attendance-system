"""
Standalone employee registration tool -- fully automatic multi-angle
face capture.

WHY THIS IS SEPARATE FROM main.py:
Registration is an admin/setup task with an entirely different
workflow than continuous monitoring: it needs a live preview, explicit
per-angle prompts ("now turn slightly left"), and pass/fail feedback
per capture. Bundling that into the main monitoring loop would
complicate both. This mirrors the project's existing pattern of
keeping distinct concerns in distinct entry points.

HOW CAPTURE WORKS (fully automatic, with REAL pose detection):
For each angle (front, left, right by default), the system first
shows a few seconds of countdown ("Get ready: LEFT... turn now") --
a mandatory pause with no capture logic running, giving real time to
physically turn before anything is checked. After the countdown, the
system watches continuously and GEOMETRICALLY MEASURES your head's
yaw (turn angle) from facial keypoints every frame -- it will not
accept a "front" capture unless you're actually facing forward, and
will not accept a "left"/"right" capture unless a real head turn is
measured, turned in the OPPOSITE direction from any other turn angle
already captured (so left and right can't both accidentally be the
same turn). This is on top of the existing quality gates (one clear
face, good detection confidence, close enough to the camera) and an
embedding-distinctness check as a secondary safety net. Once every
check passes and HOLDS for a short continuous window, it captures
automatically and moves to the next angle. SPACE is still available as
a manual override the instant it's ready, and Q cancels at any time
(including during the countdown).

USAGE:
    python register_employee.py
"""

import logging
import sys
import time

import cv2

from camera.factory import create_video_source
from config.settings import CONFIG
from database.connection import initialize_database
from database.employee_repository import DuplicateEmployeeCodeError, employee_code_exists
from face_id.embedder import FaceEmbedder
from face_id.registration import evaluate_capture_quality, register_new_employee

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("register_employee")

WINDOW_NAME = "Employee Registration"

ANGLE_INSTRUCTIONS = {
    "front": "Look straight at the camera",
    "left": "Turn your head clearly to the LEFT (about 30-45 degrees) and hold",
    "right": "Turn your head clearly to the RIGHT (about 30-45 degrees) and hold",
}

# How long a candidate frame must CONTINUOUSLY pass every quality gate
# before it auto-captures. Deliberately elapsed TIME, not a frame
# count -- a frame count silently assumes a roughly fixed frame rate,
# which breaks down badly between a fast local webcam and a laggier
# phone camera over Wi-Fi (DroidCam). See the full explanation on
# FaceRecognitionConfig.registration_auto_capture_hold_seconds in
# config/settings.py.
AUTO_CAPTURE_HOLD_SECONDS = CONFIG.face_recognition.registration_auto_capture_hold_seconds

CAPTURE_CONFIRM_SECONDS = 0.9
REJECT_FLASH_SECONDS = 1.0

# A transient camera read hiccup (one dropped frame) shouldn't abort
# the whole registration session -- only a sustained failure should.
MAX_CONSECUTIVE_READ_FAILURES = 45


def _draw_overlay(frame, angle_label: str, status_text: str, status_color, debug_text: str = ""):
    instruction = ANGLE_INSTRUCTIONS.get(angle_label, angle_label)

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (25, 25, 25), -1)
    cv2.putText(frame, f"Capturing: {angle_label.upper()}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, instruction, (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h - 60), (w, h), (25, 25, 25), -1)
    cv2.putText(frame, status_text, (12, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)
    if debug_text:
        cv2.putText(frame, debug_text, (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "Auto-captures when ready   [SPACE] Capture now   [Q] Cancel", (w - 430, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def _show_transient_message(video_source, window_name, base_frame, text, color, seconds):
    """
    Freeze on `base_frame` with a big confirmation banner for a fixed
    duration -- makes a capture (or rejection) unmistakable, rather
    than a one-frame state change easy to miss while watching your own
    face in the preview. Still polls the camera (so the stream doesn't
    fall behind) but ignores frames for display during the freeze.
    """
    h, w = base_frame.shape[:2]
    overlay = base_frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 45), (w, h // 2 + 45), color, -1)
    cv2.addWeighted(overlay, 0.85, base_frame, 0.15, 0, base_frame)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
    cv2.putText(base_frame, text, (max(10, (w - text_size[0]) // 2), h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    end_time = time.time() + seconds
    while time.time() < end_time:
        cv2.imshow(window_name, base_frame)
        cv2.waitKey(1)
        video_source.read_frame()  # drain the stream so it doesn't fall behind


def _run_prep_countdown(video_source, window_name, angle_label: str, seconds: float) -> bool:
    """
    Block for `seconds`, showing a live camera preview with a large
    countdown and the angle instruction, so the person has real time
    to physically get into position BEFORE any capture logic even
    looks at their pose. Returns False if the user pressed Q to cancel
    during the countdown.
    """
    instruction = ANGLE_INSTRUCTIONS.get(angle_label, angle_label)
    deadline = time.time() + seconds

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return True

        frame = video_source.read_frame()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 90), (25, 25, 25), -1)
        cv2.putText(frame, f"Get ready: {angle_label.upper()}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, instruction, (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 230), 2, cv2.LINE_AA)

        count_text = f"{remaining:.1f}"
        text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 4)[0]
        cv2.putText(frame, count_text, ((w - text_size[0]) // 2, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 200, 230), 4, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False


def run_registration() -> None:
    initialize_database()

    full_name = input("Employee full name: ").strip()
    if not full_name:
        print("Name is required. Aborting.")
        return

    # Validate the employee code for uniqueness BEFORE opening the
    # camera at all. Catching a duplicate only when saving (at the very
    # end, after several minutes of guided face capture) wastes the
    # registering person's time and the employee's patience -- a real,
    # recurring cost at thousands-of-employees scale where code
    # collisions (typos, re-registration attempts) are routine, not
    # rare. See database/employee_repository.py's employee_code_exists()
    # for the full reasoning.
    employee_code = None
    while True:
        raw_code = input("Employee code (optional, press Enter to skip): ").strip()
        if not raw_code:
            break
        if employee_code_exists(raw_code):
            print(f"  Employee code '{raw_code}' is already in use by another employee. Try a different one.")
            continue
        employee_code = raw_code
        break

    department = input("Department (optional, press Enter to skip): ").strip() or None

    embedder = FaceEmbedder(det_size=CONFIG.face_recognition.detection_size)
    video_source = create_video_source(CONFIG.camera)

    captured = []
    captured_embeddings = []  # accepted embeddings so far this session, for the distinctness check
    captured_turn_signs = []  # yaw signs already used by turn (non-frontal) angles, to force opposite turns
    angles = list(CONFIG.face_recognition.registration_angles)
    frontal_angle = angles[0] if angles else None  # first angle is treated as the "facing forward" reference
    max_similarity = CONFIG.face_recognition.registration_max_similarity_to_previous_angle
    prep_seconds = CONFIG.face_recognition.registration_prep_seconds
    max_frontal_yaw = CONFIG.face_recognition.registration_max_frontal_yaw_ratio
    min_turn_yaw = CONFIG.face_recognition.registration_min_turn_yaw_ratio

    try:
        video_source.open()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        for angle_label in angles:
            is_frontal_angle = angle_label == frontal_angle
            print(f"\n--- Capturing angle: {angle_label} ---")
            print(ANGLE_INSTRUCTIONS.get(angle_label, ""))

            if not _run_prep_countdown(video_source, WINDOW_NAME, angle_label, prep_seconds):
                print("Registration cancelled.")
                return

            accepted_this_angle = False
            consecutive_read_failures = 0
            ready_since = None  # timestamp when quality first became continuously acceptable

            while not accepted_this_angle:
                frame = video_source.read_frame()
                if frame is None:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                        logger.error(
                            "Camera stopped delivering frames (%d consecutive failed reads). "
                            "Run 'python diagnose_camera.py' to check the camera connection.",
                            consecutive_read_failures,
                        )
                        return
                    continue  # transient hiccup -- just try again next loop
                consecutive_read_failures = 0

                # Continuously evaluate so the live status text updates
                # every frame, even before capture -- real-time
                # feedback ("move closer", "turn further, too similar
                # to your front capture") instead of a surprise failure.
                # require_frontal / require_turned / forbidden_yaw_signs
                # are what actually make this DETECT the angle -- see
                # face_id/registration.py's evaluate_capture_quality().
                result = evaluate_capture_quality(
                    frame, embedder,
                    min_detection_confidence=CONFIG.face_recognition.registration_min_detection_confidence,
                    min_face_size_px=CONFIG.face_recognition.registration_min_face_size_px,
                    previous_embeddings=captured_embeddings,
                    max_similarity_to_previous=max_similarity,
                    require_frontal=is_frontal_angle,
                    require_turned=not is_frontal_angle,
                    min_turn_yaw_ratio=min_turn_yaw,
                    max_frontal_yaw_ratio=max_frontal_yaw,
                    forbidden_yaw_signs=captured_turn_signs,
                )

                now = time.time()
                if result.accepted:
                    if ready_since is None:
                        ready_since = now
                    held_seconds = now - ready_since
                else:
                    ready_since = None
                    held_seconds = 0.0

                ready_to_capture = result.accepted and held_seconds >= AUTO_CAPTURE_HOLD_SECONDS

                if not result.accepted:
                    status_text = f"Adjust: {result.reason}"
                    status_color = (60, 140, 240)
                elif not ready_to_capture:
                    status_text = f"Hold steady... {held_seconds:.1f}s / {AUTO_CAPTURE_HOLD_SECONDS:.1f}s"
                    status_color = (0, 200, 230)
                else:
                    status_text = "Capturing..."
                    status_color = (80, 200, 120)

                debug_text = ""
                if result.face is not None:
                    x1, y1, x2, y2 = result.face.bbox
                    debug_text = f"confidence={result.face.detection_confidence:.2f}  face_size={min(x2 - x1, y2 - y1)}px"
                    if result.measured_yaw_ratio is not None:
                        debug_text += f"  yaw={result.measured_yaw_ratio:+.2f}"
                    if result.similarity_to_previous is not None:
                        debug_text += f"  similarity_to_prev_angle={result.similarity_to_previous:.3f}"

                display_frame = _draw_overlay(frame.copy(), angle_label, status_text, status_color, debug_text)
                if result.face is not None:
                    x1, y1, x2, y2 = result.face.bbox
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), status_color, 2)

                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Registration cancelled.")
                    return

                manual_capture_requested = key == ord(" ") and ready_to_capture
                manual_reject_feedback = key == ord(" ") and not ready_to_capture

                if ready_to_capture or manual_capture_requested:
                    captured.append((angle_label, result.face))
                    captured_embeddings.append(result.face.embedding)
                    if not is_frontal_angle and result.measured_yaw_ratio is not None:
                        # Record which direction this turn angle used so
                        # the NEXT turn angle is required to go the
                        # opposite way (see forbidden_yaw_signs above) --
                        # this is what guarantees "left" and "right"
                        # can't both accidentally be the same turn.
                        captured_turn_signs.append(1 if result.measured_yaw_ratio > 0 else -1)
                    accepted_this_angle = True
                    print(f"  Captured '{angle_label}' (detection confidence: {result.face.detection_confidence:.2f}"
                          f"{f', yaw={result.measured_yaw_ratio:+.2f}' if result.measured_yaw_ratio is not None else ''})")
                    _show_transient_message(
                        video_source, WINDOW_NAME, display_frame.copy(),
                        f"CAPTURED: {angle_label.upper()}", (60, 170, 60), CAPTURE_CONFIRM_SECONDS,
                    )
                elif manual_reject_feedback:
                    # Pressing SPACE before it's actually ready still
                    # gives clear, unmissable feedback on exactly why
                    # (not silent -- this was the original "pressing
                    # space does nothing" bug fixed in an earlier pass).
                    print(f"  Not ready yet: {status_text}")
                    _show_transient_message(
                        video_source, WINDOW_NAME, display_frame.copy(),
                        f"NOT READY: {status_text}", (40, 40, 200), REJECT_FLASH_SECONDS,
                    )

        # Retry loop here is defense-in-depth against the small race
        # window in the upfront employee_code_exists() check (see its
        # docstring) -- if it still collides here, ask for a new code
        # WITHOUT discarding the face captures already completed;
        # re-running the whole camera flow over a code typo would be a
        # real, avoidable waste of the registering person's time.
        while True:
            try:
                employee_id = register_new_employee(
                    full_name, captured, employee_code=employee_code, department=department
                )
                break
            except DuplicateEmployeeCodeError as exc:
                print(f"\n  {exc} Your captured face data is safe -- just enter a different code.")
                new_code = input("Employee code (Enter to skip): ").strip()
                employee_code = new_code or None

        print(f"\nSuccess! Registered '{full_name}' as employee_id={employee_id} "
              f"with {len(captured)} angle(s) captured.")
        print("They will be recognized the next time main.py runs (embeddings are loaded at startup).")

    finally:
        video_source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(run_registration() or 0)
