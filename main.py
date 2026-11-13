"""
Application entry point.

Execution flow per frame:

    camera -> person_detection -> person_tracking
        -> entrance_line (crossing / direction, with commitment-distance
           filtering so hesitation/partial crossings don't count)
        -> wave_detection (per-hand gesture analysis, full landmark detail)
        -> face recognition (Phase 2: identify WHO is waving, cached and
           re-checked periodically rather than every frame)
        -> attendance_decision (the ONE place that decides whether a
           Suggested Action should actually fire AND persists it to the
           database -- state-machine gated by employee_id, not by the
           ephemeral tracking ID, so it survives tracking ID churn)
        -> display (render everything)

Run with:
    python main.py

To register a new employee's face first, run:
    python register_employee.py

IMPORTANT: keyboard shortcuts below only work while the VIDEO WINDOW
has focus (click on it first) -- this is an OpenCV/OS limitation, not
a bug. If key presses seem to do nothing, click the video window
first, then press the key again.

Controls (also shown live at the bottom of the video window):
    C - calibrate the entrance line: move the mouse to slide a vertical
        line to match your real doorway, click to confirm
    F - flip which side counts as IN vs OUT, if they come out backwards
    M - toggle fullscreen / maximized window
    V - toggle mirror preview (OFF by default -- see DisplayConfig.
        mirror_preview for why the unmirrored view is actually correct
        for a real entrance camera, not a bug). Forces recalibration
        when toggled, since it changes what left/right means in pixels.
    R - reload known faces from the database (use after registering a
        new employee with register_employee.py, without restarting)
    S - cycle wave-detection sensitivity: strict -> balanced -> relaxed
        -> strict... Takes effect immediately, no restart needed. If
        real waves are being missed, press S until you land on
        "relaxed"; if walking-past people are triggering false CHECK
        INs, go the other way. Current level is always shown on-screen.
    D - toggle the wave DEBUG/ANALYSIS overlay: shows live hand
        tracking status, swing magnitude vs. threshold, reversal
        count, timing regularity, and body-stillness state for every
        tracked hand -- use this to see exactly WHY a wave is or isn't
        being detected. See wave_detection/oscillation.py and
        wave_detection/body_motion.py for what each number means.
    Q - quit

Wave sensitivity can also be set at startup instead of cycling live:
    python main.py --sensitivity relaxed

On first run (no saved calibration yet), calibration mode starts
automatically so the line is never silently "just wherever the default
happens to be" -- you're required to place it once before normal
detection begins. After that, it's saved to
config/entrance_line_calibration.json and reloaded automatically, so
you only calibrate once per camera position.
"""

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2

from camera.factory import create_video_source
from config.settings import CONFIG, WAVE_SENSITIVITY_PRESETS, apply_wave_sensitivity
from database.connection import initialize_database
from direction_estimation.entrance_line import EntranceGate
from direction_estimation.line_calibration import load_calibration, save_calibration
from direction_estimation.line_calibrator import LineCalibrator
from display.renderer import Renderer
from face_id.embedder import FaceEmbedder
from face_id.identified_attendance import IdentifiedAttendanceDecisionEngine, status_label_for_employee
from face_id.live_recognition import LiveFaceRecognitionCache
from face_id.recognizer import FaceRecognizer
from person_detection.detector import PersonDetector
from person_tracking.tracker import PersonTracker
from utils.event_log import EventLog
from utils.geometry import bbox_center
from wave_detection.debug_overlay import draw_wave_debug_panel
from wave_detection.detector import WaveDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# Persistent, rotating log file -- a real company deployment needs an
# audit trail that survives closing the terminal window, not just
# console output. Rotation caps disk usage automatically so this
# doesn't need manual maintenance. Added on top of (not replacing) the
# console handler above, so behavior when running interactively is
# unchanged.
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)
_file_handler = RotatingFileHandler(_log_dir / "attendance_system.log", maxBytes=5_000_000, backupCount=5)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_file_handler)


def _build_entrance_gate():
    """
    Build the entrance gate, preferring a previously-saved on-screen
    calibration (config/entrance_line_calibration.json) over the
    hardcoded default in config/settings.py.

    Returns (gate, needs_first_run_calibration).
    """
    gate = EntranceGate(CONFIG.entrance_line, CONFIG.camera.frame_width, CONFIG.camera.frame_height)
    saved = load_calibration()
    if saved:
        try:
            p1_ratio = tuple(saved["line_start"])
            p2_ratio = tuple(saved["line_end"])
            inside_left = saved["inside_is_left_of_line"]
            p1_px = (p1_ratio[0] * CONFIG.camera.frame_width, p1_ratio[1] * CONFIG.camera.frame_height)
            p2_px = (p2_ratio[0] * CONFIG.camera.frame_width, p2_ratio[1] * CONFIG.camera.frame_height)
            gate.set_line(p1_px, p2_px)
            gate._inside_is_left = inside_left
            logger.info("Using saved entrance line calibration.")
            return gate, False
        except (KeyError, TypeError, IndexError):
            logger.warning("Saved calibration file was malformed; forcing recalibration.")
            return gate, True
    else:
        logger.info(
            "No saved entrance line calibration found. Calibration mode will "
            "start automatically -- move the mouse to slide the line into "
            "place, then click to confirm. Press 'C' any time afterward to redo it."
        )
        return gate, True


_SENSITIVITY_CYCLE = ["strict", "balanced", "relaxed"]


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the attendance detection system.")
    parser.add_argument(
        "--sensitivity", choices=list(WAVE_SENSITIVITY_PRESETS), default="balanced",
        help="Starting wave-detection sensitivity (can still be cycled live with the 'S' key).",
    )
    return parser.parse_args()


def run() -> None:
    args = _parse_args()
    initialize_database()

    video_source = create_video_source(CONFIG.camera)
    detector = PersonDetector(CONFIG.detection)
    tracker = PersonTracker(CONFIG.tracking)
    gate, needs_calibration = _build_entrance_gate()
    wave_detector = WaveDetector(CONFIG.wave)
    apply_wave_sensitivity(CONFIG.wave, args.sensitivity)
    sensitivity_level = args.sensitivity
    logger.info("Wave sensitivity starting at '%s'.", sensitivity_level)
    debug_overlay_on = CONFIG.wave.debug_overlay_default_on

    face_embedder = FaceEmbedder(det_size=CONFIG.face_recognition.detection_size)
    face_recognizer = FaceRecognizer(
        min_similarity=CONFIG.face_recognition.min_match_similarity,
        min_margin_over_runner_up=CONFIG.face_recognition.min_margin_over_runner_up,
    )
    known_count = face_recognizer.reload_known_faces()
    if known_count == 0:
        logger.warning(
            "No registered employees found -- everyone will show as UNRECOGNIZED and no "
            "attendance will be recorded. Run 'python register_employee.py' to register people."
        )
    recognition_cache = LiveFaceRecognitionCache(CONFIG.face_recognition, face_embedder, face_recognizer)
    decision_engine = IdentifiedAttendanceDecisionEngine(CONFIG.entrance_line, CONFIG.action_guard, CONFIG.camera.frame_width)

    renderer = Renderer(CONFIG.display)
    event_log = EventLog(CONFIG.display.event_log_max_entries)
    calibrator = LineCalibrator(CONFIG.display.window_name, CONFIG.camera.frame_height)

    if needs_calibration:
        calibrator.start()

    previously_seen_ids = set()
    window_created = False
    is_fullscreen = False
    mirror_preview = CONFIG.display.mirror_preview

    try:
        video_source.open()
        logger.info("Attendance detection system started. Press 'q' to quit.")

        consecutive_read_failures = 0
        # A single dropped frame (network hiccup on an IP/DroidCam
        # source) shouldn't end the whole session -- only a sustained
        # failure should, matching the same fix already applied to
        # register_employee.py's capture loop.
        max_consecutive_read_failures = 45  # ~1.5s worth at 30fps

        while True:
            frame = video_source.read_frame()
            if frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_consecutive_read_failures:
                    logger.warning(
                        "No frame received for %d consecutive reads; stopping. "
                        "Run 'python diagnose_camera.py' to check the camera connection.",
                        consecutive_read_failures,
                    )
                    break
                continue
            consecutive_read_failures = 0

            if mirror_preview:
                frame = cv2.flip(frame, 1)

            # 1. Detect all people in this frame.
            detections = detector.detect(frame)

            # 2. Track people across frames (stable IDs + position history).
            tracked_people = tracker.update(detections)
            current_ids = {p.track_id for p in tracked_people}

            # Clean up per-person state for anyone who has left the frame.
            # Note: this clears the FACE RECOGNITION CACHE and the wave/
            # gate/UI state for the track -- it does NOT reset attendance
            # status, because that now lives in the database keyed by
            # employee_id, not in memory keyed by track_id. That's the
            # whole point of this phase's upgrade.
            dropped_ids = previously_seen_ids - current_ids
            for track_id in dropped_ids:
                wave_detector.remove_track(track_id)
                renderer.remove_track(track_id)
                gate.remove_track(track_id)
                recognition_cache.remove_track(track_id)
            previously_seen_ids = current_ids

            gate_states = {}
            wave_results = {}
            status_labels = {}
            display_names = {}

            # 3. For each tracked person: identify them, update entrance-line
            #    side, check for a wave, and run it through the attendance
            #    decision engine.
            if not calibrator.is_active:
                for person in tracked_people:
                    center = bbox_center(person.bbox)
                    gate_state = gate.update(person.track_id, center)
                    gate_states[person.track_id] = gate_state

                    if gate_state.just_crossed:
                        event_log.add(f"ID {person.track_id}: crossed -> {gate_state.last_direction.value}")

                    wave_result = wave_detector.process(frame, person)
                    wave_results[person.track_id] = wave_result

                    recognition = recognition_cache.get_recognition(frame, person)
                    employee_id = recognition.employee_id if recognition else None
                    employee_name = recognition.full_name if recognition else None
                    match_confidence = recognition.similarity if recognition else 0.0

                    display_names[person.track_id] = employee_name or f"ID {person.track_id}"
                    status_labels[person.track_id] = status_label_for_employee(employee_id)

                    decision = decision_engine.evaluate(
                        employee_id=employee_id,
                        employee_name=employee_name,
                        gate_state=gate_state,
                        wave_confirmed=wave_result.wave_confirmed,
                        position=center,
                        match_confidence=match_confidence,
                        source_track_id=person.track_id,
                    )

                    if decision.action is not None:
                        renderer.show_suggested_action(person.track_id, decision.action, gate_state.last_direction, employee_name or "")
                        event_log.add(f"{employee_name} (ID:{employee_id}): {decision.action.label}")
                        logger.info(
                            "Attendance recorded | Employee: %s (ID:%s) | Direction: %s | Action: %s | Confidence: %.2f",
                            employee_name, employee_id, gate_state.last_direction.value, decision.action.label, match_confidence,
                        )
                        # Status label just changed as a result of this
                        # action -- refresh it immediately so the UI
                        # doesn't show stale status for one extra frame.
                        status_labels[person.track_id] = status_label_for_employee(employee_id)
                    elif wave_result.wave_confirmed and decision.reason_if_none:
                        # Routine, expected non-events (already checked in
                        # today and waved again) stay silent on-screen.
                        # GENUINE reasons (not recognized, no recent line
                        # crossing) are shown, so a stuck user can see WHY
                        # a green wave produced no check-in instead of
                        # getting no feedback at all.
                        if not decision.reason_is_routine:
                            who = f"{employee_name} (ID:{employee_id})" if employee_name else f"ID {person.track_id} (unrecognized)"
                            event_log.add(f"{who}: waved but not checked in -- {decision.reason_if_none}")
                        logger.info("Wave confirmed but no action for ID %d: %s", person.track_id, decision.reason_if_none)

            # 4. Draw everything and show the frame.
            output_frame = renderer.render(
                frame,
                tracked_people,
                gate_states,
                wave_results,
                gate,
                event_log,
                CONFIG.entrance_line.min_frames_to_confirm_side,
                calibrator,
                status_labels,
                display_names,
                sensitivity_level,
            )

            if debug_overlay_on:
                for panel_idx, person in enumerate(tracked_people):
                    wr = wave_results.get(person.track_id)
                    if wr is not None:
                        draw_wave_debug_panel(output_frame, person.bbox, wr, panel_index=panel_idx)

            if not window_created:
                cv2.namedWindow(CONFIG.display.window_name, cv2.WINDOW_NORMAL)

            cv2.imshow(CONFIG.display.window_name, output_frame)

            if not window_created:
                calibrator.attach()
                window_created = True

            completed_line = calibrator.consume_completed_line()
            if completed_line:
                gate.set_line(completed_line[0], completed_line[1])
                p1_ratio, p2_ratio, inside_left = gate.get_calibration()
                save_calibration(p1_ratio, p2_ratio, inside_left)
                event_log.add("Entrance line calibrated")
                logger.info("Entrance line calibrated and saved.")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit key pressed. Shutting down.")
                break
            elif key == ord("c"):
                calibrator.start()
                logger.info("Calibration mode started -- slide the line into place, then click.")
            elif key == ord("f"):
                new_side = gate.flip_inside_side()
                p1_ratio, p2_ratio, inside_left = gate.get_calibration()
                save_calibration(p1_ratio, p2_ratio, inside_left)
                event_log.add("Flipped IN/OUT sides")
                logger.info("Flipped IN/OUT side. inside_is_left_of_line is now %s", new_side)
            elif key == ord("m"):
                is_fullscreen = not is_fullscreen
                prop_value = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(CONFIG.display.window_name, cv2.WND_PROP_FULLSCREEN, prop_value)
                logger.info("Fullscreen toggled: %s", is_fullscreen)
            elif key == ord("v"):
                mirror_preview = not mirror_preview
                logger.warning(
                    "Mirror preview toggled to %s. Left/right pixel meaning just changed -- "
                    "any saved entrance line calibration is now WRONG. Press 'C' to recalibrate.",
                    mirror_preview,
                )
                event_log.add(f"Mirror {'ON' if mirror_preview else 'OFF'} -- recalibrate line!")
                calibrator.start()
            elif key == ord("r"):
                known_count = face_recognizer.reload_known_faces()
                event_log.add(f"Reloaded {known_count} face embeddings")
                logger.info("Face embeddings reloaded from database: %d entries.", known_count)
            elif key == ord("s"):
                current_index = _SENSITIVITY_CYCLE.index(sensitivity_level)
                sensitivity_level = _SENSITIVITY_CYCLE[(current_index + 1) % len(_SENSITIVITY_CYCLE)]
                apply_wave_sensitivity(CONFIG.wave, sensitivity_level)
                event_log.add(f"Wave sensitivity -> {sensitivity_level.upper()}")
                logger.info("Wave sensitivity changed to '%s'.", sensitivity_level)
            elif key == ord("d"):
                debug_overlay_on = not debug_overlay_on
                event_log.add(f"Wave debug overlay {'ON' if debug_overlay_on else 'OFF'}")
                logger.info("Wave debug overlay toggled %s.", "ON" if debug_overlay_on else "OFF")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception:
        logger.exception("Unexpected error in main loop.")
    finally:
        video_source.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    sys.exit(run() or 0)
