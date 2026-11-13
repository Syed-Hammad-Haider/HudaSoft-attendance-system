"""
Renders all live overlays: person boxes, hand skeletons, movement
trail, direction arrow, entrance line, status text (including
attendance state), Suggested Action banners, recent event log, and FPS.

Design decision: box/element colors are STATE-DRIVEN (crossing the
line, entering, leaving, waving, etc.) rather than one fixed color for
everything. This is what makes the difference between "a debug overlay"
and "a product UI" -- color communicates state at a glance without
reading text.

Design decision: hands are drawn as an actual 21-point skeleton
(landmarks + connecting lines), not just a bounding box. A box only
tells you "MediaPipe found something here"; a skeleton visibly shows
finger/wrist positions, which is both a clearer confirmation of
tracking precision and looks like a real product feature rather than a
debug rectangle.
"""

from typing import Dict, List, Optional

import cv2

from config.settings import DisplayConfig
from direction_estimation.entrance_line import Direction, PersonGateState, Side
from display.suggested_action import SuggestedAction
from person_tracking.tracked_person import TrackedPerson
from utils.event_log import EventLog
from utils.timing import ExpiringMessage, FPSCounter
from wave_detection.wave_result import WaveResult

# Standard MediaPipe Hands 21-landmark skeleton connections (pairs of
# landmark indices to draw a line between). This is a fixed, well-known
# topology (thumb, four fingers, palm), not something that needs to
# come from MediaPipe itself at runtime -- hardcoding it here keeps the
# display layer from needing a MediaPipe import.
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


class Renderer:
    def __init__(self, config: DisplayConfig):
        self._config = config
        self._messages: Dict[int, ExpiringMessage] = {}
        self._active_action: Dict[int, str] = {}  # track_id -> "CHECK IN" / "CHECK OUT", cleared on expiry
        self._fps_counter = FPSCounter()

    # -- public API used by main.py -----------------------------------
    def show_suggested_action(self, track_id: int, action: SuggestedAction, direction: Direction, display_name: str = "") -> None:
        if track_id not in self._messages:
            self._messages[track_id] = ExpiringMessage(self._config.message_display_seconds)
        name_part = f"{display_name}: " if display_name else ""
        text = f"{action.icon} {name_part}{action.label}"
        self._messages[track_id].set(text)
        self._active_action[track_id] = action.label

    def remove_track(self, track_id: int) -> None:
        self._messages.pop(track_id, None)
        self._active_action.pop(track_id, None)

    def render(
        self,
        frame,
        tracked_people: List[TrackedPerson],
        gate_states: Dict[int, PersonGateState],
        wave_results: Dict[int, WaveResult],
        gate,
        event_log: EventLog,
        min_history_for_stable: int,
        calibrator=None,
        status_labels: Optional[Dict[int, str]] = None,
        display_names: Optional[Dict[int, str]] = None,
        wave_sensitivity: str = "balanced",
        web_mode: bool = False,
    ):
        # web_mode=True strips the overlays that only make sense for the
        # OpenCV DESKTOP window (main.py): the keyboard-shortcut control
        # bar ([C] Calibrate / [Q] Quit / ...), the baked-in RECENT
        # EVENTS panel, the FPS counter, and the per-person status text
        # block. The web dashboard renders all of those as real HTML
        # panels beside the video instead, so baking them into the JPEG
        # was duplicated clutter that also made the feed look like a
        # terminal. The genuinely useful VISION overlays -- entrance
        # line, person boxes + names, movement trail, direction arrow,
        # the live hand/wave skeleton, and the big CHECK IN/OUT banner
        # -- are kept in both modes, because those convey what the
        # system is actually seeing and can't be shown any other way.
        colors = self._config.colors
        status_labels = status_labels or {}
        display_names = display_names or {}

        if calibrator is None or not calibrator.is_active:
            self._draw_entrance_line(frame, gate, colors)

        for person in tracked_people:
            gate_state = gate_states.get(person.track_id)
            wave_result = wave_results.get(person.track_id, WaveResult())
            status_label = status_labels.get(person.track_id, "NOT CHECKED IN")
            display_name = display_names.get(person.track_id, f"ID {person.track_id}")
            self._draw_trail(frame, person, colors)
            self._draw_person_box(frame, person, gate_state, gate, colors, min_history_for_stable, display_name)
            self._draw_direction_arrow(frame, person, gate_state, colors)
            self._draw_hands(frame, wave_result, colors)
            if not web_mode:
                self._draw_status_block(frame, person, gate_state, wave_result, status_label)
            self._draw_active_message(frame, person, colors)

        if not web_mode:
            self._draw_event_log(frame, event_log, colors)
            if self._config.show_fps:
                self._draw_fps(frame)
            self._draw_controls_hint(frame, colors, calibrator, wave_sensitivity)

        if calibrator is not None and calibrator.is_active:
            self._draw_calibration_overlay(frame, calibrator, colors)

        return frame

    # -- entrance line ---------------------------------------------------
    def _draw_entrance_line(self, frame, gate, colors) -> None:
        p1, p2 = gate.line_points_px()
        cv2.line(frame, p1, p2, colors.entrance_line, 2, cv2.LINE_AA)

        in_point, out_point = gate.label_points_px()
        self._draw_pill_label(frame, "IN", in_point, colors.person_box_entering)
        self._draw_pill_label(frame, "OUT", out_point, colors.person_box_leaving)

    def _draw_pill_label(self, frame, text: str, center, bg_color) -> None:
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x, y = center
        x1, y1 = x - text_w // 2 - 10, y - text_h // 2 - 8
        x2, y2 = x + text_w // 2 + 10, y + text_h // 2 + 8
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.putText(frame, text, (x1 + 10, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # -- movement trail ---------------------------------------------------
    def _draw_trail(self, frame, person: TrackedPerson, colors) -> None:
        points = list(person.position_history)[-self._config.trail_length:]
        if len(points) < 2:
            return
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            thickness = max(1, int(2 * (i / len(points))))
            cv2.line(frame, pt1, pt2, colors.movement_trail, thickness, cv2.LINE_AA)

    # -- person box --------------------------------------------------------
    def _box_color_for_person(self, gate_state: Optional[PersonGateState], gate, colors) -> tuple:
        if gate_state is None:
            return colors.person_box_default
        if gate_state.just_crossed or gate.is_near_line(gate_state.signed_distance):
            return colors.person_box_crossing
        if gate_state.last_direction == Direction.ENTERING:
            return colors.person_box_entering
        if gate_state.last_direction == Direction.LEAVING:
            return colors.person_box_leaving
        return colors.person_box_default

    def _draw_person_box(self, frame, person: TrackedPerson, gate_state, gate, colors, min_history_for_stable: int, display_name: str) -> None:
        x1, y1, x2, y2 = person.bbox
        color = self._box_color_for_person(gate_state, gate, colors)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._config.box_thickness)

        status = "STABLE" if len(person.position_history) >= min_history_for_stable else "ACQUIRING"
        # An unrecognized person still shows a display_name of "ID N"
        # (set by main.py when face recognition hasn't identified them)
        # -- this label doubles as the visual cue: a real name means
        # recognized, "ID N" means not, without needing a separate flag.
        label = f"{display_name}  [{status}]"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self._config.font_scale, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - text_h - 10)), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    self._config.font_scale, (0, 0, 0), 2, cv2.LINE_AA)

    # -- direction arrow ----------------------------------------------------
    def _draw_direction_arrow(self, frame, person: TrackedPerson, gate_state, colors) -> None:
        if gate_state is None or gate_state.last_direction == Direction.UNKNOWN:
            return
        if len(person.position_history) < 2:
            return

        x1, y1, x2, y2 = person.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        length = int((x2 - x1) * self._config.arrow_length_ratio)

        recent = list(person.position_history)[-5:]
        dx = recent[-1][0] - recent[0][0] if len(recent) >= 2 else 0
        direction_sign = 1 if dx >= 0 else -1

        end_point = (cx + direction_sign * length, cy)
        cv2.arrowedLine(frame, (cx, cy), end_point, colors.direction_arrow, 3, cv2.LINE_AA, tipLength=0.3)

    # -- hands: skeleton + subtle bbox ------------------------------------------
    def _draw_hands(self, frame, wave_result: WaveResult, colors) -> None:
        for hand in wave_result.hands:
            if hand.is_waving:
                color = colors.hand_valid_wave
            elif hand.confidence > 0.0:
                color = colors.hand_detected
            else:
                color = colors.hand_normal

            if hand.landmarks:
                self._draw_hand_skeleton(frame, hand.landmarks, color)
            else:
                # Fallback if landmarks weren't available for some reason.
                x1, y1, x2, y2 = hand.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_x, label_y = (hand.landmarks[0] if hand.landmarks else (hand.bbox[0], hand.bbox[1]))
            label = f"{int(hand.confidence * 100)}%"
            cv2.putText(frame, label, (label_x, max(12, label_y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)

    def _draw_hand_skeleton(self, frame, landmarks_px, color) -> None:
        for a, b in _HAND_CONNECTIONS:
            if a < len(landmarks_px) and b < len(landmarks_px):
                cv2.line(frame, landmarks_px[a], landmarks_px[b], color, 2, cv2.LINE_AA)
        for point in landmarks_px:
            cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
        # Slightly larger markers on the 5 landmarks the wave-detection
        # math actually tracks (wrist + the four MCP knuckles -- their
        # average is the "palm centroid" the oscillation detector
        # follows; see wave_detection/detector.py's module docstring
        # for why these points, not just the wrist alone).
        for idx in (0, 5, 9, 13, 17):
            if idx < len(landmarks_px):
                cv2.circle(frame, landmarks_px[idx], 5, color, 2, cv2.LINE_AA)

    # -- per-person status text block -----------------------------------------
    def _draw_status_block(self, frame, person: TrackedPerson, gate_state, wave_result: WaveResult, status_label: str) -> None:
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = person.bbox
        direction = gate_state.last_direction.value if gate_state else Direction.UNKNOWN.value
        wave_status = "WAVING" if wave_result.wave_confirmed or any(h.is_waving for h in wave_result.hands) \
            else ("HAND" if wave_result.hands else "NO HAND")

        lines = [
            f"Dir:{direction}  Wave:{wave_status}",
            f"Det:{person.confidence:.2f}  Wave:{wave_result.wave_confidence:.2f}",
            f"Status:{status_label}",
        ]

        line_h = 16
        block_h = line_h * len(lines)
        if y2 + block_h + 6 > frame_h:
            text_y = max(line_h, y1 - block_h - 4)
        else:
            text_y = y2 + line_h

        text_x = max(0, min(x1, frame_w - 220))
        for line in lines:
            cv2.putText(frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            text_y += line_h

    # -- suggested action banner ------------------------------------------------
    def _draw_active_message(self, frame, person: TrackedPerson, colors) -> None:
        message = self._messages.get(person.track_id)
        if message is None:
            return
        text = message.get_active_text()
        if text is None:
            self._active_action.pop(person.track_id, None)
            return

        action_label = self._active_action.get(person.track_id, "")
        bg_color = colors.suggested_checkout_bg if "OUT" in action_label else colors.suggested_checkin_bg

        x1, y1, x2, y2 = person.bbox
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self._config.font_scale, 2)

        banner_y1 = y2 + 64
        banner_x1, banner_x2 = x1, x1 + text_w + 14
        banner_y2 = banner_y1 + text_h + 16

        cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2), bg_color, -1)
        cv2.putText(frame, text, (banner_x1 + 7, banner_y2 - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    self._config.font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # -- event log panel -----------------------------------------------------
    def _draw_event_log(self, frame, event_log: EventLog, colors) -> None:
        entries = event_log.recent()
        if not entries:
            return

        frame_h, frame_w = frame.shape[:2]
        panel_w = 300
        line_h = 20
        panel_h = 24 + line_h * len(entries)
        x1 = frame_w - panel_w - 10
        y1 = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), colors.panel_bg, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, "RECENT EVENTS", (x1 + 8, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colors.text_color, 1, cv2.LINE_AA)

        y = y1 + 18
        for entry in entries:
            y += line_h
            cv2.putText(frame, entry.text[:38], (x1 + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, colors.text_color, 1, cv2.LINE_AA)

    # -- fps ---------------------------------------------------------------
    def _draw_fps(self, frame) -> None:
        fps = self._fps_counter.tick()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # -- calibration mode: live vertical slider preview -----------------------
    def _draw_calibration_overlay(self, frame, calibrator, colors) -> None:
        frame_h, frame_w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        msg = "CALIBRATION: move the mouse to slide the line, click to confirm"
        (text_w, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, msg, (max(10, (frame_w - text_w) // 2), 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        preview = calibrator.preview_line_px()
        if preview is not None:
            p1, p2 = preview
            # Dashed-looking preview line (alternating short segments) so
            # it visually reads as "not yet confirmed" versus the solid
            # magenta line used once calibration is done.
            self._draw_dashed_line(frame, p1, p2, (0, 255, 255))

    def _draw_dashed_line(self, frame, p1, p2, color, dash_len=12, gap_len=8) -> None:
        x1, y1 = p1
        x2, y2 = p2
        total_len = max(1, abs(y2 - y1))
        y = y1
        while y < y2:
            y_end = min(y + dash_len, y2)
            cv2.line(frame, (x1, y), (x2, y_end), color, 2, cv2.LINE_AA)
            y += dash_len + gap_len

    # -- persistent controls hint --------------------------------------------
    def _draw_controls_hint(self, frame, colors, calibrator, wave_sensitivity: str = "balanced") -> None:
        if calibrator is not None and calibrator.is_active:
            return  # calibration overlay already shows its own instructions

        frame_h, frame_w = frame.shape[:2]
        text = (
            f"[C] Calibrate line   [F] Flip IN/OUT   [M] Fullscreen   [V] Mirror   "
            f"[S] Wave sensitivity: {wave_sensitivity.upper()}   [D] Wave debug   [Q] Quit"
        )
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        x1 = (frame_w - text_w) // 2 - 10
        y2 = frame_h - 8
        y1 = y2 - text_h - 14
        x2 = x1 + text_w + 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colors.panel_bg, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colors.text_color, 1, cv2.LINE_AA)
