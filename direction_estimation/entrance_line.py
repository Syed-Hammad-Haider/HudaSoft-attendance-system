"""
Entrance Line (Virtual Gate) system.

WHY THIS REPLACES PURE MOVEMENT-DIRECTION ESTIMATION:
The earlier approach guessed direction from whether a person's net
horizontal movement was positive or negative. That has two real
problems: (1) it only works if the camera happens to face the entrance
such that "left/right" maps cleanly to "in/out" -- a camera at any other
angle breaks it -- and (2) if a person pauses, net displacement can drop
below threshold and the label flickers back to UNKNOWN and then
re-emerges, which looks broken in a live product.

A virtual gate fixes both: we define an actual line across the frame
(matching the real entrance/doorway location), and we only care about
which SIDE of that line someone is on and when they switch sides. This
works for a camera at ANY angle, because the line itself is drawn to
match wherever the real entrance appears in that specific camera's
view -- horizontal doorway, vertical doorway, or diagonal, doesn't
matter, because the line is what gets configured to match, not the
detection logic.

COMPUTER VISION CONCEPT USED: signed distance from a point to a line,
computed via a 2D cross product. This is standard, cheap (a few
multiplications), and gives us both (a) which side a point is on
(the sign) and (b) how close it is to the line (the magnitude) for the
same calculation -- the magnitude is reused to decide the "crossing
soon" UI highlight.

DEBOUNCING: raw per-frame side classification is noisy right at the
line (detection box jitter can flip the sign frame to frame). We
require a side to be observed for `min_frames_to_confirm_side`
consecutive frames before treating it as the person's real side. This
also means the reported Direction is STICKY -- it holds the last
confirmed crossing direction and does not flicker back to UNKNOWN
between crossings, which is more trustworthy for a live product than
a value that flickers.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from config.settings import EntranceLineConfig


class Side(str, Enum):
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"
    UNKNOWN = "UNKNOWN"


class Direction(str, Enum):
    ENTERING = "ENTERING"
    LEAVING = "LEAVING"
    UNKNOWN = "UNKNOWN"


@dataclass
class PersonGateState:
    track_id: int
    confirmed_side: Side = Side.UNKNOWN
    pending_side: Side = Side.UNKNOWN
    pending_count: int = 0
    last_direction: Direction = Direction.UNKNOWN
    last_crossing_time: Optional[float] = None
    signed_distance: float = 0.0
    just_crossed: bool = False  # True only on the exact frame a crossing is confirmed

    def direction_is_recent(self, window_seconds: float) -> bool:
        """
        Whether `last_direction` reflects a crossing that happened
        recently enough to still be actionable. Without this check, a
        wave from someone standing well inside the building (long after
        their last crossing) would incorrectly still suggest CHECK IN,
        just because that's the last thing they ever crossed as.
        """
        if self.last_crossing_time is None:
            return False
        return (time.time() - self.last_crossing_time) <= window_seconds




class EntranceGate:
    def __init__(self, config: EntranceLineConfig, frame_width: int, frame_height: int):
        self._config = config
        self._frame_w = frame_width
        self._frame_h = frame_height
        # Copied from config at startup, but mutable at runtime via
        # set_line()/flip_inside_side() -- this is what lets someone
        # calibrate the line interactively on-site (e.g. against a real
        # CCTV entrance) without editing config.py at all.
        self._inside_is_left = config.inside_is_left_of_line
        self._p1: Tuple[float, float] = (
            config.line_start[0] * frame_width,
            config.line_start[1] * frame_height,
        )
        self._p2: Tuple[float, float] = (
            config.line_end[0] * frame_width,
            config.line_end[1] * frame_height,
        )
        self._states: Dict[int, PersonGateState] = {}
        # Precomputed once: how far (in pixels) past the line someone
        # must actually be for a crossing to count as committed, not
        # just a momentary lean/hesitation across it.
        self._commit_distance_px = config.min_crossing_commit_distance_ratio * max(frame_width, frame_height)

    def line_points_px(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return the line's two endpoints in pixel coordinates, for drawing."""
        return (
            (int(self._p1[0]), int(self._p1[1])),
            (int(self._p2[0]), int(self._p2[1])),
        )

    def set_line(self, p1_px: Tuple[float, float], p2_px: Tuple[float, float]) -> None:
        """
        Move the entrance line to new pixel coordinates -- used by the
        interactive on-screen calibrator. Clears all per-person crossing
        state, since "which side of the line" is meaningless once the
        line itself has moved.
        """
        self._p1 = p1_px
        self._p2 = p2_px
        self._states.clear()

    def flip_inside_side(self) -> bool:
        """
        Swap which side of the line counts as INSIDE. Used when CHECK
        IN/OUT come out backwards for a given camera. Returns the new
        value so the caller can persist it.
        """
        self._inside_is_left = not self._inside_is_left
        self._states.clear()
        return self._inside_is_left

    def get_calibration(self) -> Tuple[Tuple[float, float], Tuple[float, float], bool]:
        """Return the current line as frame-ratio points (0-1) plus the
        inside-side flag, ready to be saved to disk."""
        p1_ratio = (self._p1[0] / self._frame_w, self._p1[1] / self._frame_h)
        p2_ratio = (self._p2[0] / self._frame_w, self._p2[1] / self._frame_h)
        return p1_ratio, p2_ratio, self._inside_is_left

    def label_points_px(self, offset: int = 45) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Return ((in_x, in_y), (out_x, out_y)) pixel positions for drawing
        "IN" / "OUT" text labels on the correct respective sides of the
        line. Uses the SAME sign convention as the actual crossing
        classification, so the labels can never mismatch the real logic.
        """
        x1, y1 = self._p1
        x2, y2 = self._p2
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length == 0:
            return (int(mx), int(my)), (int(mx), int(my))

        # Unit vector perpendicular to the line.
        perp_x, perp_y = -dy / length, dx / length
        point_a = (mx + perp_x * offset, my + perp_y * offset)
        point_b = (mx - perp_x * offset, my - perp_y * offset)

        dist_a = self._signed_distance(point_a)
        if (dist_a > 0) == self._inside_is_left:
            inside_point, outside_point = point_a, point_b
        else:
            inside_point, outside_point = point_b, point_a

        return (
            (int(inside_point[0]), int(inside_point[1])),
            (int(outside_point[0]), int(outside_point[1])),
        )

    def _signed_distance(self, point: Tuple[float, float]) -> float:
        x1, y1 = self._p1
        x2, y2 = self._p2
        x, y = point
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return cross / length if length > 0 else 0.0

    def update(self, track_id: int, center_point: Tuple[float, float]) -> PersonGateState:
        """
        Feed this person's current center point in for this frame.
        Returns their current gate state (side, sticky direction, and
        whether a crossing was JUST confirmed this frame).
        """
        state = self._states.setdefault(track_id, PersonGateState(track_id=track_id))
        distance = self._signed_distance(center_point)
        state.signed_distance = distance
        state.just_crossed = False

        raw_side = Side.INSIDE if ((distance > 0) == self._inside_is_left) else Side.OUTSIDE

        if raw_side == state.pending_side:
            state.pending_count += 1
        else:
            state.pending_side = raw_side
            state.pending_count = 1

        if (
            state.pending_count >= self._config.min_frames_to_confirm_side
            and state.confirmed_side != raw_side
            and abs(distance) >= self._commit_distance_px
        ):
            previous_side = state.confirmed_side
            state.confirmed_side = raw_side

            if previous_side == Side.OUTSIDE and raw_side == Side.INSIDE:
                state.last_direction = Direction.ENTERING
                state.last_crossing_time = time.time()
                state.just_crossed = True
            elif previous_side == Side.INSIDE and raw_side == Side.OUTSIDE:
                state.last_direction = Direction.LEAVING
                state.last_crossing_time = time.time()
                state.just_crossed = True
            # first-ever confirmed side (previous_side == UNKNOWN) is not
            # a crossing -- it's just establishing where they started.
            #
            # NOTE: if pending_count's debounce is satisfied but the
            # commit-distance requirement is NOT yet met, we deliberately
            # do nothing here rather than resetting pending_count -- the
            # person is still moving in a consistent direction (that's
            # why raw_side hasn't flipped), just hasn't traveled far
            # enough past the line yet. The very next frame where they
            # do will confirm it. If they instead turn back, raw_side
            # will flip on its own and reset pending_side/pending_count
            # via the branch above, correctly discarding the hesitation.

        return state

    def is_near_line(self, distance: float) -> bool:
        threshold = self._config.crossing_highlight_distance_ratio * max(self._frame_w, self._frame_h)
        return abs(distance) <= threshold

    def remove_track(self, track_id: int) -> None:
        self._states.pop(track_id, None)
