"""
Wave gesture detector built on MediaPipe Hands.

Design decision: MediaPipe Hands runs on each tracked person's cropped
bounding box (not the whole frame). This does two things at once:
  1. Cheaper inference (small crop vs full frame).
  2. Automatically associates detected hands with the correct tracking
     ID -- no separate hand-to-person matching step needed.

Design decision: ONE MediaPipe Hands instance per tracked person
(created lazily, destroyed when the track disappears, capped at
`max_concurrent_wave_trackers`). MediaPipe's Hands has internal temporal
smoothing that assumes a continuous video stream of the SAME subject;
sharing one instance across different people's crops would feed it a
discontinuous stream and degrade tracking quality. Per-person instances
keep each person's gesture history clean and independent, required for
"two people wave together -> two independent Suggested Actions."

Design decision: EVERY detected hand is tracked (not just the first),
matched frame-to-frame by nearest palm-centroid position. This fixes a
real bug in an earlier version, which only ever looked at hand index
0 -- a second hand (either the same person's other hand, or in rare
crowded cases a neighboring hand at the crop edge) was silently
ignored.

Design decision: crop margin is kept modest (see config) specifically
to avoid pulling a NEIGHBORING person's hand into this person's crop
in a crowded entrance -- a wider margin would catch more raised hands
but at the cost of misattributing waves between people standing close
together.

Design decision (performance without sacrificing responsiveness): each
hand crop is downscaled to a capped pixel size before being handed to
MediaPipe (`hand_crop_max_dimension_px`). MediaPipe's landmark model
works at a small fixed internal resolution regardless of input size, so
feeding it an unnecessarily large crop only adds our own color-convert
and copy overhead for no accuracy benefit. This -- plus defaulting to
tracking one hand instead of two (`max_num_hands`) -- is what keeps
per-frame cost low WITHOUT needing to skip frames, so wave/hand-box
detection can run at full frame rate for maximum responsiveness.

Design decision: every detected hand's full 21-point landmark skeleton
is captured and exposed (HandObservation.landmarks), not just a
bounding box. This lets the UI draw an actual hand skeleton, which is
both a clearer visual confirmation of tracking precision and a more
"premium product" look than a plain rectangle.

====================================================================
REDESIGNED WAVE ALGORITHM -- see wave_detection/oscillation.py and
wave_detection/body_motion.py for the full root-cause analysis and
design rationale. Summary of what changed and why:

  OLD: tracked wrist position only, on the horizontal axis only,
  normalized to the person's crop width (which changes with camera
  distance, crop margin config, and even reactively as the tracked
  bbox shifts). This systematically under-measured exactly the most
  natural wave style (wrist-pivot waves, where the wrist itself barely
  moves), missed non-horizontal waves entirely, and tied the required
  physical hand motion to something that isn't the hand.

  NEW: tracks the PALM CENTROID (average of wrist + four MCP knuckle
  landmarks -- a point that moves substantially under essentially all
  common wave styles), in full 2D, normalized to the HAND'S OWN
  detected size (wrist-to-middle-MCP distance, recomputed fresh every
  frame) rather than the crop. This is measured in "hand-widths": a
  scale- and distance-invariant unit. See oscillation.py for the
  vector-based reversal-counting algorithm operating on this signal.

  ADDED: a body-stillness gate (body_motion.py). Testing found hand
  kinematics ALONE cannot fully separate a deliberate wave from
  vigorous walking/jogging arm swing at overlapping amplitude/cadence
  -- this uses the person's own tracked bbox (already computed for
  entrance-line crossing) as a second, orthogonal signal: a wave must
  also happen while the person's body is relatively stationary.

Both the amplitude+timing signal and the body-stillness signal are
independently unit-tested with realistic synthetic trajectories in
wave_detection/test_oscillation.py -- run it directly:
    python wave_detection/test_oscillation.py
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp

from config.settings import WaveDetectionConfig
from person_tracking.tracked_person import TrackedPerson
from utils.geometry import expand_bbox
from utils.timing import Cooldown
from wave_detection.body_motion import BodyStillnessTracker
from wave_detection.oscillation import OscillationTracker
from wave_detection.wave_result import HandObservation, WaveResult
from wave_detection.wave_state import HandTrack, WaveState

logger = logging.getLogger(__name__)

# Palm-centroid landmarks: wrist + the four MCP (base) knuckles of the
# index/middle/ring/pinky fingers. Chosen specifically because these
# points sit well away from the wrist pivot -- they sweep a
# meaningfully larger arc than the wrist alone under a relaxed,
# wrist-pivoted wave (the most common natural style), while still
# moving correctly under a full-forearm or elbow-pivoted wave too.
_PALM_CENTROID_INDICES = (0, 5, 9, 13, 17)
# The standard "palm length" scale reference: wrist to middle-finger
# MCP knuckle.
_HAND_SCALE_INDEX_A = 0
_HAND_SCALE_INDEX_B = 9


class WaveDetector:
    def __init__(self, config: WaveDetectionConfig, debug_overlay: Optional[bool] = None):
        # FAIL FAST, WITH A CLEAR MESSAGE, on a real installation problem
        # this project has actually hit: newer mediapipe releases (>=
        # 0.10.3x, and the 1.x line) removed the legacy `mp.solutions`
        # API this module is built on (`mp.solutions.hands.Hands`) --
        # `mp.solutions` simply doesn't exist on those versions, only
        # the newer Tasks API (`mp.tasks`) does. requirements.txt pins
        # `mediapipe==0.10.14` (which still has `mp.solutions`), but if
        # that exact pin wasn't honored at install time -- a fresh
        # `pip install mediapipe` with no version, a wheel-availability
        # fallback on a newer Python, or some other package pulling in
        # a newer mediapipe as a dependency -- hand detection silently
        # never runs at all: no crash inside the main loop (this class
        # would only blow up the instant it's constructed, before the
        # camera even opens), no per-frame error, no visible hand box,
        # no skeleton, nothing -- which looks EXACTLY like "wave
        # detection just isn't working," even though every part of the
        # actual detection/decision logic downstream (see oscillation.py,
        # body_motion.py) is correct and fully unit-tested. Checking
        # this explicitly, with a message that names the exact fix,
        # turns a confusing silent failure into an obvious one.
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This installed mediapipe version ({version}) does not have "
                "mp.solutions.hands -- the legacy Hands API this project's wave "
                "detection depends on was removed in newer mediapipe releases. "
                "Fix: pip install mediapipe==0.10.14 (the exact version pinned "
                "in requirements.txt), then restart. If a virtual environment "
                "isn't already in use, create one first so this doesn't get "
                "overridden by another package's mediapipe requirement later.".format(
                    version=getattr(mp, "__version__", "unknown")
                )
            )

        self._config = config
        self._mp_hands = mp.solutions.hands
        self._hands_instances: Dict[int, "mp.solutions.hands.Hands"] = {}
        self._states: Dict[int, WaveState] = {}
        # Caches the last real WaveResult per person, returned unchanged
        # on frames we deliberately skip (only relevant if
        # wave_processing_frame_interval > 1; default is 1, i.e. never
        # skip).
        self._last_results: Dict[int, WaveResult] = {}
        self._call_counters: Dict[int, int] = {}
        self.debug_overlay = config.debug_overlay_default_on if debug_overlay is None else debug_overlay
        # Snapshot of every sensitivity-relevant config field, taken at
        # construction and refreshed every time it's applied -- see
        # _sync_config_if_changed() for why this exists: it's what
        # lets the 'S' hotkey (or a preset/CLI change) actually reach
        # people who are ALREADY being tracked, not just new people
        # who appear afterward.
        self._last_applied_snapshot = self._config_snapshot()

    def _config_snapshot(self) -> tuple:
        c = self._config
        return (
            c.min_detection_confidence,
            c.min_tracking_confidence,
            c.max_num_hands,
            c.min_swing_amplitude_hand_widths,
            c.min_reversals_required,
            c.max_seconds_between_reversals,
            c.max_reversal_gap_regularity_cv,
            c.wave_smoothing_alpha,
            c.max_body_speed_fraction_per_second,
        )

    def _sync_config_if_changed(self) -> None:
        """
        REAL BUG FIX: every sensitivity-affecting value (MediaPipe's own
        min_detection_confidence/min_tracking_confidence, plus the
        oscillation and body-stillness thresholds) used to be baked in
        ONCE -- at the moment a person's MediaPipe Hands instance /
        OscillationTracker was first created -- and never revisited
        after that. Pressing 'S' to cycle sensitivity (or starting with
        --sensitivity, or the dashboard's sensitivity dropdown) mutates
        CONFIG.wave in place, which DOES reach brand-new people (their
        state is created fresh, from current config, the first time
        they're seen) -- but does nothing at all for anyone ALREADY
        being tracked at that moment, since their Hands instance and
        OscillationTracker objects were already built with the OLD
        values and neither is re-read from config afterward. In
        practice this is exactly the scenario someone testing wave
        sensitivity hits hardest: stand in frame, wave, it doesn't
        register, press 'S' to loosen it, wave again -- and NOTHING
        changes for that same still-in-frame person, because their
        detector objects never got the memo. Fixed by detecting a
        config change once per call to process() and, if one happened,
        recreating every active person's Hands instance (with the new
        confidence thresholds) and clearing every hand's
        OscillationTracker (forcing a fresh one, built from current
        config, on the very next update) -- plus pushing the new
        body-stillness speed cap into every already-existing
        BodyStillnessTracker. This does discard in-progress swing
        history for anyone mid-gesture the instant sensitivity changes,
        which is an acceptable, minor cost (equivalent to them starting
        their wave over) for actually applying the new settings live,
        which is the entire point of the hotkey existing.
        """
        snapshot = self._config_snapshot()
        if snapshot == self._last_applied_snapshot:
            return
        self._last_applied_snapshot = snapshot

        for track_id in list(self._hands_instances.keys()):
            old_hands = self._hands_instances.pop(track_id)
            old_hands.close()
            self._hands_instances[track_id] = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self._config.max_num_hands,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )

        for state in self._states.values():
            for hand_track in state.hands.values():
                # Force a fresh OscillationTracker next update -- it
                # will be built from _make_oscillation_tracker(), which
                # always reads current config values.
                hand_track.oscillation = None
            if state.body_stillness is not None:
                state.body_stillness.set_max_speed(self._config.max_body_speed_fraction_per_second)

    def _get_or_create_state(self, track_id: int) -> Optional[WaveState]:
        if track_id in self._states:
            return self._states[track_id]

        if len(self._hands_instances) >= self._config.max_concurrent_wave_trackers:
            # Capacity reached. Detection/tracking are unaffected; this
            # person simply won't get wave analysis until a slot frees
            # up (e.g. someone else leaves frame). Prevents unbounded
            # CPU growth in a crowded entrance.
            return None

        state = WaveState(
            track_id=track_id,
            cooldown=Cooldown(seconds=self._config.wave_cooldown_seconds),
            body_stillness=BodyStillnessTracker(
                window_seconds=self._config.body_stillness_window_seconds,
                max_speed_fraction_of_height_per_sec=self._config.max_body_speed_fraction_per_second,
            ),
        )
        self._states[track_id] = state
        self._hands_instances[track_id] = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self._config.max_num_hands,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        return state

    def remove_track(self, track_id: int) -> None:
        hands = self._hands_instances.pop(track_id, None)
        if hands is not None:
            hands.close()
        self._states.pop(track_id, None)
        self._last_results.pop(track_id, None)
        self._call_counters.pop(track_id, None)

    def _make_oscillation_tracker(self) -> OscillationTracker:
        return OscillationTracker(
            min_swing_amplitude_hand_widths=self._config.min_swing_amplitude_hand_widths,
            min_reversals_required=self._config.min_reversals_required,
            max_seconds_between_reversals=self._config.max_seconds_between_reversals,
            max_regularity_cv=self._config.max_reversal_gap_regularity_cv,
            smoothing_alpha=self._config.wave_smoothing_alpha,
        )

    def process(self, frame, tracked_person: TrackedPerson) -> WaveResult:
        """
        Run wave detection for one tracked person on the current frame.
        Returns a WaveResult with per-hand observations (bbox +
        landmarks, for drawing) and whether a NEW wave was just
        confirmed this frame.
        """
        self._sync_config_if_changed()

        track_id = tracked_person.track_id
        state = self._get_or_create_state(track_id)
        if state is None:
            return WaveResult()

        # Update the body-stillness signal every call this person is
        # processed, regardless of whether a hand is currently detected
        # -- it needs a continuous history of the person's own bbox to
        # judge whether they've been translating through the frame.
        now = time.time()
        px1, py1, px2, py2 = tracked_person.bbox
        body_center_x = (px1 + px2) / 2.0
        body_center_y = (py1 + py2) / 2.0
        body_height = max(1.0, py2 - py1)
        state.body_stillness.update(body_center_x, body_center_y, body_height, now)

        call_count = self._call_counters.get(track_id, 0)
        self._call_counters[track_id] = call_count + 1

        if call_count % self._config.wave_processing_frame_interval != 0:
            cached = self._last_results.get(track_id, WaveResult())
            return WaveResult(wave_confirmed=False, wave_confidence=cached.wave_confidence, hands=cached.hands)

        frame_h, frame_w = frame.shape[:2]
        crop_bbox = expand_bbox(
            tracked_person.bbox, frame_w, frame_h, margin_ratio=self._config.hand_crop_margin_ratio
        )
        cx1, cy1, cx2, cy2 = crop_bbox
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            result = WaveResult(body_debug=state.body_stillness.debug)
            self._last_results[track_id] = result
            return result

        crop_h, crop_w = crop.shape[:2]

        # Downscale before feeding to MediaPipe -- see module docstring.
        # Landmark coordinates come back normalized (0-1) regardless of
        # the input size, so mapping back to global frame pixels still
        # uses the ORIGINAL crop_w/crop_h, not the downscaled size.
        longer_side = max(crop_w, crop_h)
        if longer_side > self._config.hand_crop_max_dimension_px:
            scale = self._config.hand_crop_max_dimension_px / longer_side
            mp_input = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)))
        else:
            mp_input = crop

        crop_rgb = cv2.cvtColor(mp_input, cv2.COLOR_BGR2RGB)
        results = self._hands_instances[track_id].process(crop_rgb)

        result = WaveResult(body_debug=state.body_stillness.debug)

        if not results.multi_hand_landmarks:
            self._last_results[track_id] = result
            return result

        # For every detected hand this frame: map all 21 landmarks to
        # GLOBAL FRAME PIXEL coordinates (not MediaPipe's normalized
        # 0-1 crop-relative coordinates). Using pixel coordinates for
        # the palm centroid AND the hand-scale reference keeps both
        # measurements in the SAME, aspect-ratio-correct unit -- see
        # oscillation.py's docstring for why this matters (mixing
        # normalized x and y directly would distort distances whenever
        # the crop isn't square).
        detected_hands = []
        for landmarks in results.multi_hand_landmarks:
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]

            bbox_px = (
                cx1 + int(min(xs) * crop_w),
                cy1 + int(min(ys) * crop_h),
                cx1 + int(max(xs) * crop_w),
                cy1 + int(max(ys) * crop_h),
            )
            landmark_points_px = [
                (cx1 + lm.x * crop_w, cy1 + lm.y * crop_h) for lm in landmarks.landmark
            ]

            palm_x = sum(landmark_points_px[i][0] for i in _PALM_CENTROID_INDICES) / len(_PALM_CENTROID_INDICES)
            palm_y = sum(landmark_points_px[i][1] for i in _PALM_CENTROID_INDICES) / len(_PALM_CENTROID_INDICES)
            ax, ay = landmark_points_px[_HAND_SCALE_INDEX_A]
            bx, by = landmark_points_px[_HAND_SCALE_INDEX_B]
            hand_scale_px = math.hypot(bx - ax, by - ay)

            # Integer pixel points for drawing (skeleton overlay).
            landmark_points_int = [(int(x), int(y)) for x, y in landmark_points_px]

            detected_hands.append((palm_x, palm_y, hand_scale_px, bbox_px, landmark_points_int))

        assigned_slots = self._assign_hands_to_slots(state, detected_hands)

        any_confirmed = False
        best_confidence = 0.0
        body_ok = state.body_stillness.is_relatively_stationary()

        for slot_id, (palm_x, palm_y, hand_scale_px, bbox_px, landmark_points_int) in assigned_slots.items():
            track = state.hands[slot_id]
            track.last_bbox_px = bbox_px
            track.last_palm_xy = (palm_x, palm_y)
            track.last_seen_time = now

            if track.oscillation is None:
                track.oscillation = self._make_oscillation_tracker()

            hand_confirmed = track.oscillation.update(palm_x, palm_y, hand_scale_px, now)
            if hand_confirmed:
                # ALWAYS reset the instant the hand's own gesture-shape
                # check completes -- regardless of body_ok or cooldown
                # below. This is the fix for a real, tested bug:
                # OscillationTracker._check_confirmed() keeps returning
                # True on every subsequent call once the reversal count
                # is reached; it does not reset itself. If reset were
                # only called deeper in the "actually confirmed and
                # allowed to fire" branch, a wave-SHAPED hand motion
                # that happens to complete while the person is still
                # WALKING (body_ok False) would leave the tracker
                # permanently "primed": the moment they later simply
                # STOP WALKING for any reason -- with zero further hand
                # motion -- body_ok would flip True and this stale state
                # would fire a spurious CHECK IN/OUT. Resetting here,
                # unconditionally, means only an ACTUAL new wave-shaped
                # motion (checked fresh from this point on) can ever
                # lead to a confirmation, however body-stillness or
                # cooldown play out afterward.
                track.oscillation.reset_progress()

            # A wave only actually confirms if the person's BODY has
            # also been relatively stationary AT THE MOMENT the hand
            # completed its swing -- see module docstring and
            # body_motion.py for why this is required, not optional,
            # on top of the hand-motion signal.
            confirmed = hand_confirmed and body_ok
            if hand_confirmed and not body_ok:
                logger.info(
                    "Track %d: hand-oscillation confirmed a wave shape, but body-stillness "
                    "gate rejected it (speed=%.2f height/s, cap=%.2f) -- treating "
                    "as walking/jogging arm swing, not a deliberate wave.",
                    track_id,
                    state.body_stillness.debug.speed_fraction_of_height_per_sec or 0.0,
                    self._config.max_body_speed_fraction_per_second,
                )

            if confirmed:
                track.confirmed_display_until = now + self._config.wave_confirmed_display_seconds

            is_waving_now = track.confirmed_display_until is not None and now < track.confirmed_display_until
            progress = min(1.0, track.oscillation.debug.reversal_count / max(1, self._config.min_reversals_required))
            best_confidence = max(best_confidence, 1.0 if confirmed else progress)

            result.hands.append(HandObservation(
                bbox=bbox_px, is_waving=is_waving_now, confidence=progress,
                landmarks=landmark_points_int, debug=track.oscillation.debug,
            ))

            # Cooldown gates only whether the ACTUAL attendance action
            # fires this instant -- never whether gesture state gets
            # cleared (that already happened, unconditionally, above).
            # If a wave-shaped motion completes while a PREVIOUS wave's
            # cooldown is still active, it's simply not actioned (and
            # not remembered) -- the person must wave again after the
            # cooldown clears, exactly as intended, with no stale
            # leftover state to spuriously fire later.
            if confirmed and state.cooldown.is_ready():
                state.cooldown.trigger()
                any_confirmed = True

        result.wave_confirmed = any_confirmed
        result.wave_confidence = best_confidence
        self._last_results[track_id] = result
        return result

    def _assign_hands_to_slots(
        self, state: WaveState, detected_hands: List[Tuple[float, float, float, Tuple[int, int, int, int], list]]
    ) -> Dict[int, Tuple[float, float, float, Tuple[int, int, int, int], list]]:
        """
        Nearest-neighbor match each detected hand this frame to an
        existing hand slot (by 2D palm-centroid pixel distance, scaled
        by hand size so the gate is meaningful regardless of distance
        from camera), or create a new slot if there's room. Keeps each
        hand's oscillation history continuous across frames even if
        MediaPipe reorders its output.
        """
        available_slots = list(state.hands.keys())
        assigned: Dict[int, Tuple[float, float, float, Tuple[int, int, int, int], list]] = {}

        for palm_x, palm_y, hand_scale_px, bbox_px, landmarks_px in detected_hands:
            best_slot = None
            best_dist = None
            for slot_id in available_slots:
                prev_xy = state.hands[slot_id].last_palm_xy
                if prev_xy is None:
                    continue
                dist = math.hypot(prev_xy[0] - palm_x, prev_xy[1] - palm_y)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_slot = slot_id

            # A hand shouldn't jump more than a few hand-widths between
            # consecutive frames at a reasonable frame rate -- gating
            # by a multiple of the CURRENT hand's own detected size
            # (rather than a fixed pixel distance) keeps this
            # assignment threshold meaningful regardless of how close
            # the person is to the camera.
            match_gate_px = max(20.0, 4.0 * hand_scale_px)

            if best_slot is not None and best_dist is not None and best_dist < match_gate_px:
                available_slots.remove(best_slot)
                assigned[best_slot] = (palm_x, palm_y, hand_scale_px, bbox_px, landmarks_px)
            else:
                new_slot_id = max(state.hands.keys(), default=-1) + 1
                state.hands[new_slot_id] = HandTrack()
                assigned[new_slot_id] = (palm_x, palm_y, hand_scale_px, bbox_px, landmarks_px)

        return assigned
