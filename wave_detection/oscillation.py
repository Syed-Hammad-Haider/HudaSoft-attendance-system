"""
Pure, camera-independent core of wave-gesture detection: given a
stream of (x, y, hand_scale, timestamp) samples, decides whether the
motion is a genuine wave.

WHY THIS IS A SEPARATE, MEDIAPIPE-FREE MODULE:
Every other part of wave detection (cropping, MediaPipe inference,
coordinate mapping) depends on a real camera and can only really be
verified by a human waving at one. The actual DECISION LOGIC --
"is this sequence of hand positions a wave?" -- does not depend on any
of that. Keeping it in its own module with no MediaPipe/OpenCV import
means it can be, and is, unit-tested directly with synthetic
trajectories representing real waving patterns, walking gait, static
raises, pointing, and random motion (see the test procedure documented
in WAVE_DETECTION_REDESIGN.md). That is what actually gives confidence
this logic works, rather than "it compiles."

====================================================================
WHY THE OLD DESIGN MISSED GENUINE WAVES (root-cause analysis)
====================================================================

The previous version tracked ONE landmark (the wrist), on ONE axis
(horizontal x), normalized to the CROP width (the person's bounding
box, freshly recomputed every frame). Three independent, compounding
problems:

1. THE WRIST IS THE WORST POINT TO TRACK, BY DESIGN.
   In the most natural, relaxed "hello" wave, people pivot the hand at
   the WRIST -- the forearm barely moves, only the palm/fingers swing
   through an arc. The wrist is, almost by definition, the point on
   the hand that moves LEAST in a wrist-pivoted wave (it's the pivot,
   not the thing sweeping). Tracking wrist position specifically
   captured the smallest possible signal for exactly the most common,
   natural wave style -- a person waving normally, close to the
   camera, with clearly visible hand motion, could produce almost no
   wrist displacement at all, and the detector would see ~nothing.

   FIX: track the PALM CENTROID (average of the wrist plus the four
   MCP knuckles -- landmarks 0, 5, 9, 13, 17) instead. These points sit
   further from the wrist pivot and sweep a meaningfully larger arc
   under a wrist-pivoted wave, while still moving correctly under a
   full-forearm or elbow-pivoted wave too. This one change alone
   substantially increases the measured signal for the style of wave
   most people default to.

2. HORIZONTAL-ONLY TRACKING MISSES NON-HORIZONTAL WAVES.
   Many genuine waves aren't purely side-to-side -- a "royal wave"
   (wrist rotation), an up-down wave, or anything at a diagonal angle
   has real Y-axis motion the old x-only detector never looked at,
   however clear and close the hand was.

   FIX: track BOTH axes independently and register a reversal on
   WHICHEVER axis shows a genuine swing. A pure horizontal wave is
   caught on X; a pure vertical wave on Y; a diagonal wave contributes
   to both. See WaveAxisTracker below.

3. NORMALIZING BY A CONSTANTLY-CHANGING PERSON CROP IS INHERENTLY
   UNSTABLE. The crop is recomputed fresh every frame from the
   person's live-tracked bounding box, which itself can shift and
   resize as the person moves (including in reaction to a raised arm).
   Measuring "how far did the hand move" as a FRACTION OF THAT CROP'S
   WIDTH ties the required physical hand motion to something that
   isn't the hand at all, and that keeps moving under it -- direct,
   documented evidence of this in the old config: an earlier change to
   the crop margin (0.14 -> 0.25) silently made the SAME physical wave
   score ~15% smaller, requiring a manual threshold correction. That
   is a symptom of a genuinely unstable measurement, not a tuning gap.

   FIX: normalize by the HAND'S OWN detected size instead --
   specifically, the distance between the wrist and the middle-finger
   MCP knuckle (landmarks 0 and 9), recomputed fresh every frame from
   that same frame's landmarks. Motion is now measured in "hand
   widths": a person's hand naturally appears bigger when close to the
   camera, but their hand's OWN motion also appears bigger by the same
   factor, so the ratio is close to invariant to distance, crop margin,
   camera resolution, or any bounding-box jitter. This is the standard
   scale-normalization approach for hand landmark motion and is the
   real, permanent fix for the fragility documented above -- not
   another number to retune every time something else about the crop
   changes.

====================================================================
WHY min_reversals_required + max_seconds_between_reversals ARE KEPT
====================================================================

This project's existing timing-based confirmation logic (require N
direction reversals, each within a bounded time of the last) is sound
and was already validated against real gait data (see
config/settings.py's WaveDetectionConfig history) -- a static raised
hand produces 0 reversals, walking/jogging arm swing was tested and
found to need 3 reversals within a tight window to reliably exclude.
That part of the design isn't being replaced, because it isn't the
part that was broken -- what was broken was the SIGNAL fed into it
(see above). This redesign fixes the signal and keeps the proven
timing logic, plus adds one new, genuinely complementary safety net:
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple


@dataclass
class OscillationDebugInfo:
    """Everything needed to render the DEBUG/ANALYSIS overlay for one hand."""
    samples_analyzed: int = 0
    hand_scale_px: Optional[float] = None
    palm_x: Optional[float] = None
    palm_y: Optional[float] = None
    last_swing_magnitude_hand_widths: float = 0.0
    reversal_count: int = 0
    reversals_required: int = 0
    regularity_cv: Optional[float] = None
    regularity_ok: bool = True
    seconds_since_last_reversal: Optional[float] = None
    state: str = "idle"  # "idle" | "tracking" | "ready_pending_regularity" | "confirmed" | "cooldown"


class OscillationTracker:
    """
    Feed this real (x, y, hand_scale, timestamp) samples, one per
    frame, via update(). It decides whether the recent motion is a
    genuine wave, using hand-width-normalized 2D VECTOR oscillation
    counting with wall-clock timing (see module docstring for why).

    WHY A UNIFIED 2D VECTOR, NOT TWO INDEPENDENT X/Y TRACKERS:
    An earlier version of this class tracked X and Y as two completely
    independent 1D extremum trackers, each needing to independently
    clear the amplitude threshold. Testing (wave_detection/
    test_oscillation.py) caught a real bug this causes: a genuine
    diagonal wave splits its motion across both axes, so EACH axis
    individually can end up below threshold even though the true,
    combined 2D displacement clearly clears it -- a diagonal wave could
    be silently under-counted for no real reason. This version instead
    tracks the swing as a single 2D vector from the last reversal
    point: a reversal fires when the hand has moved at least the
    threshold distance AND the direction of that motion is roughly
    opposite (negative dot product) to the direction of the previous
    swing. This correctly handles horizontal, vertical, and diagonal
    motion uniformly, using the TRUE combined magnitude rather than
    two independently-gated components.
    """

    def __init__(
        self,
        min_swing_amplitude_hand_widths: float,
        min_reversals_required: int,
        max_seconds_between_reversals: float,
        max_regularity_cv: float,
        smoothing_alpha: float,
    ):
        self._min_swing = min_swing_amplitude_hand_widths
        self._min_reversals = min_reversals_required
        self._max_gap = max_seconds_between_reversals
        self._max_regularity_cv = max_regularity_cv
        self._smoothing_alpha = smoothing_alpha

        self._smoothed: Optional[Tuple[float, float]] = None
        # The point (in the same pixel space as the input) from which
        # the CURRENT swing's distance is measured -- either the start
        # of tracking, or the point of the last registered reversal.
        self._extremum: Optional[Tuple[float, float]] = None
        # Unit vector of the current swing's direction, or None before
        # the first threshold-crossing swing has been established.
        self._direction: Optional[Tuple[float, float]] = None

        self._reversal_times: Deque[float] = deque(maxlen=10)
        # Edge-triggered guard: once a confirmation has been reported
        # for the CURRENT streak of reversals, don't report it again
        # until either reset_progress() is called or a genuinely NEW
        # reversal extends the streak. Defense-in-depth on top of
        # (not a replacement for) the caller correctly calling
        # reset_progress() -- found by direct testing that
        # _check_confirmed() alone keeps returning True on every
        # subsequent call once the reversal count is reached, which is
        # a footgun for any future caller of this class, not just the
        # one caller that exists today. See
        # wave_detection/test_oscillation.py's "does not stay stuck"
        # test.
        self._already_reported = False

        self.samples_analyzed = 0
        self.debug = OscillationDebugInfo(reversals_required=min_reversals_required)

    def reset_progress(self) -> None:
        """Clear accumulated reversal progress (e.g. after a confirmed wave)
        without discarding the tracker's current position -- a new wave
        should be measured from wherever the hand currently is, not
        forced back through an artificial re-initialization."""
        self._reversal_times.clear()
        self._already_reported = False

    def update(self, x: float, y: float, hand_scale: Optional[float], now: float) -> bool:
        """
        Process one frame's palm-centroid position. Returns True if a
        wave was just confirmed on THIS call (caller is responsible for
        cooldown and any additional real-world gating, e.g. requiring
        the person's BODY to be relatively stationary -- see
        wave_detection/detector.py -- this class only reasons about the
        hand's own motion).
        """
        self.samples_analyzed += 1
        self.debug.samples_analyzed = self.samples_analyzed

        if hand_scale is None or hand_scale <= 1e-6:
            # Can't measure motion in hand-widths without a valid scale
            # this frame (e.g. a degenerate/edge-of-crop detection).
            # Don't penalize progress for one bad frame -- just skip it.
            self.debug.state = "tracking" if self._reversal_times else "idle"
            return False

        # Light exponential smoothing: reduces per-frame landmark
        # jitter (occasional single-frame outliers from partial
        # occlusion or motion blur) without meaningfully delaying real
        # motion. Deliberately LIGHT (high alpha, i.e. trusts the new
        # sample heavily) -- testing found a heavier smoothing value
        # measurably damped fast waves and low-FPS sampling below the
        # detection threshold. This is safe to keep light because the
        # amplitude threshold itself (0.5 hand-widths) already sits
        # roughly 15-20x above typical single-frame landmark jitter
        # (a few percent of hand-width) -- the threshold, not heavy
        # smoothing, is what actually rejects jitter.
        a = self._smoothing_alpha
        if self._smoothed is None:
            self._smoothed = (x, y)
        else:
            self._smoothed = (
                a * x + (1 - a) * self._smoothed[0],
                a * y + (1 - a) * self._smoothed[1],
            )
        sx, sy = self._smoothed

        min_swing_units = self._min_swing * hand_scale
        reversed_now = False

        if self._extremum is None:
            self._extremum = (sx, sy)
        else:
            ex, ey = self._extremum
            dx, dy = sx - ex, sy - ey
            dist = math.hypot(dx, dy)

            if dist >= min_swing_units:
                if self._direction is None:
                    # First real swing away from the starting point --
                    # establishes a direction, but there's nothing to
                    # have reversed FROM yet.
                    self._direction = (dx / dist, dy / dist)
                    self._extremum = (sx, sy)
                else:
                    dot = (dx / dist) * self._direction[0] + (dy / dist) * self._direction[1]
                    if dot < 0:
                        # Moved back past the threshold in roughly the
                        # opposite direction from the current swing --
                        # a genuine reversal.
                        reversed_now = True
                        self._direction = (dx / dist, dy / dist)
                        self._extremum = (sx, sy)
                    else:
                        # Still extending in the same general direction
                        # -- ratchet the extremum forward so the NEXT
                        # reversal is measured from the true peak, not
                        # an earlier intermediate point.
                        self._extremum = (sx, sy)

                self.debug.last_swing_magnitude_hand_widths = dist / hand_scale

        self.debug.hand_scale_px = hand_scale
        self.debug.palm_x, self.debug.palm_y = sx, sy

        if reversed_now:
            self._register_reversal(now)

        raw_confirmed = self._check_confirmed(now)
        # Edge-triggered: only report True the first time this streak
        # satisfies confirmation, not on every subsequent call -- see
        # _already_reported's field comment for why.
        confirmed = raw_confirmed and not self._already_reported
        if raw_confirmed:
            self._already_reported = True

        self.debug.reversal_count = len(self._reversal_times)
        if self._reversal_times:
            self.debug.seconds_since_last_reversal = now - self._reversal_times[-1]
        self.debug.state = (
            "confirmed" if raw_confirmed else
            ("ready_pending_regularity" if self._reversal_times and len(self._reversal_times) >= self._min_reversals else
             ("tracking" if self._reversal_times else "idle"))
        )
        return confirmed

    def _register_reversal(self, now: float) -> None:
        if self._reversal_times and (now - self._reversal_times[-1]) > self._max_gap:
            # Too long since the last reversal for this to be the same
            # continuous gesture -- start counting fresh. (Wall-clock
            # time, not frame count -- see wave_state.py's module
            # docstring for why that distinction matters on variable-FPS
            # hardware.)
            self._reversal_times.clear()
        self._reversal_times.append(now)
        # A genuinely NEW reversal is worth re-evaluating even if a
        # confirmation was already reported for the previous streak
        # state (relevant if a caller adds MORE reversals after a
        # confirmation without calling reset_progress() -- shouldn't
        # normally happen given detector.py's discipline, but keeps
        # this class correct on its own terms regardless).
        self._already_reported = False

    def _regularity_cv(self) -> Optional[float]:
        """
        Coefficient of variation (stddev / mean) of the timing GAPS
        between consecutive reversals in the current streak. A genuine
        rhythmic wave keeps a fairly steady tempo even if not
        metronomic; truly random, uncoordinated hand motion tends to
        have much more erratic timing between direction changes. This
        is a SECONDARY safety net on top of the primary amplitude +
        count + max-gap checks -- specifically targeting "random hand
        movement that happens to zig-zag enough times, at wildly
        inconsistent intervals."

        Needs at least 3 reversals (2 gaps) to be meaningful; returns
        None before that, and the caller treats None as "not enough
        data to judge, don't block on it."
        """
        if len(self._reversal_times) < 3:
            return None
        gaps = [b - a for a, b in zip(self._reversal_times, list(self._reversal_times)[1:])]
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap <= 1e-6:
            return 0.0
        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        return math.sqrt(variance) / mean_gap

    def _check_confirmed(self, now: float) -> bool:
        if len(self._reversal_times) < self._min_reversals:
            self.debug.regularity_cv = None
            self.debug.regularity_ok = True
            return False

        cv = self._regularity_cv()
        self.debug.regularity_cv = cv
        regularity_ok = cv is None or cv <= self._max_regularity_cv
        self.debug.regularity_ok = regularity_ok
        if not regularity_ok:
            return False

        return True

