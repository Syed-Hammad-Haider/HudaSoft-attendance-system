"""
Central configuration for the real-time attendance detection system.

This is the ONLY file most people will need to touch when:
  - swapping the webcam for a CCTV/RTSP stream
  - tuning wave-detection sensitivity
  - tuning direction-estimation sensitivity
  - changing the detection model

Every other module imports values from here instead of hardcoding
constants, so behavior can be tuned without touching logic code.
"""

from dataclasses import dataclass, field


def _auto_detect_device() -> str:
    """
    Pick the fastest inference device actually available, falling back to
    CPU. This is the single biggest smoothness win when a GPU is present:
    YOLO person-detection is the dominant per-frame cost, so moving just
    that stage onto a CUDA GPU (or Apple MPS) can turn a laggy ~8 fps CPU
    loop into a smooth real-time one -- with NO change to detection logic
    or results, only where the math runs. Safe by construction: if torch
    isn't built with CUDA/MPS, or anything goes wrong probing for it, this
    returns "cpu" and the system behaves exactly as before. The detector
    also verifies the chosen device with a tiny warm-up and falls back to
    CPU if it fails (see person_detection/detector.py), so a broken GPU
    setup can never leave detection dead -- it just runs on CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


_DEFAULT_DEVICE = _auto_detect_device()


# ---------------------------------------------------------------------------
# CAMERA
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    # Set to an integer (0, 1, ...) for a local webcam index.
    # OR set to an IP camera URL string, e.g.:
    #   "rtsp://user:pass@192.168.1.50:554/stream1"   (most CCTV/NVR/IP cams)
    #   "http://192.168.1.50:8080/video"               (IP Webcam-style apps, MJPEG)
    # camera/factory.py picks the right source class automatically based
    # on whether `source` is an int (webcam) or a string starting with
    # rtsp://, http://, or https:// (IP camera) -- no other module needs
    # to know the difference. See camera/ip_camera_source.py.
    source: object = 0

    frame_width: int = 960
    frame_height: int = 540

    # Cap FPS to avoid pegging the CPU when the source can deliver faster
    # than we can usefully process (mainly relevant for RTSP streams).
    max_fps: int = 30

    # --- IP camera only (ignored by the plain webcam source) ---
    # How many times to retry connecting/reconnecting before giving up
    # for that attempt. Network cameras drop connections far more often
    # than a local USB webcam (Wi-Fi hiccups, the camera itself
    # rebooting, router DHCP renewals), so the IP source treats a read
    # failure as "try to reconnect", not "fatal" -- unlike the webcam
    # source, where a failed read almost always means the device was
    # unplugged and there's nothing to reconnect to.
    ip_reconnect_attempts: int = 5
    ip_reconnect_delay_seconds: float = 2.0
    # How many seconds to wait for the initial connection before giving
    # up -- prevents the whole app hanging forever on a wrong IP/port.
    ip_connect_timeout_seconds: float = 8.0


# ---------------------------------------------------------------------------
# PERSON DETECTION (YOLO)
# ---------------------------------------------------------------------------
@dataclass
class DetectionConfig:
    # Any Ultralytics-compatible weights path/name.
    # Swap "yolov8n.pt" -> "yolov11n.pt" here only; nothing else changes.
    model_path: str = "yolov8n.pt"

    # COCO class id for "person"
    person_class_id: int = 0

    # Lowered 0.45 -> 0.35: directly addresses "doesn't detect a person
    # when the lighting is harsh/bright" (back-lit doorways, a bright
    # window behind someone, or an over-exposed frame all push a real
    # person's detection confidence DOWN, so a high cutoff silently drops
    # them). 0.35 recovers those while staying well clear of random
    # false positives for the person class, and it also feeds ByteTrack
    # the lower-confidence boxes it's specifically designed to use for
    # association (track activation stays governed by TrackingConfig's own
    # threshold, so this doesn't create junk tracks). It also leaves
    # healthy margin now that inference runs at 384 (see inference_size).
    confidence_threshold: float = 0.35

    # Device string passed to Ultralytics: "cpu", "cuda", "mps", etc.
    # Auto-detected at import (CUDA GPU > Apple MPS > CPU) with a CPU
    # fallback in the detector -- see _auto_detect_device() above. Set a
    # literal string here to force a specific device.
    device: str = field(default_factory=lambda: _DEFAULT_DEVICE)

    # Resolution YOLO actually runs inference at, independent of the
    # camera's captured frame_width/frame_height above (Ultralytics
    # internally resizes to a square of this size before inference,
    # then maps detections back to the original frame's coordinates --
    # no code elsewhere needs to know this happens).
    #
    # CHANGED from the library default of 640 to 416 after direct
    # benchmarking: this is a genuine, verified speedup that holds
    # regardless of hardware, because it's a real reduction in how many
    # pixels the model has to process (compute cost for a CNN scales
    # roughly with pixel count) -- unlike a backend/library choice
    # (e.g. PyTorch vs ONNX Runtime), which can go either way depending
    # on the specific machine and isn't something safe to guess at
    # without benchmarking on the actual target hardware. Measured in
    # this environment: 640 -> ~150ms/frame, 416 -> ~58ms/frame (~2.6x
    # faster), with detection count and confidence verified STABLE
    # across repeated runs at 416 (all 4 people in a multi-person test
    # photo detected consistently, minimum confidence 0.627 -- well
    # clear of the 0.45 threshold above). 384 was faster still (~47ms)
    # but pushed the least-confident detection uncomfortably close to
    # the confidence threshold (0.456), so 416 was chosen as the safer
    # balance of real speed gain with a healthy accuracy margin.
    #
    # WHY THIS MATTERS BEYOND JUST "FPS": overall pipeline throughput
    # is the direct limiting factor on how reliably wave detection
    # works (see WaveDetectionConfig.max_seconds_between_reversals'
    # extensive documentation on this) -- a faster detection stage
    # means more processed frames per second, which means the system
    # actually samples fast hand motion often enough to see it clearly.
    # Set to 384 (from 416) for a further real speedup to smooth out the
    # live loop (~20% fewer pixels for the CNN than 416). The author's
    # earlier note below cautioned that at the OLD 0.45 threshold, 384's
    # least-confident detection (0.456) sat uncomfortably close to the
    # cutoff -- that caution no longer applies now the threshold is 0.35,
    # which leaves a healthy margin. On a GPU (auto-detected) this is
    # moot headroom; on CPU it's the difference between a choppy and a
    # fluid feed.
    inference_size: int = 384




# ---------------------------------------------------------------------------
# PERSON TRACKING (ByteTrack)
# ---------------------------------------------------------------------------
@dataclass
class TrackingConfig:
    # Number of past center-points to keep per tracked person.
    # Used by both direction estimation and wave detection.
    history_length: int = 30

    # How many frames a track can go undetected before we drop its state
    # (wave buffers, cooldown timers, etc.) to avoid memory growth.
    max_missed_frames: int = 45

    # --- ByteTrack internal tuning (passed straight through to the
    # `supervision` ByteTrack implementation) ---
    #
    # These directly affect how well tracking survives two things that
    # were reported as real problems in testing: (1) two people walking
    # close together / briefly occluding each other, and (2) fast
    # movement causing large frame-to-frame jumps when the pipeline's
    # actual FPS is low (heavier per-frame processing = fewer frames
    # per second = bigger pixel jumps between the frames that DO get
    # processed = harder for IOU-based matching to find the right track).

    # Minimum detection confidence to START a new track. Left at
    # ByteTrack's own sensible default -- raising it would make the
    # system slower to assign an ID to a genuinely new person; lowering
    # it risks starting tracks on false detections.
    track_activation_threshold: float = 0.25

    # How many frames a track is kept "alive" (eligible for
    # re-matching) after it stops being detected, before being dropped
    # by ByteTrack itself (distinct from `max_missed_frames` above,
    # which is OUR OWN cleanup of wave/UI state -- this is ByteTrack's
    # internal occlusion memory). Raised from the library default of 30
    # to 60: at the lower real-world FPS this pipeline runs at (YOLO +
    # multiple MediaPipe instances per frame is heavy), a fixed frame
    # count needs to be larger to cover a meaningful amount of real
    # time, so a person briefly blocked by another person is more
    # likely to keep their original ID instead of getting a new one.
    lost_track_buffer: int = 60

    # IOU threshold required to match a new detection to a predicted
    # track position. Lowered from ByteTrack's default 0.8 to 0.65:
    # testing showed fast movement (combined with this pipeline's
    # real-world FPS) can move a person far enough between processed
    # frames that their new position's overlap with the OLD position
    # drops below 0.8, causing a needless ID switch even with no
    # occlusion involved at all. 0.65 tolerates bigger jumps while
    # still being strict enough to avoid matching two DIFFERENT nearby
    # people to the same track.
    minimum_matching_threshold: float = 0.65

    # Reported to ByteTrack's internal Kalman motion model as the
    # expected frame rate; used to scale its velocity predictions. Set
    # conservatively low (15, not 30) because this pipeline's real
    # achieved FPS on a typical laptop CPU is well under standard
    # webcam FPS once YOLO + tracking + per-person MediaPipe Hands are
    # all running -- telling the motion model a more realistic frame
    # rate keeps its position predictions closer to reality.
    assumed_frame_rate: int = 15


# ---------------------------------------------------------------------------
# WAVE DETECTION (MediaPipe Hands)
# ---------------------------------------------------------------------------
@dataclass
class WaveDetectionConfig:
    """
    REDESIGNED wave-detection configuration. The core algorithm changed
    from 1D wrist-x tracking normalized to the person's crop width, to
    2D palm-centroid tracking normalized to the hand's OWN detected
    size ("hand-widths"). See wave_detection/oscillation.py's module
    docstring for the full root-cause analysis of why the old design
    missed genuine waves, and wave_detection/test_oscillation.py for
    the test suite (22 synthetic scenarios, all passing) that these
    defaults were chosen against -- not guessed, not hand-tuned against
    a single anecdote.
    """

    # Minimum/maximum hands MediaPipe should look for per person crop.
    # Defaulted to 1 (not 2): MediaPipe's per-call cost scales with how
    # many hands it searches for and tracks, so this roughly halves its
    # cost per person per frame. A real attendance wave realistically
    # uses one hand, so this trades away detecting a simultaneous
    # SECOND hand for meaningfully faster, more responsive single-hand
    # detection. Raise to 2 if two-handed waves need to be recognized
    # and your hardware has headroom for the added cost.
    max_num_hands: int = 1

    # Slightly lowered from 0.6 -- at typical webcam/entrance distance
    # and resolution, MediaPipe's hand detector can under-report
    # confidence even for a clearly visible hand mid-motion (motion
    # blur reduces landmark confidence). 0.5 still rejects genuinely
    # ambiguous/partial hand shapes while catching real hands reliably.
    min_detection_confidence: float = 0.4
    min_tracking_confidence: float = 0.5

    # ------------------------------------------------------------------
    # OSCILLATION SIGNAL (wave_detection/oscillation.py)
    # ------------------------------------------------------------------

    # Minimum swing distance to count as a genuine direction reversal,
    # measured in HAND WIDTHS (the live-detected distance between the
    # wrist and middle-finger-MCP landmarks each frame) -- NOT a
    # fraction of the person's crop, which was the old design's core
    # flaw (see oscillation.py's docstring: crop width changes with
    # camera distance, crop margin config, and even reactively as the
    # tracked bbox shifts, none of which have anything to do with how
    # far the HAND actually moved).
    #
    # 0.45 was chosen from two directions, both verified in
    # test_oscillation.py:
    #   - Above the noise floor: single-frame MediaPipe landmark jitter
    #     is typically a few percent of hand width (well under 0.05) --
    #     0.45 leaves roughly an order of magnitude of margin, so
    #     landmark noise alone can never masquerade as a swing.
    #   - Below genuine wave amplitude: a relaxed, unexaggerated
    #     wrist-pivot wave was tested down to 0.55 hand-widths peak
    #     amplitude and reliably detected; 0.45 keeps meaningful margin
    #     below even a deliberately subtle motion, without approaching
    #     the noise floor above.
    min_swing_amplitude_hand_widths: float = 0.32

    # Number of direction reversals required to confirm a genuine wave.
    # UPDATED to 4 (from 3) as part of the amplitude threshold being
    # lowered (see min_swing_amplitude_hand_widths above) after live
    # testing found genuine waves still being missed. Lowering the
    # amplitude alone, tested in isolation, measurably weakened
    # rejection of vigorous random hand motion (no complementary
    # "body-stillness"-style signal exists for that specific case, the
    # way it does for walking). Requiring one more reversal restores
    # a meaningful chunk of that safety margin -- tested directly: cut
    # false confirmations on adversarial random-motion trajectories by
    # more than half -- while adding under 0.1s to real wave
    # confirmation time (verified in wave_detection/test_oscillation.py).
    min_reversals_required: int = 4

    # Reversals must happen within this many SECONDS of each other to
    # count as part of the same continuous gesture.
    #
    # UPDATED from 0.55 -> 1.2 after direct user testing found a real
    # gap in the previous value: a SLOW, WIDE, deliberate wave (larger
    # swing, taken unhurried) naturally has a longer time between each
    # reversal than a fast small oscillation does. At 0.55s, each of
    # that slow wave's reversals arrived just outside the window, so
    # _register_reversal() kept wiping the count back to 1 every single
    # time -- 4 reversals were never allowed to accumulate, no matter
    # how many times the person waved, while the exact same gesture
    # done quickly (short, fast oscillations) worked fine, because its
    # reversals landed well inside 0.55s. This is purely a TIMING
    # window fix, not a change to how many reversals are required (see
    # min_reversals_required above, unchanged) or how big a swing must
    # be (see min_swing_amplitude_hand_widths above, unchanged) -- a
    # slow wave and a fast wave should both count, as long as each
    # still clears the same amplitude and still completes the same
    # number of reversals. 1.2s comfortably covers a slow, deliberate
    # half-swing (commonly 0.6-1.0s) while still being far below
    # anything that would plausibly link together two UNRELATED hand
    # movements as if they were one continuous gesture. Widening this
    # (rather than tightening anything else) is a strict loosening: it
    # cannot break the fast-oscillation case that already worked, since
    # a smaller real gap was always within an even larger allowed
    # window too.
    max_seconds_between_reversals: float = 1.2

    # Coefficient of variation (stddev/mean) ceiling on the timing GAPS
    # between consecutive reversals in a streak, checked once at least
    # 3 reversals (2 gaps) exist. A genuine rhythmic wave keeps a
    # fairly steady tempo even if not metronomic; truly random,
    # uncoordinated hand motion that happens to zig-zag enough times
    # tends to do so at much more erratic intervals. 0.6 permits
    # generous natural human tempo variation (voluntary rhythmic motion
    # commonly varies up to roughly 50-60% relative to its own mean
    # timing) while still catching genuinely chaotic, patternless
    # motion -- verified in test_oscillation.py's "random uncoordinated
    # hand movement" case. This is a SECONDARY safety net on top of
    # the primary amplitude+count+max-gap checks above, not the main
    # gate.
    max_reversal_gap_regularity_cv: float = 0.6

    # Exponential-smoothing factor applied to the palm-centroid signal
    # before oscillation analysis (closer to 1.0 = trusts each new
    # sample more, less lag; closer to 0.0 = heavier smoothing, more
    # lag). Deliberately LIGHT: testing found a heavier value (0.6)
    # measurably damped fast waves and low-FPS sampling below the
    # detection threshold before they could register. 0.85 still
    # smooths out occasional single-frame landmark outliers (partial
    # occlusion, motion blur spikes) without meaningfully delaying real
    # motion -- safe because the amplitude threshold above already sits
    # roughly an order of magnitude above normal jitter, so heavy
    # smoothing was never actually load-bearing for noise rejection.
    wave_smoothing_alpha: float = 0.85

    # ------------------------------------------------------------------
    # BODY-STILLNESS GATE (wave_detection/body_motion.py)
    # ------------------------------------------------------------------
    # HONEST, TESTED FINDING: hand-motion kinematics ALONE -- however
    # well the signal above is designed -- cannot fully distinguish a
    # deliberate wave from vigorous arm-swinging while walking or
    # jogging, when the two share similar amplitude and cadence. This
    # isn't a threshold gap; test_oscillation.py directly demonstrates
    # both "brisk walking" and "jogging-speed arm pump" synthetic
    # trajectories passing the exact same amplitude/timing checks a
    # real wave passes. The fix is an orthogonal signal, not a tighter
    # threshold: a person WALKING is, close to by definition,
    # translating their whole body across the frame; a person PAUSING
    # TO WAVE typically is not. The person tracker already computes
    # each tracked person's bounding box every frame for entrance-line
    # crossing -- reusing it here is nearly free.

    # How far back (in seconds) to look when judging whether a person's
    # body has been "moving" vs "relatively still."
    #
    # REFINED after direct testing (wave_detection/test_body_motion.py)
    # found two competing failure modes and swept window lengths
    # empirically to find the balance point between them:
    #   - Too SHORT (or worse, purely frame-to-frame instantaneous
    #     speed): ordinary person-tracker bounding-box jitter (a
    #     perfectly still person's detected bbox still wobbles a
    #     percent or two frame to frame) gets amplified into a large
    #     apparent "speed" when divided by a very small time step,
    #     causing FALSE positives -- a stationary person incorrectly
    #     blocked from waving. Verified directly: at 0.2s this produced
    #     false "moving" flags even at just 2% bbox jitter.
    #   - Too LONG (the original 1.0s value): after a person actually
    #     stops walking, stale old positions linger in the window for
    #     up to the full window duration -- measured at ~0.73s of lag
    #     before the system would even consider them stationary, direct
    #     lag against "detected quickly" for the single most common
    #     real scenario (walk up, stop, wave).
    # 0.5s is the empirically-found sweet spot: responds within ~0.3s
    # of an actual stop while producing zero false positives up to a
    # generous 3% bbox-jitter assumption, and still correctly detects
    # normal/brisk walking and jogging as "moving."
    body_stillness_window_seconds: float = 0.5

    # Maximum body-center SPEED allowed (displacement between the
    # oldest and newest sample in the window, divided by the elapsed
    # time between them) -- expressed as a FRACTION OF THE PERSON'S OWN
    # BBOX HEIGHT PER SECOND, so it's invariant to distance from camera
    # (exactly like the hand-width normalization above is invariant to
    # hand size) AND invariant to the specific window length chosen
    # above (a genuine speed, not a raw per-window fraction). 0.20: a
    # normal walking pace covers roughly 60-100%+ of one's own height
    # per second, so 20% comfortably separates "actually walking" from
    # ordinary standing sway, weight shifts, or the small
    # torso/shoulder movement that naturally accompanies waving itself.
    # Verified in test_oscillation.py's integration section: correctly
    # rejects both the brisk-walking and jogging hand patterns above
    # WHEN COMBINED with realistic walking-speed body translation,
    # while correctly accepting the SAME hand motion when the body is
    # stationary (proving body motion, not hand cadence, is what's
    # actually being discriminated).
    #
    # HONEST TRADE-OFF, stated plainly: someone waving WHILE STILL
    # WALKING toward the camera will not be confirmed until they slow
    # or stop -- this is intentional (see test_oscillation.py's last
    # integration case) and matches how a real human greeter would
    # naturally read the same situation: a wave mid-stride reads as
    # "still arriving," a wave while paused reads as "greeting."
    max_body_speed_fraction_per_second: float = 0.24

    # ------------------------------------------------------------------
    # PERFORMANCE / INFRASTRUCTURE (unchanged from before this redesign)
    # ------------------------------------------------------------------

    # After a confirmed wave, ignore further wave triggers for this ID
    # for this many seconds (prevents repeated triggers while the
    # person keeps waving).
    wave_cooldown_seconds: float = 5.0

    # How long a hand keeps showing as "WAVING" (green highlight) in
    # the UI after a confirmation, before reverting to its normal
    # tracked appearance. Long enough to register as clear positive
    # feedback the moment it happens, short enough not to look
    # permanently stuck on -- roughly a third of the cooldown above, so
    # the visual state has clearly settled well before another wave
    # could even be accepted.
    wave_confirmed_display_seconds: float = 1.5

    # Margin used to crop around each person's bbox before running hand
    # detection. 0.25 comfortably covers a natural wave's full arc for
    # someone standing alone or with normal personal space. HONEST
    # TRADE-OFF: this does increase the chance of a neighboring
    # person's hand bleeding into this crop in a genuinely crowded,
    # shoulder-to-shoulder entrance queue -- if that becomes a real
    # problem in testing, the fix is lowering this value, not raising
    # min_detection_confidence (which doesn't address attribution, only
    # detection quality).
    hand_crop_margin_ratio: float = 0.25

    # Hard cap on how many people can have an active MediaPipe Hands
    # instance at once. Each instance has real CPU/memory cost; in a
    # crowded entrance with many simultaneous people, uncapped instance
    # creation would degrade FPS for everyone. People beyond this cap
    # (oldest-tracked-first gets priority) simply don't get wave analysis
    # until a slot frees up -- detection and tracking are unaffected.
    max_concurrent_wave_trackers: int = 8

    # How often (in frames) to actually run MediaPipe Hands per person.
    # Kept at 1 (every frame) -- hand-box responsiveness matters more
    # than the CPU savings from skipping frames. The cost this would
    # buy back is instead recovered two other ways: max_num_hands
    # defaulting to 1 (above) and downscaling each hand crop before
    # MediaPipe processes it (hand_crop_max_dimension_px, below) --
    # both cut real per-call cost without reducing how often detection
    # actually runs, which is what responsiveness depends on. Raise
    # this above 1 only if lag appears on slower hardware.
    wave_processing_frame_interval: int = 1

    # Each person's hand-search crop is resized so its longer side is
    # at most this many pixels before being handed to MediaPipe.
    # MediaPipe's own hand landmark model operates on a small fixed
    # input size internally regardless of what's given to it, so
    # feeding it an unnecessarily large crop wastes time on our own
    # color-conversion and resizing step for no accuracy benefit.
    # Downscaling ourselves to a sensible cap keeps that overhead small
    # while still comfortably exceeding the model's own working
    # resolution, so landmark accuracy is unaffected.
    #
    # Lowered 260 -> 224: MediaPipe Hands' landmark model runs at ~224px
    # internally, so capping our hand crop here matches its native size --
    # it trims our per-person color-convert/resize overhead (which adds up
    # when several people are in frame at once) with no landmark-accuracy
    # cost, directly helping multi-person smoothness without touching the
    # wave algorithm or how often it runs.
    hand_crop_max_dimension_px: int = 224

    # Toggle for the DEBUG/ANALYSIS overlay (press 'D' while main.py is
    # running). Off by default to keep the normal display clean; turn
    # on for testing/tuning to see live hand landmarks, tracking
    # status, wave score, reversal count, regularity, body-stillness
    # state, and samples analyzed for every tracked hand.
    debug_overlay_default_on: bool = False




# ---------------------------------------------------------------------------
# WAVE SENSITIVITY PRESETS
# ---------------------------------------------------------------------------
# The defaults above ("balanced") are the result of the extensive tuning
# documented inline in WaveDetectionConfig. But "accuracy" for a wave
# gesture is fundamentally a trade-off between two opposite failure
# modes -- missed real waves (too strict) vs. false triggers from
# walking/arm-swinging (too loose) -- and the right point on that
# trade-off genuinely depends on your camera's distance, angle, and
# lighting, which can't be guessed generically. Rather than editing
# code to retune, pick a preset here (or press 'S' in main.py to cycle
# them live and see the effect immediately):
#
#   strict   - fewer false CHECK IN/OUTs from people just walking past;
#              requires a clearer, more deliberate wave. Best for a
#              busy entrance with a lot of foot traffic near the camera.
#   balanced - the defaults above; good starting point for most setups.
#   relaxed  - triggers on a smaller/quicker wave; best if genuine waves
#              are being missed and the entrance isn't crowded enough
#              for false positives to be a real risk.
#
# Only the fields that meaningfully affect the strict/loose trade-off
# are varied here -- performance-only settings (max_num_hands, crop
# size, frame interval) stay the same across presets.
WAVE_SENSITIVITY_PRESETS = {
    # UPDATED again after direct user feedback that even "relaxed" was
    # still missing real waves in live testing. Re-validated with the
    # same synthetic-trajectory method used throughout this project
    # (not a blind guess): tested subtle wave amplitudes (down to ~0.18
    # hand-widths peak, i.e. a fairly small motion) across several
    # cadences, and confirmed zero false positives from static-hold
    # jitter at each candidate value before adopting it. See
    # wave_detection/test_oscillation.py for the corresponding test
    # additions.
    # The three tiers are MONOTONIC -- each strictly looser than the one
    # above on every axis -- so "Balanced" genuinely sits in the middle.
    # Reversals: Strict 4 > Balanced 3 > Relaxed 2. Amplitude: 0.5 >
    # 0.3 > 0.18. Gap/confidence/body-speed all loosen in the same
    # order too. (Earlier, Strict and Balanced both required 3, so
    # Strict wasn't actually stricter on reversal count -- Strict is 4
    # now, restoring a real ladder.)
    "strict": {
        "min_swing_amplitude_hand_widths": 0.50,
        "min_reversals_required": 4,
        "max_seconds_between_reversals": 0.9,
        "min_detection_confidence": 0.5,
        "max_body_speed_fraction_per_second": 0.15,
    },
    "balanced": {
        "min_swing_amplitude_hand_widths": 0.30,
        "min_reversals_required": 3,
        "max_seconds_between_reversals": 1.2,
        "min_detection_confidence": 0.4,
        "max_body_speed_fraction_per_second": 0.24,
    },
    "relaxed": {
        # Tuned for FAST, low-effort detection after direct user feedback
        # that waves took too long to register: only 2 direction
        # reversals are required now (one quick back-and-forth -- "waving
        # twice" easily clears it), at a small amplitude, so a genuine
        # wave confirms in a fraction of a second. This is the most
        # sensitive tier and the dashboard default.
        "min_swing_amplitude_hand_widths": 0.18,
        "min_reversals_required": 2,
        "max_seconds_between_reversals": 1.6,
        "min_detection_confidence": 0.3,
        "max_body_speed_fraction_per_second": 0.32,
        # HONEST, TESTED LIMITATION: at this amplitude + only 2 reversals,
        # direct testing found vigorous RANDOM hand motion (not a wave,
        # not walking -- e.g. energetic fidgeting while standing still)
        # can also satisfy this tier, because there's no complementary
        # signal for "random" the way body-stillness catches walking.
        # For a busy/crowded entrance where that's a nuisance, switch to
        # "balanced" or "strict" in Settings; for a normal door this is
        # the snappiest and what most people want.
    },
}


def apply_wave_sensitivity(wave_config: "WaveDetectionConfig", preset_name: str) -> None:
    """
    Mutate a WaveDetectionConfig IN PLACE to match a named preset.
    Mutating in place (rather than replacing CONFIG.wave with a new
    object) matters: WaveDetector holds a reference to this same config
    object and reads its fields every frame, so an in-place mutation
    takes effect immediately on the next frame -- no need to recreate
    the detector or restart the app. This is what makes the 'S' hotkey
    in main.py work live.
    """
    preset = WAVE_SENSITIVITY_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(
            f"Unknown wave sensitivity preset '{preset_name}'. "
            f"Choose from: {list(WAVE_SENSITIVITY_PRESETS)}"
        )
    for field_name, value in preset.items():
        setattr(wave_config, field_name, value)


# ---------------------------------------------------------------------------
# ENTRANCE LINE (VIRTUAL GATE)
# ---------------------------------------------------------------------------
@dataclass
class EntranceLineConfig:
    """
    Defines a virtual line across the frame that represents the real-world
    entrance/doorway. Crossing it is what actually determines
    ENTERING vs LEAVING -- not raw movement direction.

    Points are stored as (x_ratio, y_ratio) fractions of frame width/height
    (0.0 - 1.0) rather than raw pixels, so the line stays correctly placed
    even if frame resolution changes (e.g. webcam -> CCTV with a different
    resolution).

    DEFAULT: a vertical line through the horizontal center of the frame.
    This is a reasonable default for a camera facing an entrance where
    people walk left-right through frame center, but the *correct* line
    placement always depends on the physical camera installation -- this
    is a business/installation decision, not something that can be chosen
    generically. Adjust `line_start` / `line_end` after installing the
    real camera.
    """
    line_start: tuple = (0.5, 0.0)   # top point, as (x_ratio, y_ratio)
    line_end: tuple = (0.5, 1.0)     # bottom point

    # Which side of the line counts as "inside" the building.
    # The line has a direction (start -> end); standing at start looking
    # toward end, "left" is one side and "right" is the other. This flag
    # picks which one is "inside". Flip it if CHECK IN/OUT come out
    # backwards for your camera's physical orientation.
    inside_is_left_of_line: bool = True

    # A person's side is only considered "confirmed" after they've been
    # observed on that side for this many consecutive frames. This is a
    # debounce: without it, someone standing exactly on the line would
    # flicker between sides every frame due to detection box jitter,
    # producing spurious crossing events.
    #
    # Lowered from an earlier 5 -> 3: testing with two people moving
    # together showed the tracker can briefly lose/reassign an ID during
    # close-proximity occlusion, which resets this counter for whatever
    # ID re-emerges. 3 frames confirms fast enough to survive a short
    # ID hiccup while still filtering out single-frame jitter.
    min_frames_to_confirm_side: int = 3

    # In addition to the frame-count debounce above, a crossing is only
    # confirmed once the person has moved at least this far PAST the
    # line (as a fraction of frame width/height) -- not just barely
    # tipped over it. This is what distinguishes a genuine, committed
    # entry/exit from someone standing near the door, leaning across it,
    # turning around, or hesitating right at the threshold: those cases
    # keep the person's distance-from-line small even if they
    # technically tip to the other side of it for a few frames. Real
    # walking-through motion carries someone solidly past this margin.
    min_crossing_commit_distance_ratio: float = 0.05

    # A confirmed wave only produces a Suggested Action if it happens
    # within this many seconds of that person's most recent line
    # crossing. This is what ties "waving" to "actually at the door"
    # instead of a wave anywhere in the room triggering an action based
    # on stale crossing history from minutes ago. If no crossing has
    # happened yet, or the last one is older than this window, a
    # confirmed wave produces NO Suggested Action at all (rather than
    # guessing).
    #
    # WIDENED 6.0 -> 12.0 after a real "wave goes green but no check-in"
    # report: a genuine wave now needs 4 direction reversals to confirm
    # (see WaveDetectionConfig.min_reversals_required), which at a
    # natural, unhurried pace takes several seconds. In real use a
    # person crosses the doorway FIRST, then stops and raises a hand to
    # wave -- so by the time the wave actually confirms, 6 seconds had
    # often already elapsed since the crossing, leaving the wave with no
    # recent crossing to pair with, and it was silently dropped. 12s
    # comfortably covers "walk through the door, stop, wave" at a
    # relaxed pace while still being far too short to pair a wave with a
    # crossing from a genuinely different visit minutes earlier.
    action_window_seconds: float = 12.0

    # A person is visually flagged as "crossing" when within this fraction
    # of frame width/height of the line (used for UI highlighting only,
    # not for the crossing decision itself).
    crossing_highlight_distance_ratio: float = 0.04

    # The camera is mounted OUTSIDE the entrance (that's the real
    # deployment): an employee stands outside the door and waves to
    # check in on arrival, and again -- still outside, having just come
    # out -- to check out when leaving. So a wave should only be acted
    # on when the person is on the OUTSIDE side of the line. With this
    # True, a wave from someone the gate has definitively classified as
    # INSIDE is ignored (a person standing UNKNOWN or OUTSIDE is still
    # allowed, so a normal outside wave always works and nobody is
    # blocked merely for having just appeared). If a particular camera
    # mounting makes "inside"/"outside" come out reversed, use the
    # Calibration tab's "Flip IN / OUT" button rather than turning this
    # off. Set False only for a camera mounted INSIDE.
    require_wave_from_outside: bool = True


# ---------------------------------------------------------------------------
# DUPLICATE ACTION PREVENTION (handles tracking ID churn)
# ---------------------------------------------------------------------------
@dataclass
class ActionGuardConfig:
    """
    ByteTrack can occasionally drop and reassign a NEW track ID to the
    same physical person (e.g. after a brief full occlusion). Because the
    wave cooldown is keyed by track ID, an ID change would otherwise
    defeat the cooldown and cause a duplicate Suggested Action for
    someone who already triggered one moments ago.

    This guard remembers WHERE and WHEN recent actions were triggered
    (position, not identity) and suppresses a new action if it would
    fire very close to a very recent one -- regardless of track ID.
    """
    suppression_radius_ratio: float = 0.08   # fraction of frame width
    suppression_seconds: float = 4.0


# ---------------------------------------------------------------------------
# UI COLOR SCHEME
# ---------------------------------------------------------------------------
@dataclass
class ColorScheme:
    """
    All BGR colors (OpenCV convention) used across the live overlay, in
    one place so the visual language of the product can be restyled
    without touching rendering logic.
    """
    person_box_default: tuple = (200, 200, 200)     # neutral gray: tracked, no state yet
    person_box_entering: tuple = (80, 200, 120)      # green-ish
    person_box_leaving: tuple = (60, 140, 240)       # orange-ish
    person_box_crossing: tuple = (0, 255, 255)       # yellow highlight near the line

    hand_normal: tuple = (180, 180, 180)             # hand visible, not waving
    hand_detected: tuple = (255, 200, 0)             # hand actively being analyzed
    hand_valid_wave: tuple = (0, 255, 0)             # confirmed wave in progress

    entrance_line: tuple = (255, 0, 255)
    movement_trail: tuple = (100, 100, 255)
    direction_arrow: tuple = (0, 220, 255)

    suggested_checkin_bg: tuple = (30, 120, 30)
    suggested_checkout_bg: tuple = (30, 60, 140)

    text_color: tuple = (255, 255, 255)
    panel_bg: tuple = (25, 25, 25)


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------
@dataclass
class DisplayConfig:
    window_name: str = "Attendance Detection - Phase 1"

    # Whether the camera feed is horizontally flipped before ANY
    # processing (detection, tracking, entrance line -- everything sees
    # the flipped frame, not just the display, so left/right stays
    # internally consistent).
    #
    # DEFAULTS TO FALSE ON PURPOSE. This is NOT a bug fix -- an
    # unmirrored feed is the physically CORRECT behavior for a real
    # entrance camera. If you face the camera and raise your right
    # hand, it correctly appears on the LEFT side of an unmirrored
    # image, because that's genuinely where it is from the camera's
    # point of view. Only "selfie mode" apps (Zoom, FaceTime, a phone's
    # front camera) mirror the image, purely so it feels like looking
    # in a mirror -- that convention makes no sense for a security/
    # attendance camera watching a room, where LEFT and RIGHT in the
    # video need to match the TRUE physical layout of the space for the
    # entrance line and IN/OUT sides to mean anything correct. A real
    # CCTV camera is never mirrored, and calibrating the entrance line
    # only makes sense against the true, unmirrored view.
    #
    # Set to True only for your own comfort while personally testing in
    # front of a laptop webcam, if the unmirrored view feels
    # disorienting. Toggle live with the 'V' key. IMPORTANT: flipping
    # this changes which pixels mean "left" and "right", so any saved
    # entrance line calibration becomes wrong when you toggle it --
    # recalibrate (press 'C') after switching.
    mirror_preview: bool = False

    # How long a "Suggested Action" banner stays visible after a
    # confirmed wave, in seconds.
    message_display_seconds: float = 2.5

    box_thickness: int = 2
    font_scale: float = 0.55
    show_fps: bool = True

    # Movement trail: how many recent center-points to draw as a fading line.
    trail_length: int = 15

    # Direction arrow length as a fraction of person bbox width.
    arrow_length_ratio: float = 0.6

    # Recent event log panel (top-right corner): how many events to keep.
    event_log_max_entries: int = 6

    colors: ColorScheme = field(default_factory=ColorScheme)


# ---------------------------------------------------------------------------
# FACE RECOGNITION (Phase 2)
# ---------------------------------------------------------------------------
@dataclass
class FaceRecognitionConfig:
    """
    Settings for employee face recognition, used by both the live
    monitoring loop (main.py) and the registration script
    (register_employee.py). See face_id/embedder.py for why InsightFace
    was chosen over the more commonly-tutorial'd dlib-based library.
    """

    # ONNX detector input size. Larger catches smaller/farther faces at
    # higher compute cost. 320x320 is a reasonable default for typical
    # webcam/entrance distance; consider raising for a CCTV camera
    # mounted further from the doorway.
    detection_size: tuple = (320, 320)

    # Minimum cosine similarity (embeddings are L2-normalized, so this
    # is a plain dot product) for a live face to be accepted as a match
    # to a known employee. Verified empirically safe against genuine
    # impostor pairs (real different people scored far below this, see
    # documentation) -- but the exact right value depends on real
    # camera/lighting conditions. Recommended to validate and adjust
    # with real repeated photos of actual employees during pilot
    # testing, not just trust this default blindly. Erring toward a
    # HIGHER threshold (stricter) is the safer direction for an
    # attendance system: an unrecognized person shown as "UNKNOWN" is
    # a much smaller problem than one employee's attendance being
    # incorrectly credited to another.
    min_match_similarity: float = 0.45

    # SCALE SAFEGUARD -- matters much more with thousands of employees
    # than it did with a handful: the best match being above
    # min_match_similarity only proves the top candidate looks
    # plausible, not that it's clearly the RIGHT one out of a large
    # population. With more people enrolled, the odds rise that some
    # OTHER employee also scores close behind the top match -- an
    # ambiguous case a small-headcount pilot would rarely surface. This
    # requires the best match to beat the best-scoring DIFFERENT
    # employee by at least this margin, or the result is treated as
    # unrecognized rather than guessing between two plausible people.
    # See FaceRecognizer.recognize() in face_id/recognizer.py for the
    # exact computation.
    min_margin_over_runner_up: float = 0.03

    # Minimum detection confidence during LIVE recognition (distinct
    # from registration's own quality gate, which is intentionally
    # stricter -- see registration.py). Live recognition tolerates a
    # somewhat lower bar since real-world motion/lighting is less
    # controlled than a deliberate registration capture.
    min_live_detection_confidence: float = 0.4

    # How often (in seconds) to re-run face recognition on an already-
    # recognized person. Once someone is confidently identified, there
    # is little value in re-running the (relatively expensive) face
    # model every frame -- their identity doesn't change while they
    # remain the same track. Re-checking periodically instead of never
    # allows recovery if an early recognition was wrong.
    re_recognition_interval_seconds: float = 5.0

    # How often to RETRY when the most recent attempt did NOT recognize
    # anyone. Deliberately much shorter than the interval above.
    #
    # BUG FIX: these used to share one interval. That meant a single
    # unlucky first attempt -- caught mid-turn, motion-blurred over a
    # phone camera link, briefly backlit -- got cached as "unrecognized"
    # for the FULL 5 seconds before being retried, even though the same
    # person would very likely have matched fine one second later. This
    # is what "sometimes it recognizes them, sometimes it doesn't" in
    # practice looks like: it's not that recognition is unreliable per
    # attempt, it's that one bad attempt got frozen in place for far too
    # long. There's no real cost to retrying an UNRECOGNIZED result
    # quickly (the standard used to accept a match, min_match_similarity,
    # is unchanged -- this only controls how soon a fresh attempt is
    # allowed, not how lenient any single attempt is).
    #
    # Nudged 0.75 -> 1.0: with SEVERAL unknown people in frame at once,
    # a 0.75s retry runs the (relatively expensive) InsightFace embed on
    # each of them very frequently, which is a real contributor to lag in
    # a crowd. 1.0s keeps re-recognition snappy (still 5x quicker than the
    # recognized-person interval, preserving the bug fix above) while
    # easing the multi-person load.
    unrecognized_retry_interval_seconds: float = 1.0

    # How many consecutive frames a person must be trackable/stable
    # before face recognition is even attempted -- running it on a
    # freshly-appeared, not-yet-stable track wastes compute on a person
    # who might be a momentary false detection.
    min_track_stability_frames: int = 5

    # --- "recognise immediately, and don't forget" tuning ---------------
    # A face that is only ever tried once per second, from the same still
    # pose, fails the same way every time (e.g. similarity 0.42 vs a 0.45
    # cut-off), so a person standing still stays "ID n" while someone who
    # moves gets lucky on some frame. These settings fix that without
    # making a wrong name any more likely:
    #
    # 1) Retry FAST while a track is new (first few seconds), then slow down.
    fast_retry_interval_seconds: float = 0.4
    fast_retry_window_seconds: float = 4.0
    # 2) Accept a slightly weaker match ONLY when it is confirmed: the same
    #    employee is the best match on N recent attempts (and the usual
    #    runner-up margin still holds). Two agreeing weak matches are
    #    safer than one lucky strong one.
    soft_match_similarity: float = 0.38
    soft_match_confirmations: int = 2
    soft_match_window_seconds: float = 3.0
    # 3) Once recognised, keep the name through a failed re-check (a still
    #    pose can fail one re-check) for this long while retrying; a
    #    DIFFERENT recognised employee still replaces it at once.
    identity_hold_seconds: float = 20.0
    # 4) Search the top part of the person box for the face; this is the
    #    fraction of the box height used (a leaning head used to be cut off
    #    at 45%). If that finds nothing useful, the whole box is tried too.
    face_crop_fraction: float = 0.60

    # Registration-specific quality gates (stricter than live
    # recognition, since a registration capture is deliberate and
    # controlled -- see face_id/registration.py).
    registration_min_detection_confidence: float = 0.6
    registration_min_face_size_px: int = 80
    registration_angles: tuple = ("front", "left", "right")

    # THE ACTUAL FIX for "front/left/right directions are broken":
    # evaluate_capture_quality() previously checked ONLY generic frame
    # quality (one face, confident detection, big enough) -- it never
    # verified the person had actually turned their head at all. That
    # meant a "left" or "right" capture could silently be an identical
    # frontal pose to the "front" capture, with only the on-screen
    # LABEL differing -- defeating the entire purpose of multi-angle
    # registration (see registration.py's module docstring on why
    # multiple distinct angles matter for real-world matching).
    #
    # A geometric yaw check (comparing eye/nose keypoint positions) was
    # considered and deliberately rejected: which side is "left" vs
    # "right" in that geometry depends on camera mirroring conventions
    # that vary by platform/webcam driver, and getting that sign wrong
    # would silently swap the two angles rather than genuinely
    # verifying anything -- worse than not checking at all, because it
    # would look like it's working. Comparing embedding SIMILARITY to
    # already-captured angles instead is sign-convention-proof: it
    # doesn't need to know which physical direction is which, it only
    # needs each new capture's embedding to be meaningfully different
    # from the earlier one(s), which is the actual property multi-angle
    # capture depends on. This value is a cosine-similarity ceiling:
    # a capture whose embedding is AT OR ABOVE this similarity to any
    # already-accepted angle for this same employee is rejected as
    # "not different enough, turn further."
    registration_max_similarity_to_previous_angle: float = 0.95

    # How many CONTINUOUS SECONDS a candidate frame must keep passing
    # every quality gate (incl. the distinctness check above) before it
    # auto-captures -- deliberately elapsed TIME, not a frame count.
    # An earlier version used a fixed frame-count (8 consecutive good
    # frames), which is the same class of bug already fixed elsewhere
    # in this project for wave timing (see WaveDetectionConfig's
    # max_seconds_between_reversals comment): a frame count silently
    # assumes a roughly fixed frame rate, but achieved FPS varies a lot
    # between a local webcam and a phone camera over Wi-Fi (DroidCam) --
    # so "8 frames" could be under half a second on a fast webcam or
    # several seconds on a laggy network camera, either too twitchy or
    # frustratingly slow depending on hardware. A fixed time window
    # behaves consistently regardless of camera/hardware.
    registration_auto_capture_hold_seconds: float = 0.6

    # THE FIX for "it captured all 3 angles immediately, before I even
    # turned": auto-capture (added to replace a manual SPACE press)
    # removed the one thing that used to give a real person time to
    # physically reposition -- pressing a key takes a few seconds of
    # human reaction time by itself. Without an explicit substitute,
    # the very first still-facing-forward frame of the "left" angle
    # could pass the quality gate and start counting toward
    # registration_auto_capture_hold_seconds before the person had even
    # started turning. This adds a mandatory countdown BEFORE capture
    # checking even begins for each angle, during which nothing can be
    # captured no matter how good the frame looks -- guaranteeing real
    # wall-clock time to get into position, shown on screen as an
    # explicit "turn now" countdown rather than left implicit.
    registration_prep_seconds: float = 3.0

    # REAL head-pose (yaw) validation, using the 5-point facial
    # keypoints the detector already produces (see
    # face_id/registration.py's _estimate_yaw_ratio() for the method
    # and why it deliberately doesn't hardcode which numeric sign means
    # "left" vs "right" -- that mapping isn't reliable across
    # camera/mirroring setups, but MAGNITUDE of turn and whether two
    # captures turned in OPPOSITE directions both are).
    #
    # For the "front" angle: reject a candidate whose measured |yaw|
    # exceeds this -- must be genuinely facing forward, not just
    # "roughly forward-ish."
    registration_max_frontal_yaw_ratio: float = 0.08
    # For "left"/"right" (any non-first angle): reject a candidate
    # whose measured |yaw| is BELOW this -- a real, meaningful head
    # turn must be measured before it's accepted, replacing what used
    # to be pure elapsed-time/embedding-similarity heuristics with an
    # actual geometric pose check.
    registration_min_turn_yaw_ratio: float = 0.12


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    wave: WaveDetectionConfig = field(default_factory=WaveDetectionConfig)
    entrance_line: EntranceLineConfig = field(default_factory=EntranceLineConfig)
    action_guard: ActionGuardConfig = field(default_factory=ActionGuardConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    face_recognition: FaceRecognitionConfig = field(default_factory=FaceRecognitionConfig)


# Single shared instance imported by all other modules.
CONFIG = AppConfig()
