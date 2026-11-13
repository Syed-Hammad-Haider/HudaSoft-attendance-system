"""Result objects returned by WaveDetector.process(), consumed by display/."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from wave_detection.body_motion import BodyMotionDebugInfo
from wave_detection.oscillation import OscillationDebugInfo


@dataclass
class HandObservation:
    """One detected hand, positioned in global frame coordinates, ready for drawing."""
    bbox: Tuple[int, int, int, int]
    is_waving: bool
    confidence: float  # 0.0 - 1.0, progress toward confirming a wave
    # All 21 MediaPipe hand landmark points, mapped to global frame pixel
    # coordinates. Used to draw an actual hand skeleton (dots + connecting
    # lines) instead of just a bounding box -- much clearer visual
    # confirmation of exactly what the system is tracking, and makes
    # detection precision visible rather than just asserted.
    landmarks: Optional[List[Tuple[int, int]]] = None
    # For the DEBUG/ANALYSIS overlay: full internal state of this
    # hand's oscillation detector this frame (samples analyzed, current
    # swing magnitude, reversal count, regularity, etc). None if this
    # HandObservation wasn't produced through the normal tracked path.
    debug: Optional[OscillationDebugInfo] = None


@dataclass
class WaveResult:
    """Everything the rest of the app needs to know about one person's hands this frame."""
    wave_confirmed: bool = False
    wave_confidence: float = 0.0
    hands: List[HandObservation] = field(default_factory=list)
    # For the DEBUG/ANALYSIS overlay: this person's body-stillness gate
    # state this frame. None if debug wasn't requested or no hands are
    # being tracked for this person.
    body_debug: Optional[BodyMotionDebugInfo] = None
