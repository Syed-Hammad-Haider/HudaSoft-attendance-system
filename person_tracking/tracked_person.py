"""Data structure representing a person tracked across multiple frames."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

from utils.geometry import BBox, bbox_center


@dataclass
class TrackedPerson:
    """
    A person with a stable tracking ID, plus a short rolling history of
    center-point positions. The history is the shared data source used
    by BOTH direction_estimation and wave_detection, so tracking is the
    one place per-person temporal state lives.
    """
    track_id: int
    bbox: BBox
    confidence: float
    history_length: int = 30
    position_history: Deque[Tuple[float, float]] = field(default=None)
    frames_since_seen: int = 0

    def __post_init__(self) -> None:
        if self.position_history is None:
            self.position_history = deque(maxlen=self.history_length)

    def update(self, bbox: BBox, confidence: float) -> None:
        """Update this track with a new detection for the current frame."""
        self.bbox = bbox
        self.confidence = confidence
        self.position_history.append(bbox_center(bbox))
        self.frames_since_seen = 0

    def mark_missed(self) -> None:
        """Called when this track was not matched to any detection this frame."""
        self.frames_since_seen += 1
