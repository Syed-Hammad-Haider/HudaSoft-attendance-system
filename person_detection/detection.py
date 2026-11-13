"""Data structure representing a single raw person detection in a frame."""

from dataclasses import dataclass
from utils.geometry import BBox


@dataclass
class Detection:
    """A single detected person before any tracking ID is assigned."""
    bbox: BBox          # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
