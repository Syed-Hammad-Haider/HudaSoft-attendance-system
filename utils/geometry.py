"""Small geometry helpers shared across modules."""

from typing import Tuple


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    """Return the (x, y) center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_width(bbox: BBox) -> float:
    x1, _, x2, _ = bbox
    return float(x2 - x1)


def bbox_height(bbox: BBox) -> float:
    _, y1, _, y2 = bbox
    return float(y2 - y1)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def expand_bbox(bbox: BBox, frame_w: int, frame_h: int, margin_ratio: float = 0.1) -> BBox:
    """
    Slightly expand a bounding box (by margin_ratio of its size) so that
    hand crops used for wave detection include hands that extend just
    outside the tight person box. Clamped to frame bounds.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    nx1 = int(clamp(x1 - mx, 0, frame_w - 1))
    ny1 = int(clamp(y1 - my, 0, frame_h - 1))
    nx2 = int(clamp(x2 + mx, 0, frame_w - 1))
    ny2 = int(clamp(y2 + my, 0, frame_h - 1))
    return (nx1, ny1, nx2, ny2)
