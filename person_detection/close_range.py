"""
Close-range fallback for person detection.

PROBLEM: YOLOv8n was trained on whole-body people. When someone steps
very close to the camera only their head and shoulders are in frame, the
"person" score can drop under the confidence threshold, no track is
created, and nothing downstream (face recognition, wave detection,
attendance) runs at all.

FIX: when YOLO finds NOBODY in a frame, look for FACES instead (the face
detector copes well with large close-up faces) and turn each face into a
person-shaped box (the face widened sideways and extended down over the
shoulders/torso). That box goes through the same tracker, recognition and
wave pipeline as a normal YOLO detection.

COST: it only runs when YOLO returned nothing, and only on every second
such frame, so a normal frame (someone visible to YOLO) pays nothing.
"""

from typing import List, Tuple

from person_detection.detection import Detection


def face_to_person_box(face: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    fx1, fy1, fx2, fy2 = face
    fw, fh = fx2 - fx1, fy2 - fy1
    x1 = int(fx1 - 1.2 * fw)
    x2 = int(fx2 + 1.2 * fw)
    y1 = int(fy1 - 0.5 * fh)
    y2 = int(fy1 + 5.0 * fh)
    return (max(0, x1), max(0, y1), min(frame_w - 1, x2), min(frame_h - 1, y2))


class CloseRangeFallback:
    def __init__(self, embedder, every_n_frames: int = 2,
                 min_face_height_ratio: float = 0.08, min_det_score: float = 0.6):
        self._embedder = embedder
        self._every = max(1, every_n_frames)
        self._min_ratio = min_face_height_ratio
        self._min_score = min_det_score
        self._n = 0
        self._last: List[Detection] = []

    def augment(self, frame, detections: List[Detection]) -> List[Detection]:
        if detections:                       # YOLO sees someone: nothing to do
            self._last = []
            return detections
        self._n += 1
        if self._n % self._every != 0:
            return list(self._last)          # reuse the previous result between runs
        h, w = frame.shape[:2]
        found: List[Detection] = []
        try:
            faces = self._embedder.detect_faces_only(frame)
        except Exception:
            faces = []
        for (x1, y1, x2, y2, score) in faces:
            if score < self._min_score or (y2 - y1) < self._min_ratio * h:
                continue
            found.append(Detection(bbox=face_to_person_box((x1, y1, x2, y2), w, h), confidence=float(score)))
        self._last = found
        return found
