"""
Face detection + embedding extraction, wrapping InsightFace.

WHY INSIGHTFACE (over the more commonly-tutorial'd dlib/`face_recognition`
library): dlib requires compiling a C++ library from source with no
prebuilt wheel for recent Python versions on this platform -- a real,
practical installation blocker, not a style preference. InsightFace
ships prebuilt ONNX models via onnxruntime, installs cleanly, and its
recognition model (part of the "buffalo_s" pack used here) is generally
regarded as more accurate than the older dlib ResNet-based embeddings
it would have replaced -- so this substitution is a genuine upgrade,
not just a workaround.

WHY THE "buffalo_s" (small) MODEL PACK SPECIFICALLY: InsightFace offers
several model packs trading size/speed for accuracy (buffalo_s/m/l).
Matching this project's established CPU-first philosophy (YOLOv8n over
larger variants, MediaPipe over heavier alternatives), buffalo_s is the
right default -- comfortably fast on CPU for a handful of people in
frame. `allowed_modules=['detection', 'recognition']` skips loading the
pack's landmark/age/gender models entirely, since none of those are
used here -- faster startup, lower memory, for zero functional loss.

EMBEDDING: each detected face produces a 512-dimensional vector.
Two face crops of the SAME person produce embeddings that are close
together (small distance / high cosine similarity); different people
produce embeddings far apart. Recognition is therefore just "find the
closest stored embedding, and check if it's close enough to trust."
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in the coordinate space of the image passed in
    embedding: np.ndarray  # shape (512,), float32, L2-normalized
    detection_confidence: float
    # 5-point landmarks (left_eye, right_eye, nose, mouth_left, mouth_right),
    # each (x, y) in the coordinate space of the image passed in. These
    # come from the DETECTION model itself (SCRFD), not a separate
    # landmark model -- available even with allowed_modules restricted
    # to ['detection', 'recognition'], at zero extra inference cost.
    # Used for real head-pose (yaw) estimation during registration --
    # see face_id/registration.py's _estimate_yaw_ratio(). None if the
    # installed model/version doesn't provide keypoints.
    keypoints: Optional[np.ndarray] = None


class FaceEmbedder:
    def __init__(self, det_size: Tuple[int, int] = (320, 320), providers=None):
        self._app = FaceAnalysis(
            name="buffalo_s",
            providers=providers or ["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=0, det_size=det_size)
        logger.info("FaceEmbedder ready (buffalo_s, CPU).")

    def detect_and_embed(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        """
        Run face detection + embedding extraction on a BGR image
        (typically a person's cropped bounding box, not the whole
        frame -- same "crop first" pattern used by wave detection, for
        the same reasons: cheaper, and avoids matching the wrong
        person's face in a crowded scene).
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        faces = self._app.get(image_bgr)
        results = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            embedding = face.embedding.astype(np.float32)
            # L2-normalize so downstream cosine-similarity comparisons
            # reduce to a simple dot product.
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            results.append(DetectedFace(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                embedding=embedding,
                detection_confidence=float(face.det_score),
                keypoints=getattr(face, "kps", None),
            ))
        return results

    def detect_faces_only(self, image_bgr: np.ndarray):
        """Fast face DETECTION only (no embedding): list of (x1, y1, x2, y2, score).
        Used by the close-range fallback to find a person whose body YOLO cannot see."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        det = self._app.models.get("detection")
        if det is None:
            return []
        bboxes, _ = det.detect(image_bgr, max_num=0, metric="default")
        if bboxes is None:
            return []
        return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])) for b in bboxes]

    def largest_face(self, image_bgr: np.ndarray) -> Optional[DetectedFace]:
        """
        Convenience for registration: when capturing one employee at a
        time, assume the largest detected face is the intended subject
        (guards against a stray face in the background of the capture
        frame).
        """
        faces = self.detect_and_embed(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
