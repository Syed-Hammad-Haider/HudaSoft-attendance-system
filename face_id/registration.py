"""
Multi-angle employee registration workflow.

WHY MULTIPLE ANGLES, NOT ONE FRONTAL PHOTO:
A single frontal embedding is fragile against how people actually
appear to a real entrance camera -- turning their head slightly while
walking, camera mounted above eye level, etc. Capturing front/left/
right (and storing each as a separate embedding rather than averaging
them, see recognizer.py) gives the live matching step multiple
reference points to compare against, meaningfully improving real-world
match reliability without needing a different, heavier model.

QUALITY GUARDS during capture (each angle must pass before being
accepted, rather than blindly saving whatever the camera saw):
  - Exactly one face detected in the capture frame (guards against
    capturing a bystander in the background by mistake).
  - Detection confidence above a minimum (guards against a blurry/
    poorly-lit capture producing a low-quality embedding).
  - The face must be a reasonable size in frame (too small/far away
    produces a less reliable embedding).

This module contains the CAPTURE LOGIC only. The actual guided UI loop
(showing a live preview, prompting "now turn left", etc.) lives in
register_employee.py at the project root, kept as a separate,
deliberately simple entry point from main.py -- registration is an
admin/setup task with a different workflow than continuous monitoring,
and keeping them as separate scripts matches this project's existing
pattern of separating concerns by module.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from database.employee_repository import register_employee as db_register_employee
from database.employee_repository import save_embedding
from face_id.embedder import DetectedFace, FaceEmbedder

logger = logging.getLogger(__name__)

DEFAULT_ANGLES = ["front", "left", "right"]


def _estimate_yaw_ratio(keypoints: Optional[np.ndarray]) -> Optional[float]:
    """
    Estimate head yaw (left/right turn) from the 5-point keypoints,
    returning a signed ratio: ~0 for a straight-on frontal face,
    increasingly positive/negative the further the head is turned.

    METHOD: compare the nose tip's horizontal position to the midpoint
    between the two eyes, normalized by inter-eye distance (so it's
    scale-invariant -- works the same whether the face is close to or
    far from the camera). When frontal, the nose sits roughly centered
    between the eyes. When the head turns, the nose visually shifts
    toward one side relative to the eyes. This is a standard, cheap
    yaw proxy that needs no extra model -- just the keypoints the
    detector already produces.

    DELIBERATE DESIGN CHOICE -- WHY THIS DOESN'T HARDCODE WHICH SIGN
    MEANS "LEFT" VS "RIGHT": whether a positive value corresponds to
    the subject's actual left or right depends on camera mirroring
    conventions that vary by platform/webcam driver/app -- getting
    that mapping wrong would silently swap the two labels while
    looking like it's working correctly, which is worse than not
    claiming an absolute direction at all. Instead, the caller
    (register_employee.py) uses this value in a way that doesn't
    depend on knowing the absolute mapping:
      - magnitude (abs value) confirms a REAL turn happened at all,
        directly replacing what used to be a blind timer/no pose check
      - sign is used only to confirm two different turn angles (e.g.
        "left" and "right") are actually turned in OPPOSITE directions
        from each other, not verifying which specific physical
        direction each one is.
    """
    if keypoints is None or len(keypoints) < 3:
        return None

    left_eye, right_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    inter_eye_dist = abs(float(right_eye[0]) - float(left_eye[0]))
    if inter_eye_dist < 1e-3:
        return None

    eye_mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
    return (float(nose[0]) - eye_mid_x) / inter_eye_dist


@dataclass
class CaptureQualityResult:
    accepted: bool
    reason: Optional[str]  # human-readable rejection reason, if not accepted
    face: Optional[DetectedFace]
    similarity_to_previous: Optional[float] = None  # for on-screen debug, regardless of accept/reject
    measured_yaw_ratio: Optional[float] = None       # for on-screen debug, regardless of accept/reject


def evaluate_capture_quality(
    frame_bgr: np.ndarray,
    embedder: FaceEmbedder,
    min_detection_confidence: float = 0.6,
    min_face_size_px: int = 80,
    previous_embeddings: Optional[List[np.ndarray]] = None,
    max_similarity_to_previous: float = 0.97,
    require_frontal: bool = False,
    require_turned: bool = False,
    min_turn_yaw_ratio: float = 0.12,
    max_frontal_yaw_ratio: float = 0.08,
    forbidden_yaw_signs: Optional[Sequence[int]] = None,
) -> CaptureQualityResult:
    """
    Run detection on a single candidate capture frame and decide
    whether it's good enough to accept for this angle. Called
    repeatedly (e.g. once per live preview frame) by the guided capture
    loop until a frame passes.

    POSE VALIDATION (this is the actual fix for "the system should
    detect the angle, not just wait on a timer"): if `require_frontal`
    is set, a candidate whose measured yaw exceeds
    `max_frontal_yaw_ratio` is rejected -- it must be genuinely facing
    forward. If `require_turned` is set, a candidate whose measured yaw
    magnitude is BELOW `min_turn_yaw_ratio` is rejected -- a real head
    turn must be measured, not just assumed to have happened. If
    `forbidden_yaw_signs` is given (the signs already used by other
    turn angles captured this session), a candidate turned the SAME
    direction as one already captured is rejected too, so "left" and
    "right" are guaranteed to be genuinely opposite turns.

    `previous_embeddings`, if given, remain a SECONDARY safety net on
    top of the pose check: a candidate whose embedding is too similar
    to any already-accepted angle is still rejected even if the
    measured yaw looks fine, catching anything the yaw estimate alone
    might miss (e.g. a very subtle/partial turn).
    """
    faces = embedder.detect_and_embed(frame_bgr)

    if len(faces) == 0:
        return CaptureQualityResult(accepted=False, reason="No face detected", face=None)
    if len(faces) > 1:
        return CaptureQualityResult(
            accepted=False, reason=f"{len(faces)} faces detected -- only the registering person should be in frame", face=None
        )

    face = faces[0]
    if face.detection_confidence < min_detection_confidence:
        return CaptureQualityResult(
            accepted=False,
            reason=f"Detection confidence too low ({face.detection_confidence:.2f} < {min_detection_confidence})",
            face=face,
        )

    x1, y1, x2, y2 = face.bbox
    face_size = min(x2 - x1, y2 - y1)
    if face_size < min_face_size_px:
        return CaptureQualityResult(
            accepted=False,
            reason=f"Face too small in frame ({face_size}px < {min_face_size_px}px) -- move closer to the camera",
            face=face,
        )

    yaw_ratio = _estimate_yaw_ratio(face.keypoints)

    if require_frontal:
        if yaw_ratio is None:
            # No keypoints available (older model/version) -- can't
            # verify frontality geometrically. Fall through to the
            # remaining (similarity-based) checks rather than blocking
            # registration entirely on a missing capability.
            pass
        elif abs(yaw_ratio) > max_frontal_yaw_ratio:
            return CaptureQualityResult(
                accepted=False,
                reason=f"Head turned too much for a FRONT capture (yaw={yaw_ratio:+.2f}) -- look straight at the camera",
                face=face, measured_yaw_ratio=yaw_ratio,
            )

    if require_turned:
        if yaw_ratio is None:
            return CaptureQualityResult(
                accepted=False,
                reason="Could not measure head turn -- reposition so your face and eyes are clearly visible",
                face=face,
            )
        if abs(yaw_ratio) < min_turn_yaw_ratio:
            return CaptureQualityResult(
                accepted=False,
                reason=f"Not turned enough yet (yaw={yaw_ratio:+.2f}, need magnitude >= {min_turn_yaw_ratio:.2f}) -- turn your head further",
                face=face, measured_yaw_ratio=yaw_ratio,
            )
        sign = 1 if yaw_ratio > 0 else -1
        if forbidden_yaw_signs and sign in forbidden_yaw_signs:
            return CaptureQualityResult(
                accepted=False,
                reason=f"This is turned the SAME direction as an angle already captured (yaw={yaw_ratio:+.2f}) -- turn the OTHER way",
                face=face, measured_yaw_ratio=yaw_ratio,
            )

    similarity_to_previous = None
    if previous_embeddings:
        similarity_to_previous = max(float(np.dot(face.embedding, prev)) for prev in previous_embeddings)
        if similarity_to_previous >= max_similarity_to_previous:
            return CaptureQualityResult(
                accepted=False,
                reason=(
                    f"Too similar to an angle already captured (similarity={similarity_to_previous:.3f}) "
                    "-- turn your head further for a genuinely different angle"
                ),
                face=face,
                similarity_to_previous=similarity_to_previous,
                measured_yaw_ratio=yaw_ratio,
            )

    return CaptureQualityResult(
        accepted=True, reason=None, face=face,
        similarity_to_previous=similarity_to_previous, measured_yaw_ratio=yaw_ratio,
    )


def register_new_employee(
    full_name: str,
    captured_faces: List[Tuple[str, DetectedFace]],
    employee_code: Optional[str] = None,
    department: Optional[str] = None,
    db_path=None,
) -> int:
    """
    Persist a new employee and all their accepted angle captures.
    `captured_faces` is a list of (angle_label, DetectedFace) pairs,
    already quality-checked by the caller via evaluate_capture_quality.
    Returns the new employee_id.
    """
    if not captured_faces:
        raise ValueError("At least one accepted face capture is required to register an employee.")

    kwargs = {"db_path": db_path} if db_path is not None else {}
    employee_id = db_register_employee(full_name, employee_code=employee_code, department=department, **kwargs)

    for angle_label, face in captured_faces:
        save_embedding(employee_id, angle_label, face.embedding, **kwargs)

    logger.info(
        "Registered employee '%s' (id=%d) with %d angle(s): %s",
        full_name, employee_id, len(captured_faces), [a for a, _ in captured_faces],
    )
    return employee_id
