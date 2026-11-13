"""
LiveFaceRecognitionCache: manages WHEN face recognition actually runs
during live monitoring, and caches the result per tracked person.

WHY THIS EXISTS AS ITS OWN CLASS:
Running face detection + embedding on every stable tracked person
EVERY frame would add real, avoidable CPU cost on top of YOLO
detection, tracking, and per-person MediaPipe hand analysis --
consistent with this project's established pattern (see
WaveDetectionConfig.wave_processing_frame_interval) of only re-running
an expensive model as often as the underlying reality can actually
change. A person's identity does not change frame to frame once
they're recognized, so recognition is only re-attempted:
  - the first time a track becomes stable enough to bother (avoids
    wasting a recognition attempt on a track that might be a momentary
    false detection), and
  - periodically afterward (re_recognition_interval_seconds), which
    both amortizes cost AND allows recovery if an early recognition
    attempt was wrong or unrecognized.

FACE CROP STRATEGY: rather than running a separate face detector on
the whole frame, this reuses the SAME "crop to person, then search
within the crop" pattern wave_detection/detector.py already uses for
hands -- cheaper than full-frame search, and naturally associates the
right face with the right tracked person with no extra matching step.
The crop is the top portion of the person's bounding box (where a
head/face will be for a person standing/walking upright), not the
whole body box.

IDENTITY HOLD + FAST START + CONFIRMED WEAK MATCHES (added after "I stand
still and it shows ID n, it only names me when I move"):
  - a new track is retried every ~0.4s for its first few seconds, so a
    registered person is named almost immediately;
  - a name is KEPT through a failed re-check (a still pose can fail once)
    for `identity_hold_seconds`, instead of flipping back to "ID n";
  - a slightly weaker match (>= soft_match_similarity) is accepted when the
    same employee wins on `soft_match_confirmations` recent attempts and the
    usual runner-up margin holds -- two agreeing weak matches are safer
    than one lucky strong one;
  - the face search crop is taller (face_crop_fraction), and if that finds
    no usable face the whole person box is tried too.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from config.settings import FaceRecognitionConfig
from face_id.embedder import FaceEmbedder
from face_id.recognizer import FaceRecognizer, RecognitionResult
from person_tracking.tracked_person import TrackedPerson

logger = logging.getLogger(__name__)

_NOBODY = RecognitionResult(employee_id=None, full_name=None, similarity=0.0, matched_angle=None)


@dataclass
class _CacheEntry:
    result: RecognitionResult                 # what the rest of the app sees
    last_checked: float
    first_seen: float
    last_attempt_ok: bool = False             # did the most recent attempt recognise someone?
    last_good: Optional[RecognitionResult] = None
    last_good_at: float = 0.0
    weak: Deque[Tuple[float, int, float]] = field(default_factory=lambda: deque(maxlen=6))  # (time, employee_id, similarity)
    last_log: float = 0.0


class LiveFaceRecognitionCache:
    def __init__(self, config: FaceRecognitionConfig, embedder: FaceEmbedder, recognizer: FaceRecognizer):
        self._config = config
        self._embedder = embedder
        self._recognizer = recognizer
        self._cache: Dict[int, _CacheEntry] = {}

    def get_recognition(self, frame, person: TrackedPerson) -> Optional[RecognitionResult]:
        """
        Return the current best-known recognition result for this tracked
        person, re-running face recognition only when warranted. Returns
        None if the track isn't stable enough yet to attempt recognition
        (distinct from an attempted-but-unrecognized result).
        """
        cfg = self._config
        if len(person.position_history) < cfg.min_track_stability_frames:
            return None

        now = time.time()
        entry = self._cache.get(person.track_id)
        if entry is None:
            entry = _CacheEntry(result=_NOBODY, last_checked=0.0, first_seen=now)
            self._cache[person.track_id] = entry
            needs_check = True
        else:
            if entry.last_attempt_ok:
                interval = cfg.re_recognition_interval_seconds
            elif now - entry.first_seen < cfg.fast_retry_window_seconds:
                interval = cfg.fast_retry_interval_seconds       # new track: be quick
            else:
                interval = cfg.unrecognized_retry_interval_seconds
            needs_check = (now - entry.last_checked) >= interval

        if not needs_check:
            return entry.result

        raw = self._attempt_recognition(frame, person)
        entry.last_checked = now
        self._apply(entry, raw, now, person.track_id)
        return entry.result

    # ------------------------------------------------------------------
    def _apply(self, entry: _CacheEntry, raw: RecognitionResult, now: float, track_id: int) -> None:
        cfg = self._config
        accepted: Optional[RecognitionResult] = None

        if raw.is_recognized:
            accepted = raw
        elif (raw.rejected_reason == "below_threshold" and raw.candidate_employee_id is not None
              and raw.similarity >= cfg.soft_match_similarity):
            # weak match: only accept once the SAME employee keeps winning
            margin_ok = (raw.runner_up_similarity is None or
                         raw.similarity - raw.runner_up_similarity >= self._recognizer._min_margin_over_runner_up)
            if margin_ok:
                entry.weak.append((now, raw.candidate_employee_id, raw.similarity))
                recent = [w for w in entry.weak
                          if now - w[0] <= cfg.soft_match_window_seconds and w[1] == raw.candidate_employee_id]
                if len(recent) >= cfg.soft_match_confirmations:
                    accepted = RecognitionResult(
                        employee_id=raw.candidate_employee_id, full_name=raw.candidate_name,
                        similarity=max(w[2] for w in recent), matched_angle=raw.matched_angle,
                        runner_up_similarity=raw.runner_up_similarity, rejected_reason=None,
                    )

        if accepted is not None:
            entry.result = accepted
            entry.last_good = accepted
            entry.last_good_at = now
            entry.last_attempt_ok = True
            return

        # Not recognised this time.
        entry.last_attempt_ok = False
        if entry.last_good is not None and (now - entry.last_good_at) <= cfg.identity_hold_seconds:
            entry.result = entry.last_good        # keep the name through a failed re-check
        else:
            entry.result = raw

        if now - entry.last_log >= 3.0:
            entry.last_log = now
            logger.info(
                "Track %d not recognised: best similarity %.3f (candidate=%s), runner-up=%s, reason=%s%s",
                track_id, raw.similarity, raw.candidate_name,
                f"{raw.runner_up_similarity:.3f}" if raw.runner_up_similarity is not None else "n/a",
                raw.rejected_reason or "no usable face found",
                " (holding previous name)" if entry.result is entry.last_good else "",
            )

    # ------------------------------------------------------------------
    def _crop(self, frame, bbox, fraction: float):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = int((x2 - x1) * 0.08)
        cy2 = y1 + int((y2 - y1) * fraction)
        return frame[max(0, y1):min(h, cy2), max(0, x1 - pad):min(w, x2 + pad)]

    def _attempt_recognition(self, frame, person: TrackedPerson) -> RecognitionResult:
        """
        Try the face region first (top part of the box). If that does not
        produce a recognition, also try the whole person box and keep the
        better of the two results -- costs a second face pass only on
        attempts that were failing anyway.
        """
        best: Optional[RecognitionResult] = None
        for fraction in (self._config.face_crop_fraction, 1.0):
            crop = self._crop(frame, person.bbox, fraction)
            result = self._recognize_crop(crop)
            if result.is_recognized:
                return result
            if best is None or result.similarity > best.similarity:
                best = result
        return best if best is not None else _NOBODY

    def _recognize_crop(self, crop) -> RecognitionResult:
        if crop is None or crop.size == 0:
            return _NOBODY
        faces = self._embedder.detect_and_embed(crop)
        if not faces:
            return _NOBODY
        best_face = max(faces, key=lambda f: f.detection_confidence)
        if best_face.detection_confidence < self._config.min_live_detection_confidence:
            return _NOBODY
        result = self._recognizer.recognize(best_face.embedding)
        if result.is_recognized:
            logger.debug("Recognized employee_id=%d (%s), similarity=%.3f",
                         result.employee_id, result.full_name, result.similarity)
        return result

    def remove_track(self, track_id: int) -> None:
        self._cache.pop(track_id, None)
