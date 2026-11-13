"""
Multi-object tracker built on ByteTrack (via the `supervision` library).

Why ByteTrack (over DeepSORT):
  - No appearance re-identification network to run per frame, so it is
    significantly lighter on CPU -> better fit for real-time webcam/CCTV.
  - Associates low-confidence detections with existing tracks instead of
    discarding them outright, which makes it more robust to brief
    occlusion (e.g. one person briefly blocking another) without extra
    tuning.
  - Well maintained standalone implementation in `supervision`, decoupled
    from any specific detector -> keeps this module independent from
    person_detection/, matching the modular architecture.

This module owns ALL per-person temporal state (via TrackedPerson), which
direction_estimation and wave_detection both read from. It also owns
track lifecycle cleanup (dropping stale tracks) so state doesn't leak.
"""

import logging
from typing import Dict, List

import numpy as np
import supervision as sv

from config.settings import TrackingConfig
from person_detection.detection import Detection
from person_tracking.tracked_person import TrackedPerson

logger = logging.getLogger(__name__)


class PersonTracker:
    def __init__(self, config: TrackingConfig):
        self._config = config
        self._tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.assumed_frame_rate,
        )
        self._tracks: Dict[int, TrackedPerson] = {}

    def update(self, detections: List[Detection]) -> List[TrackedPerson]:
        """
        Feed this frame's detections through ByteTrack and return the
        full list of currently-active TrackedPerson objects (updated
        in place where possible so downstream buffers persist).
        """
        sv_detections = self._to_sv_detections(detections)
        tracked_sv = self._tracker.update_with_detections(sv_detections)

        matched_ids = set()

        for i in range(len(tracked_sv)):
            track_id = int(tracked_sv.tracker_id[i])
            xyxy = tracked_sv.xyxy[i]
            bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            confidence = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 0.0

            if track_id not in self._tracks:
                self._tracks[track_id] = TrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    history_length=self._config.history_length,
                )
            self._tracks[track_id].update(bbox, confidence)
            matched_ids.add(track_id)

        # Any existing track not matched this frame gets marked missed,
        # and dropped entirely once it has been missing too long. This
        # prevents unbounded growth of wave-buffers / cooldown timers
        # for people who have left the frame.
        stale_ids = []
        for track_id, track in self._tracks.items():
            if track_id not in matched_ids:
                track.mark_missed()
                if track.frames_since_seen > self._config.max_missed_frames:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            del self._tracks[track_id]
            logger.debug("Dropped stale track ID %d", track_id)

        return [self._tracks[tid] for tid in matched_ids]

    def _to_sv_detections(self, detections: List[Detection]) -> "sv.Detections":
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        # ByteTrack requires class_id to be present; every detection here
        # is already filtered to "person" upstream, so this is constant.
        class_id = np.zeros(len(detections), dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
