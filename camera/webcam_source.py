"""
Local webcam implementation of BaseVideoSource.

Later, a CCTV/RTSP source (e.g. camera/rtsp_source.py) can implement the
exact same interface. To switch, only config/settings.py's
CameraConfig.source needs to change (index -> RTSP URL), plus picking the
right source class in camera/factory.py. No other module changes.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from camera.base_source import BaseVideoSource
from camera.latest_frame_reader import LatestFrameReader
from config.settings import CameraConfig

logger = logging.getLogger(__name__)


class WebcamSource(BaseVideoSource):
    """Reads frames from a local webcam via OpenCV's VideoCapture."""

    def __init__(self, config: CameraConfig):
        self._config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._reader: Optional[LatestFrameReader] = None

    def open(self) -> None:
        self._capture = cv2.VideoCapture(self._config.source)

        # Request a specific resolution; the camera may not honor it
        # exactly, but this keeps downstream processing predictable.
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
        self._capture.set(cv2.CAP_PROP_FPS, self._config.max_fps)
        # Best-effort only -- many backends (notably DirectShow, which
        # a DroidCam-style virtual webcam typically uses on Windows)
        # ignore this. The LatestFrameReader thread below is the part
        # that actually guarantees no lag buildup regardless of
        # whether the backend honors this setting. See
        # camera/latest_frame_reader.py for the full explanation.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam source '{self._config.source}'. "
                "Check that the camera is connected and not in use by "
                "another application."
            )

        self._reader = LatestFrameReader(self._capture)
        if not self._reader.start():
            self._capture.release()
            self._capture = None
            self._reader = None
            raise RuntimeError(
                f"Opened webcam source '{self._config.source}' but no frame arrived "
                "within the startup timeout. If this is a DroidCam/virtual webcam, "
                "make sure the DroidCam Client (or OBS Virtual Camera) is actually "
                "running and connected before starting this program."
            )

        logger.info("Webcam source opened (source=%s)", self._config.source)

    def read_frame(self) -> Optional[np.ndarray]:
        if self._reader is None:
            return None
        return self._reader.get_latest()

    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def release(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Webcam source released.")
