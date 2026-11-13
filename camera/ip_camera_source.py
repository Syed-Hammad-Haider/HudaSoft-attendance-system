"""
IP camera (RTSP / HTTP-MJPEG) implementation of BaseVideoSource.

WHY THIS IS A SEPARATE CLASS FROM WebcamSource, NOT A FLAG ON IT:
a network stream fails differently than a local device. A USB webcam
either opens or it doesn't -- if a read ever fails afterward, the
device was almost certainly unplugged, and there's nothing useful to
retry. An IP camera's connection can drop mid-stream for reasons that
have nothing to do with the camera being "gone" (Wi-Fi hiccup, router
DHCP renewal, the camera itself rebooting, a brief network stall) --
and in every one of those cases, the right response is to reconnect,
not to treat one failed read as fatal. That reconnect-on-failure
behavior is the whole reason this class exists separately.

USAGE (config/settings.py):
    CONFIG.camera.source = "rtsp://user:pass@192.168.1.50:554/stream1"
    # or for an MJPEG-over-HTTP style IP camera app:
    CONFIG.camera.source = "http://192.168.1.50:8080/video"

camera/factory.py picks this class automatically whenever `source` is
a string starting with rtsp://, http://, or https:// -- no other
module needs to change.

FINDING YOUR CAMERA'S URL:
Every IP/CCTV camera or NVR brand exposes its stream differently.
Check the camera/NVR's own settings page or manual for its exact RTSP
path -- there is no universal one. A few common patterns to try:
    rtsp://<user>:<pass>@<ip>:554/stream1
    rtsp://<user>:<pass>@<ip>:554/h264/ch1/main/av_stream
    rtsp://<ip>:554/live
Most brands also list their default RTSP path on their support site if
you search "<brand> RTSP URL".
"""

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

from camera.base_source import BaseVideoSource
from camera.latest_frame_reader import LatestFrameReader
from config.settings import CameraConfig

logger = logging.getLogger(__name__)

# Bound FFMPEG's network open/read so an unreachable or WRONG IP fails in a
# few seconds instead of blocking the detection thread for ~30s+ (OpenCV's
# default socket timeout). Without this, typing a wrong DroidCam IP could
# freeze the live loop long enough that even switching back to a correct
# source -- or to the PC webcam -- appeared to "not connect". The value is
# in MICROSECONDS (5s). Set once, before any VideoCapture is created, and
# only if the operator hasn't already set their own override.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000")


class IPCameraSource(BaseVideoSource):
    """Reads frames from a network camera (RTSP or HTTP-MJPEG) via OpenCV."""

    def __init__(self, config: CameraConfig):
        self._config = config
        self._url = config.source
        self._capture: Optional[cv2.VideoCapture] = None
        self._reader: Optional[LatestFrameReader] = None

    def _connect(self) -> bool:
        # cv2.CAP_FFMPEG is explicit (rather than letting OpenCV guess a
        # backend) because it's the backend that reliably handles RTSP;
        # letting OpenCV auto-pick can silently choose a backend that
        # doesn't understand the URL scheme on some platforms.
        capture = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)

        # Best-effort only -- see camera/latest_frame_reader.py for why
        # this alone doesn't reliably prevent lag buildup on every
        # backend; the LatestFrameReader thread started below is what
        # actually guarantees it.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        deadline = time.time() + self._config.ip_connect_timeout_seconds
        while time.time() < deadline:
            if capture.isOpened():
                self._capture = capture
                reader = LatestFrameReader(capture)
                if not reader.start(startup_timeout_seconds=max(0.0, deadline - time.time())):
                    capture.release()
                    self._capture = None
                    return False
                self._reader = reader
                return True
            time.sleep(0.2)

        capture.release()
        return False

    def open(self) -> None:
        logger.info("Connecting to IP camera: %s", self._url)
        if not self._connect():
            raise RuntimeError(
                f"Could not connect to IP camera at '{self._url}' within "
                f"{self._config.ip_connect_timeout_seconds}s. Common causes: "
                "wrong IP/port/path in the URL, wrong username/password, the "
                "camera and this computer are on different networks (e.g. "
                "camera on Wi-Fi guest network), or a firewall blocking the "
                "port. Test the same URL in VLC (Media > Open Network "
                "Stream) first -- if VLC can't play it either, the problem "
                "is the URL/network, not this program."
            )
        logger.info("IP camera connected.")

    def _reconnect(self) -> bool:
        logger.warning("IP camera stream lost -- attempting to reconnect...")
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        for attempt in range(1, self._config.ip_reconnect_attempts + 1):
            if self._connect():
                logger.info("IP camera reconnected (attempt %d).", attempt)
                return True
            logger.warning(
                "Reconnect attempt %d/%d failed; retrying in %.1fs.",
                attempt, self._config.ip_reconnect_attempts, self._config.ip_reconnect_delay_seconds,
            )
            time.sleep(self._config.ip_reconnect_delay_seconds)

        logger.error(
            "Gave up reconnecting to IP camera after %d attempts. Check the "
            "camera is powered on and reachable on the network.",
            self._config.ip_reconnect_attempts,
        )
        return False

    def read_frame(self) -> Optional[np.ndarray]:
        if self._capture is None or self._reader is None:
            return None

        # Reconnect only on a genuinely dead stream (no successful read
        # in several seconds), not on every momentary blip -- the
        # background reader thread already retries transient failures
        # on its own without needing a full reconnect. Being too
        # trigger-happy here would tear down and rebuild the connection
        # constantly on a slightly-flaky-but-fine Wi-Fi link.
        if self._reader.seconds_since_last_success() > 5.0:
            if not self._reconnect():
                return None

        return self._reader.get_latest() if self._reader is not None else None

    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def release(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("IP camera source released.")
