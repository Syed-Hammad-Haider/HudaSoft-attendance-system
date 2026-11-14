"""
"This browser's webcam" as a camera source.

WHY: when the app runs on a cloud server (AWS), the server has no webcam of
its own. Instead, the dashboard page in someone's browser opens THEIR
webcam (with their permission) and uploads small JPEG frames to
/api/browser_camera/frame. This module keeps the most recent uploaded
frame, and BrowserCameraSource hands it to the normal detection pipeline
exactly like any other camera -- nothing else in the app changes.

Only ONE browser feeds the camera at a time: the one that last chose
"Use this browser's webcam" in Settings (identified by a random sender id
that page keeps). Frames from any other page are refused, which tells
that page to stop sending.

Browsers only allow webcam access on https:// pages (or localhost), so on
a server this needs the HTTPS link (see deploy/Caddyfile, SITE_ADDRESS).
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np

from camera.base_source import BaseVideoSource
from config.settings import CameraConfig

BROWSER_SOURCE = "browser"     # the value stored in Settings -> camera_source
_FRESH_SECONDS = 3.0           # older than this = the browser stopped sending

_lock = threading.Lock()
_latest_frame: Optional[np.ndarray] = None
_latest_time: float = 0.0
_active_sender: str = ""
_latest_seq: int = -1


def is_browser_source(source) -> bool:
    return isinstance(source, str) and source.strip().lower() == BROWSER_SOURCE


def set_active_sender(sender_id: str) -> None:
    """Called when a page picks 'Use this browser's webcam'. Old frames are
    dropped so the previous browser's last picture never lingers."""
    global _active_sender, _latest_frame, _latest_time
    global _latest_seq
    with _lock:
        _active_sender = (sender_id or "").strip()
        _latest_frame = None
        _latest_time = 0.0
        _latest_seq = -1


def get_active_sender() -> str:
    with _lock:
        return _active_sender


def push_frame(sender_id: str, jpeg_bytes: bytes, seq: int = -1) -> bool:
    """Store one uploaded frame. Returns False if this sender is not the
    active one (so that page knows to stop). `seq` is the page's frame
    counter: several uploads travel at once, so a slower older frame that
    arrives after a newer one is ignored instead of jumping back in time."""
    global _latest_frame, _latest_time, _latest_seq
    global _active_sender
    with _lock:
        if not _active_sender and sender_id:
            # After a server restart nobody is registered yet: the first
            # page that is still sending (it chose this mode earlier) takes over.
            _active_sender = sender_id
            _latest_seq = -1
        if sender_id != _active_sender:
            return False
    data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("not a valid JPEG image")
    with _lock:
        if sender_id != _active_sender:
            return False
        if 0 <= seq <= _latest_seq:
            return True            # out-of-order older frame: skip it
        if seq >= 0:
            _latest_seq = seq
        _latest_frame = frame
        _latest_time = time.time()
    return True


def seconds_since_last_frame() -> float:
    with _lock:
        return (time.time() - _latest_time) if _latest_time else float("inf")


def _fit(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale into frame_width x frame_height keeping the shape (black bars
    fill leftover space) -- the app's geometry assumes exactly this size."""
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h) or w == 0 or h == 0:
        return frame
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    if (new_w, new_h) == (target_w, target_h):
        return resized
    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    x0, y0 = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


class BrowserCameraSource(BaseVideoSource):
    def __init__(self, config: CameraConfig):
        self._config = config
        self._opened = False

    def open(self) -> None:
        # Nothing to connect to: frames arrive by upload. Never raises, so
        # the detection loop simply waits until a browser starts sending.
        self._opened = True

    def read_frame(self) -> Optional[np.ndarray]:
        with _lock:
            frame = _latest_frame
            fresh = _latest_time and (time.time() - _latest_time) < _FRESH_SECONDS
        if frame is None or not fresh:
            time.sleep(0.05)   # nothing new yet: don't spin the detection loop
            return None
        return _fit(frame, int(self._config.frame_width), int(self._config.frame_height))

    def is_opened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self._opened = False
