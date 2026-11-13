"""
Background thread that continuously reads from a cv2.VideoCapture and
keeps only the SINGLE most recent frame, discarding everything else.

WHY THIS EXISTS -- this is the actual fix for "the laptop screen lags
behind the phone camera":

OpenCV's VideoCapture.read() pulls from an internal frame queue. If
frames arrive faster than this pipeline can process them -- very
plausible here, since DroidCam adds its own capture+encode+transmit
delay on top of this pipeline running YOLO detection + tracking +
per-person MediaPipe hand tracking + face recognition, all on EVERY
frame -- that queue backs up. Each read() call then returns the
OLDEST unconsumed frame, not the newest, so the displayed video falls
further and further behind the real, live camera feed the longer the
app runs. This is standard OpenCV behavior, not something specific to
this project or to DroidCam.

Setting CAP_PROP_BUFFERSIZE=1 (also done in webcam_source.py and
ip_camera_source.py) helps on backends that honor it, but several
common backends -- including DirectShow on Windows, which is what a
DroidCam-style virtual webcam typically registers as -- ignore it
entirely. So that setting alone is not a reliable fix by itself.

THE ACTUAL FIX: decouple "reading from the camera" from "processing a
frame" using a dedicated background thread whose only job is to call
.read() in a tight loop and overwrite a single shared "latest frame"
slot. The main pipeline thread then always grabs whatever is CURRENTLY
in that slot via get_latest(), instead of pulling from a backed-up
queue. Worst case, if processing is briefly slower than the camera,
the same frame gets read twice in a row -- but the video never
accumulates a growing lag, which is what was actually reported.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LatestFrameReader:
    def __init__(self, capture):
        self._capture = capture
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_read_ok = True
        self._last_success_time: Optional[float] = None

    def start(self, startup_timeout_seconds: float = 8.0) -> bool:
        """
        Start the background reader thread and block until the first
        real frame has arrived (or the timeout elapses). Blocking here
        -- rather than letting the caller's first read_frame() return
        None -- matters because main.py treats a None frame as fatal
        ("no frame received; stopping") by design; that check exists
        to catch a genuinely dead camera mid-session, and shouldn't be
        confused with the ordinary one-time startup delay before the
        very first frame arrives.
        """
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        deadline = time.time() + startup_timeout_seconds
        while time.time() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return True
            time.sleep(0.05)
        return False

    def _run(self) -> None:
        while self._running:
            try:
                success, frame = self._capture.read()
            except Exception:
                logger.exception("Unexpected error reading from capture in background thread.")
                success, frame = False, None

            with self._lock:
                self._last_read_ok = bool(success and frame is not None)
                if self._last_read_ok:
                    self._latest_frame = frame
                    self._last_success_time = time.time()

            if not success:
                # Avoid a tight spin loop hammering the CPU if the
                # source has genuinely gone away. Reconnect logic (see
                # ip_camera_source.py) lives in the owning source
                # class, not here -- this thread just keeps trying to
                # read from whatever capture object it's holding at
                # the time.
                time.sleep(0.05)

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def is_healthy(self) -> bool:
        with self._lock:
            return self._last_read_ok

    def seconds_since_last_success(self) -> float:
        """
        Used by IPCameraSource to decide when a stale-but-non-None
        cached frame means the stream has actually died (vs. a normal,
        momentary hiccup the background thread's own retry loop will
        recover from on its own without needing a full reconnect).
        """
        with self._lock:
            if self._last_success_time is None:
                return float("inf")
            return time.time() - self._last_success_time

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
