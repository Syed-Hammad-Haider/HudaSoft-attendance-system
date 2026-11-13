"""
Abstract interface for any video source (webcam, RTSP/CCTV, video file...).

Every concrete source implements this same contract, so the rest of the
pipeline (detection, tracking, etc.) never needs to know or care where
frames are coming from. This is the seam that lets a CCTV camera replace
the webcam later with zero changes outside this module.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseVideoSource(ABC):
    """Contract that all video sources must implement."""

    @abstractmethod
    def open(self) -> None:
        """Open/connect to the video source. Raise on failure."""
        raise NotImplementedError

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Return the next frame as a BGR numpy array, or None if no frame
        is currently available (end of stream / read failure).
        """
        raise NotImplementedError

    @abstractmethod
    def is_opened(self) -> bool:
        """Return True if the source is currently open and readable."""
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """Release any underlying resources (camera handle, socket, etc.)."""
        raise NotImplementedError

    # Context-manager support so sources can be used with `with ... as cam:`
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
