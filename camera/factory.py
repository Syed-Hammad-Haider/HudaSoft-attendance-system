"""
Factory for constructing the configured video source.

This is the single switch point for changing the video input type.
The decision is made purely from the TYPE/SHAPE of `config.source`:
an int (or numeric-looking value) means a local webcam index; a string
starting with rtsp://, http://, or https:// means an IP/CCTV camera.
Nothing outside camera/ needs to know this decision is being made --
main.py and register_employee.py just call create_video_source() and
get back something implementing BaseVideoSource either way.
"""

from camera.base_source import BaseVideoSource
from camera.ip_camera_source import IPCameraSource
from camera.webcam_source import WebcamSource
from config.settings import CameraConfig

_IP_URL_PREFIXES = ("rtsp://", "http://", "https://")


def create_video_source(config: CameraConfig) -> BaseVideoSource:
    """Return a BaseVideoSource instance appropriate for the given config."""
    source = config.source
    if isinstance(source, str) and source.lower().startswith(_IP_URL_PREFIXES):
        return IPCameraSource(config)
    return WebcamSource(config)
