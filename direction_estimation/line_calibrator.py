"""
Interactive vertical-slider entrance line calibrator.

WHY A SLIDER INSTEAD OF TWO ARBITRARY CLICKED POINTS:
An earlier version required clicking two arbitrary points, which could
produce any angled line -- flexible, but more than most installs need
and easy to get accidentally crooked. For the common case (a camera
facing an entrance more or less head-on, which is most CCTV/webcam
setups), what's actually needed is simpler: ONE vertical line that
spans the full height of the frame, adjustable only by sliding it left
or right to match where the doorway actually is in that camera's view.

HOW IT WORKS:
While calibration mode is active, moving the mouse over the video
previews a vertical line at the mouse's current X position, spanning
the full frame height automatically -- there is nothing to set for the
vertical extent, it always matches the camera screen's full height, as
requested. A single left-click confirms that X position as the new
entrance line.

This still hands back two (x, y) points to EntranceGate.set_line() --
just always with matching X and Y running from 0 to frame_height --
so no changes were needed to the entrance-line math itself, only to
how those two points get chosen.
"""

import cv2


class LineCalibrator:
    def __init__(self, window_name: str, frame_height: int):
        self._window_name = window_name
        self._frame_height = frame_height
        self._active = False
        self._preview_x = None
        self._confirmed_x = None
        self._attached = False

    def attach(self) -> None:
        """Must be called once, after the OpenCV window has been created
        (i.e. after the first cv2.imshow call), to start listening for
        mouse movement and clicks."""
        if not self._attached:
            cv2.setMouseCallback(self._window_name, self._on_mouse)
            self._attached = True

    def start(self) -> None:
        """Enter calibration mode: mouse movement previews the line, next click confirms it."""
        self._active = True
        self._preview_x = None
        self._confirmed_x = None

    def cancel(self) -> None:
        self._active = False
        self._preview_x = None
        self._confirmed_x = None

    @property
    def is_active(self) -> bool:
        return self._active

    def preview_line_px(self):
        """
        Return ((x, 0), (x, frame_height)) for the line currently
        following the mouse, or None if the mouse hasn't moved over the
        video yet this calibration session.
        """
        if self._preview_x is None:
            return None
        return (self._preview_x, 0), (self._preview_x, self._frame_height)

    def consume_completed_line(self):
        """
        If the line's position has been confirmed by a click, return it
        as ((x, 0), (x, frame_height)) and reset calibration mode.
        Otherwise return None.
        """
        if self._active and self._confirmed_x is not None:
            x = self._confirmed_x
            line = ((x, 0), (x, self._frame_height))
            self._active = False
            self._preview_x = None
            self._confirmed_x = None
            return line
        return None

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if not self._active:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self._preview_x = x
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._confirmed_x = x
