"""
Standalone camera diagnostic tool.

WHY THIS EXISTS:
"The camera doesn't work" can mean a dozen different underlying
problems (wrong index, another app has it locked, a driver issue, a
permissions prompt you missed, an IP camera URL that's simply wrong),
and main.py's error message alone can't distinguish between them. This
script isolates the camera from the rest of the pipeline (no YOLO, no
face recognition, no wave detection) so you can find out FIRST whether
the problem is the camera itself, before troubleshooting anything else.

USAGE:
    python diagnose_camera.py                 # scan for local webcams
    python diagnose_camera.py --index 1       # test one specific index
    python diagnose_camera.py --url "rtsp://user:pass@192.168.1.50:554/stream1"
                                                # test an IP camera URL
"""

import argparse
import logging
import platform
import sys
import time

import cv2

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("diagnose_camera")


def _print_header(text: str) -> None:
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


def scan_webcam_indices(max_index: int = 5) -> list:
    """
    Try opening webcam indices 0..max_index and report which ones
    actually produce frames. Returns the list of working indices.
    """
    _print_header("Scanning for local webcams (indices 0-%d)" % max_index)
    working = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        opened = cap.isOpened()
        frame_ok = False
        w = h = fps = 0
        if opened:
            success, frame = cap.read()
            if success and frame is not None:
                frame_ok = True
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if frame_ok:
            print(f"  [{index}] OK -- {w}x{h} @ {fps:.0f}fps (reported)")
            working.append(index)
        elif opened:
            print(f"  [{index}] Opened but could not read a frame (device busy or driver issue)")
        else:
            print(f"  [{index}] Not available")

    if not working:
        print(
            "\nNo working webcam found. See the troubleshooting checklist "
            "below -- this is almost always fixable."
        )
    else:
        print(f"\nWorking webcam index(es): {working}")
        print(f"Set CONFIG.camera.source = {working[0]} in config/settings.py to use it.")
    return working


def test_specific_source(source, duration_seconds: float = 5.0) -> bool:
    """
    Open one specific source (int index or URL string) and stream from
    it for a few seconds, reporting actual achieved FPS -- this matters
    because wave detection's accuracy depends directly on real
    throughput (see WaveDetectionConfig.max_seconds_between_reversals
    in config/settings.py for why).
    """
    _print_header(f"Testing source: {source!r}")

    is_ip = isinstance(source, str) and source.lower().startswith(("rtsp://", "http://", "https://"))
    backend = cv2.CAP_FFMPEG if is_ip else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend) if is_ip else cv2.VideoCapture(source)

    if not cap.isOpened():
        print("FAILED to open this source.")
        cap.release()
        return False

    print("Opened successfully. Reading frames for %.0fs to measure real throughput..." % duration_seconds)
    frame_count = 0
    start = time.time()
    last_frame = None
    while time.time() - start < duration_seconds:
        success, frame = cap.read()
        if success and frame is not None:
            frame_count += 1
            last_frame = frame
        else:
            print("  (a read failed mid-test -- this is worth noting for IP cameras: "
                  "occasional drops are normal and handled by reconnect logic in "
                  "camera/ip_camera_source.py, but frequent drops point to a weak "
                  "network link)")

    elapsed = time.time() - start
    achieved_fps = frame_count / elapsed if elapsed > 0 else 0
    cap.release()

    print(f"\nFrames read: {frame_count} in {elapsed:.1f}s -> {achieved_fps:.1f} fps actual")
    if last_frame is not None:
        h, w = last_frame.shape[:2]
        print(f"Frame size: {w}x{h}")

    if achieved_fps < 8:
        print(
            "\nWARNING: achieved FPS is below ~8. main.py's wave detection "
            "specifically depends on sampling fast enough to catch hand "
            "motion -- below this, real waves can be missed no matter how "
            "sensitivity is tuned. See config/settings.py "
            "WaveDetectionConfig.max_seconds_between_reversals for the full "
            "explanation. If this is a webcam, try a lower "
            "frame_width/frame_height in CameraConfig. If this is an IP "
            "camera, try a lower-resolution sub-stream if your camera "
            "offers one (often called 'stream2' or 'sub')."
        )
    else:
        print("\nThroughput looks healthy for real-time wave detection.")

    return frame_count > 0


def print_checklist() -> None:
    system = platform.system()
    _print_header("Troubleshooting checklist")
    print("1. Is another app using the camera right now? Close Zoom, Teams, "
          "Skype, browser tabs with camera access, or any other webcam app, "
          "then re-run this script. Most OSes only let ONE process use a "
          "webcam at a time -- this is the single most common cause.")
    print("2. Try a different index: some laptops expose a virtual/IR "
          "camera at index 0 and the real color camera at index 1 (or vice "
          "versa). This script's scan above tries indices 0-5 automatically.")

    if system == "Windows":
        print("3. Windows: Settings > Privacy & security > Camera -> make sure "
              "'Camera access' and 'Let apps access your camera' are ON, and "
              "that Python/your terminal isn't blocked.")
        print("4. Windows: Device Manager > Cameras -- if there's a yellow "
              "warning icon, update or reinstall the driver.")
    elif system == "Darwin":
        print("3. macOS: System Settings > Privacy & Security > Camera -> "
              "make sure Terminal (or your IDE/Python) is checked. If you "
              "don't see it listed, run this script once first -- macOS "
              "adds the entry after the first access attempt, and you may "
              "need to check the box and re-run.")
    else:
        print("3. Linux: check `ls /dev/video*` -- if nothing lists, the "
              "kernel isn't seeing the camera at all (driver/hardware "
              "issue). If devices ARE listed, check you're in the "
              "`video` group: `groups $USER`, and if not, "
              "`sudo usermod -aG video $USER` then log out/in.")

    print("5. If NOTHING opens at all: unplug/replug (for USB webcams) or "
          "restart the laptop -- this genuinely fixes a surprising number "
          "of 'camera won't open' cases caused by a stuck driver handle.")
    print("6. For an IP camera that won't connect: test the exact same URL "
          "in VLC first (Media > Open Network Stream). If VLC can't play "
          "it either, the problem is the URL/network/credentials, not this "
          "program -- fix it there first, then retry here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose camera connectivity issues.")
    parser.add_argument("--index", type=int, default=None, help="Test one specific webcam index.")
    parser.add_argument("--url", type=str, default=None, help="Test one specific IP camera URL.")
    parser.add_argument("--max-scan-index", type=int, default=5, help="Highest webcam index to scan (default 5).")
    args = parser.parse_args()

    print(f"OpenCV version: {cv2.__version__}  |  Platform: {platform.platform()}")

    ok = True
    if args.url is not None:
        ok = test_specific_source(args.url)
    elif args.index is not None:
        ok = test_specific_source(args.index)
    else:
        working = scan_webcam_indices(args.max_scan_index)
        if working:
            test_specific_source(working[0])
        ok = bool(working)

    if not ok:
        print_checklist()

    sys.exit(0 if ok else 1)
