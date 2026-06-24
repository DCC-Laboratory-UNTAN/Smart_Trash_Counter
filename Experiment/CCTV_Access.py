"""
RTSP Webcam Stream Viewer
=========================
Connects to an RTSP stream, displays the live feed, and optionally saves
frames or a video recording.

Requirements:
    pip install opencv-python

Usage examples:
    # Basic view (press 'q' to quit)
    python rtsp_stream.py --url "rtsp://username:password@192.168.1.100:554/stream"
    # Successfully Tested: 
    python3 rtsp_stream.py --url 'rtsp://admin:D1p4TL26!@192.168.1.103:554/stream1'

    # Snapshot every 10 seconds
    python rtsp_stream.py --url "rtsp://..." --snapshot-interval 10

    # Record to a video file
    python rtsp_stream.py --url "rtsp://..." --record output.avi

    # Run headless (no window; useful on servers)
    python rtsp_stream.py --url "rtsp://..." --headless --record output.avi

Common RTSP URL formats:
    Generic  : rtsp://user:pass@<IP>:<port>/stream
    Hikvision: rtsp://user:pass@<IP>:554/Streaming/Channels/101
    Dahua    : rtsp://user:pass@<IP>:554/cam/realmonitor?channel=1&subtype=0
    Reolink  : rtsp://user:pass@<IP>:554//h264Preview_01_main
    Axis     : rtsp://user:pass@<IP>/axis-media/media.amp
"""

import argparse
import os
import sys
import time
from datetime import datetime

try:
    import cv2
except ImportError:
    sys.exit(
        "OpenCV is required. Install it with:\n"
        "    pip install opencv-python"
    )


# ---------------------------------------------------------------------------
# Configuration defaults (override via CLI arguments)
# ---------------------------------------------------------------------------
DEFAULT_PORT = 554
RECONNECT_DELAY_SEC = 5   # seconds to wait before reconnecting on failure
MAX_RECONNECTS = 10        # 0 = unlimited


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_rtsp_url(
    ip: str,
    port: int = DEFAULT_PORT,
    username: str = "",
    password: str = "",
    path: str = "/stream",
) -> str:
    """Construct an RTSP URL from components (useful when not passing a full URL)."""
    creds = f"{username}:{password}@" if username else ""
    return f"rtsp://{creds}{ip}:{port}{path}"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_capture(url: str, transport: str = "tcp") -> cv2.VideoCapture:
    """Open an RTSP stream with the specified transport protocol."""
    # Force TCP transport to reduce packet loss over Wi-Fi / VPN
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return cap


# ---------------------------------------------------------------------------
# Core streaming function
# ---------------------------------------------------------------------------

def stream(
    url: str,
    *,
    headless: bool = False,
    record_path: str | None = None,
    snapshot_interval: float = 0,   # seconds; 0 = disabled
    transport: str = "tcp",
    max_reconnects: int = MAX_RECONNECTS,
    window_title: str = "RTSP Stream  |  press Q to quit",
) -> None:
    """
    Open and display (or process) an RTSP stream.

    Parameters
    ----------
    url               : Full RTSP URL.
    headless          : Skip imshow; useful on servers without a display.
    record_path       : File path to save a video recording (e.g. 'out.avi').
    snapshot_interval : Save a JPEG snapshot every N seconds (0 = disabled).
    transport         : 'tcp' (default, more reliable) or 'udp'.
    max_reconnects    : How many times to retry on disconnect (0 = unlimited).
    window_title      : OpenCV window title shown when not headless.
    """

    print(f"[INFO] Connecting to: {url}")
    print(f"[INFO] Transport    : {transport.upper()}")
    print(f"[INFO] Headless     : {headless}")
    if record_path:
        print(f"[INFO] Recording to : {record_path}")
    if snapshot_interval:
        print(f"[INFO] Snapshots    : every {snapshot_interval}s")

    reconnect_count = 0
    video_writer: cv2.VideoWriter | None = None
    last_snapshot_time = time.time()
    frame_count = 0

    try:
        while True:
            cap = open_capture(url, transport)

            if not cap.isOpened():
                reconnect_count += 1
                print(
                    f"[WARN] Could not open stream "
                    f"(attempt {reconnect_count}"
                    + (f"/{max_reconnects}" if max_reconnects else "")
                    + ")"
                )
                if max_reconnects and reconnect_count >= max_reconnects:
                    print("[ERROR] Max reconnects reached. Exiting.")
                    break
                print(f"[INFO] Retrying in {RECONNECT_DELAY_SEC}s …")
                time.sleep(RECONNECT_DELAY_SEC)
                continue

            # Stream metadata
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] Stream opened — {width}x{height} @ {fps:.1f} fps")
            reconnect_count = 0  # reset counter on successful connect

            # Set up video writer once we know the frame size
            if record_path and video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_writer = cv2.VideoWriter(record_path, fourcc, fps, (width, height))
                print(f"[INFO] VideoWriter initialised → {record_path}")

            # ---- Main read loop ----
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("[WARN] Frame read failed — attempting reconnect …")
                    break  # break inner loop → reconnect

                frame_count += 1

                # --- Optional: record to file ---
                if video_writer is not None:
                    video_writer.write(frame)

                # --- Optional: periodic snapshots ---
                now = time.time()
                if snapshot_interval and (now - last_snapshot_time) >= snapshot_interval:
                    snap_name = f"snapshot_{timestamp()}.jpg"
                    cv2.imwrite(snap_name, frame)
                    print(f"[INFO] Snapshot saved → {snap_name}")
                    last_snapshot_time = now

                # --- Display ---
                if not headless:
                    # Overlay connection info on frame
                    overlay = frame.copy()
                    label = f"Frame: {frame_count}  |  {width}x{height}  |  {fps:.1f} fps"
                    cv2.putText(
                        overlay, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                    cv2.imshow(window_title, overlay)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[INFO] 'Q' pressed — stopping.")
                        return  # clean exit
                    elif key == ord("s"):
                        snap_name = f"snapshot_{timestamp()}.jpg"
                        cv2.imwrite(snap_name, frame)
                        print(f"[INFO] Manual snapshot → {snap_name}")

            cap.release()
            time.sleep(RECONNECT_DELAY_SEC)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    finally:
        if "cap" in dir() and cap.isOpened():
            cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Video saved → {record_path}")
        if not headless:
            cv2.destroyAllWindows()
        print(f"[INFO] Total frames processed: {frame_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTSP Webcam Stream Viewer / Recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Connection ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url", "-u",
        help='Full RTSP URL (e.g. rtsp://user:pass@192.168.1.1:554/stream)',
    )
    group.add_argument(
        "--ip",
        help="Camera IP (used together with --port, --username, --password, --path)",
    )

    parser.add_argument("--port",     type=int, default=DEFAULT_PORT)
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--path",     default="/stream", help="RTSP path (default: /stream)")

    # --- Behaviour ---
    parser.add_argument(
        "--transport", choices=["tcp", "udp"], default="tcp",
        help="RTSP transport protocol (default: tcp)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Disable the display window (useful on headless servers)",
    )
    parser.add_argument(
        "--record", metavar="FILE",
        help="Save stream to a video file (e.g. output.avi)",
    )
    parser.add_argument(
        "--snapshot-interval", type=float, default=0, metavar="SECONDS",
        help="Auto-save a JPEG snapshot every N seconds (0 = disabled)",
    )
    parser.add_argument(
        "--max-reconnects", type=int, default=MAX_RECONNECTS,
        help=f"Max reconnection attempts (0 = unlimited, default: {MAX_RECONNECTS})",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.url:
        url = args.url
    else:
        url = build_rtsp_url(
            ip=args.ip,
            port=args.port,
            username=args.username,
            password=args.password,
            path=args.path,
        )

    stream(
        url,
        headless=args.headless,
        record_path=args.record,
        snapshot_interval=args.snapshot_interval,
        transport=args.transport,
        max_reconnects=args.max_reconnects,
    )


if __name__ == "__main__":
    main()
