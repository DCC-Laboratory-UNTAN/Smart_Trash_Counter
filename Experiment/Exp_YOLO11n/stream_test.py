#!/usr/bin/env python3
"""
Smart Trash Counter — YOLOv8n / YOLO11n
Using Ultralytics Docker on Jetson Nano

DCC lab - HAM

Usage (RTSP stream):
    python3 stream_test.py --rtsp=rtsp://user:pass@192.168.1.x:554/stream

Usage (video file):
    python3 stream_test.py --video=trash_test.mp4

Requirements (inside Ultralytics Docker):
    - ultralytics
    - opencv-python
    - Your trained model:
        models/smart-trash-detector/best.engine
"""

import cv2
import argparse
import time
import json
import os
import math
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════
# CONFIG — edit these
# ═══════════════════════════════════════════════════════════════
MODEL       = "best.engine"
THRESHOLD   = 0.50           # detection confidence threshold
IMGSZ       = 320            # must match the size used when exporting best.engine
LINE_POS    = 0.55           # virtual line position (0.0=top, 1.0=bottom)
SAVE_LOG    = True           # save count log to JSON
LOG_FILE    = "trash_counts_yolo.json"
DISPLAY     = False          # show OpenCV window — set False for headless

# Overlay colours (BGR)
COLOR_LINE      = (50,  50,  255)
COLOR_BOX       = (50,  200, 50)
COLOR_CROSSED   = (50,  50,  255)
COLOR_TEXT_BG   = (20,  20,  20)
COLOR_COUNT     = (50,  220, 255)
COLOR_TOTAL     = (50,  220, 255)

# ── Recording ──────────────────────────────────────────────────
RECORD       = True           # save annotated video to disk
RECORD_DIR   = "recordings"   # folder where .mp4 files are saved
RECORD_FPS   = 15             # output video FPS (match your camera or lower)
RECORD_SPLIT = 10             # split into a new file every N minutes (0 = no split)
# ═══════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp",      type=str, default=None,
                        help="RTSP URL, e.g. rtsp://user:pass@192.168.1.10:554/stream")
    parser.add_argument("--video",     type=str, default=None,
                        help="Path to a local video file, e.g. trash_test.mp4")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--line-pos",  type=float, default=LINE_POS)
    parser.add_argument("--headless",  action="store_true")
    return parser.parse_args()


# ─── Video Recorder ────────────────────────────────────────────

class VideoRecorder:
    """
    Wraps cv2.VideoWriter.  Call .write(bgr_frame) each frame.
    Handles timestamped filenames and automatic split-by-minutes.
    """
    def __init__(self, out_dir, fps, split_minutes=0):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir       = out_dir
        self.fps           = fps
        self.split_sec     = split_minutes * 60 if split_minutes > 0 else None
        self.writer        = None
        self.current_file  = None
        self.segment_start = None

    def _new_writer(self, w, h):
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        path   = os.path.join(self.out_dir, f"trash_yolo_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"[REC] Cannot open VideoWriter for {path}")
        print(f"[REC] Recording → {path}")
        self.current_file  = path
        self.segment_start = time.time()
        return writer

    def write(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        if self.writer is None:
            self.writer = self._new_writer(w, h)

        # Auto-split
        if self.split_sec and (time.time() - self.segment_start) >= self.split_sec:
            self.release()
            self.writer = self._new_writer(w, h)

        self.writer.write(bgr_frame)

    def release(self):
        if self.writer:
            self.writer.release()
            print(f"[REC] Saved → {self.current_file}")
            self.writer = None


# ─── Object Tracker ────────────────────────────────────────────

class CentroidTracker:
    """
    Simple centroid tracker — assigns persistent IDs to detections
    and detects when an object crosses the virtual line.
    """
    def __init__(self, max_lost=15, max_dist=90):
        self.next_id  = 0
        self.objects  = {}   # id -> {cx, cy, label, crossed}
        self.lost     = {}
        self.max_lost = max_lost
        self.max_dist = max_dist

    def update(self, boxes, line_y):
        """
        boxes : list of (cx, cy, label, conf)
        Returns list of (label, id) that crossed the line this frame.
        """
        matched_ids = set()

        if self.objects:
            for cx, cy, label, conf in boxes:
                best_id, best_dist = None, self.max_dist
                for oid, obj in self.objects.items():
                    if oid in matched_ids:
                        continue
                    d = math.hypot(cx - obj["cx"], cy - obj["cy"])
                    if d < best_dist:
                        best_dist, best_id = d, oid
                if best_id is not None:
                    self.objects[best_id].update({"cx": cx, "cy": cy, "label": label})
                    matched_ids.add(best_id)
                else:
                    self.objects[self.next_id] = {
                        "cx": cx, "cy": cy, "label": label, "crossed": False
                    }
                    matched_ids.add(self.next_id)
                    self.next_id += 1
        else:
            for cx, cy, label, conf in boxes:
                self.objects[self.next_id] = {
                    "cx": cx, "cy": cy, "label": label, "crossed": False
                }
                self.next_id += 1

        for oid in list(self.objects):
            if oid not in matched_ids:
                self.lost[oid] = self.lost.get(oid, 0) + 1
                if self.lost[oid] > self.max_lost:
                    del self.objects[oid]
                    self.lost.pop(oid, None)
            else:
                self.lost.pop(oid, None)

        crossed = []
        for oid, obj in self.objects.items():
            if not obj["crossed"] and obj["cy"] >= line_y:
                obj["crossed"] = True
                crossed.append((obj["label"], oid))

        return crossed


# ─── Overlay ───────────────────────────────────────────────────

def draw_overlay(frame, detections, tracker, line_y, counts, fps):
    H, W = frame.shape[:2]

    # Virtual line
    cv2.line(frame, (0, int(line_y)), (W, int(line_y)), COLOR_LINE, 2)
    cv2.putText(frame, "-- COUNT LINE --",
                (10, int(line_y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LINE, 1, cv2.LINE_AA)

    # Bounding boxes
    for det in detections:
        x1, y1, x2, y2, label, conf, oid = det
        cx      = (x1 + x2) // 2
        cy      = (y1 + y2) // 2
        crossed = tracker.objects.get(oid, {}).get("crossed", False)
        color   = COLOR_CROSSED if crossed else COLOR_BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Stats panel (top-left)
    panel_lines = [
        (f"FPS: {fps:.1f}",            (200, 200, 200)),
        ("=== TRASH COUNTS ===",        COLOR_COUNT),
    ]
    for label, cnt in sorted(counts.items()):
        panel_lines.append((f"  {label}: {cnt}", (100, 255, 100)))
    panel_lines.append((f"  TOTAL: {sum(counts.values())}", COLOR_TOTAL))

    pad, lh = 8, 22
    panel_h = len(panel_lines) * lh + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, panel_h), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (text, color) in enumerate(panel_lines):
        cv2.putText(frame, text,
                    (pad, pad + (i + 1) * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ─── Log ───────────────────────────────────────────────────────

def save_log(counts, log_file):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "counts":    dict(counts),
        "total":     sum(counts.values())
    }
    log = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
    log.append(entry)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[LOG] Saved to {log_file}")


# ─── Main ──────────────────────────────────────────────────────

def main():
    args      = parse_args()
    show_disp = DISPLAY and not args.headless

    print(f"[INFO] Loading model: {MODEL}")
    model = YOLO(MODEL)
    names = model.names   # {0: 'Sterofoam', 1: 'plastikk', ...}

    # Determine source
    if args.video and args.rtsp:
        raise ValueError("Use --video or --rtsp, not both.")
    if not args.video and not args.rtsp:
        raise ValueError("Provide either --video or --rtsp.")

    is_video_file = args.video is not None

    if is_video_file:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        print(f"[INFO] Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
        # Use the video's own FPS for the recorder so playback speed is correct
        src_fps = cap.get(cv2.CAP_PROP_FPS) or RECORD_FPS
    else:
        print(f"[INFO] Opening RTSP stream: {args.rtsp}")
        gst_pipeline = (
            f"rtspsrc location={args.rtsp} latency=100 ! "
            "rtph264depay ! h264parse ! omxh264dec ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("[WARN] GStreamer pipeline failed, trying direct RTSP...")
            cap = cv2.VideoCapture(args.rtsp)
        src_fps = RECORD_FPS

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source.")

    tracker     = CentroidTracker()
    counts      = defaultdict(int)
    recorder    = VideoRecorder(RECORD_DIR, src_fps, RECORD_SPLIT if not is_video_file else 0) if RECORD else None
    frame_count = 0
    fps         = 0.0
    t_start     = time.time()
    last_save   = time.time()

    print(f"[INFO] Virtual line at {args.line_pos * 100:.0f}% of frame height")
    if RECORD:
        print(f"[INFO] Recording to folder: {RECORD_DIR}/")
        if RECORD_SPLIT:
            print(f"[INFO] Auto-splitting every {RECORD_SPLIT} minutes")
    print("[INFO] Running — press Ctrl+C to stop\n")

    def open_cap(rtsp_url):
        """Try GStreamer first, fall back to direct RTSP."""
        gst = (
            f"rtspsrc location={rtsp_url} latency=200 ! "
            "rtph264depay ! h264parse ! omxh264dec ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink"
        )
        c = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if c.isOpened():
            return c
        print("[WARN] GStreamer failed, trying direct RTSP...")
        return cv2.VideoCapture(rtsp_url)

    fail_count = 0
    MAX_FAILS  = 30   # consecutive failed reads before reconnecting (RTSP only)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    print("[INFO] End of video file.")
                    break
                fail_count += 1
                print(f"[WARN] Failed to read frame ({fail_count}/{MAX_FAILS})...")
                time.sleep(0.2)
                if fail_count >= MAX_FAILS:
                    print("[WARN] Too many failures — reconnecting to stream...")
                    cap.release()
                    time.sleep(2)
                    cap = open_cap(args.rtsp)
                    if not cap.isOpened():
                        print("[ERROR] Reconnect failed, retrying in 5s...")
                        time.sleep(5)
                    else:
                        print("[INFO] Reconnected successfully.")
                    fail_count = 0
                continue
            fail_count = 0   # reset on successful read

            H, W   = frame.shape[:2]
            line_y = H * args.line_pos

            # YOLO inference
            results = model(frame,
                            conf=args.threshold,
                            verbose=False,
                            imgsz=IMGSZ)[0]

            # Parse detections
            boxes_for_tracker = []
            draw_dets         = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = names[cls]
                cx    = (x1 + x2) // 2
                cy    = (y1 + y2) // 2
                boxes_for_tracker.append((cx, cy, label, conf))
                draw_dets.append([x1, y1, x2, y2, label, conf, None])  # oid filled below

            crossed = tracker.update(boxes_for_tracker, line_y)
            for label, oid in crossed:
                counts[label] += 1
                print(f"[COUNT] {label} crossed line  |  "
                      f"Total {label}: {counts[label]}  |  "
                      f"Grand total: {sum(counts.values())}")

            # Attach object IDs to draw list (best-effort match by index)
            obj_list = list(tracker.objects.items())
            for i, det in enumerate(draw_dets):
                if i < len(obj_list):
                    det[6] = obj_list[i][0]

            # FPS
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                t_start     = time.time()

            draw_overlay(frame, draw_dets, tracker, line_y, counts, fps)

            if show_disp:
                cv2.imshow("Smart Trash Counter — YOLO", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Write annotated frame to recording
            if recorder:
                recorder.write(frame)

            # Auto-save log every 60 seconds
            if SAVE_LOG and (time.time() - last_save) >= 60:
                save_log(counts, LOG_FILE)
                last_save = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    finally:
        cap.release()
        if show_disp:
            cv2.destroyAllWindows()
        if recorder:
            recorder.release()
        if SAVE_LOG:
            save_log(counts, LOG_FILE)
        print("\n[RESULT] Final counts:")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
        print(f"  TOTAL: {sum(counts.values())}")


if __name__ == "__main__":
    main()
