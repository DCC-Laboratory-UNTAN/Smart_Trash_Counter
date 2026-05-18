# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Smart Trash Counter - SSD MobileNet V1
Using jetson-inference Docker on Jetson Nano

DCC lab - HAM

Usage (RTSP stream):
    python3 stream_test.py --rtsp=rtsp://user:pass@192.168.1.x:554/stream

Usage (video file):
    python3 stream_test.py --video=trash_test.mp4

Requirements (inside jetson-inference Docker):
    - jetson-inference
    - jetson-utils
    - Your trained model files:
        models/smart-trash-detector/ssd-mobilenet.onnx
        models/smart-trash-detector/labels.txt
"""

import jetson_inference
import jetson_utils
import argparse
import time
import json
import os
import math
import cv2
from datetime import datetime
from collections import defaultdict

# ==============================================================
# CONFIG - edit these
# ==============================================================
MODEL        = "ssd-mobilenet.onnx"
LABELS       = "labels.txt"
THRESHOLD    = 0.50      # detection confidence threshold
LINE_POS     = 0.55      # virtual line position (0.0=top, 1.0=bottom)
SAVE_LOG     = True      # save count log to JSON
LOG_FILE     = "trash_counts_ssd.json"
DISPLAY      = False     # show live output window - set False for headless

# -- Recording -------------------------------------------------
RECORD       = True      # save annotated video to disk
RECORD_DIR   = "recordings"  # folder where .mp4 files are saved
RECORD_FPS   = 15        # output FPS for RTSP (video file uses its own FPS)
RECORD_SPLIT = 10        # split every N minutes for RTSP (0 = no split)

# -- Rotation fix for portrait video via jetson_utils ----------
# jetson_utils ignores rotation metadata unlike cv2.
# Set to True only when using --video and the output appears rotated.
# For RTSP from a correctly mounted camera, leave False.
ROTATE_VIDEO = False      # rotate portrait video frames 90 degrees clockwise
# ==============================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp",      type=str, default=None,
                        help="RTSP stream URL")
    parser.add_argument("--video",     type=str, default=None,
                        help="Path to a local video file")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--line-pos",  type=float, default=LINE_POS,
                        help="Virtual line Y position as fraction of frame height (0.0-1.0)")
    parser.add_argument("--headless",  action="store_true",
                        help="Disable display window")
    return parser.parse_args()


# --------------------------------------------------------------
# Video Recorder
# --------------------------------------------------------------

class VideoRecorder:
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
        path   = os.path.join(self.out_dir, "trash_ssd_{}.mp4".format(ts))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("[REC] Cannot open VideoWriter for {}".format(path))
        print("[REC] Recording -> {}".format(path))
        self.current_file  = path
        self.segment_start = time.time()
        return writer

    def write(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        if self.writer is None:
            self.writer = self._new_writer(w, h)
        if self.split_sec and (time.time() - self.segment_start) >= self.split_sec:
            self.release()
            self.writer = self._new_writer(w, h)
        self.writer.write(bgr_frame)

    def release(self):
        if self.writer:
            self.writer.release()
            print("[REC] Saved -> {}".format(self.current_file))
            self.writer = None


# --------------------------------------------------------------
# Object Tracker
# --------------------------------------------------------------

class ObjectTracker:
    def __init__(self, max_lost=10, max_dist=80):
        self.next_id  = 0
        self.objects  = {}
        self.lost     = {}
        self.max_lost = max_lost
        self.max_dist = max_dist

    def update(self, detections, net, line_y, rotated_boxes=None):
        """
        detections    : list of jetson_inference.Detection (raw CUDA coords)
        net           : detectNet instance
        line_y        : crossing line Y in the coordinate space being tracked
        rotated_boxes : if provided, use these transformed coords instead of raw detections
        """
        if rotated_boxes is not None:
            current = []
            for b in rotated_boxes:
                cx = (b["x1"] + b["x2"]) / 2
                cy = (b["y1"] + b["y2"]) / 2
                current.append((cx, cy, b["label"]))
        else:
            current = []
            for d in detections:
                cx    = (d.Left + d.Right) / 2
                cy    = (d.Top + d.Bottom) / 2
                label = net.GetClassDesc(d.ClassID)
                current.append((cx, cy, label))

        matched_ids = set()
        if self.objects:
            for cx, cy, label in current:
                best_id, best_dist = None, self.max_dist
                for oid, obj in self.objects.items():
                    if oid in matched_ids:
                        continue
                    dist = math.hypot(cx - obj["cx"], cy - obj["cy"])
                    if dist < best_dist:
                        best_dist, best_id = dist, oid
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
            for cx, cy, label in current:
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

        crossed_this_frame = []
        for oid, obj in self.objects.items():
            if not obj["crossed"] and obj["cy"] >= line_y:
                obj["crossed"] = True
                crossed_this_frame.append((obj["label"], oid))

        return crossed_this_frame


# --------------------------------------------------------------
# Overlay - cv2 only (drawn on BGR frame for recording)
# --------------------------------------------------------------

def draw_overlay(bgr, boxes, line_y, counts, fps):
    """
    bgr   : BGR numpy frame to draw on
    boxes : list of dicts {x1,y1,x2,y2,label,conf} already in frame coordinate space
    """
    H, W = bgr.shape[:2]

    # Virtual line
    cv2.line(bgr, (0, int(line_y)), (W, int(line_y)), (50, 50, 255), 2)
    cv2.putText(bgr, "-- COUNT LINE --",
                (10, int(line_y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 1, cv2.LINE_AA)

    # Bounding boxes
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        label, conf     = b["label"], b["conf"]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (50, 200, 50), 2)
        cv2.putText(bgr, "{} {:.2f}".format(label, conf),
                    (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 50), 1, cv2.LINE_AA)

    # Stats panel
    panel_lines = [
        ("FPS: {:.1f}".format(fps),   (200, 200, 200)),
        ("=== TRASH COUNTS ===",       (255, 220, 50)),
    ]
    for label, cnt in sorted(counts.items()):
        panel_lines.append(("  {}: {}".format(label, cnt), (100, 255, 100)))
    panel_lines.append(("  TOTAL: {}".format(sum(counts.values())), (50, 220, 255)))

    pad, lh = 8, 22
    panel_h = len(panel_lines) * lh + pad * 2
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (240, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, bgr, 0.4, 0, bgr)
    for i, (text, color) in enumerate(panel_lines):
        cv2.putText(bgr, text,
                    (pad, pad + (i + 1) * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def rotate_detections_90cw(detections, net, orig_W, orig_H):
    """
    Transform detection bounding boxes from original frame space
    into 90-degree-clockwise-rotated frame space.

    Original (x, y) -> Rotated (orig_H - y, x)
    So a box (x1,y1,x2,y2) becomes:
        new_x1 = orig_H - y2
        new_y1 = x1
        new_x2 = orig_H - y1
        new_y2 = x2
    Returns list of dicts with keys: x1,y1,x2,y2,label,conf
    """
    result = []
    for d in detections:
        x1 = int(d.Left)
        y1 = int(d.Top)
        x2 = int(d.Right)
        y2 = int(d.Bottom)
        label = net.GetClassDesc(d.ClassID)
        conf  = d.Confidence
        rx1 = orig_H - y2
        ry1 = x1
        rx2 = orig_H - y1
        ry2 = x2
        result.append({"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2,
                        "label": label, "conf": conf})
    return result




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
    print("[LOG] Saved to {}".format(log_file))


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

def main():
    args         = parse_args()
    show_display = DISPLAY and not args.headless

    # Validate source
    if args.video and args.rtsp:
        raise ValueError("Use --video or --rtsp, not both.")
    if not args.video and not args.rtsp:
        raise ValueError("Provide either --video or --rtsp.")

    is_video_file = args.video is not None

    print("[INFO] Loading model: {}".format(MODEL))
    net = jetson_inference.detectNet(
        argv=[
            "--model={}".format(MODEL),
            "--labels={}".format(LABELS),
            "--input-blob=input_0",
            "--output-cvg=scores",
            "--output-bbox=boxes",
            "--threshold={}".format(args.threshold),
        ]
    )

    # Open source
    if is_video_file:
        if not os.path.exists(args.video):
            raise FileNotFoundError("Video file not found: {}".format(args.video))
        print("[INFO] Opening video file: {}".format(args.video))
        cap     = cv2.VideoCapture(args.video)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or RECORD_FPS
        print("[INFO] Video FPS: {:.4f}".format(src_fps))
        do_rotate  = ROTATE_VIDEO
        split_min  = 0
        input_stream = None
    else:
        print("[INFO] Opening RTSP stream: {}".format(args.rtsp))
        input_stream = jetson_utils.videoSource(args.rtsp, argv=["--input-codec=h264"])
        cap        = None
        src_fps    = RECORD_FPS
        do_rotate  = False
        split_min  = RECORD_SPLIT

    output_stream = jetson_utils.videoOutput("display://0") if show_display else None
    font          = jetson_utils.cudaFont()
    tracker       = ObjectTracker()
    counts        = defaultdict(int)
    recorder      = VideoRecorder(RECORD_DIR, src_fps, split_min) if RECORD else None

    frame_count = 0
    fps         = 0.0
    t_start     = time.time()
    last_save   = time.time()

    print("[INFO] Virtual line at {:.0f}% of frame height".format(args.line_pos * 100))
    if RECORD:
        print("[INFO] Recording to folder: {}/".format(RECORD_DIR))
    print("[INFO] Running - press Ctrl+C to stop\n")

    try:
        while True:

            # ── Read frame ──────────────────────────────────────
            if is_video_file:
                ret, bgr = cap.read()
                if not ret:
                    print("[INFO] End of video file.")
                    break
                if do_rotate:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                # Convert BGR -> CUDA RGBA for detectNet
                rgba  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
                img   = jetson_utils.cudaFromNumpy(rgba)
            else:
                img = input_stream.Capture()
                if img is None:
                    print("[WARN] No frame received, retrying...")
                    time.sleep(0.1)
                    continue
                bgr = None  # built later in recording block

            # ── Detection (always on CUDA img, already rotated if video) ──
            raw_H = img.height
            raw_W = img.width

            detections = net.Detect(img, overlay="none")

            # For video file: detections are already in rotated space (img was rotated)
            # For RTSP: no rotation, use raw coords directly
            if is_video_file:
                # detections already in correct space, build box dicts directly
                draw_boxes   = [{"x1": int(d.Left),  "y1": int(d.Top),
                                  "x2": int(d.Right), "y2": int(d.Bottom),
                                  "label": net.GetClassDesc(d.ClassID),
                                  "conf":  d.Confidence}
                                 for d in detections]
                track_line_y = raw_H * args.line_pos
                crossed = tracker.update(detections, net, track_line_y)
            else:
                draw_boxes   = [{"x1": int(d.Left),  "y1": int(d.Top),
                                  "x2": int(d.Right), "y2": int(d.Bottom),
                                  "label": net.GetClassDesc(d.ClassID),
                                  "conf":  d.Confidence}
                                 for d in detections]
                track_line_y = raw_H * args.line_pos
                crossed = tracker.update(detections, net, track_line_y)

            for label, obj_id in crossed:
                counts[label] += 1
                print("[COUNT] {} crossed line  |  Total {}: {}  |  Grand total: {}".format(
                    label, label, counts[label], sum(counts.values())))

            # ── FPS ─────────────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                t_start     = time.time()

            # ── Display ─────────────────────────────────────────
            if show_display and not is_video_file:
                output_stream.Render(img)
                if not input_stream.IsStreaming():
                    break

            # ── Recording ───────────────────────────────────────
            if recorder:
                if is_video_file:
                    rec_frame = bgr  # already correct orientation
                else:
                    rec_frame = jetson_utils.cudaToNumpy(img)
                    rec_frame = cv2.cvtColor(rec_frame, cv2.COLOR_RGBA2BGR)

                rec_line_y = rec_frame.shape[0] * args.line_pos
                draw_overlay(rec_frame, draw_boxes, rec_line_y, counts, fps)
                recorder.write(rec_frame)

            # ── Auto-save log (RTSP only) ────────────────────────
            if SAVE_LOG and not is_video_file and (time.time() - last_save) >= 60:
                save_log(counts, LOG_FILE)
                last_save = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    finally:
        if cap:
            cap.release()
        if recorder:
            recorder.release()
        if SAVE_LOG:
            save_log(counts, LOG_FILE)
        print("\n[RESULT] Final counts:")
        for label, count in sorted(counts.items()):
            print("  {}: {}".format(label, count))
        print("  TOTAL: {}".format(sum(counts.values())))


if __name__ == "__main__":
    main()
