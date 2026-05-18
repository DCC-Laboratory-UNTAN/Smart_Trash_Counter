# this script detects objects in all images within a folder using an SSD MobileNet V1 model
# and saves the results with bounding boxes drawn on the images.
# It also prints out the detection results and timing information for each image,
# as well as a summary at the end.
# DCC Lab - HAM

import jetson_inference
import jetson_utils
import time
import os
import glob

# -- NOTES ----------------------------------------------------------
# This code needs: 
#  - ssd-mobilenet.onnx (exported from TensorFlow)
#  - ssd-mobilenet.onnx.data (exported from TensorFlow)
#  - labels.txt (class labels for the model)

# -- Config ----------------------------------------------------------
MODEL      = "ssd-mobilenet.onnx"
LABELS     = "labels.txt"
INPUT_DIR  = "Test_Image"
OUTPUT_DIR = " results_ssd"
THRESHOLD  = 0.5
# --------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read labels
with open(LABELS, "rb") as f:
    raw_lines = f.read().splitlines()

class_names = []
for line in raw_lines:
    label = line.decode("utf-8", errors="ignore").strip()
    label = label.encode("ascii", errors="ignore").decode("ascii").strip()
    class_names.append(label if label else "class_%d" % len(class_names))

print("Loaded %d labels:" % len(class_names))
for idx, name in enumerate(class_names):
    print("  [%d] %s" % (idx, name))

net = jetson_inference.detectNet(
    network="custom",
    argv=[
        f"--model={MODEL}",
        f"--labels={LABELS}",
        "--input-blob=input_0",
        "--output-cvg=scores",
        "--output-bbox=boxes",
        f"--threshold={THRESHOLD}",
    ]
)

# Collect images
exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_files = []
for ext in exts:
    image_files += glob.glob(os.path.join(INPUT_DIR, ext))
    image_files += glob.glob(os.path.join(INPUT_DIR, ext.upper()))
image_files = sorted(set(image_files))

if len(image_files) == 0:
    print("No images found in: %s" % INPUT_DIR)
    exit(1)

print("\nFound %d images in %s\n" % (len(image_files), INPUT_DIR))

total_time   = 0.0
total_det    = 0

for i, img_path in enumerate(image_files, 1):
    filename = os.path.basename(img_path)
    out_path = os.path.join(OUTPUT_DIR, "%04d_%s" % (i, filename))

    img = jetson_utils.loadImage(img_path)

    t0         = time.perf_counter()
    detections = net.Detect(img, overlay="box,labels,conf")
    t1         = time.perf_counter()

    elapsed_ms  = (t1 - t0) * 1000
    total_time += elapsed_ms
    total_det  += len(detections)

    print("[%04d/%04d] %s  |  %.1f ms  |  %d detections" % (
        i, len(image_files), filename, elapsed_ms, len(detections)))

    for d in detections:
        cid   = d.ClassID
        label = class_names[cid] if cid < len(class_names) else "class_%d" % cid
        print("  %-20s conf=%.2f  box=(%d,%d,%d,%d)" % (
            label, d.Confidence,
            int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
        ))

    jetson_utils.saveImage(out_path, img)

avg_ms = total_time / len(image_files)
print("\n--- Summary ---")
print("Images processed : %d" % len(image_files))
print("Total detections : %d" % total_det)
print("Avg inference    : %.1f ms -> %.1f FPS" % (avg_ms, 1000 / avg_ms))
print("Results saved to : %s" % OUTPUT_DIR)
