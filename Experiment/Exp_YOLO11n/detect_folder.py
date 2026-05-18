# this scripts detects objects in all images within a folder using a YOLO model 
# and saves the results with bounding boxes drawn on the images. 
# It also prints out the detection results and timing information for each image, 
# as well as a summary at the end.
# DCC Lab - HAM

from ultralytics import YOLO
import cv2
import time
import os
import glob

# -- Config ----------------------------------------------------------
MODEL      = "best.engine"
INPUT_DIR  = "Test_Image"
OUTPUT_DIR = "results_yolo"
THRESHOLD  = 0.5
# --------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL, task='detect')

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

print("Found %d images in %s\n" % (len(image_files), INPUT_DIR))

pre_total  = 0.0
inf_total  = 0.0
post_total = 0.0
total_det  = 0

for i, img_path in enumerate(image_files, 1):
    filename = os.path.basename(img_path)
    out_path = os.path.join(OUTPUT_DIR, "%04d_%s" % (i, filename))

    results = model(img_path, conf=THRESHOLD, verbose=False)
    r       = results[0]

    pre_total  += r.speed['preprocess']
    inf_total  += r.speed['inference']
    post_total += r.speed['postprocess']
    total_det  += len(r.boxes)

    total_ms = r.speed['preprocess'] + r.speed['inference'] + r.speed['postprocess']
    print("[%04d/%04d] %s  |  %.1f ms  |  %d detections" % (
        i, len(image_files), filename, total_ms, len(r.boxes)))

    for box in r.boxes:
        cls   = int(box.cls[0])
        conf  = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        label = r.names[cls]
        print("  %-20s conf=%.2f  box=(%d,%d,%d,%d)" % (
            label, conf, x1, y1, x2, y2))

    # Draw and save
    frame = cv2.imread(img_path)
    for box in r.boxes:
        cls   = int(box.cls[0])
        conf  = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        label = r.names[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 50), 2)
        cv2.putText(frame, "%s %.2f" % (label, conf),
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 50), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, frame)

n          = len(image_files)
avg_pre    = pre_total  / n
avg_inf    = inf_total  / n
avg_post   = post_total / n
avg_total  = avg_pre + avg_inf + avg_post

print("\n--- Summary ---")
print("Images processed : %d" % n)
print("Total detections : %d" % total_det)
print("Avg preprocess   : %.1f ms" % avg_pre)
print("Avg inference    : %.1f ms" % avg_inf)
print("Avg postprocess  : %.1f ms" % avg_post)
print("Avg total        : %.1f ms -> %.1f FPS" % (avg_total, 1000 / avg_total))
print("Results saved to : %s" % OUTPUT_DIR)
