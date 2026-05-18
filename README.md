# 🗑️ Smart Trash Counter

> A collaborative project between the **Electrical Engineering** and **Environmental Engineering** departments, leveraging computer vision to automate trash detection and counting using SSD MobileNet and YOLOv8 on NVIDIA Jetson hardware.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
  - [SSD MobileNet (Jetson-Inference Docker)](#1-ssd-mobilenet--jetson-inference-docker)
  - [YOLOv8 (Ultralytics Docker)](#2-yolov8--ultralytics-docker)
- [Docker Convenience Aliases](#docker-convenience-aliases)
- [Running the Experiments](#running-the-experiments)
  - [YOLO11n Experiments](#yolo11n-experiments)
  - [SSD MobileNetV1 Experiments](#ssd-mobilenetv1-experiments)
- [Custom Object Training](#custom-object-training)
- [License](#license)
- [Team](#team)

---

## Overview

The **Smart Trash Counter** uses real-time object detection models deployed on an NVIDIA Jetson device to identify and count trash objects in a given environment. Two model architectures are supported:

- **SSD MobileNetV1** — via the [`jetson-inference`](https://github.com/dusty-nv/jetson-inference) framework
- **YOLOv8 / YOLO11n** — via the [Ultralytics](https://docs.ultralytics.com/) framework with TensorRT acceleration

Both models were trained using custom datasets sourced from [Roboflow](https://roboflow.com/) and can be fine-tuned using the provided training notebooks.

---

## Repository Structure

```
Smart_Trash_Counter/
├── Experiment/
│   ├── Exp_YOLO11n/
│   |   ├── Test_Image/            # Sample images for model validation
│   │   ├── detect_folder.py       # Batch image detection using YOLO TensorRT engine
│   │   ├── model_converter.py     # Converts YOLO .pt model to TensorRT .engine format
│   │   └── stream_test.py         # Real-time trash counting from RTSP stream or video file
│   └── Exp_SSDMobilenetV1/
│       ├── Test_Image/            # Sample images for model validation
│       ├── detect_image_folder.py # Batch image detection using SSD MobileNet
│       └── stream_test.py         # Real-time trash counting from RTSP stream or video file
├── TrainingNotebook/
│   ├── Test_Image/                # Sample images for model validation
│   ├── SSDMobilenetV1_Roboflow.ipynb  # Training notebook for SSD MobileNetV1
│   └── YOLO_Roboflow.ipynb            # Training notebook for YOLOv8 / YOLO11n
├── LICENSE
└── README.md
```

---

## Environment Setup

> **Prerequisites:** An NVIDIA Jetson device flashed with [JetPack](https://github.com/dusty-nv/jetson-inference/blob/master/docs/jetpack-setup-2.md). Both setups below assume JetPack is already configured.

### 1. SSD MobileNet — Jetson-Inference Docker

Follow the official [Jetson-Inference](https://github.com/dusty-nv/jetson-inference/tree/master) setup guide.

**Step 1:** Set up the Jetson with JetPack by following the [official guide](https://github.com/dusty-nv/jetson-inference/blob/master/docs/jetpack-setup-2.md).

**Step 2:** Clone the repository and launch the Docker container:

```bash
git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
cd jetson-inference
docker/run.sh
```

---

### 2. YOLOv8 / YOLO11n — Ultralytics Docker

Follow the official [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson#use-tensorrt-on-nvidia-jetson).

**Step 1:** Ensure JetPack is installed (see SSD setup Step 1 if not done).

**Step 2:** Pull and run the Ultralytics Docker container:

```bash
t=ultralytics/ultralytics:latest-jetson-jetpack4
sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
```

---

## Docker Convenience Aliases

To simplify starting, accessing, and stopping Docker containers, you can set up shell aliases in `.bashrc`.

**Step 1:** Find your container names by running:

```bash
sudo docker ps -a
```

The container name appears in the rightmost column (e.g., `clever_noyce` for YOLO, `jetson-inference` for SSD). Replace the names in Step 2 if yours differ.

**Step 2:** Append the aliases to `.bashrc`:

```bash
cat << 'EOF' >> ~/.bashrc

# Docker aliases — Smart Trash Counter
alias docker_yolo='docker start clever_noyce && docker exec -it clever_noyce /bin/bash'
alias docker_ssd='docker start jetson-inference && docker exec -it jetson-inference /bin/bash'
alias stop_yolo='docker stop clever_noyce'
alias stop_ssd='docker stop jetson-inference'
EOF

source ~/.bashrc
```

**Step 3:** Add your user to the `docker` group to avoid using `sudo` each time:

```bash
sudo usermod -aG docker $USER
```

**Step 4:** Reboot the device to apply group changes:

```bash
sudo reboot
```

**Step 5:** After rebooting, use the following commands to manage your containers:

| Command | Description |
|---|---|
| `docker_yolo` | Start and enter the YOLOv8 / YOLO11n container |
| `docker_ssd` | Start and enter the SSD MobileNet container |
| `stop_yolo` | Stop the YOLOv8 container |
| `stop_ssd` | Stop the SSD MobileNet container |

---

## Running the Experiments

All experiment scripts are located in the [`/Experiment`](./Experiment/) directory. Copy the relevant scripts and your trained model files into the appropriate Docker container before running.

> **Note:** Edit the `CONFIG` section at the top of each script to point to your model file, input directory, and adjust thresholds as needed.

---

### YOLO11n Experiments

All scripts below must be run **inside the Ultralytics Docker container**.

#### Step 0 — Convert your trained model to TensorRT (run once)

Place your `best.pt` file in the same directory as the script, then run:

```bash
python3 model_converter.py
```

This exports `best.pt` to `best.engine` (TensorRT FP16), optimized for inference on Jetson. The output file is used by all other YOLO scripts.

---

#### Batch Image Detection

Runs detection on all images in a folder and saves annotated results.

```bash
python3 detect_folder.py
```

Expected folder layout:

```
Exp_YOLO11n/
├── best.engine       # TensorRT model
├── Test_Image/       # Input images (.jpg / .png / .bmp)
├── detect_folder.py
└── results_yolo/     # Output images (auto-created)
```

Detection results and per-image timing are printed to the terminal. A summary of average FPS and total detections is shown at the end.

---

#### Real-Time Stream Detection & Counting

Counts trash objects crossing a virtual line in a live RTSP stream or video file.

```bash
# From an RTSP stream
python3 stream_test.py --rtsp=rtsp://user:pass@192.168.1.x:554/stream

# From a local video file
python3 stream_test.py --video=trash_test.mp4
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--threshold` | `0.50` | Detection confidence threshold |
| `--line-pos` | `0.55` | Virtual counting line position (0.0 = top, 1.0 = bottom) |
| `--headless` | off | Disable the OpenCV display window |

Annotated video is saved to the `recordings/` folder. Count logs are saved to `trash_counts_yolo.json`.

---

### SSD MobileNetV1 Experiments

All scripts below must be run **inside the Jetson-Inference Docker container**. Place your `ssd-mobilenet.onnx`, `ssd-mobilenet.onnx.data` and `labels.txt` model files in the same directory as the scripts.

---

#### Batch Image Detection

Runs detection on all images in a folder and saves annotated results.

```bash
python3 detect_image_folder.py
```

Expected folder layout:

```
Exp_SSDMobilenetV1/
├── ssd-mobilenet.onnx   # Trained SSD model
├── labels.txt           # Class labels
├── Test_Image/          # Input images
├── detect_image_folder.py
└── results_ssd/         # Output images (auto-created)
```

---

#### Real-Time Stream Detection & Counting

Counts trash objects crossing a virtual line in a live RTSP stream or video file.

```bash
# From an RTSP stream
python3 stream_test.py --rtsp=rtsp://user:pass@192.168.1.x:554/stream

# From a local video file
python3 stream_test.py --video=trash_test.mp4
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--threshold` | `0.50` | Detection confidence threshold |
| `--line-pos` | `0.55` | Virtual counting line position (0.0 = top, 1.0 = bottom) |
| `--headless` | off | Disable the display window |

Annotated video is saved to the `recordings/` folder. Count logs are saved to `trash_counts_ssd.json`.

---

## Custom Object Training

Training notebooks are located in the [`/TrainingNotebook`](./TrainingNotebook/) directory. They are optimized for **Google Colab** to take advantage of free GPU sessions.

| Notebook | Model | Framework |
|---|---|---|
| `SSDMobilenetV1_Roboflow.ipynb` | SSD MobileNetV1 | Jetson-Inference |
| `YOLO_Roboflow.ipynb` | YOLOv8 / YOLO11n | Ultralytics |

> **Note:** Both notebooks assume the dataset is fetched via the **Roboflow API**. Adapt the data-loading section if using a different dataset source.

---

## 📝 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## Team

| Role | Details |
|---|---|
| **Repository** | Smart Trash Counter |
| **Maintained By** | DCC Laboratory Development Team × Environmental Engineering Team |
| **Person in Charge** | HAM, ... |
| **Year** | 2026 |