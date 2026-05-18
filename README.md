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
- [Custom Object Training](#custom-object-training)
- [License](#license)
- [Team](#team)

---

## Overview

The **Smart Trash Counter** uses real-time object detection models deployed on an NVIDIA Jetson device to identify and count trash objects in a given environment. Two model architectures are supported:

- **SSD MobileNetV1** — via the [`jetson-inference`](https://github.com/dusty-nv/jetson-inference) framework
- **YOLOv8** — via the [Ultralytics](https://docs.ultralytics.com/) framework with TensorRT acceleration

Both models were trained using custom datasets sourced from [Roboflow](https://roboflow.com/) and can be fine-tuned using the provided training notebooks.

---

## Repository Structure

```
Smart_Trash_Counter/
├── TrainingNotebook/
│   ├── Test_Image/                   # Sample images for model validation
│   ├── SSDMobilenetV1_Roboflow.ipynb # Training notebook for SSD MobileNetV1
│   └── YOLO_Roboflow.ipynb           # Training notebook for YOLOv8
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

### 2. YOLOv8 — Ultralytics Docker

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
| `docker_yolo` | Start and enter the YOLOv8 container |
| `docker_ssd` | Start and enter the SSD MobileNet container |
| `stop_yolo` | Stop the YOLOv8 container |
| `stop_ssd` | Stop the SSD MobileNet container |

---

## Deploying Code to Containers

After Docker is set up, copy the necessary inference scripts and model files into the respective containers.

> 🚧 **Work in Progress** — Deployment instructions will be added in a future update.

---

## Custom Object Training

Training notebooks are located in the [`/TrainingNotebook`](./TrainingNotebook/) directory. They are optimized for **Google Colab** to take advantage of free GPU sessions.

| Notebook | Model | Framework |
|---|---|---|
| `SSDMobilenetV1_Roboflow.ipynb` | SSD MobileNetV1 | Jetson-Inference |
| `YOLO_Roboflow.ipynb` | YOLOv8 | Ultralytics |

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