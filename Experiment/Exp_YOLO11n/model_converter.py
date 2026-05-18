# This script loads a YOLO26n PyTorch model and exports it to TensorRT format.
from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("best.pt")

# Export the model to TensorRT
model.export(format="engine", half=True, imgsz=320)  # creates 'yolo11n.engine'
