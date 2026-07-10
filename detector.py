import torch
from ultralytics import YOLO


class VehicleDetector:
    """
    Loads the YOLO model and picks the compute device (CUDA if available,
    else CPU). self.device is read by tracker.py's update() to run tracking
    on the same device, and can be read by anything else that needs it.
    """

    def __init__(self, model_size="yolov8x.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_size)
        self.model.to(self.device)

        if self.device == "cuda":
            print(f"YOLO model loaded: {model_size}  [CUDA: {torch.cuda.get_device_name(0)}]")
        else:
            print(f"YOLO model loaded: {model_size}  [CPU -- no CUDA GPU detected]")



