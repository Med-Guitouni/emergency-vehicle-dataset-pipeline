from ultralytics import YOLO


class VehicleDetector:
    """
    Loads the YOLO model. That is its only job.


    """

    def __init__(self, model_size="yolov8x.pt"):
        self.model = YOLO(model_size)
        print(f"YOLO model loaded: {model_size}")



