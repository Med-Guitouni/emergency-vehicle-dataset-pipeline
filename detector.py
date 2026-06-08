from ultralytics import YOLO


class VehicleDetector:
    """
    Loads the YOLO model. That is its only job.

    NOTE: detection is actually run by tracker.py, which calls
    model.track(...) directly on the .model attribute below (ByteTrack needs
    to drive detection itself so it can keep track IDs across frames). So this
    class deliberately does NOT have a detect() method any more - it would
    never be called. main.py passes d.model straight into the tracker.
    """

    def __init__(self, model_size="yolov8x.pt"):
        self.model = YOLO(model_size)
        print(f"YOLO model loaded: {model_size}")



