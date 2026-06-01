from ultralytics import YOLO

class VehicleDetector:

    """""
It does one job: detect vehicles using YOLO and return bounding boxes
"""
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(self, model_size="yolov8m.pt"):
        self.model = YOLO(model_size)
        print("YOLO model loaded")

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id not in self.VEHICLE_CLASSES:
                continue
            if confidence < 0.25:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "width": x2 - x1,
                "height": y2 - y1,
                "class_id": class_id,
                "type": self.VEHICLE_CLASSES[class_id],
                "confidence": confidence,
                "is_emergency": False
            })

        return detections



