from ultralytics import YOLO

class VehicleTracker:

    def __init__(self):
        print("ByteTrack tracker ready")

    def update(self, model, frame):
        """Use YOLOv8 built-in ByteTrack tracking"""
        results = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)[0]
        tracked = []

        for box in results.boxes:
            if box.id is None:
                continue
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
            if class_id not in VEHICLE_CLASSES:
                continue
            if confidence < 0.25:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            tracked.append({
                "track_id": int(box.id[0]),
                "type": VEHICLE_CLASSES[class_id],
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy]
            })

        return tracked

