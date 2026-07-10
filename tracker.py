import numpy as np
import cv2
from ultralytics import YOLO


class VehicleTracker:
    """
    YOLO detection + BoT-SORT tracking.


    -----------------------------------------


    The previous fix (EMAP) was supposed to compensate for ego-motion before
    the Kalman predict step
    BoT-SORT (Aharon et al. 2022, arXiv 2206.14651) solves this correctly:
      1. ReID appearance model — matches vehicles by what they look like, not
         just where they are predicted to be. A vehicle that moved 30 m in
         1 s is still recognised by appearance and keeps its ID.
      2. Camera-motion compensation (CMC) via optical flow — built in and
         integrated before matching, unlike the broken EMAP setup.

    Usage: change tracker="bytetrack.yaml" -> tracker="botsort.yaml".
    Ultralytics ships botsort.yaml and the ReID weights; nothing extra to
    install. Track confidence and IoU thresholds remain the same.
    """

    def __init__(self):
        print("BoT-SORT tracker ready")

    @staticmethod
    def _iou(boxA, boxB):
        """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(aA + aB - inter)

    def update(self, model, frame):
        """
        Run one tracking step on the current frame.

        model: loaded YOLOv8 model from detector.py
        frame: spatially cropped BGR frame

        Returns list of dicts, one per tracked vehicle:
            track_id, type, bbox [x1,y1,x2,y2], center [cx,cy]
        """
        results = model.track(
            frame,
            tracker="botsort.yaml",
            persist=True,
            verbose=False,
        )[0]

        VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        tracked = []
        for box in results.boxes:
            if box.id is None:
                continue
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASSES:
                continue
            if float(box.conf[0]) < 0.25:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tracked.append({
                "track_id": int(box.id[0]),
                "type":     VEHICLE_CLASSES[class_id],
                "bbox":     [x1, y1, x2, y2],
                "center":   [(x1 + x2) // 2, (y1 + y2) // 2],
            })

        return tracked
