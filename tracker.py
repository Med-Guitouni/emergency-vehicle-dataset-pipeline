import numpy as np
import cv2
from ultralytics import YOLO


class VehicleTracker:
    """
    YOLO detection + BoT-SORT tracking.

    WHY BoT-SORT INSTEAD OF BYTETRACK + EMAP
    -----------------------------------------
    At 1 Hz, the ambulance moves ~30 m between frames. ByteTrack matches
    detections purely by bounding-box overlap (IoU). At 1 Hz the predicted
    box position is almost never close enough to the new detection for a
    good IoU match, so the same vehicle gets a new ID every second.

    The previous fix (EMAP) was supposed to compensate for ego-motion before
    the Kalman predict step, but it ran AFTER ByteTrack's internal association
    had already finished — too late to improve matching at all.

    BoT-SORT (Aharon et al. 2022, arXiv 2206.14651) solves this correctly:
      1. ReID appearance model — matches vehicles by what they look like, not
         just where they are predicted to be. A vehicle that moved 30 m in
         1 s is still recognised by appearance and keeps its ID.
      2. Camera-motion compensation (CMC) via optical flow — built in and
         integrated before matching, unlike the broken EMAP setup.

    Usage: change tracker="bytetrack.yaml" -> tracker="botsort.yaml".
    Ultralytics ships botsort.yaml and the ReID weights; nothing extra to
    install. Track confidence and IoU thresholds remain the same.

    EDGE STRIP DETECTION
    --------------------
    Close-range vehicles that fill most of the left or right side of frame
    look like partial blobs to full-frame YOLO. Running a second detection
    pass on 40% side strips presents them at a more normal scale.

    Edge track IDs (9000+) are now PERSISTENT: each detection is matched
    against the previous frame's edge detections by IoU. If IoU > 0.25 the
    same ID is kept; otherwise a new one is issued. This gives edge tracks
    stable IDs across consecutive frames so the smoother and annotator can
    accumulate their history.
    """

    def __init__(self):
        # counter for synthetic IDs assigned to edge-strip detections.
        # starts at 9000 to avoid any conflict with BoT-SORT IDs (1-8999).
        self._edge_id_counter = 9000

        # previous frame's edge detections — list of {"track_id", "bbox"} —
        # used to assign stable IDs by IoU matching each frame.
        self._prev_edge = []

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

    def _detect_edge_vehicles(self, model, frame, existing_tracked):
        """
        Detect vehicles in the left and right 40% strips of the frame.

        Detections overlapping an already-tracked box (IoU > 0.30) are
        discarded to avoid double-counting.

        Each remaining detection is matched against self._prev_edge by IoU.
        Match threshold: 0.25. If matched, the previous frame's ID is reused;
        if not, a new synthetic ID is issued. self._prev_edge is updated at
        the end of the call for the next frame.
        """
        fh, fw = frame.shape[:2]
        STRIP_W = int(fw * 0.40)
        CONF = 0.20
        IOU_THR_EXISTING = 0.30
        IOU_THR_PREV = 0.25

        VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        existing_boxes = [v["bbox"] for v in existing_tracked]
        supplemental = []

        strips = [
            ("left",  0,           STRIP_W),
            ("right", fw - STRIP_W, fw),
        ]

        for _, x_start, x_end in strips:
            strip = frame[:, x_start:x_end]
            results = model.predict(strip, verbose=False, conf=CONF)[0]

            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASSES:
                    continue
                if float(box.conf[0]) < CONF:
                    continue

                sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
                full_box = [sx1 + x_start, sy1, sx2 + x_start, sy2]

                # skip if already captured by main tracking pass
                if any(self._iou(full_box, eb) > IOU_THR_EXISTING
                       for eb in existing_boxes):
                    continue

                # match against previous frame's edge detections for ID stability
                matched_id = None
                for prev in self._prev_edge:
                    if self._iou(full_box, prev["bbox"]) > IOU_THR_PREV:
                        matched_id = prev["track_id"]
                        break

                if matched_id is None:
                    self._edge_id_counter += 1
                    matched_id = self._edge_id_counter

                x1, y1, x2, y2 = full_box
                supplemental.append({
                    "track_id": matched_id,
                    "type":     VEHICLE_CLASSES[class_id],
                    "bbox":     full_box,
                    "center":   [(x1 + x2) // 2, (y1 + y2) // 2],
                })

        # store this frame's edge detections for next-frame matching
        self._prev_edge = [
            {"track_id": d["track_id"], "bbox": d["bbox"]}
            for d in supplemental
        ]
        return supplemental

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

        edge_detections = self._detect_edge_vehicles(model, frame, tracked)
        tracked.extend(edge_detections)
        return tracked

