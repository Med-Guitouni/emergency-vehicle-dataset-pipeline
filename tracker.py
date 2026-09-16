class VehicleTracker:
    """
    YOLOv8x detection + BoT-SORT tracking at TRACK_FPS (30 Hz, see main.py).

  


      1. ReID appearance model -- a vehicle is recognised by what it looks
         like from frame to frame, so association does not depend on
         modelling the camera's own motion at all.
      2. Camera-motion compensation via sparse optical flow, built in and
         applied before matching, and lighter-weight than a full
         depth-conditioned correction.

    Config lives in botsort.yaml; Ultralytics ships the tracker and its ReID
    weights, so nothing extra needs installing.
    """

    def __init__(self):
        print("BoT-SORT tracker ready")

    def update(self, model, frame, device=None):
        """
        Run one tracking step on the current frame.

        model: loaded YOLOv8 model from detector.py
        frame: spatially cropped BGR frame
        device: "cuda" or "cpu" -- pass detector.py's VehicleDetector.device
                here so tracking runs on the same device the model was
                loaded to. If None, ultralytics falls back to its own
                auto-detection (usually fine, but explicit is safer).

        Returns list of dicts, one per tracked vehicle:
            track_id, type, bbox [x1,y1,x2,y2], center [cx,cy]
        """
        results = model.track(
            frame,
            tracker="botsort.yaml",
            persist=True,
            verbose=False,
            device=device,
        )[0]

        VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        MIN_CONFIDENCE = 0.25

        tracked = []
        for box in results.boxes:
            if box.id is None:
                continue
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASSES:
                continue
            if float(box.conf[0]) < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tracked.append({
                "track_id": int(box.id[0]),
                "type":     VEHICLE_CLASSES[class_id],
                "bbox":     [x1, y1, x2, y2],
                "center":   [(x1 + x2) // 2, (y1 + y2) // 2],
            })

        return tracked
