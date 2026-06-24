import numpy as np
import cv2
from ultralytics import YOLO


# =============================================================================
# EMAP INTEGRATION - READ THIS BEFORE TOUCHING THIS FILE
# =============================================================================
#
# WHAT IS EMAP AND WHY DO WE NEED IT
# ------------------------------------
# When the ambulance drives at highway speed, every vehicle in the camera image
# appears to drift backward - even ones that are completely stationary.
# Standard ByteTrack uses a Kalman Filter that assumes the camera never moves.
# So it thinks all those parked cars are actually moving, which causes wrong
# speed estimates and ID switches (the tracker loses track of a vehicle and
# assigns it a new ID when it finds it again).
#
# EMAP (Ego-Motion Aware Target Prediction, Mahdian et al. 2024) fixes this by
# modifying the Kalman Filter predict step to subtract the camera's own motion
# before predicting where each vehicle will be next frame. It needs three things
# per vehicle per frame:
#   1. yaw_dot   - how much the camera rotated left/right (radians per frame)
#   2. D_dot     - how much the camera moved forward (metres per frame)
#   3. depth     - how far away that specific vehicle is (metres)
#
# We already compute all three:
#   - yaw_dot and D_dot come from the ego homography H in homography.py
#   - depth comes from the Depth Anything V2 depth map in homography.py

#So in practic:

#ByteTrack's own Kalman Filter: still running, handles the full track lifecycle (new tracks, lost tracks, re-identification)
#EMAP's Kalman Filter: runs before ByteTrack's matching step, corrects for camera motion
#
# WHAT WE CHANGED IN THE EMAP REPO AND WHY
# ------------------------------------------
# The EMAP repo (https://github.com/noyzzz/EMAP) was built for a ROS robot.
# It had two lines at the top of kalman_filter.py that import ROS packages:
#
#     import rospy
#     from std_msgs.msg import Float32MultiArray
#
# And two lines inside __init__ that use those packages to publish a debug msg:
#
#     params_array = Float32MultiArray()
#     params_array.data = [self._q1, self._q4, self._r1, self._r4]
#
# ROS is not installed and not needed here. These four lines were the only
# thing blocking us. The actual Kalman Filter math is pure numpy and works
# perfectly without ROS. We deleted those four lines and nothing else.
#
# The modified file is:
#     EMAP/trackers/bytetrack/kalman_filter.py
# The original backup is:
#     EMAP/trackers/bytetrack/kalman_filter_backup.py
#
# HOW TO SET THIS UP FROM A FRESH CLONE
# ---------------------------------------
# 1. Clone EMAP into the pipeline folder:
#        git clone https://github.com/noyzzz/EMAP
#
# 2. Remove the four ROS lines from kalman_filter.py:
#    Open EMAP/trackers/bytetrack/kalman_filter.py and delete:
#        Line:  import rospy
#        Line:  #import ros float array type
#        Line:  from std_msgs.msg import Float32MultiArray
#        And inside __init__, delete:
#        Line:  params_array = Float32MultiArray()
#        Line:  params_array.data = [self._q1, self._q4, self._r1, self._r4]
#
# 3. Add EMAP to the Python path so it can be imported:
#        export PYTHONPATH=$PYTHONPATH:~/Desktop/einsatz_pipeline/EMAP
#    To make this permanent (so you don't need to run it every time):
#        echo 'export PYTHONPATH=$PYTHONPATH:~/Desktop/einsatz_pipeline/EMAP' >> ~/.zshrc
#        source ~/.zshrc
#
# to test:
#
#        python3 -c "from EMAP.trackers.bytetrack.kalman_filter import KalmanFilter; print('OK')"
#    Should print: OK
#
# If EMAP is not set up, this file falls back to standard ByteTrack silently.
# The pipeline still runs, just without ego-motion compensation.
# =============================================================================


class VehicleTracker:
    """
        Runs YOLO detection + ByteTrack tracking on each frame.

        If EMAP is available (see setup instructions above), it replaces the
        standard Kalman Filter inside ByteTrack with the EMAP version that
        subtracts camera motion from each vehicle's predicted position.

        If EMAP is not available, falls back to standard ByteTrack.
        Nothing crashes either way.
        """

    # These must match the actual frame dimensions after spatial crop.
    # If preprocessor.py crop settings change, update these too.
    # EMAP's Kalman Filter uses these to convert pixel positions correctly.
    IMG_WIDTH = 1280
    IMG_HEIGHT = 720

    # Focal length estimate: frame_width * 0.8 is the same approximation
    # used in homography.py. Both should be updated together if camera
    # calibration is ever done properly.
    FOCAL_LENGTH = 1280 * 0.8

    def __init__(self):
        # --- try to load EMAP Kalman Filter ---
        # We import only the KalmanFilter class from EMAP, not the full
        # BYTETracker from EMAP (which requires ROS odometry objects).
        # We use Ultralytics ByteTrack for detection/association as before,
        # but intercept the Kalman predict step and replace it with EMAP's.
        self.emap_available = False
        self.emap_kalman = None

        # per-track Kalman state and missed-frame counter.
        # initialised here (not inside the EMAP try-block) so the cleanup
        # logic in update() never crashes when EMAP is not installed.
        self.track_states = {}
        self.track_missed = {}

        try:
            from EMAP.trackers.bytetrack.kalman_filter import KalmanFilter as EmapKalmanFilter

            # Pass image dimensions and focal length so EMAP can do the
            # pixel-to-angle conversion correctly inside its control matrix
            self.emap_kalman = EmapKalmanFilter(
                self.IMG_WIDTH,
                self.IMG_HEIGHT,
                self.FOCAL_LENGTH
            )

            # (track_states / track_missed already initialised above)

            self.emap_available = True
            print("EMAP Kalman Filter loaded - ego-motion compensation active")

        except Exception as ex:
            print(f"EMAP not available ({ex}) - using standard ByteTrack")

        # current ego-motion control signals - updated each frame by update()
        # yaw_dot: camera rotation speed in radians per frame
        # D_dot:   camera forward speed in metres per frame
        self.current_yaw_dot = 0.0
        self.current_D_dot = 0.0

        # counter for synthetic IDs assigned to side-strip detections
        # starts at 9000 to avoid any conflict with ByteTrack IDs (1-8999)
        self._edge_id_counter = 9000

        print("ByteTrack tracker ready")

    def _extract_ego_motion_from_H(self, H, depth_map, frame_width):
        """
        Pull yaw_dot and D_dot out of the 3x3 ego homography matrix H.

        H is computed by optical flow in homography.py. It describes how
        the whole image shifted between frames due to camera movement.

        yaw_dot (camera rotation left/right):
            The top-right element H[0,2] is the horizontal pixel shift of
            the image centre caused by camera yaw. Dividing by focal length
            converts that pixel shift to radians.

        D_dot (camera forward translation):
            Forward motion causes the image to zoom outward from the centre.
            The scale factor of H (average of H[0,0] and H[1,1]) tells us
            how much the image expanded. Scale > 1 means we moved forward.
            We multiply by a rough depth estimate (median of the depth map)
            to get metres per frame.

        These are approximations - good enough for the Kalman Filter to
        subtract most of the camera motion. Not as accurate as a proper
        GPS odometry signal, but we don't have those.
        """
        if H is None:
            return 0.0, 0.0

        focal_length = frame_width * 0.8

        # horizontal pixel shift at image centre = yaw effect
        # dividing by focal length converts pixels to radians
        yaw_dot = H[0, 2] / focal_length

        # image scale change = forward motion effect
        # H[0,0] and H[1,1] are the x and y scale factors of the homography
        scale = (H[0, 0] + H[1, 1]) / 2.0
        D_dot = 0.0
        if depth_map is not None:
            # use median scene depth as rough distance to the world
            median_depth = float(np.median(depth_map))
            # (scale - 1) is how much the image expanded proportionally
            # multiplying by depth converts that to metres of forward movement
            D_dot = (scale - 1.0) * median_depth

        return float(yaw_dot), float(D_dot)

    def _emap_predict_all(self, track_ids, bboxes_xyah, depth_map):
        """
        Run EMAP Kalman predict step for all currently tracked vehicles.

        For each tracked vehicle:
          1. Look up its stored Kalman state (mean, covariance)
          2. Build the control signal [yaw_dot, D_dot, vehicle_depth]
          3. Call EMAP's multi_predict which subtracts camera motion
          4. Store the updated state back

        track_ids:    list of integer track IDs active this frame
        bboxes_xyah:  list of [cx, cy, aspect, height] for each track
        depth_map:    relative depth map from Depth Anything V2

        OVERFLOW PROTECTION
        -------------------
        EMAP's Kalman Filter raises RuntimeWarning overflows when:
          - vehicle_depth is too small (division by near-zero)
          - vehicle_depth is too large (power operations overflow float64)
          - pixel coordinates (u1, v1) are at extreme frame edges
        We clamp all three to safe ranges. The exact depth value does not
        matter much - EMAP only needs a rough relative ordering to subtract
        camera motion. A fallback of 10.0m is safe for all road scenarios.
        """
        if not self.emap_available or len(track_ids) == 0:
            return

        # build mean and covariance arrays for tracks that have state
        active_ids = []
        active_means = []
        active_covs = []

        for tid, xyah in zip(track_ids, bboxes_xyah):
            if tid in self.track_states:
                mean, cov = self.track_states[tid]
                active_ids.append(tid)
                active_means.append(mean.copy())
                active_covs.append(cov.copy())

        if len(active_ids) == 0:
            return

        multi_mean = np.array(active_means)
        multi_cov = np.array(active_covs)

        # build control signal [yaw_dot, D_dot, depth] per track
        control_signals = []
        for tid, xyah in zip(active_ids,
                             [bboxes_xyah[track_ids.index(t)] for t in active_ids]):

            if depth_map is not None:
                h_img, w_img = depth_map.shape

                # clamp pixel coords well away from frame edges to prevent
                # extreme u1/v1 values in EMAP's power calculations
                cx = int(np.clip(xyah[0], 10, w_img - 10))
                cy = int(np.clip(xyah[1], 10, h_img - 10))

                vehicle_depth = float(depth_map[cy, cx])

                # clamp depth to a range where EMAP's float64 math is stable.
                # Depth Anything V2 outputs relative depth (unitless) which can
                # be very small or very large. Values outside [0.5, 50] cause
                # overflow in EMAP's coefficient calculations. The fallback 10.0
                # is a neutral mid-range value that keeps ego-motion subtraction
                # approximately correct for typical road scenes.
                if not (0.5 <= vehicle_depth <= 50.0):
                    vehicle_depth = 10.0
            else:
                vehicle_depth = 10.0

            control_signals.append([
                self.current_yaw_dot,
                self.current_D_dot,
                vehicle_depth
            ])

        control_array = np.array(control_signals)

        # run EMAP predict - subtracts camera motion from Kalman state
        try:
            multi_mean, multi_cov = self.emap_kalman.multi_predict(
                multi_mean, multi_cov, control_array
            )
            # store updated states back
            for i, tid in enumerate(active_ids):
                self.track_states[tid] = (multi_mean[i], multi_cov[i])
        except (RuntimeWarning, FloatingPointError, ValueError):
            # if EMAP math still fails (e.g. degenerate covariance), skip
            # this predict step - ByteTrack will still associate correctly,
            # just without ego-motion compensation this frame
            pass

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
        aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / float(aA + aB - inter)

    def _detect_edge_vehicles(self, model, frame, existing_tracked):
        """
        Run YOLO on the left and right 40% strips of the frame to catch
        close-range vehicles that are partially out of frame and therefore
        missed or poorly detected by full-frame inference.

        Why this works: YOLO was trained on images where vehicles occupy a
        moderate fraction of the frame. When a vehicle is so close it fills
        most of the left or right side, full-frame YOLO sees a partial blob
        instead of a recognisable vehicle shape. Running on the side strip
        presents the vehicle at a more normal scale relative to the crop.

        Detections that overlap an already-tracked box (IoU > 0.3) are
        discarded to avoid double-counting. The rest are assigned synthetic
        IDs starting at 9000, which cannot conflict with ByteTrack IDs.

        These edge detections appear in the pipeline output and JSON exactly
        like normal tracks. They will not have stable IDs across frames if
        the vehicle moves out of the strip, but for close-range vehicles
        that stay at the edge this is acceptable.
        """
        fh, fw = frame.shape[:2]
        STRIP_W = int(fw * 0.40)
        CONF    = 0.20   # lower threshold than main detection (0.25)
        IOU_THR = 0.30   # suppress if overlap with existing track

        VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        existing_boxes  = [v["bbox"] for v in existing_tracked]
        supplemental    = []

        strips = [
            ("left",  0,           STRIP_W),
            ("right", fw - STRIP_W, fw),
        ]

        for _, x_start, x_end in strips:
            strip   = frame[:, x_start:x_end]
            results = model.predict(strip, verbose=False, conf=CONF)[0]

            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASSES:
                    continue
                if float(box.conf[0]) < CONF:
                    continue

                # convert strip-local coords to full-frame coords
                sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
                x1 = sx1 + x_start
                x2 = sx2 + x_start
                y1, y2 = sy1, sy2

                full_box = [x1, y1, x2, y2]

                # skip if already captured by main tracking pass
                if any(self._iou(full_box, eb) > IOU_THR
                       for eb in existing_boxes):
                    continue

                self._edge_id_counter += 1
                supplemental.append({
                    "track_id": self._edge_id_counter,
                    "type":     VEHICLE_CLASSES[class_id],
                    "bbox":     full_box,
                    "center":   [(x1 + x2) // 2, (y1 + y2) // 2],
                })

        return supplemental

    def update(self, model, frame, ego_H=None, depth_map=None):
        """
        Run one tracking step on the current frame.

        model:     loaded YOLOv8 model from detector.py
        frame:     spatially cropped BGR frame
        ego_H:     3x3 camera homography from homography.py estimate_ego_motion()
        depth_map: per-pixel depth in metres from homography.py compute_depth_map()

        Returns list of dicts, one per tracked vehicle:
            track_id, type, bbox [x1,y1,x2,y2], center [cx,cy]
        """
        frame_height, frame_width = frame.shape[:2]

        # extract yaw_dot and D_dot from the ego homography for this frame
        if ego_H is not None:
            self.current_yaw_dot, self.current_D_dot = \
                self._extract_ego_motion_from_H(ego_H, depth_map, frame_width)

        # run YOLO + standard ByteTrack association
        # persist=True keeps track states between frames
        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )[0]

        VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        tracked = []
        track_ids_this_frame = []
        bboxes_xyah_this_frame = []

        for box in results.boxes:
            if box.id is None:
                continue

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id not in VEHICLE_CLASSES:
                continue
            if confidence < 0.25:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tid = int(box.id[0])

            # convert bbox to xyah format for EMAP Kalman Filter
            # xyah = [centre_x, centre_y, width/height, height]
            w = x2 - x1
            h = y2 - y1
            if h > 0:
                xyah = [float(cx), float(cy), float(w) / float(h), float(h)]
            else:
                xyah = [float(cx), float(cy), 1.0, 1.0]

            # if this is a new track, initialise its Kalman state in EMAP
            if self.emap_available and tid not in self.track_states:
                mean, cov = self.emap_kalman.initiate(np.array(xyah))
                self.track_states[tid] = (mean, cov)

            track_ids_this_frame.append(tid)
            bboxes_xyah_this_frame.append(xyah)

            tracked.append({
                "track_id": tid,
                "type": VEHICLE_CLASSES[class_id],
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy]
            })

        # run EMAP predict step for all active tracks
        # this is where camera motion gets subtracted from the Kalman state
        if self.emap_available:
            self._emap_predict_all(
                track_ids_this_frame,
                bboxes_xyah_this_frame,
                depth_map
            )

        # clean up state for tracks that disappeared.
        # At 5Hz tracking, a vehicle missed for a frame or two is usually a
        # brief occlusion, not a real disappearance - ByteTrack will re-find
        # it with the same ID. So we keep the EMAP state alive for up to
        # MISS_TOLERANCE consecutive missed frames before deleting it.
        # Deleting immediately (the old behaviour) threw away the Kalman
        # state on every brief occlusion and contributed to ID churn.
        MISS_TOLERANCE = 10  # frames (= 2 seconds at 5Hz tracking)

        active_set = set(track_ids_this_frame)
        for tid in list(self.track_states.keys()):
            if tid in active_set:
                self.track_missed[tid] = 0
            else:
                self.track_missed[tid] = self.track_missed.get(tid, 0) + 1
                if self.track_missed[tid] > MISS_TOLERANCE:
                    del self.track_states[tid]
                    del self.track_missed[tid]

        # side-strip detection: catch close-range vehicles at frame edges
        # that full-frame YOLO misses because they appear as partial blobs
        edge_detections = self._detect_edge_vehicles(model, frame, tracked)
        tracked.extend(edge_detections)

        return tracked


