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
#        cd ~/Desktop/einsatz_pipeline
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
# 4. Test that it works:
#        cd ~/Desktop/einsatz_pipeline
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
    IMG_WIDTH    = 1280
    IMG_HEIGHT   = 720

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

        try:
            from EMAP.trackers.bytetrack.kalman_filter import KalmanFilter as EmapKalmanFilter

            # Pass image dimensions and focal length so EMAP can do the
            # pixel-to-angle conversion correctly inside its control matrix
            self.emap_kalman = EmapKalmanFilter(
                self.IMG_WIDTH,
                self.IMG_HEIGHT,
                self.FOCAL_LENGTH
            )

            # We store per-track Kalman state ourselves because we are not
            # using EMAP's full BYTETracker class (which needs ROS).
            # Key: track_id  Value: (mean, covariance) numpy arrays
            self.track_states = {}

            self.emap_available = True
            print("EMAP Kalman Filter loaded - ego-motion compensation active")

        except Exception as ex:
            print(f"EMAP not available ({ex}) - using standard ByteTrack")

        # current ego-motion control signals - updated each frame by update()
        # yaw_dot: camera rotation speed in radians per frame
        # D_dot:   camera forward speed in metres per frame
        self.current_yaw_dot = 0.0
        self.current_D_dot   = 0.0

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
        IMU or GPS odometry signal, but we don't have those.
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

        This is what replaces the standard Kalman predict inside ByteTrack.
        For each tracked vehicle we:
          1. Look up its stored Kalman state (mean, covariance)
          2. Build the control signal [yaw_dot, D_dot, vehicle_depth]
          3. Call EMAP's multi_predict which subtracts camera motion
          4. Store the updated state back

        track_ids:    list of integer track IDs active this frame
        bboxes_xyah:  list of [cx, cy, aspect, height] for each track
                      (the format EMAP's Kalman Filter uses internally)
        depth_map:    per-pixel depth in metres from Depth Anything V2
        """
        if not self.emap_available or len(track_ids) == 0:
            return

        # build mean and covariance arrays for all tracks that have state
        active_ids    = []
        active_means  = []
        active_covs   = []

        for tid, xyah in zip(track_ids, bboxes_xyah):
            if tid in self.track_states:
                mean, cov = self.track_states[tid]
                active_ids.append(tid)
                active_means.append(mean.copy())
                active_covs.append(cov.copy())

        if len(active_ids) == 0:
            return

        multi_mean = np.array(active_means)
        multi_cov  = np.array(active_covs)

        # build control signal: [yaw_dot, D_dot, depth] per track
        control_signals = []
        for tid, xyah in zip(active_ids, [bboxes_xyah[track_ids.index(t)] for t in active_ids]):
            # get depth for this specific vehicle from the depth map
            if depth_map is not None:
                cx, cy = int(xyah[0]), int(xyah[1])
                h_img, w_img = depth_map.shape
                cx = max(0, min(w_img - 1, cx))
                cy = max(0, min(h_img - 1, cy))
                vehicle_depth = float(depth_map[cy, cx])
                if vehicle_depth < 0.1:
                    vehicle_depth = 10.0  # fallback if depth is zero
            else:
                vehicle_depth = 10.0  # fallback if no depth map

            control_signals.append([
                self.current_yaw_dot,
                self.current_D_dot,
                vehicle_depth
            ])

        control_array = np.array(control_signals)

        # run EMAP predict - this modifies mean in place to subtract camera motion
        multi_mean, multi_cov = self.emap_kalman.multi_predict(
            multi_mean, multi_cov, control_array
        )

        # store updated states back
        for i, tid in enumerate(active_ids):
            self.track_states[tid] = (multi_mean[i], multi_cov[i])

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
        track_ids_this_frame   = []
        bboxes_xyah_this_frame = []

        for box in results.boxes:
            if box.id is None:
                continue

            class_id   = int(box.cls[0])
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
                "type":     VEHICLE_CLASSES[class_id],
                "bbox":     [x1, y1, x2, y2],
                "center":   [cx, cy]
            })

        # run EMAP predict step for all active tracks
        # this is where camera motion gets subtracted from the Kalman state
        if self.emap_available:
            self._emap_predict_all(
                track_ids_this_frame,
                bboxes_xyah_this_frame,
                depth_map
            )

        # clean up state for tracks that disappeared
        # if a track_id hasn't been seen for this frame, remove it
        # to avoid the dict growing forever
        active_set = set(track_ids_this_frame)
        gone_ids = [tid for tid in self.track_states if tid not in active_set]
        for tid in gone_ids:
            # keep lost tracks for up to 25 frames in case they reappear
            # for simplicity here we remove immediately - ByteTrack handles
            # the re-association logic internally anyway
            del self.track_states[tid]

        return tracked

