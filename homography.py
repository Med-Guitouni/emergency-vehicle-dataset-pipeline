import cv2
import numpy as np
import torch


class DepthEstimator:
    """
    Wraps Depth Anything V2 to produce a per-pixel depth map for each frame.

    WHAT IT IS USED FOR NOW
    -----------------------
    Depth is NO LONGER used for distance/position. Distance comes from pure
    ground-plane geometry in HomographyEstimator (more reliable). The depth
    map here exists for ONE reason: it is handed to the EMAP tracker as a
    per-object control signal that helps it associate vehicles across frames
    when the camera is moving.

    depth is RELATIVE, not metric
    -----------------------------------------
    Depth Anything V2 (the non-metric checkpoint we use) outputs relative
    inverse depth - a unitless value that is monotonic with distance but is
    NOT in metres and is nonlinear. We deliberately do NOT convert it to
    metres any more (an earlier version did, with a near-road anchor, and it
    was wrong because the crop removes the near road). EMAP only needs the
    relative ordering of depths, so we pass the raw model output through.

    Output: a 2D numpy array [H, W], same size as the input frame, of relative
    depth values. Returns None if the model is not installed.
    """

    # Depth Anything V2 expects ImageNet normalisation (pretrained there)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    # native training resolution of Depth Anything V2 Small
    INPUT_SIZE = 518

    def __init__(self, model_path="models/depth_anything_v2_vits.pth"):
        """
        Load the model once at startup (per-frame loading would be far too slow).
        model_path points to the downloaded checkpoint (~98 MB).
        Download from: https://github.com/DepthAnything/Depth-Anything-V2
        """
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                "encoder": "vits",   # ViT-Small - fastest, good enough on CPU
                "features": 64,
                "out_channels": [48, 96, 192, 384]
            }
            self.model = DepthAnythingV2(**model_configs)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Depth Anything V2 loaded on {self.device}")

        except Exception as ex:
            # Model missing or package not installed - EMAP just runs without
            # the depth signal. Nothing else in the pipeline depends on depth.
            print(f"Depth Anything V2 not available: {ex}")
            print("EMAP will run without the depth control signal.")

    def estimate_depth(self, frame_bgr):
        """
        Run the depth model on one frame and return a relative-depth map.
        Returns a 2D numpy float32 array [H, W] of relative depth values
        (not metres - see class docstring), or None if unavailable.
        """
        if self.model is None:
            return None

        h_orig, w_orig = frame_bgr.shape[:2]

        # resize to model input size
        img = cv2.resize(frame_bgr, (self.INPUT_SIZE, self.INPUT_SIZE))

        # BGR -> RGB, scale to 0..1, then ImageNet-normalise
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        # [H, W, C] -> [1, C, H, W]
        img_tensor = torch.from_numpy(
            img.transpose(2, 0, 1)
        ).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            depth_relative = self.model(img_tensor).squeeze().cpu().numpy()

        # resize back to original frame size so pixel lookups line up
        depth_resized = cv2.resize(
            depth_relative, (w_orig, h_orig),
            interpolation=cv2.INTER_LINEAR
        )

        return depth_resized.astype(np.float32)


class HomographyEstimator:
    """
    Turns each detected vehicle's pixel position into real-world metres, then
    derives relative velocity, distance to the ambulance, heading and lane info
    from those metric positions.

    =========================================================================
    HOW POSITION IS COMPUTED - ground-plane pinhole projection
    =========================================================================
    A vehicle's tyres touch the road. The road is a flat plane a known height
    (camera_height) below the camera. Basic camera geometry gives the forward
    distance directly, with NO depth model needed:

        forward_distance = camera_height * focal_length / pixels_below_horizon

    "pixels_below_horizon" is how far the bottom-centre of the bounding box
    sits below the horizon line. The lower in the image the wheels are, the
    closer the vehicle. Metric by construction - the only unknowns are
    focal_length and camera_height, both estimated, which scale all distances
    by one constant factor that can be calibrated later.

    WHERE THE HORIZON IS
    --------------------
    For a dashcam looking roughly straight ahead the horizon sits at the
    vertical centre of the ORIGINAL image. preprocessor.py crops to rows
    [0.20H : 0.85H], so in the CROPPED frame the horizon moves to about
    (0.5 - 0.20)/(0.85 - 0.20) = 0.46 down the cropped frame. Validated up to
    0.55 against the test video (camera tilts slightly down). Tunable because
    it depends on camera tilt and on the crop fractions.

    =========================================================================
    HONEST LIMITATION - speed is RELATIVE, not absolute
    =========================================================================
    Velocity here is the change in a vehicle's position RELATIVE TO THE
    AMBULANCE per second. If the ambulance and a car both do 100 km/h, the
    car's relative velocity is ~0. Absolute speed needs metric ego-odometry
    (calibrated depth or GPS) which we do not have. Relative velocity is still
    exactly the yielding signal: a car pulling aside has a clear sideways
    relative velocity regardless of how fast anyone goes forward.

    DEPTH is still computed in process_frame() but ONLY for the EMAP tracker -
    position and velocity here are pure geometry.
    """

    LANE_WIDTH_METERS = 3.75  # German Autobahn lane width (fallback only)

    # vehicles beyond this are clamped: near the horizon a 1px change swings
    # distance by tens of metres, so the value is unreliable past ~150m.
    MAX_FORWARD_METERS = 250.0

    # how much road exists beyond the outermost lane edge (hard shoulder /
    # Standstreifen on the Autobahn, parking strip in the city). Used to clamp
    # physically impossible lateral positions.
    SHOULDER_METERS = 3.0

    def __init__(self, camera_height=1.4, focal_length_factor=0.8,
                 horizon_ratio=0.55):
        """
        camera_height: dashcam height above the road in metres (~1.4).
        focal_length_factor: focal_length_px = frame_width * this (~0.8).
        horizon_ratio: horizon position as a fraction down the cropped frame.
                       0.55 was validated against the test video.

        camera_height and focal_length_factor both scale distance linearly, so
        (camera_height * focal_length_factor) is the single calibration constant.
        To calibrate: count Autobahn lane dashes (6m line + 12m gap = 18m period)
        to a vehicle, compare to the reported distance, scale accordingly.
        """
        self.camera_height = camera_height
        self.focal_length_factor = focal_length_factor
        self.horizon_ratio = horizon_ratio

        # previous-frame grayscale image, used by optical flow for EMAP
        self.prev_gray = None

        # previous METRIC position per track_id -> (x_m, y_m), for velocity
        self.prev_positions_m = {}

        # heading history per track_id: list of (x_m, y_m) over last N frames
        # we keep 3 entries so heading is computed over a 3-second window
        # this smooths out the noise that makes single-frame arctan2 unreliable
        self.heading_history_m = {}
        self.HEADING_WINDOW = 3   # seconds / frames at 1Hz

        # previous speed / acceleration per track_id, for accel and jerk
        self.prev_speeds = {}
        self.prev_accelerations = {}

        # last camera-motion homography (for EMAP); identity = no motion
        self.current_ego_H = np.eye(3)

        # Depth Anything V2 wrapper - output used by EMAP tracker only
        self.depth_estimator = DepthEstimator()
        self.current_depth_map = None

    # ------------------------------------------------------------------
    # EGO MOTION (camera odometry) - feeds EMAP in tracker.py
    # ------------------------------------------------------------------

    def estimate_ego_motion(self, frame):
        """
        Estimate how the camera moved between the previous and current frame as
        a 3x3 homography H, using Lucas-Kanade optical flow on background
        feature points + RANSAC (rejects moving vehicles). Consumed by the EMAP
        tracker. Returns identity when there are too few stable points.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return np.eye(3)

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=8
        )
        if prev_pts is None or len(prev_pts) < 8:
            self.prev_gray = gray
            return np.eye(3)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3
        )
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]
        if len(good_prev) < 8:
            self.prev_gray = gray
            return np.eye(3)

        H, _ = cv2.findHomography(
            good_prev, good_curr, cv2.RANSAC, ransacReprojThreshold=3.0
        )
        self.prev_gray = gray
        if H is None:
            return np.eye(3)
        self.current_ego_H = H
        return H

    def compute_depth_map(self, frame):
        """
        Run Depth Anything V2 and cache the result for the EMAP tracker.
        Returns the relative-depth map (2D array) or None if unavailable.
        """
        self.current_depth_map = self.depth_estimator.estimate_depth(frame)
        return self.current_depth_map

    # ------------------------------------------------------------------
    # POSITION - ground-plane pinhole projection
    # ------------------------------------------------------------------

    def get_bev_position(self, bottom_center, frame_width, frame_height,
                         lane_info=None):
        """
        Project a vehicle's bottom-centre pixel onto the road plane. Returns:
            x_meters (lateral): + = vehicle is to the RIGHT of centre
            y_meters (forward): distance ahead, >= 0, larger = further away

            pixels_below_horizon = bottom_y - horizon_row
            y_forward = camera_height * focal_length / pixels_below_horizon
            x_lateral = (bottom_x - image_centre_x) * y_forward / focal_length

        LATERAL CLAMP (Fix 3)
        ---------------------
        x_lateral physically cannot exceed half the road width plus the
        shoulder. Values beyond that come from wrong distance estimates
        (usually edge-clipped bounding boxes) and would place vehicles in
        the grass. We clamp to the plausible bound. The bound is derived
        from lane_info so it automatically adapts: 3-lane Autobahn allows
        more lateral range than a 2-lane city street.
        """
        f = frame_width * self.focal_length_factor
        cx = frame_width / 2.0

        horizon_row = self.horizon_ratio * frame_height

        bottom_x = bottom_center[0]
        bottom_y = bottom_center[1]

        delta_y = bottom_y - horizon_row

        # a vehicle at/above the horizon is geometrically near infinity - clamp
        # delta_y so we never divide by zero or produce a negative distance
        min_delta = (self.camera_height * f) / self.MAX_FORWARD_METERS
        if delta_y < min_delta:
            delta_y = min_delta

        y_forward = (self.camera_height * f) / delta_y
        if y_forward > self.MAX_FORWARD_METERS:
            y_forward = self.MAX_FORWARD_METERS

        x_lateral = ((bottom_x - cx) * y_forward) / f

        # ---- lateral plausibility clamp (lane-aware) ----
        if lane_info is not None:
            half_road = (lane_info["lanes"] * lane_info["lane_width_meters"]) / 2.0
        else:
            half_road = (3 * self.LANE_WIDTH_METERS) / 2.0
        max_lateral = half_road + self.SHOULDER_METERS

        if x_lateral > max_lateral:
            x_lateral = max_lateral
        elif x_lateral < -max_lateral:
            x_lateral = -max_lateral

        return round(float(x_lateral), 2), round(float(y_forward), 2)

    def is_position_reliable(self, bbox, frame_width, frame_height,
                             x_m, lane_info=None):
        """
        Returns False when the computed position should not be trusted (Fix 2).

        WHY THIS EXISTS
        ---------------
        The ground-plane formula projects the bottom-centre of the bounding
        box, assuming that point is where the tyres touch the road. When the
        bbox is CLIPPED at a frame edge - the vehicle is partially outside
        the camera view, typical for trucks right beside the ambulance - the
        bbox bottom is just where the visible part ends, NOT the tyre line.
        The projected position is then wrong, sometimes by tens of metres.

        Rules:
        - bbox touches any frame edge       -> unreliable
        - lateral position sits at the clamp -> unreliable (the clamp fired,
          meaning the raw value was physically impossible)

        The flag is exported as "position_reliable" in the JSON so the
        analysis can filter these rows instead of silently trusting them.
        """
        x1, y1, x2, y2 = bbox
        EDGE = 2  # pixels of tolerance

        clipped = (x1 <= EDGE or y1 <= EDGE
                   or x2 >= frame_width - EDGE
                   or y2 >= frame_height - EDGE)
        if clipped:
            return False

        # did the lateral clamp fire?
        if lane_info is not None:
            half_road = (lane_info["lanes"] * lane_info["lane_width_meters"]) / 2.0
        else:
            half_road = (3 * self.LANE_WIDTH_METERS) / 2.0
        max_lateral = half_road + self.SHOULDER_METERS

        if abs(x_m) >= max_lateral - 0.01:
            return False

        return True

    # ------------------------------------------------------------------
    # VELOCITY - relative, derived from metric position change
    # ------------------------------------------------------------------

    def estimate_relative_velocity(self, track_id, x_m, y_m, dt=1.0):
        """
        Velocity RELATIVE TO THE AMBULANCE from the change in metric position.
        Returns:
            forward_speed_ms : along the road, m/s. + = away, - = toward ego
            lateral_speed_ms : across the road, m/s. + = right, - = left
            speed_kmh        : overall magnitude in km/h

        A braking car changes mainly forward_speed; a yielding car changes
        mainly lateral_speed. Reads AND updates the stored previous position,
        so call exactly once per vehicle per frame.
        """
        prev = self.prev_positions_m.get(track_id)
        self.prev_positions_m[track_id] = (x_m, y_m)

        if prev is None:
            return 0.0, 0.0, 0.0

        dx = x_m - prev[0]   # + = moved right
        dy = y_m - prev[1]   # + = moved further ahead (away from ego)

        lateral_speed = dx / dt
        forward_speed = dy / dt
        speed_kmh = (np.sqrt(dx * dx + dy * dy) / dt) * 3.6

        return (round(float(forward_speed), 2),
                round(float(lateral_speed), 2),
                round(float(speed_kmh), 2))

    def estimate_acceleration(self, track_id, forward_speed_ms, dt=1.0):
        """
        LONGITUDINAL acceleration in m/s² = change in forward (along-road)
        relative speed since last frame. Negative = braking relative to ego.

        CHANGED from the old magnitude-based version, which differenced
        speed_kmh (a magnitude). That produced phantom accelerations whenever
        a vehicle's relative velocity passed through zero (magnitude collapses
        and rebounds even though nothing physical happened). highD (Krajewski
        et al. 2018) defines acceleration longitudinally; so do we now. This
        also makes the -2.5 m/s² braking threshold physically meaningful.
        """
        prev = self.prev_speeds.get(track_id, forward_speed_ms)
        acceleration = round((forward_speed_ms - prev) / dt, 3)
        self.prev_speeds[track_id] = forward_speed_ms
        return acceleration

    def estimate_jerk(self, track_id, curr_acceleration, dt=1.0):
        """Jerk in m/s^3 = change in acceleration. High jerk = panic stop."""
        prev_acc = self.prev_accelerations.get(track_id, curr_acceleration)
        jerk = round((curr_acceleration - prev_acc) / dt, 3)
        self.prev_accelerations[track_id] = curr_acceleration
        return jerk

    def estimate_heading(self, track_id, x_m, y_m):
        """
        Heading angle in degrees — the direction the vehicle is travelling
        relative to the road forward axis.

        0 degrees   = travelling straight ahead (same direction as ambulance)
        positive    = drifting right
        negative    = drifting left
        ~90 or ~-90 = moving sideways (strong yielding signal)

        WHY 3-SECOND WINDOW INSTEAD OF FRAME-TO-FRAME
        -----------------------------------------------
        At 1Hz, a vehicle moving at the same speed as the ambulance shifts
        only 1-3 pixels between frames. arctan2 of 1-3 pixels of noise gives
        a random angle — useless. By using the direction from 3 seconds ago
        to now in metric space, the displacement is large enough to be
        meaningful (1-5 metres for a yielding vehicle) and the noise averages
        out over the window.

        Uses x_meters/y_meters (ground-plane metric coordinates) instead of
        pixels — consistent with how velocity is computed and more accurate.
        """
        if track_id not in self.heading_history_m:
            self.heading_history_m[track_id] = []

        history = self.heading_history_m[track_id]
        history.append((x_m, y_m))

        # keep only the last HEADING_WINDOW frames
        if len(history) > self.HEADING_WINDOW:
            history.pop(0)

        if len(history) < 2:
            # not enough history yet
            return 0.0

        # direction from oldest stored position to current
        x_old, y_old = history[0]
        dx = x_m - x_old
        dy = y_m - y_old

        if abs(dx) < 0.01 and abs(dy) < 0.01:
            # vehicle has not moved meaningfully - no heading information
            return 0.0

        # minimum total displacement to trust the heading
        # below 0.3m over 3 seconds the signal is too noisy to be meaningful
        total_disp = np.sqrt(dx * dx + dy * dy)
        if total_disp < 0.3:
            return 0.0

        # boundary case: if forward displacement is near zero but lateral is not,
        # arctan2 snaps to exactly ±90°. This happens when a vehicle sits at
        # nearly the same forward distance for 3 seconds (common in slow traffic
        # being overtaken). A vehicle that hasn't moved forward is not necessarily
        # moving sideways — it may just be stationary relative to the ambulance.
        # Require that lateral displacement is at least 2x the forward displacement
        # to confidently report a sideways heading. Otherwise report 0° (straight).
        if abs(dy) < 0.1 and abs(dx) < abs(dy) * 2:
            return 0.0

        # in our BEV coordinate system:
        #   y_m increases forward (away from ego) = road forward direction
        #   x_m increases rightward
        # heading = angle from the forward axis (y-axis)
        # arctan2(dx, dy): dx=0,dy>0 → 0° (straight ahead)
        #                  dx>0,dy=0 → 90° (moving right)
        #                  dx<0,dy=0 → -90° (moving left)
        angle = np.degrees(np.arctan2(dx, dy))
        return round(float(angle), 2)

    # ------------------------------------------------------------------
    # DISTANCE / LANE / LATERAL OFFSET
    # ------------------------------------------------------------------

    def estimate_distance_to_ego(self, x_m, y_m):
        """
        Straight-line distance to the ambulance in metres. The ambulance is the
        origin (0,0), so this is just sqrt(x^2 + y^2) of the metric position.
        """
        return round(float(np.sqrt(x_m * x_m + y_m * y_m)), 2)

    def estimate_lane_id(self, center_x, frame_width, lane_info=None):
        """
        Lane number 1 (left) to N (right) by horizontal pixel position, using
        the real lane count from lane_info (video_lanes.json). Falls back to 3
        equal lanes if lane_info is None.

        Note: surrounding.py does NOT use this - it buckets lanes from metric
        x_meters directly, which is more reliable.
        """
        if lane_info is None:
            lane_width_px = frame_width / 3
            return min(int(center_x / lane_width_px) + 1, 3)

        n_lanes = lane_info["lanes"]
        lane_width_px = frame_width / n_lanes
        lane = int(center_x / lane_width_px) + 1
        return min(max(lane, 1), n_lanes)

    def estimate_lateral_offset(self, center_x, frame_width, lane_info=None):
        """
        Distance from the vehicle's lane centre in metres. + = right of centre.
        Uses real lane count and width from lane_info; falls back to 3 lanes.
        """
        if lane_info is None:
            n_lanes = 3
            lane_width_m = self.LANE_WIDTH_METERS
        else:
            n_lanes = lane_info["lanes"]
            lane_width_m = lane_info["lane_width_meters"]

        lane_width_px = frame_width / n_lanes
        lane_id = self.estimate_lane_id(center_x, frame_width, lane_info)
        lane_center_px = (lane_id - 0.5) * lane_width_px
        scale = (n_lanes * lane_width_m) / frame_width
        return round((center_x - lane_center_px) * scale, 2)

    # ------------------------------------------------------------------
    # CONVENIENCE - run ego motion + depth once per frame for the tracker
    # ------------------------------------------------------------------

    def process_frame(self, frame):
        """
        Call once per frame before tracking. Runs ego-motion estimation and
        depth estimation, caches both, and returns (ego_H, depth_map) for the
        EMAP tracker. Position/velocity do not depend on these.
        """
        ego_H = self.estimate_ego_motion(frame)
        depth_map = self.compute_depth_map(frame)
        return ego_H, depth_map