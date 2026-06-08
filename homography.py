import cv2
import numpy as np
import torch
import torch.nn.functional as F


class DepthEstimator:
    """
    Wraps Depth Anything V2 to produce a per-pixel depth map for each frame.

    Why we need this:
    When the ambulance is moving, every vehicle in the image appears to drift
    backward even if it is actually standing still. To correct for this we need
    to know how far away each vehicle is in real meters - because the same camera
    movement causes a small pixel shift for a far object and a large pixel shift
    for a close object. Without depth we cannot separate the two.

    Which model variant we use:
    Depth Anything V2 comes in Small, Base, and Large. We use Small (vits)
    because we are running on CPU and speed matters more than precision here.
    The depth values are relative (not absolute meters) out of the box.
    We scale them into approximate real-world meters using the known camera
    height as a ground anchor - see _scale_depth_to_meters().

    What the output looks like:
    A 2D numpy array the same height and width as the input frame.
    Each cell contains the estimated distance in meters from the camera
    to whatever is at that pixel.
    """

    # Depth Anything V2 expects images normalised with these exact values
    # (same as ImageNet - the model was pretrained on ImageNet features)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    # We resize input to this before feeding the model.
    # 518x518 is the native resolution Depth Anything V2 Small was trained at.
    # Using a different size works but 518 gives the best results for this model.
    INPUT_SIZE = 518

    # Known physical camera height in meters.
    # Used to turn the model's relative depth values into real meters.
    # This is still estimated (1.4m) - replace with measured value when available.
    CAMERA_HEIGHT_METERS = 1.4

    def __init__(self, model_path="models/depth_anything_v2_vits.pth"):
        """
        Load the model once at startup. We keep it in memory for the whole run
        because loading it per frame would be far too slow.

        model_path points to the downloaded checkpoint file.
        Download from: https://github.com/DepthAnything/Depth-Anything-V2
        File to download: depth_anything_v2_vits.pth  (~98 MB)
        """
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Import here so the rest of the pipeline still works if the
            # depth_anything_v2 package is not installed yet
            from depth_anything_v2.dpt import DepthAnythingV2

            # 'vits' = ViT-Small backbone. Faster than Base/Large on CPU.
            model_configs = {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384]
            }
            self.model = DepthAnythingV2(**model_configs)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()  # turn off dropout / batchnorm training behaviour
            print(f"Depth Anything V2 loaded on {self.device}")

        except Exception as ex:
            # If the model file is missing or the package is not installed,
            # we fall back to None. estimate_depth() will return None and
            # the rest of the code falls back to the old lane-width scale method.
            print(f"Depth Anything V2 not available: {ex}")
            print("Falling back to lane-width scale for distance estimation.")

    def estimate_depth(self, frame_bgr):
        """
        Run the depth model on one frame and return a depth map in meters.

        Returns a 2D numpy float32 array [H, W] where each value is the
        estimated distance in meters from the camera lens to that pixel.
        Returns None if the model failed to load.

        Steps:
        1. Resize the frame to the model's expected input size
        2. Normalise pixel values the same way the model was trained
        3. Run the model forward pass
        4. Resize the output back to the original frame size
        5. Scale the relative depth output into approximate real meters
        """
        if self.model is None:
            return None

        h_orig, w_orig = frame_bgr.shape[:2]

        # --- step 1: resize ---
        img = cv2.resize(frame_bgr, (self.INPUT_SIZE, self.INPUT_SIZE))

        # --- step 2: normalise ---
        # Convert BGR (OpenCV default) to RGB (what the model expects)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        # Rearrange from [H, W, C] to [1, C, H, W] for PyTorch
        img_tensor = torch.from_numpy(
            img.transpose(2, 0, 1)
        ).unsqueeze(0).float().to(self.device)

        # --- step 3: run model ---
        with torch.no_grad():
            # Output shape: [1, H, W] - one depth value per pixel
            depth_relative = self.model(img_tensor).squeeze().cpu().numpy()

        # --- step 4: resize output back to original frame dimensions ---
        depth_resized = cv2.resize(
            depth_relative, (w_orig, h_orig),
            interpolation=cv2.INTER_LINEAR
        )

        # --- step 5: scale to real meters ---
        depth_meters = self._scale_depth_to_meters(depth_resized)

        return depth_meters.astype(np.float32)

    def _scale_depth_to_meters(self, depth_relative):
        """
        Depth Anything V2 outputs values between 0 and 1 (or some arbitrary
        relative range). We need real meters.

        The trick: we know the camera is mounted at ~1.4 metres above the ground.
        The bottom rows of the frame are the road surface directly in front of
        the ambulance, which is at approximately that distance.

        So we read the median depth value from the bottom 10% of the frame
        (which is the road close in front of us) and set a scale factor so that
        area maps to CAMERA_HEIGHT_METERS.

        This is a rough calibration. It will be off on hills or ramps.
        Replace with proper metric depth if a calibrated stereo baseline
        or LiDAR reference becomes available.
        """
        h = depth_relative.shape[0]

        # Take the bottom 10% of rows - that is the road right in front of us
        bottom_strip = depth_relative[int(h * 0.9):, :]
        median_near = np.median(bottom_strip)

        if median_near < 1e-6:
            # Avoid division by zero if the model returned all zeros
            return depth_relative * self.CAMERA_HEIGHT_METERS

        # Scale so that the close road surface equals camera height
        scale = self.CAMERA_HEIGHT_METERS / median_near
        return depth_relative * scale

    def get_vehicle_depth(self, depth_map, bbox):
        """
        Given a depth map and a bounding box [x1, y1, x2, y2],
        return the median depth in meters for that vehicle.

        We use the median (not mean) because the bounding box often includes
        background pixels around the vehicle edges, which have wrong depths.
        The median ignores outliers better than the mean.
        """
        if depth_map is None:
            return None

        x1, y1, x2, y2 = bbox
        # Clamp to image boundaries in case bbox slightly overshoots
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        vehicle_region = depth_map[y1:y2, x1:x2]
        if vehicle_region.size == 0:
            return None

        return float(np.median(vehicle_region))


class HomographyEstimator:
    """
    Turns each detected vehicle's pixel position into real-world metres, then
    derives relative velocity, distance to the ambulance, heading and lane info
    from those metric positions.

    =========================================================================
    HOW POSITION IS COMPUTED  (this is the part that was fixed)
    =========================================================================
    OLD (broken) approach:
      - A single scale "3 lanes = 11.25m across the whole frame width" was used
        for everything, OR a depth map from Depth Anything V2 was scaled to
        metres using a near-road anchor.
      - Both failed. The lane-width scale ignores that far objects cover more
        ground per pixel than near ones. The depth anchor assumed the bottom of
        the frame was road 1.4m ahead - but preprocessor.py crops away the
        bottom 15% of the image, so the bottom row is actually road ~20-40m
        ahead. Anchoring a far point to 1.4m squashed every distance into a
        tiny 0.1-2.5m range. Depth Anything also outputs RELATIVE inverse depth
        (unitless, nonlinear), which no single multiply can convert to metres.

    NEW (correct) approach - ground-plane pinhole projection:
      A vehicle's tyres touch the road. The road is a flat plane a known height
      (camera_height) below the camera. Basic camera geometry then gives the
      forward distance directly, with NO depth model needed:

          forward_distance = camera_height * focal_length / pixels_below_horizon

      "pixels_below_horizon" is how far the bottom-centre of the bounding box
      sits below the horizon line in the image. The further below the horizon a
      vehicle's wheels are, the closer it is. This is metric by construction -
      the only unknowns are focal_length and camera_height, which are estimated
      and scale all distances by one constant factor we can calibrate later.

    WHERE THE HORIZON IS  (the key detail)
      The horizon is where the road meets the sky - the vanishing line. For a
      dashcam looking roughly straight ahead, the horizon sits at the vertical
      centre of the ORIGINAL image. But preprocessor.py crops the frame to
      rows [0.20*H : 0.85*H]. So in the CROPPED frame the horizon is no longer
      at the centre. Its position in the cropped frame is:

          (0.5 - 0.20) / (0.85 - 0.20) = 0.30 / 0.65 = 0.46

      i.e. about 46% of the way down the cropped frame. That is the default
      value of self.horizon_ratio below. It is exposed as a tunable parameter
      because: (a) the dashcam may be tilted slightly down (which moves the
      horizon up), and (b) if the crop fractions in preprocessor.py change,
      this ratio must change with them.

    =========================================================================
    HONEST LIMITATION - speed is RELATIVE, not absolute
    =========================================================================
    Velocity here is the change in a vehicle's position RELATIVE TO THE
    AMBULANCE per second. If the ambulance and a car both travel at 100 km/h,
    the car's relative velocity is ~0 - it is not moving relative to us.
    Getting ABSOLUTE speed would require knowing the ambulance's own metric
    speed each frame (ego odometry), which needs calibrated depth or GPS that
    we do not have. EMAP (in tracker.py) improves TRACKING robustness but does
    not give us reliable metric ego-speed from a single uncalibrated camera.
    Relative velocity is still exactly the signal we need for yielding: a car
    pulling aside has a clear sideways (lateral) relative velocity regardless
    of how fast anyone is going forward.

    =========================================================================
    DEPTH IS STILL COMPUTED - but only for EMAP
    =========================================================================
    process_frame() still runs Depth Anything V2 and returns the depth map.
    That map is no longer used for distance here; it is passed to the EMAP
    tracker as a per-object control signal to help association. Position and
    velocity in THIS class are pure geometry.
    """

    LANE_WIDTH_METERS = 3.75  # standard German Autobahn lane width

    # vehicles farther than this (metres) are clamped - beyond ~150m a 1px
    # change near the horizon swings distance by tens of metres, so the value
    # is unreliable. We keep the detection but cap the reported distance.
    MAX_FORWARD_METERS = 250.0

    def __init__(self, camera_height=1.4, focal_length_factor=0.8,
                 horizon_ratio=0.46):
        """
        camera_height: dashcam height above the road in metres. Estimate 1.4m.
                       Scales ALL distances linearly - see calibration note.

        focal_length_factor: focal_length_pixels = frame_width * this factor.
                       0.8 is a rough estimate. Also scales all distances
                       linearly, so (camera_height * focal_length_factor) is the
                       single calibration constant for the whole system.

        horizon_ratio: where the horizon sits in the CROPPED frame, as a
                       fraction of cropped height from the top. Default 0.46 is
                       derived from the 0.20/0.85 crop in preprocessor.py (see
                       class docstring). Increase it if distances read too SMALL
                       (horizon assumed too high); decrease if too LARGE.

        CALIBRATION NOTE:
        To calibrate against the video, find a vehicle whose real distance you
        can estimate - German Autobahn lane dashes are 6m painted + 12m gap =
        18m period, so you can literally count dashes to a car. If the code
        reports d_measured but the real distance is d_real, multiply
        focal_length_factor by (d_real / d_measured) and rerun. One good
        reference fixes every distance in the dataset.
        """
        self.camera_height = camera_height
        self.focal_length_factor = focal_length_factor
        self.horizon_ratio = horizon_ratio

        # previous-frame grayscale image, used by optical flow for EMAP
        self.prev_gray = None

        # previous METRIC position per track_id -> (x_m, y_m)
        # used to compute relative velocity by differencing positions
        self.prev_positions_m = {}

        # previous PIXEL centre per track_id -> [px, py]
        # used only by estimate_heading (a pixel-space angle)
        self.prev_positions_px = {}

        # previous speed / acceleration per track_id, for accel and jerk
        self.prev_speeds = {}
        self.prev_accelerations = {}

        # last camera-motion homography (for EMAP); identity = no motion
        self.current_ego_H = np.eye(3)

        # Depth Anything V2 wrapper - still loaded, used by EMAP only
        self.depth_estimator = DepthEstimator()
        self.current_depth_map = None

    # ------------------------------------------------------------------
    # EGO MOTION (camera odometry) - feeds EMAP in tracker.py
    # ------------------------------------------------------------------

    def estimate_ego_motion(self, frame):
        """
        Estimate how the camera moved between the previous frame and this one,
        as a 3x3 homography H. Uses Lucas-Kanade optical flow on background
        feature points, then RANSAC to fit H while rejecting moving vehicles.

        This H is consumed by the EMAP tracker to subtract camera motion from
        the Kalman Filter prediction. It is NOT used for distance any more.
        Returns identity (no motion) when there are too few stable points.
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
        Run Depth Anything V2 and cache the result. Used by EMAP only.
        Returns the depth map (2D array) or None if the model is unavailable.
        """
        self.current_depth_map = self.depth_estimator.estimate_depth(frame)
        return self.current_depth_map

    # ------------------------------------------------------------------
    # POSITION - ground-plane pinhole projection (the fix)
    # ------------------------------------------------------------------

    def get_bev_position(self, bottom_center, frame_width, frame_height, bbox=None):
        """
        Project a vehicle's bottom-centre pixel onto the road plane and return
        its real-world position relative to the ambulance, in metres:

            x_meters  (lateral): + = vehicle is to the RIGHT of centre
            y_meters  (forward): distance straight ahead, always >= 0,
                                 larger = further away

        Geometry (see class docstring for the full explanation):
            pixels_below_horizon = bottom_y - horizon_row
            y_forward = camera_height * focal_length / pixels_below_horizon
            x_lateral = (bottom_x - image_centre_x) * y_forward / focal_length

        bbox is accepted for signature compatibility but no longer used -
        position is pure geometry now, no depth.
        """
        f = frame_width * self.focal_length_factor
        cx = frame_width / 2.0

        # horizon row inside the cropped frame
        horizon_row = self.horizon_ratio * frame_height

        bottom_x = bottom_center[0]
        bottom_y = bottom_center[1]

        # how many pixels below the horizon the wheels sit
        delta_y = bottom_y - horizon_row

        # a vehicle at or above the horizon line is geometrically at/near
        # infinity - clamp delta_y so we never divide by zero or go negative
        min_delta = (self.camera_height * f) / self.MAX_FORWARD_METERS
        if delta_y < min_delta:
            delta_y = min_delta

        y_forward = (self.camera_height * f) / delta_y
        if y_forward > self.MAX_FORWARD_METERS:
            y_forward = self.MAX_FORWARD_METERS

        x_lateral = ((bottom_x - cx) * y_forward) / f

        return round(float(x_lateral), 2), round(float(y_forward), 2)

    # ------------------------------------------------------------------
    # VELOCITY - relative, derived from metric position change
    # ------------------------------------------------------------------

    def estimate_relative_velocity(self, track_id, x_m, y_m, dt=1.0):
        """
        Compute the vehicle's velocity RELATIVE TO THE AMBULANCE by how far its
        metric position moved since the last frame. Returns three values:

            forward_speed_ms : along the road, m/s.
                               + = moving AWAY from ego, - = moving TOWARD ego
            lateral_speed_ms : across the road, m/s.
                               + = moving RIGHT, - = moving LEFT
            speed_kmh        : overall magnitude in km/h (for backward
                               compatibility with the old speed_kmh field)

        Why split it:
        A braking car changes mainly forward_speed. A yielding car changes
        mainly lateral_speed. One combined number hides which is happening;
        the split makes the yield signal directly readable.

        This both reads AND updates the stored previous metric position, so it
        must be called exactly once per vehicle per frame.
        """
        prev = self.prev_positions_m.get(track_id)
        self.prev_positions_m[track_id] = (x_m, y_m)

        if prev is None:
            # first sighting - no previous position to difference against
            return 0.0, 0.0, 0.0

        dx = x_m - prev[0]   # + = moved right
        dy = y_m - prev[1]   # + = moved further ahead (away from ego)

        lateral_speed = dx / dt
        forward_speed = dy / dt
        speed_kmh = (np.sqrt(dx * dx + dy * dy) / dt) * 3.6

        return (round(float(forward_speed), 2),
                round(float(lateral_speed), 2),
                round(float(speed_kmh), 2))

    def estimate_acceleration(self, track_id, curr_speed, dt=1.0):
        """
        Acceleration in m/s^2 = change in speed_kmh since last frame, converted
        to m/s. Large negative value = hard braking.
        """
        prev_speed = self.prev_speeds.get(track_id, curr_speed)
        speed_diff_ms = (curr_speed - prev_speed) / 3.6
        acceleration = round(speed_diff_ms / dt, 3)
        self.prev_speeds[track_id] = curr_speed
        return acceleration

    def estimate_jerk(self, track_id, curr_acceleration, dt=1.0):
        """
        Jerk in m/s^3 = change in acceleration since last frame. High jerk =
        sudden change in braking force (panic stop or release).
        """
        prev_acc = self.prev_accelerations.get(track_id, curr_acceleration)
        jerk = round((curr_acceleration - prev_acc) / dt, 3)
        self.prev_accelerations[track_id] = curr_acceleration
        return jerk

    def estimate_heading(self, track_id, curr_center_px):
        """
        Heading angle in degrees from pixel motion. 0 = straight, + = drifting
        right, - = left. Kept in pixel space because it is only a direction.
        Reads and updates its own pixel-position store.
        """
        prev = self.prev_positions_px.get(track_id)
        self.prev_positions_px[track_id] = curr_center_px
        if prev is None:
            return 0.0
        dx = curr_center_px[0] - prev[0]
        dy = curr_center_px[1] - prev[1]
        return round(float(np.degrees(np.arctan2(dy, dx))), 2)

    # ------------------------------------------------------------------
    # DISTANCE / LANE / LATERAL OFFSET
    # ------------------------------------------------------------------

    def estimate_distance_to_ego(self, x_m, y_m):
        """
        Straight-line distance from the vehicle to the ambulance in metres.
        The ambulance is the origin (0,0) of our coordinate system, so this is
        simply sqrt(x^2 + y^2) of the vehicle's metric position. No separate
        approximation needed any more.
        """
        return round(float(np.sqrt(x_m * x_m + y_m * y_m)), 2)

    def estimate_lane_id(self, center_x, frame_width, lane_info=None):
        """
        Lane number 1 (left) to N (right) by horizontal pixel position.

        Now uses the real lane count and lane width from lane_info (read from
        video_lanes.json) instead of the old hardcoded equal-thirds assumption.

        lane_info: dict with keys 'lanes' and 'lane_width_meters', returned by
                   LaneConfig.get_lane_info(). If None, falls back to 3 equal
                   lanes across the frame (old behaviour, kept as safety net).

        Note: surrounding.py does NOT rely on this - it buckets lanes from
        metric x_meters directly, which is more reliable.
        """
        if lane_info is None:
            # old fallback - equal thirds
            lane_width_px = frame_width / 3
            lane = int(center_x / lane_width_px) + 1
            return min(lane, 3)

        n_lanes = lane_info["lanes"]
        lane_width_px = frame_width / n_lanes
        lane = int(center_x / lane_width_px) + 1
        return min(max(lane, 1), n_lanes)

    def estimate_lateral_offset(self, center_x, frame_width, lane_info=None):
        """
        Distance from the centre of the vehicle's lane in metres.
        + = right of lane centre, - = left of lane centre.

        Uses real lane count and width from lane_info when available.
        Falls back to 3-lane equal-thirds assumption if lane_info is None.
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
        # scale: total road width in metres / frame width in pixels
        scale = (n_lanes * lane_width_m) / frame_width
        return round((center_x - lane_center_px) * scale, 2)

    # ------------------------------------------------------------------
    # CONVENIENCE - run ego motion + depth once per frame for the tracker
    # ------------------------------------------------------------------

    def process_frame(self, frame):
        """
        Call once per frame before tracking. Runs ego-motion estimation and
        depth estimation, caches both, and returns (ego_H, depth_map) so main.py
        can hand them to the EMAP tracker. Position/velocity no longer depend on
        these - they are purely for EMAP.
        """
        ego_H = self.estimate_ego_motion(frame)
        depth_map = self.compute_depth_map(frame)
        return ego_H, depth_map