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
    Converts pixel positions of detected vehicles into real-world coordinates
    (metres), and computes speed, acceleration, jerk, heading, lane ID,
    lateral offset, and distance to the ambulance per vehicle per frame.

    What changed from the old version:
    The old version used a single scale factor (3 lanes = 11.25m across the
    whole frame width) for everything. This caused severe underestimation
    because objects far away subtend few pixels but cover large distances.

    The new version uses Depth Anything V2 depth maps + EMAP ego-motion
    compensation to fix both distance and speed. When the depth model is not
    available (model file missing, package not installed) the code falls back
    to the old lane-width method automatically so nothing breaks.

    EMAP integration:
    EMAP (Ego-Motion Aware Target Prediction, Mahdian et al. 2024) modifies
    the Kalman Filter inside ByteTrack to subtract the camera's own motion
    before predicting where each vehicle will be next frame. This stops
    vehicles appearing to move when actually the ambulance is moving.
    We feed EMAP the camera odometry matrix H that estimate_ego_motion()
    already computes from optical flow.
    EMAP lives in tracker.py. This class computes the odometry and depth map
    and passes them to the tracker each frame via process_frame().
    """

    LANE_WIDTH_METERS = 3.75  # standard German Autobahn lane width

    def __init__(self, camera_height=1.4, focal_length_factor=0.8):
        """
        camera_height: how high the dashcam is above the road in meters.
                       1.4m is a reasonable estimate for a windshield-mounted
                       dashcam. Replace with measured value for better accuracy.

        focal_length_factor: focal_length = frame_width * this factor.
                             0.8 is a rough estimate. Proper calibration
                             from the dashcam spec sheet would improve all
                             distance and speed numbers significantly.
        """
        self.camera_height = camera_height
        self.focal_length_factor = focal_length_factor

        # stores the previous frame's grayscale image for optical flow
        self.prev_gray = None

        # stores the last known position of each vehicle (keyed by track_id)
        # used to compute how far it moved between frames
        self.prev_positions_px  = {}  # pixel positions (used as fallback)
        self.prev_positions_m   = {}  # metric positions (used when depth available)

        # stores the last known speed and acceleration per vehicle
        # needed to compute acceleration and jerk
        self.prev_speeds        = {}
        self.prev_accelerations = {}

        # the camera odometry matrix from the last frame
        # 3x3 homography that describes how the camera moved
        # identity matrix = camera did not move at all
        self.current_ego_H = np.eye(3)

        # depth estimator - loads Depth Anything V2 if available
        self.depth_estimator = DepthEstimator()

        # cache the depth map for the current frame so we do not run
        # the model twice (once in process_frame and once in get_bev_position)
        self.current_depth_map = None

    # ------------------------------------------------------------------
    # STEP 1 - EGO MOTION (camera odometry)
    # ------------------------------------------------------------------

    def estimate_ego_motion(self, frame):
        """
        Figures out how the camera itself moved between this frame and the
        last frame. Returns a 3x3 homography matrix H.

        How it works:
        We pick ~200 corner-like points in the previous frame (background
        features like road markings, barriers, trees - things that are not
        moving themselves). We then find where those same points ended up
        in the current frame using Lucas-Kanade optical flow. From those
        point correspondences we estimate the homography (a matrix that
        describes rotation + translation of the camera between frames).

        Why this matters for EMAP:
        EMAP takes this H matrix and uses it to predict where each tracked
        vehicle WOULD appear in the new frame if only the camera had moved
        and the vehicle had stayed still. It then compares that prediction
        to where the vehicle actually appeared. The difference is the
        vehicle's own motion. This is what EMAP subtracts from the
        Kalman Filter state.

        Returns np.eye(3) (identity = no movement) if there are not enough
        feature points to estimate motion reliably.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            # First frame - nothing to compare against yet
            self.prev_gray = gray
            return np.eye(3)

        # Find corner-like points in the previous frame.
        # We avoid areas near the bottom (where vehicles are) to get
        # mostly background (road surface, barriers, sky edge).
        # qualityLevel=0.01 means accept any point with at least 1% of
        # the quality of the best point found.
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=8
        )

        if prev_pts is None or len(prev_pts) < 8:
            # Not enough stable background points - cannot estimate motion.
            # This can happen in tunnels or low-texture scenes.
            self.prev_gray = gray
            return np.eye(3)

        # Track those points into the current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None,
            winSize=(21, 21),   # search window around each point
            maxLevel=3          # pyramid levels - handles large motion better
        )

        # Keep only points that were successfully tracked (status == 1)
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 8:
            self.prev_gray = gray
            return np.eye(3)

        # Estimate homography from the matched point pairs.
        # RANSAC automatically discards outliers (moving vehicles that
        # snuck into our background point set).
        H, inlier_mask = cv2.findHomography(
            good_prev, good_curr,
            cv2.RANSAC,
            ransacReprojThreshold=3.0  # pixels - stricter than old version's 5.0
        )

        self.prev_gray = gray

        if H is None:
            return np.eye(3)

        self.current_ego_H = H
        return H

    # ------------------------------------------------------------------
    # STEP 2 - DEPTH MAP
    # ------------------------------------------------------------------

    def compute_depth_map(self, frame):
        """
        Run Depth Anything V2 on the current frame and cache the result.
        Call this once per frame before calling get_bev_position or estimate_speed.

        Returns the depth map (2D array in meters) or None if unavailable.
        """
        self.current_depth_map = self.depth_estimator.estimate_depth(frame)
        return self.current_depth_map

    # ------------------------------------------------------------------
    # STEP 3 - POSITION IN METERS (BEV)
    # ------------------------------------------------------------------

    def get_bev_position(self, bottom_center, frame_width, frame_height, bbox=None):
        """
        Convert a vehicle's pixel position to real-world metres (x = lateral,
        y = forward distance from the ambulance).

        If a depth map is available and a bbox is provided, we use the depth
        value at the vehicle location to compute distance directly.
        Otherwise we fall back to the old pinhole geometry formula.

        bottom_center: [px, py] pixel coordinate of the bottom-centre of the bbox
        bbox: [x1, y1, x2, y2] full bounding box - used to sample the depth map
        """
        focal_length = frame_width * self.focal_length_factor
        cx = frame_width / 2

        # --- Method A: depth map available ---
        if self.current_depth_map is not None and bbox is not None:
            depth_m = self.depth_estimator.get_vehicle_depth(
                self.current_depth_map, bbox
            )
            if depth_m is not None and depth_m > 0.1:
                # With real depth we can unproject pixel position directly.
                # Lateral offset: how far left/right of centre the vehicle is.
                # forward_dist = depth (we already have it from the depth map)
                px_offset = bottom_center[0] - cx
                lateral_m = (px_offset * depth_m) / focal_length
                return round(float(lateral_m), 2), round(float(depth_m), 2)

        # --- Method B: fallback pinhole geometry (no depth map) ---
        # This is the old method - works but underestimates at distance.
        cy = frame_height / 2
        py = bottom_center[1] - cy
        if py <= 0:
            return 0.0, 0.0
        distance_forward = (self.camera_height * focal_length) / py
        px_offset = bottom_center[0] - cx
        distance_lateral = (px_offset * distance_forward) / focal_length
        return round(float(distance_lateral), 2), round(float(distance_forward), 2)

    # ------------------------------------------------------------------
    # STEP 4 - SPEED (ego-motion compensated)
    # ------------------------------------------------------------------

    def estimate_speed(self, track_id, curr_center_px, frame_width,
                       frame_height, bbox=None, dt=1.0):
        """
        Compute the true speed of a vehicle in km/h by:
        1. Finding where the vehicle was last frame (in pixels)
        2. Compensating for how much of that movement was caused by the
           ambulance moving (ego motion)
        3. Converting the remaining pixel movement to real metres using
           either the depth map or the lane-width scale
        4. Dividing by time (1 second at 1Hz) and converting to km/h

        track_id: which vehicle we are computing for
        curr_center_px: [px, py] bounding box centre in pixels this frame
        bbox: full bounding box [x1,y1,x2,y2] for depth sampling
        dt: time between frames in seconds (always 1.0 at 1Hz)
        """
        prev_px = self.prev_positions_px.get(track_id)

        # First time we see this vehicle - no previous position to compare
        if prev_px is None:
            self.prev_positions_px[track_id] = curr_center_px
            return 0.0

        # --- ego motion compensation ---
        # The ego homography H tells us: if the camera moved by H between
        # frames, then a stationary point at prev_px would now appear at
        # projected_px. So (curr_center_px - projected_px) is the vehicle's
        # OWN movement in pixel space, with the camera motion removed.
        prev_pt = np.array([[[float(prev_px[0]), float(prev_px[1])]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(prev_pt, self.current_ego_H)
        projected_px = projected[0][0]  # where prev point ended up due to camera motion alone

        # True vehicle motion in pixels = actual position minus camera-caused drift
        dx_px = curr_center_px[0] - projected_px[0]
        dy_px = curr_center_px[1] - projected_px[1]

        # --- convert pixels to metres ---
        if self.current_depth_map is not None and bbox is not None:
            # Use depth to get accurate scale at this vehicle's distance
            depth_m = self.depth_estimator.get_vehicle_depth(
                self.current_depth_map, bbox
            )
            if depth_m is not None and depth_m > 0.1:
                focal_length = frame_width * self.focal_length_factor
                # At distance D, one pixel = D / focal_length metres
                metres_per_pixel = depth_m / focal_length
                dx_m = dx_px * metres_per_pixel
                dy_m = dy_px * metres_per_pixel
            else:
                dx_m, dy_m = self._pixels_to_metres_fallback(dx_px, dy_px, frame_width)
        else:
            # Fallback: assume uniform scale using lane width
            # This is the old method - still better than nothing
            dx_m, dy_m = self._pixels_to_metres_fallback(dx_px, dy_px, frame_width)

        distance_m = np.sqrt(dx_m**2 + dy_m**2)
        speed_kmh = round((distance_m / dt) * 3.6, 2)

        # Update stored position for next frame
        self.prev_positions_px[track_id] = curr_center_px

        return speed_kmh

    def _pixels_to_metres_fallback(self, dx_px, dy_px, frame_width):
        """
        Old scale method: assume 3 lanes fill the full frame width.
        3 lanes × 3.75m = 11.25m across frame_width pixels.
        This is wrong at any depth other than the lane centre plane,
        but it is the best we can do without a depth map.
        """
        scale = (3 * self.LANE_WIDTH_METERS) / frame_width
        return dx_px * scale, dy_px * scale

    def estimate_split_velocity(self, track_id, curr_center_px, frame_width,
                                frame_height, bbox=None, dt=1.0):
        """
        Split the vehicle's motion into two separate speeds instead of one:

          forward_speed  = how fast it moves ALONG the road (toward/away ego)
          lateral_speed  = how fast it moves ACROSS the road (left/right)

        Why we want this:
        A single overall speed cannot tell us WHAT kind of motion happened.
        A braking car slows down -> change shows up in forward_speed.
        A yielding car pulls to the side -> change shows up in lateral_speed.
        With one combined number both look the same. Splitting makes the
        yielding signal directly visible: "this car moved 2 m/s to the left"
        is a far cleaner yield indicator than our heading-angle approximation.

        Sign convention (matches image axes after BEV reasoning):
          forward_speed > 0  -> moving AWAY from ego (up the image, smaller y px)
          forward_speed < 0  -> moving TOWARD ego (down the image, getting closer)
          lateral_speed > 0  -> moving RIGHT (larger x px)
          lateral_speed < 0  -> moving LEFT (smaller x px)

        Units: metres per second (NOT km/h - m/s is the natural unit for the
        yielding thresholds the annotator uses, e.g. "lateral > 0.5 m/s").

        This reuses the exact same ego-motion compensation and depth-based
        pixel-to-metre conversion as estimate_speed(). The only difference is
        we keep the x and y components separate instead of combining them
        with sqrt(dx² + dy²).

        IMPORTANT: this method reads prev_positions_px but does NOT update it.
        estimate_speed() is responsible for updating that store. So call
        estimate_speed() FIRST each frame, then this method, so both see the
        same previous position. (main.py does this in the right order.)
        """
        prev_px = self.prev_positions_px.get(track_id)
        if prev_px is None:
            # first sighting - no previous frame to compare against
            return 0.0, 0.0

        # --- ego-motion compensation (same as estimate_speed) ---
        # project the previous pixel position forward by the camera motion H.
        # whatever movement is LEFT OVER after that is the vehicle's own motion.
        prev_pt = np.array([[[float(prev_px[0]), float(prev_px[1])]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(prev_pt, self.current_ego_H)
        projected_px = projected[0][0]

        dx_px = curr_center_px[0] - projected_px[0]   # + = moved right
        dy_px = curr_center_px[1] - projected_px[1]   # + = moved down the image

        # --- convert pixel movement to metres (same scale logic as speed) ---
        if self.current_depth_map is not None and bbox is not None:
            depth_m = self.depth_estimator.get_vehicle_depth(
                self.current_depth_map, bbox
            )
            if depth_m is not None and depth_m > 0.1:
                focal_length = frame_width * self.focal_length_factor
                metres_per_pixel = depth_m / focal_length
                dx_m = dx_px * metres_per_pixel
                dy_m = dy_px * metres_per_pixel
            else:
                dx_m, dy_m = self._pixels_to_metres_fallback(dx_px, dy_px, frame_width)
        else:
            dx_m, dy_m = self._pixels_to_metres_fallback(dx_px, dy_px, frame_width)

        # lateral speed = horizontal motion (across the road)
        lateral_speed = dx_m / dt

        # forward speed = vertical motion, sign flipped so that moving UP the
        # image (away from ego, smaller y) is POSITIVE forward speed.
        # in image coords y grows downward, so we negate.
        forward_speed = -dy_m / dt

        return round(float(forward_speed), 2), round(float(lateral_speed), 2)

    # ------------------------------------------------------------------
    # STEP 5 - ACCELERATION, JERK, HEADING (unchanged in logic)
    # ------------------------------------------------------------------

    def estimate_acceleration(self, track_id, curr_speed, dt=1.0):
        """
        Acceleration = how much speed changed since last frame, in m/s².
        We store the previous speed per vehicle and subtract.
        High negative acceleration = hard braking.
        """
        prev_speed = self.prev_speeds.get(track_id, curr_speed)
        # speed is in km/h, convert difference to m/s before dividing by time
        speed_diff_ms = (curr_speed - prev_speed) / 3.6
        acceleration = round(speed_diff_ms / dt, 3)
        self.prev_speeds[track_id] = curr_speed
        return acceleration

    def estimate_jerk(self, track_id, curr_acceleration, dt=1.0):
        """
        Jerk = how much acceleration changed since last frame, in m/s³.
        High jerk = sudden change in braking force = panic stop or release.
        Used by annotator Rule 2 to distinguish smooth deceleration from
        abrupt braking.
        """
        prev_acc = self.prev_accelerations.get(track_id, curr_acceleration)
        jerk = round((curr_acceleration - prev_acc) / dt, 3)
        self.prev_accelerations[track_id] = curr_acceleration
        return jerk

    def estimate_heading(self, track_id, curr_center_px):
        """
        Heading angle in degrees: direction the vehicle is travelling.
        0° = straight ahead (moving up in image = moving away from ambulance).
        Positive = drifting right, negative = drifting left.

        We use raw pixel positions here (not ego-compensated) because heading
        is used for lane-change detection, and a gradual lateral drift is
        what we are looking for whether or not the ambulance is also moving.
        """
        prev = self.prev_positions_px.get(track_id)
        if prev is None:
            return 0.0
        dx = curr_center_px[0] - prev[0]
        dy = curr_center_px[1] - prev[1]
        angle = round(float(np.degrees(np.arctan2(dy, dx))), 2)
        return angle

    # ------------------------------------------------------------------
    # STEP 6 - DISTANCE, LANE, LATERAL OFFSET
    # ------------------------------------------------------------------

    def estimate_distance_to_ego(self, vehicle_center_px, frame_width,
                                  frame_height, bbox=None):
        """
        Distance in metres from the vehicle to the ambulance (ego).

        If depth is available, use the vehicle's depth value directly -
        this is the forward distance and is already in real metres.

        Fallback: measure pixel distance from vehicle centre to the bottom
        centre of the frame (where the ambulance bonnet is), scaled by the
        lane-width factor. This was the old method and underestimates badly
        at range.

        Bug fix from old version: old code used frame_width as the y
        coordinate for ego position instead of frame_height. Fixed here.
        """
        if self.current_depth_map is not None and bbox is not None:
            depth_m = self.depth_estimator.get_vehicle_depth(
                self.current_depth_map, bbox
            )
            if depth_m is not None and depth_m > 0.1:
                # Lateral distance from centre
                focal_length = frame_width * self.focal_length_factor
                px_offset = vehicle_center_px[0] - frame_width / 2
                lateral_m = (px_offset * depth_m) / focal_length
                # Total 3D distance = sqrt(forward² + lateral²)
                return round(float(np.sqrt(depth_m**2 + lateral_m**2)), 2)

        # Fallback (old method, bug fixed)
        ego_center = [frame_width // 2, frame_height]  # was frame_width, now frame_height
        scale = (3 * self.LANE_WIDTH_METERS) / frame_width
        dx = (vehicle_center_px[0] - ego_center[0]) * scale
        dy = (vehicle_center_px[1] - ego_center[1]) * scale
        return round(float(np.sqrt(dx**2 + dy**2)), 2)

    def estimate_lane_id(self, center_x, frame_width):
        """
        Which of the 3 lanes is the vehicle in?
        Lane 1 = leftmost, Lane 2 = centre, Lane 3 = rightmost.

        This is still the equal-thirds approximation because lane detection
        failed (see lane_detector.py). Do not change this without first
        fixing lane_detector.py.
        """
        lane_width_px = frame_width / 3
        lane = int(center_x / lane_width_px) + 1
        return min(lane, 3)

    def estimate_lateral_offset(self, center_x, frame_width):
        """
        How far is the vehicle from the centre of its lane, in metres?
        Positive = right of lane centre, negative = left.

        Used by annotator Rule 1 (lateral speed > 0.5 m/s) and Rule 3
        (cumulative lateral drift > 0.8m over 3s).
        """
        scale = (3 * self.LANE_WIDTH_METERS) / frame_width
        lane_width_px = frame_width / 3
        lane_id = self.estimate_lane_id(center_x, frame_width)
        lane_center_px = (lane_id - 0.5) * lane_width_px
        offset = round((center_x - lane_center_px) * scale, 2)
        return offset

    # ------------------------------------------------------------------
    # CONVENIENCE: process one full frame
    # ------------------------------------------------------------------

    def process_frame(self, frame):
        """
        Call this once per frame at the start of the pipeline loop.
        It runs ego motion estimation and depth estimation together and
        caches both results internally.

        Returns (ego_H, depth_map) so the caller (main.py) can pass
        ego_H to the EMAP tracker if needed.

        main.py should call this BEFORE calling tracker.update() so that
        the tracker gets the freshest ego_H for EMAP compensation.
        """
        ego_H = self.estimate_ego_motion(frame)
        depth_map = self.compute_depth_map(frame)
        return ego_H, depth_map
