import cv2
import numpy as np

class HomographyEstimator:
    """
    Handles all spatial calculations for detected vehicles.
    Converts pixel positions from the dashcam image into real world meters
    using camera geometry and lane width assumptions. Also computes speed,
    acceleration, jerk, heading and lane position per vehicle per frame.

    ---

    CURRENT LIMITATIONS

    Speed is relative to the ambulance not absolute. When the ambulance moves
    at 100 km/h and a car ahead moves at 95 km/h, the car appears to move at
    only 5 km/h in our output. This makes speed and acceleration values
    scientifically unreliable for the distributional analysis.

    Camera intrinsics are estimated not measured. Focal length is set to
    frame_width times 0.8 and mount height is fixed at 1.4 meters. These are
    reasonable defaults for a dashcam but should be calibrated from the actual
    device specs for production quality results.

    Lane detection assumes exactly 3 equal lanes always visible across the frame.
    This breaks at intersections, on-ramps, and when the ambulance changes lanes.
    attempted to fix this using UFLD v2 and YOLOP but both failed because
    vehicles on the shoulder physically cover the lane markings at exactly the
    moments we care about most. This is left as a known limitation.

    Lateral offset is measured from an estimated lane center not from actual
    road markings. A proper lane detection model with static boundary estimation
    would improve this significantly.

    Distance to ego assumes the ambulance is at the bottom center of the frame.
    This is a rough approximation depending on dashcam mount position.

    All calculations assume a flat ground plane which fails on hills and ramps.

    ---

    OPTION 2 - EGO MOTION COMPENSATION - IN PROGRESS - NOT YET IMPLEMENTED

    To fix the speed problem two steps are needed:

    Step 1 - Add Depth Anything V2
    Run Depth Anything V2 on each frame to get a per-pixel depth map that tells
    us how far each vehicle is from the camera in real meters. Without this,
    ego motion compensation cannot be done accurately because the math requires
    knowing the actual distance to each object.
    Model: https://github.com/DepthAnything/Depth-Anything-V2
    Scientific basis: Cristian suggested depth estimation models.
    The EMAP paper (Mahdian et al. 2024) confirms depth maps are required for
    proper ego motion decoupling in the Kalman Filter.

    Step 2 - Integrate EMAP with ByteTrack
    EMAP (Ego-Motion Aware Target Prediction) reformulates the Kalman Filter
    inside ByteTrack to decouple camera rotation and translation from object motion.
    The estimate_ego_motion method in this class already computes camera motion
    using optical flow - this output can be fed directly into EMAP as odometry.
    On the KITTI dataset EMAP reduced identity switches in ByteTrack by 33 percent
    and improved HOTA tracking accuracy by more than 5 percent.
    Open source code: https://github.com/noyzzz/EMAP


    Files to modify when implementing:
    - main.py: add Depth Anything V2 inference per frame before tracking
    - tracker.py: replace standard ByteTrack with ByteTrack + EMAP
    - homography.py: pass depth map and camera odometry to EMAP module
    """

    # standard German highway lane width used as scale reference
    LANE_WIDTH_METERS = 3.75

    def __init__(self):
        self.prev_gray = None
        self.prev_positions = {}
        self.prev_speeds = {}
        self.prev_accelerations = {}

    def estimate_ego_motion(self, frame):
        # tracks how the camera itself moved between frames using background points
        # needed to separate camera movement from vehicle movement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return np.eye(3)

        prev_points = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10
        )

        if prev_points is None or len(prev_points) < 4:
            self.prev_gray = gray
            return np.eye(3)

        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_points, None
        )

        good_prev = prev_points[status == 1]
        good_curr = curr_points[status == 1]

        if len(good_prev) < 4:
            self.prev_gray = gray
            return np.eye(3)

        H, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        self.prev_gray = gray
        return H if H is not None else np.eye(3)

    def get_bev_position(self, bottom_center, frame_width, frame_height):
        # projects bounding box bottom center to real world meters using camera geometry
        # assumes dashcam mounted at 1.4m height and flat ground plane
        # gives forward and lateral distance of each vehicle from the ambulance
        focal_length = frame_width * 0.8
        camera_height = 1.4

        cx = frame_width / 2
        cy = frame_height / 2

        px = bottom_center[0] - cx
        py = bottom_center[1] - cy

        if py <= 0:
            return 0.0, 0.0

        distance_forward = (camera_height * focal_length) / py
        distance_lateral = (px * distance_forward) / focal_length

        return round(float(distance_lateral), 2), round(float(distance_forward), 2)

    def estimate_speed(self, track_id, curr_center, frame_width, dt=1.0):
        # measures pixel displacement between frames and converts to km/h
        # uses lane width as scale reference
        # note: relative speed only - ego motion compensation not yet implemented
        scale = (3 * self.LANE_WIDTH_METERS) / frame_width
        prev = self.prev_positions.get(track_id)
        if prev is None:
            self.prev_positions[track_id] = curr_center
            return 0.0
        dx = (curr_center[0] - prev[0]) * scale
        dy = (curr_center[1] - prev[1]) * scale
        distance = np.sqrt(dx**2 + dy**2)
        speed_kmh = round((distance / dt) * 3.6, 2)
        self.prev_positions[track_id] = curr_center
        return speed_kmh

    def estimate_acceleration(self, track_id, curr_speed, dt=1.0):
        # speed change between frames in m/s2
        prev_speed = self.prev_speeds.get(track_id, curr_speed)
        acceleration = round((curr_speed - prev_speed) / dt, 3)
        self.prev_speeds[track_id] = curr_speed
        return acceleration

    def estimate_jerk(self, track_id, curr_acceleration, dt=1.0):
        # acceleration change between frames in m/s3
        # measures smoothness of braking - high jerk means sudden harsh braking
        prev_acc = self.prev_accelerations.get(track_id, curr_acceleration)
        jerk = round((curr_acceleration - prev_acc) / dt, 3)
        self.prev_accelerations[track_id] = curr_acceleration
        return jerk

    def estimate_heading(self, track_id, curr_center):
        # direction the vehicle is moving in degrees
        # 0 = straight ahead, positive = right, negative = left
        prev = self.prev_positions.get(track_id)
        if prev is None:
            return 0.0
        dx = curr_center[0] - prev[0]
        dy = curr_center[1] - prev[1]
        angle = round(float(np.degrees(np.arctan2(dy, dx))), 2)
        return angle

    def estimate_distance_to_ego(self, vehicle_center, frame_width):
        # distance in meters between vehicle and ambulance
        # ambulance assumed at bottom center of frame where dashcam is mounted
        ego_center = [frame_width // 2, frame_width]
        scale = (3 * self.LANE_WIDTH_METERS) / frame_width
        dx = (vehicle_center[0] - ego_center[0]) * scale
        dy = (vehicle_center[1] - ego_center[1]) * scale
        return round(float(np.sqrt(dx**2 + dy**2)), 2)

    def estimate_lane_id(self, center_x, frame_width):
        # assigns lane number 1 to 3 based on horizontal position in frame
        # assumes 3 equal lanes visible across the full frame width
        lane_width_px = frame_width / 3
        lane = int(center_x / lane_width_px) + 1
        return min(lane, 3)

    def estimate_lateral_offset(self, center_x, frame_width):
        # distance from lane center in meters
        # positive = right of lane center, negative = left
        scale = (3 * self.LANE_WIDTH_METERS) / frame_width
        lane_width_px = frame_width / 3
        lane_id = self.estimate_lane_id(center_x, frame_width)
        lane_center_px = (lane_id - 0.5) * lane_width_px
        offset = round((center_x - lane_center_px) * scale, 2)
        return offset
