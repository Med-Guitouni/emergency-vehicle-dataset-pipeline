import cv2
import numpy as np


class HomographyEstimator:
    """
    Converts each detected vehicle's pixel position to real-world metres, then
    derives relative velocity, distance to ego, lane and lateral offset from
    those metric positions.

    =========================================================================
    HOW POSITION IS COMPUTED — ground-plane pinhole projection
    =========================================================================
    A vehicle's tyres touch the road. The road is a flat plane a known height
    (camera_height) below the camera. Basic camera geometry gives the forward
    distance directly, with no depth model needed:

        y_forward = camera_height * focal_length / pixels_below_horizon

    "pixels_below_horizon" is how far the bottom-centre of the bounding box
    sits below the horizon line. The lower in the image the wheels are, the
    closer the vehicle. Metric by construction — the only unknowns are
    focal_length, camera_height and cx (optical centre column), all estimated
    from the lane-dash calibration protocol.

    =========================================================================
    CALIBRATION
    =========================================================================
    camera_height    — 1.4 m, confirmed by lane-width measurement protocol
                       (mean 1.40 m across 6 frames, ~10 % spread).
    focal_length_factor — frame_width * 0.72. Mean measured ~0.77 with high
                       (~31 %) spread across frames; treat as directionally
                       confirmed but not precisely nailed down.
    horizon_ratio    — 0.60 (fraction down the cropped frame). Compromise
                       across frames that genuinely disagreed (0.55–0.62),
                       most likely due to real road-grade differences.
    CX_RATIO         — 0.47. Measured from vanishing point of lane lines
                       across 6 frames. Below 0.5 means the camera's forward
                       axis points slightly right of image centre.

    camera_height and focal_length_factor both scale distance linearly, so
    (camera_height * focal_length_factor) is the single calibration constant.
    Override attributes (override_focal_px, override_cx, override_horizon_row)
    can be set externally for per-video calibration from calibration_log.json.

    =========================================================================
    LIMITATION — speed is RELATIVE, not absolute
    =========================================================================
    Velocity is the change in a vehicle's position RELATIVE TO THE AMBULANCE
    per second. A car matching our speed reads ~0; a car we overtake reads
    negative forward speed. Relative speed is still the correct yielding
    signal: a car pulling aside has a clear sideways relative velocity
    regardless of absolute speeds.
    """

    LANE_WIDTH_METERS = 3.75   # German Autobahn lane width (fallback only)
    MAX_FORWARD_METERS = 250.0  # beyond this, 1 px ≈ tens of metres — unreliable
    SHOULDER_METERS = 3.0       # hard shoulder / Standstreifen beyond outermost lane
    CX_RATIO = 0.47             # optical centre column as fraction of frame width

    def __init__(self, camera_height=1.4, focal_length_factor=0.72,
                 horizon_ratio=0.60):
        self.camera_height = camera_height
        self.focal_length_factor = focal_length_factor
        self.horizon_ratio = horizon_ratio

        # optional per-video calibration overrides (e.g. from calibration_log.json)
        self.override_focal_px = None
        self.override_cx = None
        self.override_horizon_row = None

        # previous METRIC position per track_id -> (x_m, y_m), for velocity
        self.prev_positions_m = {}

        # previous forward speed and acceleration per track_id, for accel/jerk
        self.prev_speeds = {}
        self.prev_accelerations = {}

    # ------------------------------------------------------------------
    # POSITION — ground-plane pinhole projection
    # ------------------------------------------------------------------

    def get_vehicle_position(self, bbox, vehicle_type, frame_width,
                             frame_height, lane_info=None):
        """
        Project a vehicle's bounding box onto the road plane.
        Returns (x_meters, y_meters, position_reliable).

        Reliability rules (Stein, Mobileye, IEEE IV 2003; Tuohy IV 2010):
          TOP CLIPPED  — only roof missing, tyres visible → RELIABLE.
          SIDE CLIPPED — bottom_x centre is biased; better to flag and let
                         RTS smoother interpolate → UNRELIABLE.
          BOTTOM CLIPPED — projection input missing → UNRELIABLE.
          LATERAL CLAMP FIRED — physically impossible value → UNRELIABLE.

        x_meters: + = right of ambulance centre, − = left
        y_meters: distance ahead (always ≥ 0, larger = further)
        """
        x1, y1, x2, y2 = bbox
        EDGE = 2

        top_clipped    = y1 <= EDGE
        bottom_clipped = y2 >= frame_height - EDGE
        left_clipped   = x1 <= EDGE
        right_clipped  = x2 >= frame_width - EDGE
        side_clipped   = left_clipped or right_clipped

        f  = (self.override_focal_px
              if self.override_focal_px is not None
              else frame_width * self.focal_length_factor)
        cx = (self.override_cx
              if self.override_cx is not None
              else frame_width * self.CX_RATIO)
        horizon_row = (self.override_horizon_row
                       if self.override_horizon_row is not None
                       else self.horizon_ratio * frame_height)

        delta_y  = y2 - horizon_row
        min_delta = (self.camera_height * f) / self.MAX_FORWARD_METERS
        if delta_y < min_delta:
            delta_y = min_delta
        y_forward = (self.camera_height * f) / delta_y
        if y_forward > self.MAX_FORWARD_METERS:
            y_forward = self.MAX_FORWARD_METERS

        x_lateral = ((x1 + x2) / 2.0 - cx) * y_forward / f

        # lateral plausibility clamp (lane-aware)
        if lane_info is not None:
            half_road = (lane_info["lanes"] * lane_info["lane_width_meters"]) / 2.0
        else:
            half_road = (3 * self.LANE_WIDTH_METERS) / 2.0
        max_lateral = half_road + self.SHOULDER_METERS

        clamped = False
        if x_lateral > max_lateral:
            x_lateral = max_lateral
            clamped = True
        elif x_lateral < -max_lateral:
            x_lateral = -max_lateral
            clamped = True

        reliable = not bottom_clipped and not side_clipped and not clamped

        return (round(float(x_lateral), 2),
                round(float(y_forward), 2),
                reliable)

    # ------------------------------------------------------------------
    # VELOCITY — relative, derived from metric position change
    # ------------------------------------------------------------------

    def estimate_relative_velocity(self, track_id, x_m, y_m, dt=1.0):
        """
        Velocity RELATIVE TO THE AMBULANCE from the change in metric position.

        forward_speed_ms : along the road, m/s. + = away from ego, − = toward
        lateral_speed_ms : across the road, m/s. + = right, − = left
        speed_kmh        : overall magnitude in km/h

        Reads AND updates the stored previous position, so call exactly once
        per vehicle per frame.
        """
        prev = self.prev_positions_m.get(track_id)
        self.prev_positions_m[track_id] = (x_m, y_m)

        if prev is None:
            return 0.0, 0.0, 0.0

        dx = x_m - prev[0]
        dy = y_m - prev[1]

        lateral_speed = dx / dt
        forward_speed = dy / dt
        speed_kmh     = (np.sqrt(dx * dx + dy * dy) / dt) * 3.6

        return (round(float(forward_speed), 2),
                round(float(lateral_speed), 2),
                round(float(speed_kmh), 2))

    def estimate_acceleration(self, track_id, forward_speed_ms, dt=1.0):
        """
        Longitudinal acceleration in m/s² = change in signed forward speed.

        Using signed forward speed (not magnitude) avoids the phantom-braking
        artifact that occurred when relative velocity crossed zero: the old
        magnitude-based version collapsed to 0 and rebounded, manufacturing a
        hard-brake + acceleration even though nothing physical happened.
        Negative = braking relative to ego (highD style).
        """
        prev = self.prev_speeds.get(track_id, forward_speed_ms)
        acceleration = round((forward_speed_ms - prev) / dt, 3)
        self.prev_speeds[track_id] = forward_speed_ms
        return acceleration

    def estimate_jerk(self, track_id, curr_acceleration, dt=1.0):
        """
        Jerk in m/s³ = change in acceleration.
        High jerk = sudden onset (panic stop). Used by the brake-onset rule.
        """
        prev_acc = self.prev_accelerations.get(track_id, curr_acceleration)
        jerk = round((curr_acceleration - prev_acc) / dt, 3)
        self.prev_accelerations[track_id] = curr_acceleration
        return jerk

    # ------------------------------------------------------------------
    # DISTANCE / LANE / LATERAL OFFSET
    # ------------------------------------------------------------------

    def estimate_distance_to_ego(self, x_m, y_m):
        """Straight-line distance to the ambulance (the origin) in metres."""
        return round(float(np.sqrt(x_m * x_m + y_m * y_m)), 2)

    def estimate_lane_id(self, x_meters, lane_info=None):
        """
        Lane number 1 (leftmost) to N (rightmost) from METRIC lateral position.

        Uses x_meters (road-plane coordinates) instead of pixel centre_x.
        This matches the lane assignment used by surrounding.py so that the
        exported lane_id and the surrounding-vehicle relationships are
        consistent.

        The ambulance sits at x=0. Lanes are assumed symmetric: the road
        extends from −half_road to +half_road.
        """
        if lane_info is None:
            n_lanes   = 3
            lane_width = self.LANE_WIDTH_METERS
        else:
            n_lanes   = lane_info["lanes"]
            lane_width = lane_info["lane_width_meters"]

        half_road = (n_lanes * lane_width) / 2.0
        # shift x so lane 1 starts at 0
        x_shifted = x_meters + half_road
        lane = int(x_shifted / lane_width) + 1
        return min(max(lane, 1), n_lanes)

    def estimate_lateral_offset(self, x_meters, lane_info=None):
        """
        Distance from the vehicle's lane centre in metres. + = right of centre.
        Derived from metric x_meters (same coordinate system as estimate_lane_id).
        """
        if lane_info is None:
            n_lanes   = 3
            lane_width = self.LANE_WIDTH_METERS
        else:
            n_lanes   = lane_info["lanes"]
            lane_width = lane_info["lane_width_meters"]

        half_road   = (n_lanes * lane_width) / 2.0
        lane_id     = self.estimate_lane_id(x_meters, lane_info)
        lane_centre = -half_road + (lane_id - 0.5) * lane_width
        return round(float(x_meters - lane_centre), 2)

    def estimate_ttc_to_ego(self, y_meters, forward_speed_ms):
        """
        Seconds until this vehicle reaches the ambulance at current closing speed.
        Only meaningful for vehicles ahead (y_meters > 0) that are approaching
        (forward_speed_ms < 0, i.e. the gap is shrinking).
        Returns None for vehicles behind, moving away, or closing too slowly
        to distinguish from noise (< 0.1 m/s).
        """
        closing_speed = -forward_speed_ms
        if y_meters <= 0 or closing_speed < 0.1:
            return None
        return round(y_meters / closing_speed, 2)