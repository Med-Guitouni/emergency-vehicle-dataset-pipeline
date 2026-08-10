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

    # NEAR-HORIZON: GROUND-PLANE PROJECTION IS UNSTABLE, SWITCH ESTIMATORS
    # y_forward = camera_height*f / delta_y has derivative -camera_height*f/delta_y^2,
    # so relative error in y_forward ≈ (relative error in delta_y). A small
    # absolute jitter in delta_y (detection edge wobble, see smoother.py's
    # "a few pixels" characterisation) produces a LARGE relative error in
    # y_forward once delta_y itself is small. Verified against real output
    # (compare_distance_estimators.py): a box-height-based estimate is ~2.4x
    # more frame-to-frame stable than ground-plane for exactly these
    # vehicles, since box height has no such small-denominator singularity.
    # Below NEAR_HORIZON_MIN_DELTA_PX, y_forward switches to
    #     y_forward_alt = VEHICLE_HEIGHTS_M[type] * f / box_height_px
    # instead of being computed (badly) from the ground-plane formula.
    # ASSUMED_JITTER_PX / MAX_RELATIVE_Y_ERROR is the same derivation used
    # for both thresholds -- the relative-error formula has the same shape
    # for either denominator (delta_y or box_height_px), so the same
    # tolerance assumption gives the same numeric threshold for both.
    ASSUMED_JITTER_PX     = 3.0    # px, "a few pixels" per smoother.py
    MAX_RELATIVE_Y_ERROR  = 0.15   # tolerate up to 15% relative y_forward error
    NEAR_HORIZON_MIN_DELTA_PX = ASSUMED_JITTER_PX / MAX_RELATIVE_Y_ERROR  # = 20px

    # Below this box height, even the box-height estimator is too noisy to
    # trust — a 15px-tall box has ~20% relative height error from the same
    # 3px jitter, producing ~20% distance error. Above this, the box-height
    # estimator is empirically 2.4x more stable than ground-plane in the
    # near-horizon zone (validated via compare_distance_estimators.py), so
    # near-horizon observations with a box taller than this are marked
    # RELIABLE rather than blanket-unreliable as before.
    MIN_BOX_HEIGHT_PX = 15

    # assumed real-world vehicle heights (m) for the box-height estimator --
    # same assumption set validated in compare_distance_estimators.py
    VEHICLE_HEIGHTS_M = {
        "car":        1.5,
        "truck":      3.5,
        "bus":        3.0,
        "motorcycle": 1.2,
    }

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

        # per-track calibrated real-world height (m) for the near-horizon
        # box-height estimator -- see get_vehicle_position's calibration
        # logic. Falls back to VEHICLE_HEIGHTS_M's generic per-type constant
        # until a track has at least one trustworthy near-boundary
        # ground-plane frame to calibrate from.
        self.calibrated_heights = {}

    # ------------------------------------------------------------------
    # POSITION — ground-plane pinhole projection
    # ------------------------------------------------------------------

    # Calibration zone: ground-plane frames within this multiple of
    # NEAR_HORIZON_MIN_DELTA_PX (but still outside it, i.e. still reliable)
    # are used to calibrate a track's own effective height, rather than
    # frames from very close up where box proportions/perspective differ.
    CALIBRATION_ZONE_MULTIPLIER = 3.0  # = up to 60px from the boundary

    def get_vehicle_position(self, bbox, vehicle_type, frame_width,
                             frame_height, lane_info=None, track_id=None):
        """
        Project a vehicle's bounding box onto the road plane.
        Returns (x_meters, y_meters, position_reliable).

        Reliability rules (Stein, Mobileye, IEEE IV 2003; Tuohy IV 2010):
          TOP CLIPPED  — only roof missing, tyres visible → RELIABLE.
          SIDE CLIPPED — bottom_x centre is biased; better to flag and let
                         RTS smoother interpolate → UNRELIABLE.
          BOTTOM CLIPPED — projection input missing → UNRELIABLE.
          LATERAL CLAMP FIRED — physically impossible value → UNRELIABLE.
          NEAR HORIZON — ground-plane projection switches to a box-height
                         estimate instead (see NEAR_HORIZON_MIN_DELTA_PX
                         above). The box-height estimator is empirically
                         2.4× more stable (compare_distance_estimators.py)
                         and is now marked RELIABLE as long as the box is
                         tall enough to measure accurately (box_height_px
                         >= MIN_BOX_HEIGHT_PX = 15px). Only tiny boxes
                         (< 15px tall) in the near-horizon zone are flagged
                         unreliable, since at that size even box-height has
                         ~20% relative error from a few pixels of jitter.
                         Previously ALL near-horizon observations were
                         blanket-flagged unreliable — this overcorrected,
                         marking 61% of observations unreliable when only
                         ~9% had a second, independent cause (diagnosed via
                         diagnose_two.py on real 4-video output).

        PER-TRACK HEIGHT CALIBRATION (accuracy, not just stability)
        VEHICLE_HEIGHTS_M is a population-level constant per vehicle type --
        a real SUV isn't the same height as a real sedan, so it's a source
        of systematic bias the stability validation didn't measure. Many
        vehicles enter the near-horizon zone by RECEDING, meaning we often
        have a trustworthy ground-plane reading of that SAME vehicle right
        before it crosses into the unstable zone. Whenever a track has a
        reliable ground-plane frame within CALIBRATION_ZONE_MULTIPLIER of
        the boundary, this vehicle's own effective height is back-solved
        from that frame's (trusted) y_forward and box_height_px, and used
        for its own subsequent near-horizon frames instead of the generic
        constant. Tracks with no such frame (near-horizon from their first
        observation) fall back to VEHICLE_HEIGHTS_M as before. track_id=None
        (caller doesn't have one) also falls back, unconditionally.

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

        delta_y_raw = y2 - horizon_row
        near_horizon = delta_y_raw < self.NEAR_HORIZON_MIN_DELTA_PX
        box_height_px = max(y2 - y1, 1e-6)

        if near_horizon:
            # ground-plane is unstable this close to the horizon -- use
            # box-height instead (no delta_y-style singularity). Prefer this
            # track's own calibrated height if we have one (see docstring).
            if track_id is not None and track_id in self.calibrated_heights:
                h_real = self.calibrated_heights[track_id]
            else:
                h_real = self.VEHICLE_HEIGHTS_M.get(vehicle_type, self.VEHICLE_HEIGHTS_M["car"])
            y_forward = (h_real * f) / box_height_px
            if y_forward > self.MAX_FORWARD_METERS:
                y_forward = self.MAX_FORWARD_METERS
        else:
            min_delta = (self.camera_height * f) / self.MAX_FORWARD_METERS
            delta_y = delta_y_raw
            if delta_y < min_delta:
                delta_y = min_delta
            y_forward = (self.camera_height * f) / delta_y
            if y_forward > self.MAX_FORWARD_METERS:
                y_forward = self.MAX_FORWARD_METERS

        # x_lateral from similar triangles -- this relation is general and
        # doesn't depend on which estimator produced y_forward above
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

        reliable = (not bottom_clipped and not side_clipped
                    and not clamped
                    and not (near_horizon and box_height_px < self.MIN_BOX_HEIGHT_PX))

        # CALIBRATION: a trustworthy ground-plane frame within the
        # calibration zone gives us this specific vehicle's own effective
        # height -- store it for this track's future near-horizon frames,
        # replacing the generic per-type constant. Overwritten each
        # qualifying frame (most recent calibration wins).
        if (track_id is not None and reliable and not near_horizon
                and delta_y_raw <= self.NEAR_HORIZON_MIN_DELTA_PX * self.CALIBRATION_ZONE_MULTIPLIER):
            self.calibrated_heights[track_id] = (y_forward * box_height_px) / f

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

    def estimate_lane_position_norm(self, x_meters, lane_info=None):
        """
        Normalised position WITHIN the vehicle's own lane, in [-1, +1].
        0 = lane centre, -1 = left lane edge, +1 = right lane edge.
        This is lateral_offset divided by half the lane width, so it means
        the same thing regardless of lane width (a 1m offset is a bigger
        deal in a 3m lane than a 4m lane). Transfers across road types;
        raw metres do not. (MTP-GO expects this normalised form.)
        """
        if lane_info is None:
            lane_width = self.LANE_WIDTH_METERS
        else:
            lane_width = lane_info["lane_width_meters"]
        offset = self.estimate_lateral_offset(x_meters, lane_info)
        half_lane = lane_width / 2.0
        if half_lane <= 0:
            return 0.0
        return round(max(-1.0, min(1.0, offset / half_lane)), 3)

    def estimate_road_position_norm(self, x_meters, lane_info=None):
        """
        Normalised position across the WHOLE road, in [-1, +1].
        0 = road centre (ego centreline), -1 = left road edge,
        +1 = right road edge (i.e. onto the shoulder). Directly relevant to
        "how far right am I on the entire road" -- i.e. pulling onto the
        hard shoulder to let the ambulance pass. (MTP-GO feature.)
        """
        if lane_info is None:
            n_lanes   = 3
            lane_width = self.LANE_WIDTH_METERS
        else:
            n_lanes   = lane_info["lanes"]
            lane_width = lane_info["lane_width_meters"]
        half_road = (n_lanes * lane_width) / 2.0
        if half_road <= 0:
            return 0.0
        return round(max(-1.0, min(1.0, x_meters / half_road)), 3)

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
