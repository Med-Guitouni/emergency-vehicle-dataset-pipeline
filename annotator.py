import numpy as np


class HeuristicAnnotator:
    """
     ─────────────────────────────────────────────────────────────────────────






    Outdated : thresholds need recalibration if we want automatic use .
                Manual review will replace it









    ─────────────────────────────────────────────────────────────────────────

    ─────────────────────────────────────────────────────────────────────────
    BRAKING RULES (take priority over yield rules-manual review cant detect barking)
    ─────────────────────────────────────────────────────────────────────────
    BRAKE     — acceleration ≤ −2.5 m/s²
                (converging value across braking literature)
    BRAKE ONSET — acceleration ≤ −1.5 m/s² AND jerk ≤ −3.0 m/s³
                  catches a panic stop split across two 1 Hz frames.

    ─────────────────────────────────────────────────────────────────────────
    YIELD RULES
    ─────────────────────────────────────────────────────────────────────────
    RULE 1 — SUSTAINED lateral speed ≥ 0.5 m/s for ≥ YIELD_PERSIST consecutive
             EXPORTED frames, AND the motion is AWAY from x = 0 (the
             ambulance's path).

             Threshold: Pierson et al. 2019 (highD German highway).


    RULE 3 — cumulative lateral ≥ 0.8 m over CUMULATIVE_WINDOW, monotonic
             Window: highD lane-change durations (Krajewski et al. 2018);
             0.8 m threshold empirical.

    RULE X — OUT OF ROAD BOUNDARY (x_meters at the lateral clamp limit)


    ─────────────────────────────────────────────────────────────────────────
    FAILED-TO-YIELD
    ─────────────────────────────────────────────────────────────────────────
    Within 20 m, observed for ≥ MIN_OBSERVED_FRAMES (export frames), nothing
    triggered.

    ─────────────────────────────────────────────────────────────────────────
    REMOVED RULES
    ─────────────────────────────────────────────────────────────────────────
    Rule 2 (heading ≥ 15°) and Rule 5 (heading increasing 3 frames) were
    removed

    Rule 4 (speed drop ≥ 5 km/h) was removed. It operated on speed_kmh (a
    magnitude), which carries the zero-crossing artifact already fixed for
    acceleration.
    """

    # This pipeline's export rate -- see main.py's EXPORT_FPS. Duplicated
    # here (rather than imported) to keep this file runnable standalone;
    # keep in sync with main.py if that ever changes.
    EXPORT_FPS = 10

    # Lateral speed threshold — Pierson et al. 2019 (highD)
    YIELD_LATERAL_SPEED  = 0.5    # m/s
    YIELD_PERSIST         = 2 * EXPORT_FPS  # = 2 real seconds sustained, in export frames

    # Directional filter: vehicles within this of x=0 yield in any direction.
    # Outside this band, lateral motion toward centre is not counted.
    CENTRE_DEAD_BAND     = 0.5    # metres

    # Cumulative lateral drift rule — empirical; window from highD durations
    YIELD_CUMULATIVE      = 0.8    # metres
    CUMULATIVE_WINDOW      = 3 * EXPORT_FPS  # = 3 real seconds, in export frames

    # Out-of-road-boundary rule — same values as lane_config.py's LANE_WIDTHS
    # and homography.py's SHOULDER_METERS. Duplicated here because the
    # vehicle dict carries lanes_total/road_type but not lane_width_meters.
    LANE_WIDTHS = {
        "highway":      3.75,
        "urban":        3.00,
        "intersection": 3.00,
        "roundabout":   3.00,
        "unknown":      3.00,
    }
    SHOULDER_METERS          = 3.0    # metres, matches homography.py
    BOUNDARY_TOLERANCE       = 0.01   # metres, float-rounding slack

    # Braking thresholds
    ABRUPT_BRAKE_THRESHOLD = -2.5  # m/s²
    BRAKE_ONSET_ACCEL      = -1.5  # m/s²
    BRAKE_ONSET_JERK       = -3.0  # m/s³

    # Proximity limits
    PROXIMITY_THRESHOLD   = 50.0   # metres — outer limit for any annotation
    FAILED_YIELD_PROXIMITY = 20.0  # metres — inner limit for failed_to_yield
    MIN_OBSERVED_FRAMES    = 3 * EXPORT_FPS  # = 3 real seconds, in export frames

    def __init__(self):
        self.lateral_history = {}   # track_id -> deque of lateral_offset values
        self.frames_seen     = {}   # track_id -> count of frames observed
        self.lateral_run     = {}   # track_id -> consecutive frames above threshold

    def annotate(self, vehicle, emergency_active):
        if not emergency_active:
            return "normal"

        tid          = vehicle["track_id"]
        lateral_spd  = vehicle.get("lateral_speed_ms", 0.0)   # signed
        x_pos        = vehicle.get("x_meters", 0.0)
        acceleration = vehicle.get("acceleration", 0.0)        # longitudinal, signed
        jerk         = vehicle.get("jerk", 0.0)
        distance     = vehicle.get("distance_to_ego", 999.0)
        curr_lateral = vehicle.get("lateral_offset", 0.0)

        self.frames_seen[tid] = self.frames_seen.get(tid, 0) + 1

        # history for Rule 3
        self.lateral_history.setdefault(tid, []).append(curr_lateral)
        if len(self.lateral_history[tid]) > self.CUMULATIVE_WINDOW:
            self.lateral_history[tid].pop(0)

        # ── directional lateral speed (Rule 1) ──────────────────────────
        # Vehicles outside the centre dead band must be moving AWAY from x=0.
        # Motion toward the ambulance's path is not a yield.
        if (abs(x_pos) > self.CENTRE_DEAD_BAND
                and np.sign(lateral_spd) != np.sign(x_pos)):
            yield_lateral = 0.0
        else:
            yield_lateral = abs(lateral_spd)

        # update consecutive-frame run counter
        if yield_lateral >= self.YIELD_LATERAL_SPEED:
            self.lateral_run[tid] = self.lateral_run.get(tid, 0) + 1
        else:
            self.lateral_run[tid] = 0

        if distance > self.PROXIMITY_THRESHOLD:
            return "normal"

        # ── braking takes priority over yield rules ──────────────────────
        if acceleration <= self.ABRUPT_BRAKE_THRESHOLD:
            return "braked_abruptly"
        if (acceleration <= self.BRAKE_ONSET_ACCEL
                and jerk <= self.BRAKE_ONSET_JERK):
            return "braked_abruptly"

        # ── Rule X: x_meters at the road's lateral boundary ───────────────
        # homography.py clamps x_meters to ±max_lateral; sitting at that
        # boundary means the vehicle has effectively left the road.
        lanes_total = vehicle.get("lanes_total", 3)
        road_type   = vehicle.get("road_type", "unknown")
        lane_width  = self.LANE_WIDTHS.get(road_type, self.LANE_WIDTHS["unknown"])
        max_lateral = (lanes_total * lane_width) / 2.0 + self.SHOULDER_METERS
        if abs(x_pos) >= max_lateral - self.BOUNDARY_TOLERANCE:
            return "yielded"

        # ── Rule 1: sustained lateral speed away from centre ─────────────
        if self.lateral_run.get(tid, 0) >= self.YIELD_PERSIST:
            return "yielded"

        # ── Rule 3: cumulative lateral drift over the window, monotonic ──
        if len(self.lateral_history[tid]) >= self.CUMULATIVE_WINDOW:
            history   = self.lateral_history[tid]
            total     = abs(history[-1] - history[0])
            direction = history[-1] - history[0]
            consistent = all(
                (history[i + 1] - history[i]) * direction >= 0
                for i in range(len(history) - 1)
            )
            if total >= self.YIELD_CUMULATIVE and consistent:
                return "yielded"

        # ── failed_to_yield: close, observed long enough, nothing fired ──
        if (distance <= self.FAILED_YIELD_PROXIMITY
                and self.frames_seen[tid] >= self.MIN_OBSERVED_FRAMES):
            return "failed_to_yield"

        return "normal"