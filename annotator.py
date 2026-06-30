import numpy as np


class HeuristicAnnotator:
    """
    Labels each vehicle's behaviour per frame when emergency is active.
    Labels: normal / braked_abruptly / yielded / failed_to_yield

    INPUT QUALITY: positions are RTS-smoothed (smoother.py) before any metric
    reaches here. Acceleration is LONGITUDINAL (change in signed forward speed,
    highD-style), so the braking threshold is physically meaningful.

    ─────────────────────────────────────────────────────────────────────────
    BRAKING RULES (take priority over yield rules)
    ─────────────────────────────────────────────────────────────────────────
    BRAKE     — acceleration ≤ −2.5 m/s²
                (converging value across braking literature)
    BRAKE ONSET — acceleration ≤ −1.5 m/s² AND jerk ≤ −3.0 m/s³
                  catches a panic stop split across two 1 Hz frames.

    ─────────────────────────────────────────────────────────────────────────
    YIELD RULES
    ─────────────────────────────────────────────────────────────────────────
    RULE 1 — SUSTAINED lateral speed ≥ 0.5 m/s for ≥ YIELD_PERSIST consecutive
             frames, AND the motion is AWAY from x = 0 (the ambulance's path).

             Threshold: Pierson et al. 2019 (highD German highway).
             Persistence: nuScenes validation measured the lateral-speed noise
             floor at ~0.44 m/s at 1 Hz — almost equal to the 0.5 m/s threshold.
             A noise spike lasts one frame; a real yield sustains for 2–4 s.
             Requiring YIELD_PERSIST consecutive frames separates signal from
             noise without needing a lower noise floor.
             Direction: a vehicle moving TOWARD x = 0 (toward the ambulance) is
             NOT yielding. If |x_meters| > CENTRE_DEAD_BAND and lateral_speed
             points toward centre, yield_lateral is zeroed.

    RULE 3 — cumulative lateral ≥ 0.8 m over 3 s, monotonic
             Window: highD lane-change durations (Krajewski et al. 2018);
             0.8 m threshold empirical.

    ─────────────────────────────────────────────────────────────────────────
    FAILED-TO-YIELD
    ─────────────────────────────────────────────────────────────────────────
    Within 20 m, observed for ≥ MIN_OBSERVED_FRAMES, nothing triggered.
    The 80 % co-operation rate from Cortés & Stefoni 2023 means some
    failed_to_yield labels are expected and are not annotation errors.

    ─────────────────────────────────────────────────────────────────────────
    REMOVED RULES
    ─────────────────────────────────────────────────────────────────────────
    Rule 2 (heading ≥ 15°) and Rule 5 (heading increasing 3 frames) were
    removed because estimate_heading() could not produce reliable values at
    1 Hz on an uncalibrated dashcam and was returning 0 for every frame,
    making those rules permanently silent dead code.

    Rule 4 (speed drop ≥ 5 km/h) was removed. It operated on speed_kmh (a
    magnitude), which carries the zero-crossing artifact already fixed for
    acceleration.
    """

    # Lateral speed threshold — Pierson et al. 2019 (highD)
    YIELD_LATERAL_SPEED  = 0.5    # m/s
    YIELD_PERSIST        = 2      # consecutive frames lateral speed must hold

    # Directional filter: vehicles within this of x=0 yield in any direction.
    # Outside this band, lateral motion toward centre is not counted.
    CENTRE_DEAD_BAND     = 0.5    # metres

    # Cumulative lateral drift rule — empirical; window from highD durations
    YIELD_CUMULATIVE     = 0.8    # metres
    CUMULATIVE_WINDOW    = 3      # seconds (= frames at 1 Hz)

    # Braking thresholds
    ABRUPT_BRAKE_THRESHOLD = -2.5  # m/s²
    BRAKE_ONSET_ACCEL      = -1.5  # m/s²
    BRAKE_ONSET_JERK       = -3.0  # m/s³

    # Proximity limits
    PROXIMITY_THRESHOLD   = 50.0   # metres — outer limit for any annotation
    FAILED_YIELD_PROXIMITY = 20.0  # metres — inner limit for failed_to_yield
    MIN_OBSERVED_FRAMES   = 3      # frames of history before failed_to_yield fires

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