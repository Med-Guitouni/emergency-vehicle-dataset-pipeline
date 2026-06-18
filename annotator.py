class HeuristicAnnotator:
    """
    Labels each vehicle's behaviour per frame when emergency is active.
    Labels: normal, braked_abruptly, yielded, failed_to_yield

    INPUT QUALITY NOTE: positions are RTS-smoothed (smoother.py) before any
    metric reaches this class, and acceleration is LONGITUDINAL (change in
    forward speed, signed) per highD - so the braking threshold compares
    against a physically meaningful quantity.

    BRAKE RULE — acceleration <= -2.5 m/s²  (braking literature)
    BRAKE ONSET — acceleration <= -1.5 m/s² AND jerk <= -3.0 m/s³

    RULE 1 — lateral speed >= 0.5 m/s, SUSTAINED for >= YIELD_PERSIST frames
    Threshold from Pierson et al. 2019 (highD German highway). The PERSISTENCE
    requirement is new and is the key fix: nuScenes validation measured the
    lateral-speed noise floor at ~0.44 m/s on straight-line driving at 1Hz —
    almost equal to the 0.5 m/s threshold. A single-frame trigger therefore
    fired on noise spikes (the 57.7% over-firing). A real yield sustains lateral
    motion for 2-4s; a noise spike lasts one frame. Requiring the threshold to
    hold for consecutive frames separates signal from noise without needing the
    per-frame noise to drop below 0.5 (which is not physically achievable with a
    single uncalibrated camera at 1Hz). Same principle Rule 3 already used.

    RULE 2 — heading angle >= 15 degrees  (concept: Qiu et al. 2025; value empirical)
    // still working on

    RULE 3 — cumulative lateral >= 0.8m over 3s, monotonic  (window: highD durations)
    RULE 4 — speed drop >= 5 km/h  (empirical, §38 StVO; OFF by default — uses a
             magnitude (speed_kmh) which carries the zero-crossing artifact that
             was already fixed for acceleration, and is shaky under relative
             speed. not enabled )
    RULE 5 — heading increasing over 3 consecutive frames  (Qiu et al. 2025)

    note — failed_to_yield needs MIN_OBSERVED_FRAMES of history.
    PROXIMITY — 50m outer, 20m failed-to-yield.
    """

    YIELD_LATERAL_SPEED     = 0.5    # m/s — Pierson et al. 2019 (highD)
    YIELD_PERSIST           = 2      # consecutive frames lateral speed must hold
    YIELD_HEADING_THRESHOLD = 15.0   # degrees — Qiu et al. 2025 (value empirical)
    YIELD_CUMULATIVE        = 0.8    # metres — empirical
    CUMULATIVE_WINDOW       = 3      # seconds — highD lane change durations
    ABRUPT_BRAKE_THRESHOLD  = -2.5   # m/s² — braking literature
    BRAKE_ONSET_ACCEL       = -1.5   # m/s² — empirical (with jerk condition)
    BRAKE_ONSET_JERK        = -3.0   # m/s³ — empirical, panic-brake onset
    YIELD_SPEED_DROP        = 5.0    # km/h — empirical
    ENABLE_SPEED_DROP_RULE  = False  # Rule 4 toggle — OFF pending supervisor review
    PROXIMITY_THRESHOLD     = 50.0   # metres
    FAILED_YIELD_PROXIMITY  = 20.0   # metres
    MIN_OBSERVED_FRAMES     = 3      # frames before failed_to_yield is allowed

    def __init__(self):
        self.prev_speed      = {}
        self.lateral_history = {}
        self.heading_history = {}
        self.frames_seen     = {}
        # count of consecutive frames lateral speed has been >= threshold
        self.lateral_run     = {}

    def annotate(self, vehicle, emergency_active):
        if not emergency_active:
            return "normal"

        tid          = vehicle["track_id"]
        lateral_spd  = abs(vehicle.get("lateral_speed_ms", 0.0))
        curr_speed   = vehicle.get("speed_kmh", 0.0)
        curr_heading = abs(vehicle.get("heading_angle") or 0.0)
        acceleration = vehicle.get("acceleration", 0.0)   # longitudinal, signed
        jerk         = vehicle.get("jerk", 0.0)
        distance     = vehicle.get("distance_to_ego", 999.0)
        curr_lateral = vehicle.get("lateral_offset", 0.0)

        self.frames_seen[tid] = self.frames_seen.get(tid, 0) + 1

        # histories for Rules 3 and 5
        self.lateral_history.setdefault(tid, []).append(curr_lateral)
        if len(self.lateral_history[tid]) > self.CUMULATIVE_WINDOW:
            self.lateral_history[tid].pop(0)

        self.heading_history.setdefault(tid, []).append(curr_heading)
        if len(self.heading_history[tid]) > self.CUMULATIVE_WINDOW:
            self.heading_history[tid].pop(0)

        # update the consecutive-frame run counter for lateral speed
        if lateral_spd >= self.YIELD_LATERAL_SPEED:
            self.lateral_run[tid] = self.lateral_run.get(tid, 0) + 1
        else:
            self.lateral_run[tid] = 0

        prev_spd = self.prev_speed.get(tid, curr_speed)
        self.prev_speed[tid] = curr_speed

        if distance > self.PROXIMITY_THRESHOLD:
            return "normal"

        # --- braking takes priority over yielding rules ---
        if acceleration <= self.ABRUPT_BRAKE_THRESHOLD:
            return "braked_abruptly"
        if acceleration <= self.BRAKE_ONSET_ACCEL and jerk <= self.BRAKE_ONSET_JERK:
            return "braked_abruptly"

        # --- Rule 1: SUSTAINED lateral speed (Pierson 2019 + persistence fix) ---
        # Must hold for >= YIELD_PERSIST consecutive frames. A single noisy frame
        # at ~0.5 m/s (the measured noise floor) can no longer trigger a yield.
        if self.lateral_run.get(tid, 0) >= self.YIELD_PERSIST:
            return "yielded"

        # --- Rule 2: heading angle (Qiu 2025) ---
        if curr_heading >= self.YIELD_HEADING_THRESHOLD:
            return "yielded"

        # --- Rule 3: cumulative lateral drift over the window, monotonic ---
        if len(self.lateral_history[tid]) >= self.CUMULATIVE_WINDOW:
            history    = self.lateral_history[tid]
            total      = abs(history[-1] - history[0])
            direction  = history[-1] - history[0]
            consistent = all(
                (history[i + 1] - history[i]) * direction >= 0
                for i in range(len(history) - 1)
            )
            if total >= self.YIELD_CUMULATIVE and consistent:
                return "yielded"

        # --- Rule 4: moderate speed drop (empirical, OFF by default) ---
        if self.ENABLE_SPEED_DROP_RULE and prev_spd - curr_speed >= self.YIELD_SPEED_DROP:
            return "yielded"

        # --- Rule 5: heading increasing over the window (Qiu 2025) ---
        if len(self.heading_history[tid]) >= self.CUMULATIVE_WINDOW:
            angles = self.heading_history[tid]
            if (all(angles[i] < angles[i + 1] for i in range(len(angles) - 1))
                    and curr_heading >= self.YIELD_HEADING_THRESHOLD):
                return "yielded"

        # --- failed_to_yield: close, observed long enough, nothing triggered ---
        if (distance <= self.FAILED_YIELD_PROXIMITY
                and self.frames_seen[tid] >= self.MIN_OBSERVED_FRAMES):
            return "failed_to_yield"

        return "normal"