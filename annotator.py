class HeuristicAnnotator:
    """
    Labels each vehicle's behaviour per frame when emergency is active.
    Labels: normal, braked_abruptly, yielded, failed_to_yield

    INPUT QUALITY NOTE: positions are RTS-smoothed (smoother.py) before any
    metric reaches this class, and acceleration is LONGITUDINAL (change in
    forward speed, signed) per highD - so the braking threshold compares
    against a physically meaningful quantity.

    BRAKE RULE — acceleration <= -2.5 m/s²
    From braking literature.

    BRAKE ONSET RULE — acceleration <= -1.5 m/s² AND jerk <= -3.0 m/s³
    Supplementary, empirical. At 1Hz a panic stop can straddle two frames so
    neither frame alone crosses -2.5; a violently negative jerk (sudden CHANGE
    in deceleration) captures the onset. Jerk was previously computed and
    exported but unused — this rule puts it to work.

    RULE 1 — lateral speed > 0.5 m/s
    Threshold from Pierson et al. 2019 (highD German highway analysis).

    RULE 2 — heading angle > 15 degrees
    Concept from Qiu et al. 2025 (UHasselt). 15° threshold empirical.

    RULE 3 — cumulative lateral > 0.8m over 3 seconds
    Window consistent with highD lane change durations (2.88-7.32s).
    0.8m empirical.

    RULE 4 — speed drop > 5 km/h
    Empirical, motivated by §38 StVO (yield by slowing). Pending review.

    RULE 5 — heading increasing over 3 consecutive frames
    Directly from Qiu et al. 2025.

    MINIMUM OBSERVATION GUARD
    A vehicle must have been observed for at least MIN_OBSERVED_FRAMES before
    failed_to_yield can fire. Without this, a vehicle's FIRST frame (all-zero
    history, no rule can possibly trigger) inside 20m was instantly labelled
    failed_to_yield . A vehicle
    seen for one second cannot have "failed" anything yet.

    PROXIMITY THRESHOLDS — 50m outer, 20m failed-to-yield
    from observation and tries ( can use work still)
    """

    YIELD_LATERAL_SPEED     = 0.5    # m/s — Pierson et al. 2019 (highD)
    YIELD_HEADING_THRESHOLD = 15.0   # degrees — Qiu et al. 2025 (value empirical)
    YIELD_CUMULATIVE        = 0.8    # metres — empirical
    CUMULATIVE_WINDOW       = 3      # seconds — highD lane change durations
    ABRUPT_BRAKE_THRESHOLD  = -2.5   # m/s² — braking literature
    BRAKE_ONSET_ACCEL       = -1.5   # m/s² — empirical (with jerk condition)
    BRAKE_ONSET_JERK        = -3.0   # m/s³ — empirical, panic-brake onset
    YIELD_SPEED_DROP        = 5.0    # km/h — empirical
    PROXIMITY_THRESHOLD     = 50.0   # metres — ( from observation)
    FAILED_YIELD_PROXIMITY  = 20.0   # metres — ( from observation )
    MIN_OBSERVED_FRAMES     = 3      # frames before failed_to_yield is allowed

    def __init__(self):
        self.prev_speed      = {}
        self.lateral_history = {}
        self.heading_history = {}
        self.frames_seen     = {}

    def annotate(self, vehicle, emergency_active):
        if not emergency_active:
            return "normal"

        tid          = vehicle["track_id"]
        lateral_spd  = abs(vehicle.get("lateral_speed_ms", 0.0))
        curr_speed   = vehicle.get("speed_kmh", 0.0)
        curr_heading = abs(vehicle.get("heading_angle", 0.0))
        acceleration = vehicle.get("acceleration", 0.0)   # longitudinal, signed
        jerk         = vehicle.get("jerk", 0.0)
        distance     = vehicle.get("distance_to_ego", 999.0)
        curr_lateral = vehicle.get("lateral_offset", 0.0)

        # how many frames have we seen this vehicle (for the guard below)
        self.frames_seen[tid] = self.frames_seen.get(tid, 0) + 1

        # histories for Rules 3 and 5
        self.lateral_history.setdefault(tid, []).append(curr_lateral)
        if len(self.lateral_history[tid]) > self.CUMULATIVE_WINDOW:
            self.lateral_history[tid].pop(0)

        self.heading_history.setdefault(tid, []).append(curr_heading)
        if len(self.heading_history[tid]) > self.CUMULATIVE_WINDOW:
            self.heading_history[tid].pop(0)

        prev_spd = self.prev_speed.get(tid, curr_speed)
        self.prev_speed[tid] = curr_speed

        if distance > self.PROXIMITY_THRESHOLD:
            return "normal"

        # --- braking takes priority over yielding rules ---

        if acceleration <= self.ABRUPT_BRAKE_THRESHOLD:
            return "braked_abruptly"
        # brake ONSET: moderate deceleration arriving very suddenly (jerk)
        if acceleration <= self.BRAKE_ONSET_ACCEL and jerk <= self.BRAKE_ONSET_JERK:
            return "braked_abruptly"

        # --- Rule 1: lateral speed (Pierson 2019) ---
        if lateral_spd >= self.YIELD_LATERAL_SPEED:
            return "yielded"

        # --- Rule 2: heading angle (Qiu 2025) ---
        if curr_heading >= self.YIELD_HEADING_THRESHOLD:
            return "yielded"

        # --- Rule 3: cumulative lateral drift over the window ---
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

        # --- Rule 4: moderate speed drop (empirical) ---
        if prev_spd - curr_speed >= self.YIELD_SPEED_DROP:
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