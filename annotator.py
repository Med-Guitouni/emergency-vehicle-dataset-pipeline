class HeuristicAnnotator:
    """
    Labels each vehicle's behaviour per frame when emergency is active.
    Labels: normal, braked_abruptly, yielded, failed_to_yield

    RULE 1 — lateral speed > 0.5 m/s
    Threshold from Pierson et al. 2019 (highD German highway analysis).
    Uses lateral_speed_ms directly

    RULE 2 — heading angle > 15 degrees
    Concept backed by Qiu et al. 2025 (UHasselt dashcam lane change review).
    15-degree threshold is empirical.

    RULE 3 — cumulative lateral > 0.8m over 3 seconds
    3-second window consistent with highD lane change duration data (2.88-7.32s).
    0.8m threshold empirical

    RULE 4 — speed drop > 5 km/h
    Catches moderate slowing to make space. Empirical. Set below braking
    threshold (5 km/h ≈ 1.4 m/s²) to avoid overlap with braked_abruptly.
    German law §38 StVO requires drivers to yield to emergency vehicles by
    both pulling aside and slowing.

    RULE 5 — heading angle increasing over 3 consecutive frames
    Directly backed by Qiu et al. 2025 as the most robust lane change onset
    signal from dashcam footage when lane markings are not visible.

    BRAKING THRESHOLD — -2.5 m/s²
    From braking literature (US11643058, arxiv 2502.10243, ScienceDirect 2021).
    Updated from -2.0 (empirical) to -2.5 (literature-backed).

    PROXIMITY THRESHOLDS — 50m outer, 20m failed-to-yield
    Engineering judgements, not cited. Known limitation: failed_to_yield
    produces false positives for vehicles legitimately in front of the
    ambulance that have no room to move.
    """

    # Rule 1 — lateral speed threshold in m/s
    # Backed: Pierson et al. 2019 (highD), PMC 2020 (NGSIM)
    YIELD_LATERAL_SPEED = 0.5

    # Rules 2 and 5 — heading angle threshold in degrees
    # Concept backed: Qiu et al. 2025 (UHasselt). Threshold: empirical.
    YIELD_HEADING_THRESHOLD = 15.0

    # Rule 3 — cumulative lateral displacement threshold in metres
    # Approximately backed: highD duration data, arxiv 1907.11208. Empirical value.
    YIELD_CUMULATIVE = 0.8

    # Rule 3 and 5 — window in seconds for cumulative/heading checks
    # Backed: highD shows lane changes typically 3-7 seconds. Lower bound used.
    CUMULATIVE_WINDOW = 3

    # Braking threshold in m/s²
    # Backed: US11643058, arxiv 2502.10243, ScienceDirect 2021 AEB paper
    ABRUPT_BRAKE_THRESHOLD = -2.5

    # Rule 4 — moderate speed drop threshold in km/h (yielding-by-slowing)
    # self judgement. Set to 5 km/h to avoid overlap with braking rule.
    # (~1.4 m/s² deceleration, below the -2.5 m/s² braking threshold)
    YIELD_SPEED_DROP = 5.0

    # Outer proximity limit in metres — no interaction rules applied beyond this
    # self judgement. Not directly cited.
    PROXIMITY_THRESHOLD = 50.0

    # Inner proximity limit — vehicle this close without yielding = failed_to_yield
    # Engineering judgement. Known limitation: false positives for lead vehicles.
    FAILED_YIELD_PROXIMITY = 20.0

    def __init__(self):
        self.prev_speed = {}
        self.lateral_history = {}   # for Rule 3
        self.heading_history = {}   # for Rule 5

    def annotate(self, vehicle, emergency_active):
        """
        Assigns a behaviour label to one vehicle at one timestep.

        """
        if not emergency_active:
            return "normal"

        tid          = vehicle["track_id"]
        # lateral_speed_ms: ground-plane metric lateral velocity (m/s)
        # positive = moving right, negative = moving left
        # used in Rule 1 instead of lateral_offset differencing
        lateral_spd  = abs(vehicle.get("lateral_speed_ms", 0.0))
        curr_speed   = vehicle.get("speed_kmh", 0.0)
        curr_heading = abs(vehicle.get("heading_angle", 0.0))
        acceleration = vehicle.get("acceleration", 0.0)
        distance     = vehicle.get("distance_to_ego", 999.0)
        curr_lateral = vehicle.get("lateral_offset", 0.0)

        # update lateral offset history for Rule 3
        if tid not in self.lateral_history:
            self.lateral_history[tid] = []
        self.lateral_history[tid].append(curr_lateral)
        if len(self.lateral_history[tid]) > self.CUMULATIVE_WINDOW:
            self.lateral_history[tid].pop(0)

        # update heading history for Rule 5
        if tid not in self.heading_history:
            self.heading_history[tid] = []
        self.heading_history[tid].append(curr_heading)
        if len(self.heading_history[tid]) > self.CUMULATIVE_WINDOW:
            self.heading_history[tid].pop(0)

        # store previous speed for Rule 4
        prev_spd = self.prev_speed.get(tid, curr_speed)
        self.prev_speed[tid] = curr_speed

        # ambulance not close enough yet — no interaction expected
        if distance > self.PROXIMITY_THRESHOLD:
            return "normal"

        # --- abrupt braking takes priority over all yielding rules ---
        # threshold -2.5 m/s² from literature
        if acceleration <= self.ABRUPT_BRAKE_THRESHOLD:
            return "braked_abruptly"

        # --- Rule 1: sudden lateral speed ---
        # uses lateral_speed_ms directly

        if lateral_spd >= self.YIELD_LATERAL_SPEED:
            return "yielded"

        # --- Rule 2: vehicle clearly angled away ---
        # heading angle > 15 degrees
        if curr_heading >= self.YIELD_HEADING_THRESHOLD:
            return "yielded"

        # --- Rule 3: slow cumulative lateral drift over 3 seconds ---
        # catches gradual yielding that is too slow to trigger Rule 1 each frame

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

        # --- Rule 4: moderate speed drop (yielding by slowing) ---
        # catches vehicles that slow to make space without hard braking
        # threshold 5 km/h empirical, set below the braking threshold to avoid overlap
        speed_drop = prev_spd - curr_speed
        if speed_drop >= self.YIELD_SPEED_DROP:
            return "yielded"

        # --- Rule 5: heading angle increasing over 3 consecutive frames ---
        # backed by Qiu et al. 2025 as the most robust lane-change onset signal
        # from dashcam footage when lane markings are not visible
        if len(self.heading_history[tid]) >= self.CUMULATIVE_WINDOW:
            angles     = self.heading_history[tid]
            increasing = all(
                angles[i] < angles[i + 1]
                for i in range(len(angles) - 1)
            )
            if increasing and curr_heading >= self.YIELD_HEADING_THRESHOLD:
                return "yielded"

        # --- failed_to_yield: very close and none of the rules triggered ---
        # known limitation: false positives for vehicles legitimately in front
        # that have no room to move. Threshold 20m empirical, not cited.
        if distance <= self.FAILED_YIELD_PROXIMITY:
            return "failed_to_yield"

        return "normal"