class HeuristicAnnotator:

    YIELD_LATERAL_SUDDEN = 0.5        # meters lateral change in 1 second - Rule 1
    YIELD_HEADING_THRESHOLD = 15.0    # degrees heading angle - Rule 2
    YIELD_CUMULATIVE = 0.8            # meters cumulative lateral over 3s - Rule 3
    CUMULATIVE_WINDOW = 3             # seconds to look back
    ABRUPT_BRAKE_THRESHOLD = -2.0     # m/s2 acceleration threshold
    YIELD_SPEED_DROP = 10.0           # km/h speed drop
    PROXIMITY_THRESHOLD = 50.0        # meters - how close ambulance needs to be
    FAILED_YIELD_PROXIMITY = 20.0     # meters - very close and still not yielded

    def __init__(self):
        self.prev_lateral = {}
        self.prev_speed = {}
        self.lateral_history = {}

    def annotate(self, vehicle, emergency_active):
        """
        Assigns behaviour label to each vehicle per timestep.
        Only meaningful when emergency is active.
        Labels: normal, braked_abruptly, yielded, failed_to_yield
        """
        if not emergency_active:
            return "normal"

        tid = vehicle["track_id"]
        curr_lateral = vehicle.get("lateral_offset", 0.0)
        curr_speed = vehicle.get("speed_kmh", 0.0)
        curr_heading = abs(vehicle.get("heading_angle", 0.0))
        acceleration = vehicle.get("acceleration", 0.0)
        distance = vehicle.get("distance_to_ego", 999.0)

        # update histories
        prev_lat = self.prev_lateral.get(tid, curr_lateral)
        prev_spd = self.prev_speed.get(tid, curr_speed)

        if tid not in self.lateral_history:
            self.lateral_history[tid] = []
        self.lateral_history[tid].append(curr_lateral)
        if len(self.lateral_history[tid]) > self.CUMULATIVE_WINDOW:
            self.lateral_history[tid].pop(0)

        self.prev_lateral[tid] = curr_lateral
        self.prev_speed[tid] = curr_speed

        # ambulance not close enough yet
        if distance > self.PROXIMITY_THRESHOLD:
            return "normal"

        # Rule: abrupt braking takes priority
        if acceleration <= self.ABRUPT_BRAKE_THRESHOLD:
            return "braked_abruptly"

        # Rule 1: sudden fast lateral movement
        lateral_change = abs(curr_lateral - prev_lat)
        if lateral_change >= self.YIELD_LATERAL_SUDDEN:
            return "yielded"

        # Rule 2: car clearly angled away
        if curr_heading >= self.YIELD_HEADING_THRESHOLD:
            return "yielded"

        # Rule 3: cumulative lateral displacement over 3 seconds
        if len(self.lateral_history[tid]) >= self.CUMULATIVE_WINDOW:
            history = self.lateral_history[tid]
            total = abs(history[-1] - history[0])
            direction = history[-1] - history[0]
            consistent = all(
                (history[i+1] - history[i]) * direction >= 0
                for i in range(len(history)-1)
            )
            if total >= self.YIELD_CUMULATIVE and consistent:
                return "yielded"

        # speed drop indicates yielding
        speed_drop = prev_spd - curr_speed
        if speed_drop >= self.YIELD_SPEED_DROP:
            return "yielded"

        # very close and clearly not yielding
        if distance <= self.FAILED_YIELD_PROXIMITY:
            return "failed_to_yield"

        return "normal"