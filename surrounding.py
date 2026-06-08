import numpy as np


class SurroundingVehicles:
    """
    Finds the highD-style "surrounding vehicle" relationships for every
    vehicle in a single frame.

    WHAT THIS IS
    ------------
    ByteTrack already gives every vehicle a unique ID. That tells us WHICH
    vehicles exist, but NOT how they sit relative to each other on the road.
    The highD dataset adds, for each vehicle, the IDs of its six neighbours:

        preceding         - car directly in front, SAME lane
        following         - car directly behind, SAME lane
        left_preceding    - car in front, lane to the LEFT
        left_following    - car behind, lane to the LEFT
        right_preceding   - car in front, lane to the RIGHT
        right_following   - car behind, lane to the RIGHT

    If a neighbour does not exist (e.g. nobody in front), the ID is None.

    WHY WE WANT IT
    --------------
    Yielding is a RELATIONAL behaviour. A car pulls aside to let the ambulance
    (or the car behind it) pass. To study that we need to know who was behind
    whom and in which lane. A flat list of independent vehicles cannot express
    "car 301 moved aside as car 287 approached from behind". These neighbour
    links let the analysis reconstruct those interactions later.

    HOW WE DECIDE WHO IS A NEIGHBOUR
    ---------------------------------
    We work in the real-world metre coordinates we already compute:
        x_meters = lateral position (left/right across the road, + = right)
        y_meters = forward position (distance ahead of the ambulance)

    Lane is decided by lateral position (x_meters), NOT by the lane_id field.
    We do this because lane_id comes from the broken equal-thirds assumption
    (see lane_detector.py). Lateral metre position is more reliable for simply
    asking "is this car in my lane, the lane left of me, or the lane right
    of me". We bucket by lateral distance using half a lane width as the
    boundary.

    "In front" vs "behind" is decided by forward position (y_meters):
    a larger y_meters means further ahead.

    LIMITATIONS
    -----------
    - Accuracy depends entirely on x_meters / y_meters being correct. If the
      depth scaling is off, neighbour assignment will be off too.
    - Uses lateral metre buckets, not true detected lane lines, because lane
      detection is unsolved (see lane_detector.py).
    - Assumes a roughly straight road. On sharp curves "ahead" and "lane left"
      get blurry. Acceptable for Autobahn highway footage.
    """

    # half a lane width in metres. If two vehicles are within this lateral
    # distance of each other they count as being in the SAME lane.
    # 3.75m is one German lane, so 1.875m is the half-lane boundary.
    SAME_LANE_HALF_WIDTH = 3.75 / 2.0

    def assign(self, vehicles, lane_info=None):
        """
        Add the six neighbour-ID fields to every vehicle in the list.

        vehicles: list of dicts, each MUST already contain:
            "track_id"  - the vehicle's ID
            "x_meters"  - lateral position (+ = right of ego centre)
            "y_meters"  - forward position (distance ahead of ego)

        lane_info: dict from LaneConfig.get_lane_info() with keys:
            "lanes"             - number of lanes
            "lane_width_meters" - width of one lane in metres
            If None, falls back to the default SAME_LANE_HALF_WIDTH (1.875m)
            which assumes 3.75m Autobahn lanes.

        Modifies each dict in place, adding:
            "preceding_id", "following_id",
            "left_preceding_id", "left_following_id",
            "right_preceding_id", "right_following_id"
        """
        # use real lane width from config if available, otherwise default
        if lane_info is not None:
            half_width = lane_info["lane_width_meters"] / 2.0
        else:
            half_width = self.SAME_LANE_HALF_WIDTH
        # for each vehicle, look at every OTHER vehicle and decide:
        #   - is it in my lane / left lane / right lane?  (by lateral distance)
        #   - is it ahead of me or behind me?             (by forward distance)
        # then keep the CLOSEST one in each of the six buckets.

        for v in vehicles:
            vx = v.get("x_meters", 0.0)
            vy = v.get("y_meters", 0.0)

            # best (closest) neighbour found so far in each bucket.
            # we store (distance, id) and keep the smallest distance.
            best = {
                "preceding":       (float("inf"), None),
                "following":       (float("inf"), None),
                "left_preceding":  (float("inf"), None),
                "left_following":  (float("inf"), None),
                "right_preceding": (float("inf"), None),
                "right_following": (float("inf"), None),
            }

            for other in vehicles:
                if other is v:
                    continue  # skip self
                ox = other.get("x_meters", 0.0)
                oy = other.get("y_meters", 0.0)
                oid = other.get("track_id")

                # lateral gap: how far left/right the other car is from me
                lateral_gap = ox - vx           # + = other is to my right
                # forward gap: how far ahead/behind the other car is from me
                forward_gap = oy - vy           # + = other is ahead of me

                # how far ahead/behind in absolute terms - used to pick closest
                forward_dist = abs(forward_gap)

                # decide which lane the other car is in relative to me
                if abs(lateral_gap) <= half_width:
                    lane = "same"
                elif lateral_gap < 0:
                    lane = "left"     # other car is to my left
                else:
                    lane = "right"    # other car is to my right

                # decide ahead or behind
                if forward_gap > 0:
                    pos = "preceding"   # other car is ahead of me
                elif forward_gap < 0:
                    pos = "following"   # other car is behind me
                else:
                    continue  # exactly level - ambiguous, skip

                # map (lane, pos) to one of the six buckets
                if lane == "same":
                    bucket = pos                       # preceding / following
                elif lane == "left":
                    bucket = f"left_{pos}"
                else:
                    bucket = f"right_{pos}"

                # keep this neighbour only if it is closer than the current best
                if forward_dist < best[bucket][0]:
                    best[bucket] = (forward_dist, oid)

            # write the chosen neighbour IDs into the vehicle dict
            v["preceding_id"]        = best["preceding"][1]
            v["following_id"]        = best["following"][1]
            v["left_preceding_id"]   = best["left_preceding"][1]
            v["left_following_id"]   = best["left_following"][1]
            v["right_preceding_id"]  = best["right_preceding"][1]
            v["right_following_id"]  = best["right_following"][1]

        return vehicles