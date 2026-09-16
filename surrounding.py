import numpy as np


class SurroundingVehicles:
    """
    For each vehicle in each exported frame, find up to six neighbours: the
    preceding and following vehicle in its own lane and in each adjacent
    lane, following the highD convention (Krajewski et al. 2018).

    The tracker's IDs say WHICH vehicles exist, not how they sit relative to
    each other. Storing neighbour IDs lets later analysis reconstruct
    relational interactions -- "vehicle 301 was behind vehicle 287 when it
    started pulling aside" -- instead of reading the frame as a flat list of
    independent vehicles.

    HOW A NEIGHBOUR IS DECIDED
    Everything works in the metric coordinates already computed:
        x_meters = lateral position (+ = right of the ambulance)
        y_meters = forward position (distance ahead of the ambulance)
    Lane is decided by lateral gap (same lane if |gap| <= lane_width/2),
    ahead vs behind by the sign of the forward gap, and the closest vehicle
    in each of the six buckets wins.

    LIMITATION
    Assumes a roughly straight road. On sharp curves "ahead" and "lane left"
    get blurry — acceptable for the motorway footage in the released corpus.
    """

    # fallback only - lane_info from LaneConfig always provides the correct width
    # (3.75m highway, 3.00m urban). This constant is only used if lane_info is None.
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
