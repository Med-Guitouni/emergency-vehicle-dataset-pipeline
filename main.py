import os

from preprocessor import VideoPreprocessor
from detector import VehicleDetector
from tracker import VehicleTracker
from homography import HomographyEstimator
from exporter import JSONExporter
from annotator import HeuristicAnnotator
from emergency_detector import EmergencyDetector
from scene_classifier import SceneClassifier
from surrounding import SurroundingVehicles
from lane_config import LaneConfig

"""
Pipeline overview - same as before but with two new steps added per frame:

For each video, frames are extracted at 1Hz.
The first 10 raw uncropped frames are sampled to determine if the video is
daytime or not - this controls whether blue light detection is used later.
Each frame is then spatially cropped to remove the dashboard and sky.

NEW STEP A - Ego motion estimation:
HomographyEstimator.process_frame() runs two things in one call:
  1. Optical flow on the background to figure out how the camera moved
     (ego_H = 3x3 homography matrix describing that movement)
  2. Depth Anything V2 forward pass to get a depth map in metres
     (depth_map = 2D array, one depth value per pixel)
Both results are passed downstream.

NEW STEP B - EMAP-aware tracking:
tracker.update() now receives ego_H and depth_map.
If EMAP is installed, it uses ego_H to warp the Kalman Filter state before
ByteTrack's predict step - so tracked vehicles appear stationary when they
are stationary, even when the ambulance is moving at highway speed.

For every frame, spatial metrics are computed per vehicle:
position in meters (now from depth map instead of lane-width scale),
speed (now ego-compensated instead of relative),
acceleration, jerk, heading angle, lane ID, lateral offset,
and distance to the ambulance.

NEW STEP C - Split velocity:
On top of the single combined speed, each vehicle also gets its motion
split into forward_speed_ms (along the road) and lateral_speed_ms (across
the road). This makes braking (forward change) and yielding (lateral change)
directly separable instead of hidden inside one combined number.

NEW STEP D - Surrounding vehicle IDs:
After all vehicles in a frame have their metre positions, we assign each
vehicle its six highD-style neighbours (preceding, following, and the
left/right versions of each). This lets the analysis reconstruct who
reacted to whom. See surrounding.py.

Everything else (emergency detection, scene classification, behaviour
annotation, JSON export) is unchanged.
"""

# load scene classifier once - it is a large ResNet18, loading per video
# would waste several seconds per video for no reason
sc = SceneClassifier()
lc = LaneConfig()


def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0][:30]
    print(f"\nProcessing: {video_name}")

    p  = VideoPreprocessor(video_path)
    d  = VehicleDetector()
    t  = VehicleTracker()
    h  = HomographyEstimator()
    e  = JSONExporter()
    a  = HeuristicAnnotator()
    ed = EmergencyDetector(video_path)
    sv = SurroundingVehicles()

    # reset scene classifier smoothing window for each new video
    sc.reset()

    frames = p.extract_frames(fps=1)

    # day/night detection uses the original uncropped frames
    # because brightness of the sky is what we are measuring
    sample_frames = [frames[i]['frame'] for i in range(0, min(10, len(frames)))]
    ed.daytime = ed.detect_daytime(sample_frames)

    all_frames_data = []
    prev_vehicles   = []

    for item in frames:
        timestamp = item["timestamp"]

        # scene classification needs the uncropped frame to see sky + horizon
        scenario_type = sc.classify(item["frame"])

        # everything else uses the spatially cropped frame
        frame = p.spatial_crop(item["frame"])
        frame_height = frame.shape[0]
        frame_width  = frame.shape[1]

        # NEW: run ego motion + depth estimation together before tracking
        # ego_H  = how the camera moved this frame (3x3 homography)
        # depth_map = per-pixel depth in metres from Depth Anything V2
        # Both are cached inside h so estimate_speed / get_bev_position
        # can use them without needing them passed as arguments each time
        ego_H, depth_map = h.process_frame(frame)

        # Pass ego_H and depth_map to the tracker so EMAP can compensate
        # the Kalman Filter state before ByteTrack runs matching
        tracked = t.update(d.model, frame, ego_H=ego_H, depth_map=depth_map)
        lane_info = lc.get_lane_info(video_name, timestamp, scenario_type)

        vehicles = []
        for v in tracked:
            tid    = v["track_id"]
            center = v["center"]
            bbox   = v["bbox"]

            # bottom centre of the bounding box = where the vehicle
            # touches the road - best point for ground-plane geometry
            bottom_center = [(bbox[0] + bbox[2]) // 2, bbox[3]]

            # position in metres via ground-plane pinhole projection.
            # x_m = lateral (+ = right), y_m = forward distance ahead.
            # This is the single source of truth - everything below derives
            # from it, so all metrics share one coordinate system.
            x_m, y_m = h.get_bev_position(
                bottom_center, frame_width, frame_height
            )

            # relative velocity, derived from how the metric position moved.
            # forward_speed / lateral_speed are in m/s, speed is km/h magnitude.
            # forward: + = moving away from ego, - = moving toward ego
            # lateral: + = moving right, - = moving left
            # NOTE: this is RELATIVE to the ambulance, not absolute ground speed
            # (absolute speed needs ego odometry we do not have - see homography).
            # Call exactly once per vehicle per frame: it updates the stored
            # previous position internally.
            forward_speed, lateral_speed, speed = h.estimate_relative_velocity(
                tid, x_m, y_m
            )

            acceleration = h.estimate_acceleration(tid, speed)
            jerk         = h.estimate_jerk(tid, acceleration)
            heading      = h.estimate_heading(tid, center)

            # distance to ego = straight-line distance from origin to (x_m, y_m)
            distance_to_ego = h.estimate_distance_to_ego(x_m, y_m)

            lane_id        = h.estimate_lane_id(center[0], frame_width, lane_info)
            lateral_offset = h.estimate_lateral_offset(center[0], frame_width, lane_info)

            vehicles.append({
                "track_id":       tid,
                "type":           v["type"],
                "bbox":           bbox,
                "center":         center,
                "x_meters":       x_m,
                "y_meters":       y_m,
                "speed_kmh":      speed,
                "forward_speed_ms": forward_speed,
                "lateral_speed_ms": lateral_speed,
                "acceleration":   acceleration,
                "jerk":           jerk,
                "heading_angle":  heading,
                "lane_id":        lane_id,
                "lateral_offset": lateral_offset,
                "distance_to_ego": distance_to_ego,
                "lanes_total":    lane_info["lanes"],
                "road_type":      lane_info["road_type"],
                "lane_source":    lane_info["source"]
            })

        # surrounding IDs: runs ONCE per frame after all vehicles have positions.
        # pass lane_info so it uses the correct lane width for same/left/right bucketing
        sv.assign(vehicles, lane_info)

        emergency_active, triggered_by = ed.is_emergency_active(
            timestamp, frame, vehicles, prev_vehicles
        )

        for v in vehicles:
            v["behaviour"] = a.annotate(v, emergency_active)

        all_frames_data.append({
            "timestamp":             timestamp,
            "emergency_active":      emergency_active,
            "emergency_triggered_by": triggered_by,
            "scenario_type":         scenario_type,
            "vehicles":              vehicles
        })

        prev_vehicles = vehicles

        if timestamp % 60 == 0:
            print(
                f"  t={timestamp}s"
                f" - {len(vehicles)} vehicles"
                f" - emergency={emergency_active}"
                f" - scenario={scenario_type}"
                f" - depth={'yes' if depth_map is not None else 'no'}"
            )

    e.save_batch(all_frames_data, video_name)
    return all_frames_data


if __name__ == "__main__":
    videos = [f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")]
    print(f"Found {len(videos)} videos")
    for video in videos:
        process_video(video)
    print("\nDone.")
    # rm - rf output / *  python3  main.py


# Test script - verifies emergency detection across first 60 seconds
# Run: python3 test_emergency.py

# Test script - verifies lateral offset detection across first 25 seconds
# Run: python3 test_lateral.py


