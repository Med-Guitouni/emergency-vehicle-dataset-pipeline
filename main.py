import os

from preprocessor import VideoPreprocessor
from detector import VehicleDetector
from tracker import VehicleTracker
from homography import HomographyEstimator
from exporter import JSONExporter
from annotator import HeuristicAnnotator
from scene_classifier import SceneClassifier
from surrounding import SurroundingVehicles
from lane_config import LaneConfig
from smoother import RTSSmoother

"""
What this pipeline does, in two phases:

PHASE 1 - COLLECT (streaming, one frame at a time at 1Hz):
   - Crop out the sky and dashboard
   - Run YOLO to detect vehicles, ByteTrack (+EMAP) to assign stable IDs
   - Project each vehicle's pixel position to raw real-world metres (x, y)
   - Flag positions from clipped bounding boxes as unreliable
   - Classify the scene, look up lane config and emergency state
   - Store the raw per-frame records (no metrics yet)

BETWEEN PHASES - RTS SMOOTHING:
   Each vehicle's full raw trajectory is smoothed with a RTS
   smoother (the same post-processing highD and INTERACTION use). This removes
   detection jitter and pixel quantisation noise from positions BEFORE any
   velocity/acceleration is derived from them. Unreliable (clipped-bbox)
   measurements are down-weighted so the motion model dominates there.

PHASE 2 - COMPUTE + EXPORT (sequential over the stored records):
   - Read each vehicle's SMOOTHED position
   - Derive split velocity, longitudinal acceleration, jerk, heading (internal)
   - Assign lane, distance, surrounding-vehicle IDs
   - Label behaviour (normal / yielded / braked_abruptly / failed_to_yield)
   - Write one JSON file per second to output/video_name/

Manual inputs (video_lanes.json): lane count per time window, road type,
and emergency_start_second. Heading is internal-only (not exported).
"""

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
    sv = SurroundingVehicles()
    sm = RTSSmoother()

    sc.reset()

    # =================================================================
    # PHASE 1 - collect raw tracked positions per frame
    # =================================================================
    records = []          # one entry per second
    track_obs = {}        # track_id -> [(timestamp, x_raw, y_raw, reliable)]

    for item in p.extract_frames(fps=1):
        timestamp = item["timestamp"]
        frame_raw = item["frame"]
        frame     = p.spatial_crop(frame_raw)
        frame_height, frame_width = frame.shape[:2]

        ego_H, depth_map = h.process_frame(frame)
        tracked = t.update(d.model, frame, ego_H=ego_H, depth_map=depth_map)

        scenario_type = sc.classify(frame_raw)
        lane_info = lc.get_lane_info(video_name, timestamp, scenario_type)
        emergency_active, _ = lc.is_emergency_active(video_name, timestamp)

        vehicles_raw = []
        for v in tracked:
            bbox = v["bbox"]

            # position + reliability in one edge-aware call:
            # top-clipped boxes are valid (projection uses only the bottom
            # row), side-clipped boxes get their lateral reconstructed from
            # the visible edge + a class width prior, only bottom-clipped /
            # fully-spanning boxes stay unreliable. See homography.py
            # get_vehicle_position docstring for the literature.
            x_raw, y_raw, reliable = h.get_vehicle_position(
                bbox, v["type"], frame_width, frame_height, lane_info
            )

            vehicles_raw.append({
                "track_id": v["track_id"],
                "type":     v["type"],
                "bbox":     bbox,
                "center":   v["center"],
                "reliable": reliable,
            })
            track_obs.setdefault(v["track_id"], []).append(
                (timestamp, x_raw, y_raw, reliable)
            )

        records.append({
            "timestamp":        timestamp,
            "scenario_type":    scenario_type,
            "lane_info":        lane_info,
            "emergency_active": emergency_active,
            "frame_width":      frame_width,
            "vehicles_raw":     vehicles_raw,
        })

        if timestamp % 60 == 0:
            print(f"  [phase 1] t={timestamp}s - {len(vehicles_raw)} vehicles"
                  f" - emergency={emergency_active}")

    # =================================================================
    # RTS SMOOTHING - per track, over the full video
    # =================================================================
    print(f"  [smoothing] {len(track_obs)} tracks...")
    smoothed = sm.smooth(track_obs)

    # =================================================================
    # PHASE 2 - metrics from smoothed positions, annotate, export
    # =================================================================
    all_frames_data = []
    last_seen = {}   # track_id -> last timestamp, for correct dt across gaps

    for rec in records:
        timestamp        = rec["timestamp"]
        lane_info        = rec["lane_info"]
        emergency_active = rec["emergency_active"]
        frame_width      = rec["frame_width"]

        vehicles = []
        for vr in rec["vehicles_raw"]:
            tid    = vr["track_id"]
            center = vr["center"]

            # smoothed position (falls back to raw for very short tracks)
            x_m, y_m = smoothed[tid][timestamp]

            # real time since this track was last seen (handles gaps)
            dt = max(timestamp - last_seen.get(tid, timestamp - 1), 1)
            last_seen[tid] = timestamp

            forward_speed, lateral_speed, speed = h.estimate_relative_velocity(
                tid, x_m, y_m, dt
            )
            # longitudinal acceleration (change in forward speed) - highD style
            acceleration = h.estimate_acceleration(tid, forward_speed, dt)
            jerk         = h.estimate_jerk(tid, acceleration, dt)
            heading      = h.estimate_heading(tid, x_m, y_m)  # internal only

            distance_to_ego = h.estimate_distance_to_ego(x_m, y_m)
            lane_id         = h.estimate_lane_id(center[0], frame_width, lane_info)
            lateral_offset  = h.estimate_lateral_offset(center[0], frame_width, lane_info)

            vehicles.append({
                "track_id":          tid,
                "type":              vr["type"],
                "bbox":              vr["bbox"],
                "center":            center,
                "x_meters":          x_m,
                "y_meters":          y_m,
                "position_reliable": vr["reliable"],
                "speed_kmh":         speed,
                "forward_speed_ms":  forward_speed,
                "lateral_speed_ms":  lateral_speed,
                "acceleration":      acceleration,
                "jerk":              jerk,
                "heading_angle":     heading,
                "lane_id":           lane_id,
                "lateral_offset":    lateral_offset,
                "distance_to_ego":   distance_to_ego,
                "lanes_total":       lane_info["lanes"],
                "road_type":         lane_info["road_type"],
                "lane_source":       lane_info["source"],
            })

        sv.assign(vehicles, lane_info)

        for v in vehicles:
            v["behaviour"] = a.annotate(v, emergency_active)

        all_frames_data.append({
            "timestamp":        timestamp,
            "emergency_active": emergency_active,
            "scenario_type":    rec["scenario_type"],
            "vehicles":         vehicles,
        })

        if timestamp % 120 == 0:
            print(f"  [phase 2] t={timestamp}s exported")

    e.save_batch(all_frames_data, video_name)
    return all_frames_data


if __name__ == "__main__":
    videos = [f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")]
    print(f"Found {len(videos)} videos")
    for video in videos:
        process_video(video)
    print("\nDone.")

# Test script - verifies emergency detection across first 60 seconds
# Run: python3 test_emergency.py

# Test script - verifies lateral offset detection across first 25 seconds
# Run: python3 test_lateral.py


