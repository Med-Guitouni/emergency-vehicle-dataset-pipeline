import os
import cv2

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
from review import run_review

"""
Pipeline — two phases, then an interactive review pass.

PHASE 1  (streaming at 5 Hz, exporting records at 1 Hz)
  - Crop sky and dashboard, resize to fixed 1280×720
  - Detect vehicles (YOLOv8x), track with BoT-SORT at 5 Hz so consecutive
    frames are only 6 video frames apart — close enough for GMC sparseOptFlow
    and ReID to work correctly.
  - Every 5 Hz frame: project bounding boxes to metres, feed track_obs so
    the RTS smoother sees the full 5 Hz trajectory.
  - Only on whole-second frames: classify scene, look up lane config and
    emergency state, store a record for export, AND save the cropped frame
    to review_data/frame_<timestamp>.jpg for the review UI.

BETWEEN PHASES — RTS SMOOTHING
  Full trajectory per vehicle smoothed with Rauch-Tung-Striebel smoother.

PHASE 2  (over stored 1 Hz records, after smoothing)
  - Derive split velocity, longitudinal acceleration, jerk, TTC to ego
  - Assign lane and surrounding-vehicle IDs from metric positions
  - Label behaviour
  - Write one JSON per second to output/video_name/

REVIEW  (after each video's JSON is saved)
  - Opens review.py's interactive window automatically so behaviour labels
    and boxes can be corrected by hand before moving to the next video.
  - Reads the frames saved during Phase 1 + the JSON written by Phase 2.

Manual inputs (video_lanes.json): lane count per time window, road type,
emergency_start_second.
"""

REVIEW_DIR = "review_data"

sc = SceneClassifier()
lc = LaneConfig()


def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0][:30]
    print(f"\nProcessing: {video_name}")

    os.makedirs(REVIEW_DIR, exist_ok=True)

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
    # PHASE 1 — track at 5 Hz, store export records at 1 Hz
    # =================================================================
    records   = []   # one entry per whole second
    track_obs = {}   # track_id -> [(timestamp_float, x, y, reliable)]

    for item in p.stream_frames(fps=5):
        timestamp_float  = item["timestamp"]
        timestamp        = int(round(timestamp_float))
        is_export_frame  = (round(timestamp_float * 5) % 5 == 0)

        frame_raw    = item["frame"]
        frame        = p.spatial_crop(frame_raw)
        frame_height, frame_width = frame.shape[:2]

        tracked = t.update(d.model, frame)

        if is_export_frame:
            scenario_type        = sc.classify(frame_raw)
            lane_info            = lc.get_lane_info(video_name, timestamp, scenario_type)
            emergency_active, _  = lc.is_emergency_active(video_name, timestamp)

            # save the cropped frame for the review UI
            frame_path = os.path.join(REVIEW_DIR, f"frame_{timestamp:04d}.jpg")
            cv2.imwrite(frame_path, frame)
        else:
            lane_info = None

        vehicles_raw = []
        for v in tracked:
            x_raw, y_raw, reliable = h.get_vehicle_position(
                v["bbox"], v["type"], frame_width, frame_height, lane_info
            )

            track_obs.setdefault(v["track_id"], []).append(
                (timestamp_float, x_raw, y_raw, reliable)
            )

            if is_export_frame:
                vehicles_raw.append({
                    "track_id": v["track_id"],
                    "type":     v["type"],
                    "bbox":     v["bbox"],
                    "reliable": reliable,
                })

        if is_export_frame:
            records.append({
                "timestamp":        timestamp,
                "scenario_type":    scenario_type,
                "lane_info":        lane_info,
                "emergency_active": emergency_active,
                "frame_width":      frame_width,
                "vehicles_raw":     vehicles_raw,
            })

            if timestamp % 60 == 0:
                print(f"  [phase 1] t={timestamp}s  {len(vehicles_raw)} vehicles"
                      f"  emergency={emergency_active}")

    # =================================================================
    # RTS SMOOTHING — per track, over the full video
    # =================================================================
    print(f"  [smoothing] {len(track_obs)} tracks...")
    smoothed = sm.smooth(track_obs)

    # =================================================================
    # PHASE 2 — metrics from smoothed positions, annotate, export
    # =================================================================
    all_frames_data = []
    last_seen = {}   # track_id -> last timestamp, for correct dt across gaps

    for rec in records:
        timestamp        = rec["timestamp"]
        lane_info        = rec["lane_info"]
        emergency_active = rec["emergency_active"]

        vehicles = []
        for vr in rec["vehicles_raw"]:
            tid = vr["track_id"]

            x_m, y_m = smoothed[tid][timestamp]

            dt = max(timestamp - last_seen.get(tid, timestamp - 1), 1)
            last_seen[tid] = timestamp

            forward_speed, lateral_speed, speed = h.estimate_relative_velocity(
                tid, x_m, y_m, dt
            )
            acceleration = h.estimate_acceleration(tid, forward_speed, dt)
            jerk         = h.estimate_jerk(tid, acceleration, dt)

            distance_to_ego = h.estimate_distance_to_ego(x_m, y_m)
            ttc_to_ego      = h.estimate_ttc_to_ego(y_m, forward_speed)
            lane_id         = h.estimate_lane_id(x_m, lane_info)
            lateral_offset  = h.estimate_lateral_offset(x_m, lane_info)

            vehicles.append({
                "track_id":          tid,
                "type":              vr["type"],
                "bbox":              vr["bbox"],
                "x_meters":          x_m,
                "y_meters":          y_m,
                "position_reliable": vr["reliable"],
                "speed_kmh":         speed,
                "forward_speed_ms":  forward_speed,
                "lateral_speed_ms":  lateral_speed,
                "acceleration":      acceleration,
                "jerk":              jerk,
                "ttc_to_ego":        ttc_to_ego,
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

    # =================================================================
    # REVIEW — opens automatically once this video's JSON is on disk
    # =================================================================
    run_review(video_name)

    return all_frames_data


if __name__ == "__main__":
    videos = sorted([f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")])
    print(f"Found {len(videos)} videos")

    for video in videos:
        video_name = os.path.splitext(os.path.basename(video))[0][:30]
        json_dir   = os.path.join("output", video_name)

        # skip videos that already have JSON output — lets you quit and
        # relaunch without reprocessing finished videos. Note: if a video
        # was interrupted mid-processing, its partial JSON dir will exist
        # but be incomplete — delete that folder manually before relaunch
        # if you want it redone from scratch.
        if os.path.exists(json_dir) and os.listdir(json_dir):
            print(f"Skipping {video_name} — already processed "
                  f"({len(os.listdir(json_dir))} JSON files found)")
            continue

        process_video(video)

    print("\nDone.")