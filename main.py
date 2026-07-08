import os
import cv2
import torch
import time
import json

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
from intermediate_review import run_intermediate_review

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

INTERMEDIATE REVIEW (after Phase 1, before smoothing)
  - Opens an interactive window to correct, add, or delete tracked objects
    for each 1Hz frame.
  - Changes are applied to the raw tracking data before smoothing and metric
    calculation, ensuring all fields are correctly populated for all objects.

Manual inputs (video_lanes.json): lane count per time window, road type,
emergency_start_second.
"""
REVIEW_BASE_DIR = "review_data"

sc = SceneClassifier()
lc = LaneConfig()

def load_data_from_jsons(video_name, video_path):
    """
    Loads tracking data from existing final JSON files, allowing Phase 1 to be skipped.
    This reconstructs the `track_obs` and `records` data structures.
    NOTE: The reconstructed `track_obs` will be at 1Hz and contain smoothed data,
    not the original 5Hz raw data.
    """
    json_dir = os.path.join("output", video_name)
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json") and f.startswith('t')])
    if not json_files:
        print(f"  [loader] No valid JSON files found in {json_dir}")
        return None, None

    print(f"  [loader] Loading from {len(json_files)} JSON files...")

    track_obs = {}
    records = []

    # We need frame width/height. Let's get it from the first review frame.
    video_review_dir = os.path.join("review_data", video_name)
    first_frame_path = os.path.join(video_review_dir, f"frame_{int(json_files[0][1:5]):04d}.jpg")
    if not os.path.exists(first_frame_path):
        print(f"  [loader] Warning: Review frame not found at {first_frame_path}. Reading video to get frame size.")
        p_temp = VideoPreprocessor(video_path)
        try:
            item = next(p_temp.stream_frames(fps=1))
            frame = p_temp.spatial_crop(item['frame'])
            frame_height, frame_width = frame.shape[:2]
        except StopIteration:
            print("  [loader] Error: Could not read video to get frame size.")
            return None, None
    else:
        frame = cv2.imread(first_frame_path)
        if frame is None:
            print(f"  [loader] Error: Could not read review frame at {first_frame_path}")
            return None, None
        frame_height, frame_width = frame.shape[:2]

    for file_name in json_files:
        with open(os.path.join(json_dir, file_name), 'r') as f:
            data = json.load(f)

        timestamp = data['timestamp']
        lane_info = lc.get_lane_info(video_name, timestamp, data['scenario_type'])

        vehicles_raw = []
        for v in data['vehicles']:
            tid = v['id']
            track_obs.setdefault(tid, []).append((float(timestamp), v['x_meters'], v['y_meters'], v['position_reliable']))
            vehicles_raw.append({"track_id": tid, "type": v['type'], "bbox": v['bbox'], "reliable": v['position_reliable']})

        records.append({
            "timestamp": timestamp, "scenario_type": data['scenario_type'],
            "lane_info": lane_info, "emergency_active": data['emergency_active'],
            "frame_width": frame_width, "vehicles_raw": vehicles_raw,
        })

    for tid in track_obs:
        track_obs[tid].sort(key=lambda x: x[0])

    print(f"  [loader] Loaded {len(records)} records and {len(track_obs)} tracks.")
    return track_obs, records

def get_device():
    """Checks for available hardware backends and returns the best one."""
    if torch.backends.mps.is_available():
        print("  [info] MPS (Apple Silicon GPU) backend is available. Using MPS.")
        return "mps"
    if torch.cuda.is_available():
        print("  [info] CUDA backend is available. Using CUDA.")
        return "cuda"
    print("  [info] No GPU backend found. Using CPU.")
    return "cpu"


def process_video(video_path, loaded_data=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0][:30]
    print(video_name)
    print(f"\nProcessing: {video_name}")

    video_review_dir = os.path.join(REVIEW_BASE_DIR, video_name)
    os.makedirs(video_review_dir, exist_ok=True)

    h  = HomographyEstimator()
    e  = JSONExporter()
    a  = HeuristicAnnotator()
    sv = SurroundingVehicles()
    sm = RTSSmoother()

    sc.reset()

    if loaded_data:
        print("  [info] Using pre-loaded data, skipping Phase 1 tracking.")
        track_obs, records = loaded_data
    else:
        # =================================================================
        # PHASE 1 — track at 5 Hz, store export records at 1 Hz
        # =================================================================
        print("  [info] No pre-loaded data found, running Phase 1 tracking.")
        device = get_device()
        p  = VideoPreprocessor(video_path)
        d  = VehicleDetector()
        t  = VehicleTracker()

        records   = []   # one entry per whole second
        track_obs = {}   # track_id -> [(timestamp_float, x, y, reliable)]

        for item in p.stream_frames(fps=5):
            timestamp_float  = item["timestamp"]
            timestamp        = int(round(timestamp_float))
            is_export_frame  = (round(timestamp_float * 5) % 5 == 0)

            frame_raw    = item["frame"]
            frame        = p.spatial_crop(frame_raw)
            frame_height, frame_width = frame.shape[:2]

            tracked = t.update(d.model, frame, device=device)

            if is_export_frame:
                scenario_type        = sc.classify(frame_raw)
                lane_info            = lc.get_lane_info(video_name, timestamp, scenario_type)
                emergency_active, _  = lc.is_emergency_active(video_name, timestamp)

                # save the cropped frame for the review UI
                frame_path = os.path.join(video_review_dir, f"frame_{timestamp:04d}.jpg")
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
    # INTERMEDIATE REVIEW — correct raw tracks before smoothing
    # =================================================================
    print(f"  [review] launching intermediate review for {len(records)} frames...")
    track_obs, records = run_intermediate_review(
        video_name, track_obs, records, h)

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

    return all_frames_data


if __name__ == "__main__":
    videos = sorted([f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")])
    print(f"Found {len(videos)} videos")

    for video in videos:
        video_name = os.path.splitext(os.path.basename(video))[0][:30]
        #print(video_name)
        #continue
        json_dir   = os.path.join("output", video_name)

        # skip videos that already have JSON output — lets you quit and
        # relaunch without reprocessing finished videos. Note: if a video
        # was interrupted mid-processing, its partial JSON dir will exist
        # but be incomplete — delete that folder manually before relaunch
        # if you want it redone from scratch. The new logic will load from
        # existing JSONs instead of skipping.
        if os.path.exists(json_dir) and any(f.endswith('.json') for f in os.listdir(json_dir)):
            print(f"Found existing JSONs for {video_name}. Loading data instead of re-tracking.")
            loaded_data = load_data_from_jsons(video_name, video)
            if loaded_data[0] is None:
                print("  [main] Failed to load from JSONs, processing from scratch.")
                process_video(video)
            else:
                process_video(video, loaded_data=loaded_data)
        else:
            process_video(video)

    print("\nDone.")