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
Pipeline — two phases, 30 Hz tracking / 10 Hz export



PHASE 1 (track at 30 Hz, export every 3rd frame at 10 Hz)
  - Crop sky and dashboard, resize to fixed 1280x720
  - Detect vehicles (YOLOv8x), track with BoT-SORT at 30 Hz
  - Every raw frame: project bounding boxes to metres, feed track_obs so the
    RTS smoother sees the full 30 Hz trajectory.
  - Only on export frames (every 3rd raw frame): store a record for export.
  - Scene classification (CNN) and lane/emergency lookup are recomputed only
    when the whole real second changes, 

BETWEEN PHASES — RTS SMOOTHING
  Full 30 Hz trajectory per vehicle smoothed with the RTS smoother.

PHASE 2 (over the 10 Hz export records, after smoothing)
  - Derive split velocity, longitudinal acceleration, jerk, TTC to ego using
    the REAL elapsed time between consecutive EXPORTED observations of a
    track (dt), floored at 1/EXPORT_FPS (~0.1s) -- NOT 1/TRACK_FPS. Phase 2
    operates over export records only, so consecutive observations of the
    same track are ~1/EXPORT_FPS apart, not ~1/TRACK_FPS apart. Using the
    wrong floor here would have silently distorted every speed value again,
    the same class of bug fixed previously when export was still at 1Hz.
  - Assign lane and surrounding-vehicle IDs from metric positions
  - Label behaviour
  - Write one JSON per exported frame to output/video_name/





Manual inputs (video_lanes.json): lane count per time window, road type,
emergency_start_second.
"""

TRACK_FPS  = 30
EXPORT_FPS = 10
assert TRACK_FPS % EXPORT_FPS == 0, (
    "TRACK_FPS must be an integer multiple of EXPORT_FPS -- export frame "
    "selection below uses simple frame-count arithmetic, not float-timestamp "
    "comparisons, and requires this to divide evenly."
)
EXPORT_INTERVAL_FRAMES = TRACK_FPS // EXPORT_FPS   # export every Nth raw frame
MIN_DT = 1.0 / EXPORT_FPS   # floor for dt in Phase 2 -- see docstring above

sc = SceneClassifier()
lc = LaneConfig()


def process_video(video_path, start_s=0.0, end_s=None, output_name_override=None):
    """
    start_s/end_s: optional real-time window (seconds). Default processes
    the entire video, identical to earlier behaviour.

    output_name_override: if given, JSON is written to output/<override>/
    instead of output/<video_name>/ -- used for windowed test runs, so a
    partial window never collides with or gets mistaken for a full-video
    run's output. Lane/emergency config lookups always use the real
    video_name (the video_lanes.json key), regardless of this override.

    CAVEAT for windowed runs: BoT-SORT starts cold at start_s, with no
    warm-up/track history from t=0 -- expect possibly-inflated ID churn in
    the first few seconds of the window as a result. This is a property of
    testing a slice in isolation, not of the pipeline itself.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0][:30]
    output_name = output_name_override or video_name

    window_desc = f"{start_s}s-{end_s}s" if end_s is not None else "full video"
    print(f"\nProcessing: {video_name} @ {TRACK_FPS}Hz track / {EXPORT_FPS}Hz export, "
          f"no review, window={window_desc}, output=output/{output_name}/")
    if start_s > 0:
        print("  NOTE: tracker starts cold at start_s -- no warm-up from t=0, "
              "possible inflated churn in the first few seconds of this window.")

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
    # PHASE 1 — track at 30 Hz, export every EXPORT_INTERVAL_FRAMES-th frame
    # =================================================================
    records   = []   # one entry per EXPORTED frame (10 Hz)
    track_obs = {}   # track_id -> [(timestamp_float, x, y, reliable)]  (ALL 30Hz obs)

    frame_idx  = 0    # raw tracking frame counter (30 Hz)
    export_idx = 0    # export record counter (10 Hz) -- used for JSON filenames
    cached_second               = None
    cached_scenario_type        = None
    cached_lane_info            = None
    cached_emergency_active     = None

    for item in p.stream_frames(fps=TRACK_FPS, start_s=start_s, end_s=end_s):
        timestamp_float = item["timestamp"]
        whole_second     = int(timestamp_float)
        is_export_frame  = (frame_idx % EXPORT_INTERVAL_FRAMES == 0)

        frame_raw = item["frame"]
        frame     = p.spatial_crop(frame_raw)
        frame_height, frame_width = frame.shape[:2]

        tracked = t.update(d.model, frame)

        # scene/lane/emergency lookups: recompute only on crossing into a
        # new whole second, reuse for every raw frame within it
        if whole_second != cached_second:
            cached_second               = whole_second
            cached_scenario_type        = sc.classify(frame_raw)
            cached_lane_info            = lc.get_lane_info(video_name, whole_second, cached_scenario_type)
            cached_emergency_active, _  = lc.is_emergency_active(video_name, whole_second)

        vehicles_raw = []
        for v in tracked:
            x_raw, y_raw, reliable = h.get_vehicle_position(
                v["bbox"], v["type"], frame_width, frame_height, cached_lane_info,
                track_id=v["track_id"]
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
                "export_index":     export_idx,
                "timestamp":        timestamp_float,
                "scenario_type":    cached_scenario_type,
                "lane_info":        cached_lane_info,
                "emergency_active": cached_emergency_active,
                "vehicles_raw":     vehicles_raw,
            })

            if export_idx % (EXPORT_FPS * 60) == 0:
                print(f"  [phase 1] t={timestamp_float:.2f}s  export {export_idx}  "
                      f"{len(vehicles_raw)} vehicles  emergency={cached_emergency_active}")

            export_idx += 1

        frame_idx += 1

    # =================================================================
    # RTS SMOOTHING — per track, over the full 30 Hz trajectory
    # =================================================================
    print(f"  [smoothing] {len(track_obs)} tracks...")
    smoothed = sm.smooth(track_obs)

    # =================================================================
    # PHASE 2 — metrics from smoothed positions, annotate, export
    # =================================================================
    all_frames_data = []
    last_seen = {}   # track_id -> last EXPORTED timestamp_float, for correct dt

    for rec in records:
        timestamp        = rec["timestamp"]
        lane_info         = rec["lane_info"]
        emergency_active = rec["emergency_active"]

        vehicles = []
        for vr in rec["vehicles_raw"]:
            tid = vr["track_id"]

            x_m, y_m = smoothed[tid][timestamp]

            dt = max(timestamp - last_seen.get(tid, timestamp - MIN_DT), MIN_DT)
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
            "frame_index":      rec["export_index"],
            "timestamp":        timestamp,
            "emergency_active": emergency_active,
            "scenario_type":    rec["scenario_type"],
            "vehicles":         vehicles,
        })

        if rec["export_index"] % (EXPORT_FPS * 120) == 0:
            print(f"  [phase 2] t={timestamp:.2f}s  export {rec['export_index']} exported")

    e.save_batch(all_frames_data, output_name)

    return all_frames_data


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Process video(s). No args = full batch run over videos/ "
                     "(original behaviour, unchanged). --video = single-video "
                     "test run, optionally windowed with --start/--end."
    )
    ap.add_argument("--video", default=None,
                     help="Process only this one video (name without extension, "
                          "e.g. 'video6'), instead of the full videos/ batch.")
    ap.add_argument("--start", type=float, default=0.0,
                     help="Window start in seconds (only used with --video).")
    ap.add_argument("--end", type=float, default=None,
                     help="Window end in seconds (only used with --video). "
                          "Omit for start-to-end-of-video.")
    args = ap.parse_args()

    if args.video:
        matches = [f for f in os.listdir("videos")
                   if os.path.splitext(f)[0] == args.video]
        if not matches:
            raise SystemExit(f"No video named '{args.video}' found in videos/")
        video_path = os.path.join("videos", matches[0])

        windowed = args.start > 0 or args.end is not None
        output_name_override = None
        if windowed:
            end_label = int(args.end) if args.end is not None else "end"
            output_name_override = f"{args.video}_t{int(args.start)}-{end_label}"
            print(f"Windowed test run -> output/{output_name_override}/ "
                  f"(video_lanes.json lookups still use '{args.video}')")

        process_video(video_path, start_s=args.start, end_s=args.end,
                      output_name_override=output_name_override)
        print("\nDone.")

    else:
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