"""
Pipeline — two phases, 30 Hz tracking / 5 Hz export.

TRACK_FPS = 30, EXPORT_FPS = 5. Tracking runs denser than export: every 6th
raw tracking frame becomes an export record. TRACK_FPS must be an integer
multiple of EXPORT_FPS (asserted below) so export frame selection can use
simple frame-count arithmetic instead of float-timestamp tolerance
comparisons.

PHASE 1 (track at 30 Hz, export every 6th frame at 5 Hz)
  - Crop sky and dashboard, resize to fixed 1280x720
  - Detect vehicles (YOLOv8x), track with BoT-SORT at 30 Hz
  - Every raw frame: project bounding boxes to metres, feed track_obs so the
    RTS smoother sees the full 30 Hz trajectory.
  - Only on export frames (every 6th raw frame): store a record for export.
  - Scene classification (CNN) and lane/emergency lookups are recomputed only
    when the whole real second changes, and reused across every frame within
    it -- independent of both TRACK_FPS and EXPORT_FPS.

BETWEEN PHASES — RTS SMOOTHING
  Full 30 Hz trajectory per vehicle smoothed with the RTS smoother.

PHASE 2 (over the 5 Hz export records, after smoothing)
  - Derive split velocity, longitudinal and lateral acceleration, jerk and
    TTC to ego using the REAL elapsed time between consecutive EXPORTED
    observations of a track (dt), floored at 1/EXPORT_FPS (0.2 s) -- NOT
    1/TRACK_FPS. Phase 2 operates over export records only, so consecutive
    observations of the same track are ~1/EXPORT_FPS apart. Using the wrong
    floor here would silently distort every speed value, the same class of
    bug fixed previously when export was still at 1 Hz.
  - Assign lane and surrounding-vehicle IDs from metric positions
  - Apply the kinematic behaviour rules
  - Write one JSON per exported frame to output/video_name/

BEHAVIOUR LABELS. The labels this file writes are the kinematic-rule output
of annotator.py. They are a starting point, not the released labels: the
released corpus is labelled by hand in review.py, which overwrites the
behaviour field in place. Braking is the exception -- it stays rule-derived,
since it can be confirmed visually from brake lights but not measured by eye.
No review tool is launched from here; run review.py separately after a video
finishes.

ANNOTATOR THRESHOLD NOTE: annotator.py's frame-count constants (YIELD_PERSIST,
CUMULATIVE_WINDOW, MIN_OBSERVED_FRAMES) are expressed in EXPORT frames, since
annotate() is only ever called once per exported observation (Phase 2, below).
See annotator.py's docstring.

Manual inputs (video_lanes.json): lane count per time window, road type,
emergency_start_second.
"""

import os
import torch

from preprocessor import VideoPreprocessor
from detector import VehicleDetector
from tracker import VehicleTracker
from homography import HomographyEstimator
from exporter import JSONExporter
from annotator import HeuristicAnnotator
from surrounding import SurroundingVehicles
from lane_config import LaneConfig
from smoother import RTSSmoother
from scene_classifier import SceneClassifier

TRACK_FPS  = 30
EXPORT_FPS = 5
assert TRACK_FPS % EXPORT_FPS == 0, (
    "TRACK_FPS must be an integer multiple of EXPORT_FPS -- export frame "
    "selection below uses simple frame-count arithmetic, not float-timestamp "
    "comparisons, and requires this to divide evenly."
)
EXPORT_INTERVAL_FRAMES = TRACK_FPS // EXPORT_FPS   # export every Nth raw frame
MIN_DT = 1.0 / EXPORT_FPS   # floor for dt in Phase 2 -- see docstring above

lc = LaneConfig()

# Scene classifier, loaded lazily: it downloads the Places365 weights on first
# use, and a run whose videos are all annotated in video_lanes.json never needs
# its prediction. Created once and reused across videos, with reset() clearing
# its confirmation state between them.
_sc = None


class _NoSceneClassifier:
    """
    Stand-in used when the Places365 weights cannot be loaded (no network, for
    instance). Returns "unknown", which lane_config.py treats as "no confirmed
    prediction" and answers with DEFAULT_ROAD_TYPE and
    lane_source="default_highway_3lane". A missing fallback must not abort a
    run whose videos are annotated anyway.
    """

    def classify(self, frame):
        return "unknown"

    def reset(self):
        pass


def get_scene_classifier():
    global _sc
    if _sc is None:
        try:
            _sc = SceneClassifier()
        except Exception as ex:
            print(f"WARNING: scene classifier unavailable ({ex}). Unannotated "
                  f"segments will fall back to the highway/3-lane default.")
            _sc = _NoSceneClassifier()
    return _sc


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
    sc = get_scene_classifier()
    sc.reset()   # confirmation state must not carry over from the last video

    # =================================================================
    # PHASE 1 — track at 30 Hz, export every EXPORT_INTERVAL_FRAMES-th frame
    # =================================================================
    records   = []   # one entry per EXPORTED frame (5 Hz)
    track_obs = {}   # track_id -> [(timestamp_float, x, y, reliable)]  (ALL 30Hz obs)

    frame_idx  = 0    # raw tracking frame counter (30 Hz)
    export_idx = 0    # export record counter (5 Hz) -- used for JSON filenames
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

        tracked = t.update(d.model, frame, device=d.device)

        # Lane/emergency lookups and scene classification: recomputed only on
        # crossing into a new whole second, then reused for every raw frame
        # within it. This cadence is independent of TRACK_FPS and EXPORT_FPS --
        # re-running a CNN faster than the scene can plausibly change would be
        # pure waste.
        #
        # The classifier runs on the UNCROPPED frame (frame_raw): it was trained
        # on whole scenes, and the pipeline's crop removes the sky and horizon
        # context it relies on. It votes over its top-5 Places365 predictions
        # and requires three identical predictions in a row before accepting a
        # change, so one misclassified frame cannot flip the label. It is not
        # the primary source of road type -- manual annotation in
        # video_lanes.json always wins -- it is the automatic fallback for
        # segments that have not been annotated, so the pipeline can still
        # assign a sensible lane width instead of an arbitrary guess.
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
    # SEED homography state from the smoothed trajectory BEFORE Phase 2
    # =================================================================
    # estimate_relative_velocity() differences a track's stored previous
    # metric position, kept in h.prev_positions_m. On a track's first Phase-2
    # call no prior exists, so its first exported velocity would be an
    # artefact zero. At 30 Hz tracking each track's smoother already holds
    # several raw observations BEFORE its first exported frame, so the first
    # smoothed position is pre-loaded here and the first exported frame of
    # every track gets a real velocity instead.
    for tid, ts_dict in smoothed.items():
        ts_sorted = sorted(ts_dict.keys())
        if len(ts_sorted) < 2:
            continue
        t0 = ts_sorted[0]
        x0, y0 = ts_dict[t0]
        # seed positions so estimate_relative_velocity knows the prior
        h.prev_positions_m[tid] = (x0, y0)

    # =================================================================
    # PHASE 2 — metrics from smoothed positions, annotate, export
    # =================================================================
    # ACCEL_WINDOW_S: acceleration and jerk are computed over this fixed real-
    # time window rather than the per-export-frame dt (0.2 s at 5 Hz export).
    # Dividing an ordinary few-centimetre position fluctuation by a small
    # denominator amplifies it sharply -- the per-frame version produced
    # values up to 653 m/s². A 1-second window matches highD's approach,
    # is independent of the export rate, and keeps the denominator large
    # enough for the result to stay physically meaningful.
    ACCEL_WINDOW_S = 1.0

    all_frames_data = []
    last_seen = {}        # track_id -> last EXPORTED timestamp_float, for velocity dt
    speed_history = {}    # track_id -> list of (timestamp, forward_speed_ms)
    accel_hist    = {}    # track_id -> list of (timestamp, acceleration) for jerk window
    lat_speed_hist = {}   # track_id -> list of (timestamp, lateral_speed_ms) for lateral accel

    # seed last_seen from the same t0 used to seed prev_positions_m above,
    # so dt on each track's first exported frame reflects the real elapsed
    # time since the seeded position, not the MIN_DT fallback.
    for tid, ts_dict in smoothed.items():
        ts_sorted = sorted(ts_dict.keys())
        if ts_sorted:
            last_seen[tid] = ts_sorted[0]

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

            # --- acceleration over 1-second window (not per-export-frame dt) ---
            # Uses a fixed 1-second lookback (highD-style) rather than the
            # per-export-frame dt. When the window isn't full yet (track
            # younger than 1 real second), acceleration and jerk are exported
            # as null rather than computed from a fabricated prior.
            history = speed_history.setdefault(tid, [])
            history.append((timestamp, forward_speed))
            speed_history[tid] = [(t, s) for t, s in history if t >= timestamp - 2.0]
            past = [(t, s) for t, s in speed_history[tid] if t <= timestamp - ACCEL_WINDOW_S]
            if past:
                t_past, spd_past = past[-1]
                accel_dt = max(timestamp - t_past, MIN_DT)
                acceleration = round((forward_speed - spd_past) / accel_dt, 3)

                # --- jerk over the SAME 1-second window ---
                # jerk = delta(accel) / delta(t), where delta(t) is the REAL
                # elapsed time between the two acceleration values, not
                # ACCEL_WINDOW_S. An earlier version divided a one-export-frame
                # change by a full second, making every jerk value far too
                # small and putting the brake-onset rule out of reach.
                accel_history = accel_hist.setdefault(tid, [])
                accel_history.append((timestamp, acceleration))
                accel_hist[tid] = [(t, a) for t, a in accel_history if t >= timestamp - 2.0]
                past_accel = [(t, a) for t, a in accel_hist[tid]
                              if t <= timestamp - ACCEL_WINDOW_S]
                if past_accel:
                    t_a_past, a_past = past_accel[-1]
                    jerk_dt = max(timestamp - t_a_past, MIN_DT)
                    jerk = round((acceleration - a_past) / jerk_dt, 3)
                else:
                    jerk = None   # accel window full but jerk window not yet
            else:
                acceleration = None
                jerk = None

            # --- lateral acceleration over the same 1-second window ---
            # MTP-GO wants both longitudinal AND lateral acceleration. Same
            # method as longitudinal: change in lateral_speed_ms per second,
            # over the 1s window (not the per-frame dt, same noise reasoning).
            # A strong yielding signal -- a car swerving aside has clear
            # lateral acceleration.
            lat_history = lat_speed_hist.setdefault(tid, [])
            lat_history.append((timestamp, lateral_speed))
            lat_speed_hist[tid] = [(t, s) for t, s in lat_history if t >= timestamp - 2.0]
            past_lat = [(t, s) for t, s in lat_speed_hist[tid]
                        if t <= timestamp - ACCEL_WINDOW_S]
            if past_lat:
                t_lat_past, lat_spd_past = past_lat[-1]
                lat_accel_dt = max(timestamp - t_lat_past, MIN_DT)
                lateral_acceleration = round((lateral_speed - lat_spd_past) / lat_accel_dt, 3)
            else:
                lateral_acceleration = None

            distance_to_ego = h.estimate_distance_to_ego(x_m, y_m)
            ttc_to_ego      = h.estimate_ttc_to_ego(y_m, forward_speed)
            lane_id         = h.estimate_lane_id(x_m, lane_info)
            lateral_offset  = h.estimate_lateral_offset(x_m, lane_info)
            lane_position_norm = h.estimate_lane_position_norm(x_m, lane_info)
            road_position_norm = h.estimate_road_position_norm(x_m, lane_info)

            vehicles.append({
                "track_id":            tid,
                "type":                vr["type"],
                "bbox":                vr["bbox"],
                "x_meters":            x_m,
                "y_meters":            y_m,
                "position_reliable":   vr["reliable"],
                "speed_kmh":           speed,
                "forward_speed_ms":    forward_speed,
                "lateral_speed_ms":    lateral_speed,
                "acceleration":        acceleration,
                "lateral_acceleration": lateral_acceleration,
                "jerk":                jerk,
                "ttc_to_ego":          ttc_to_ego,
                "lane_id":             lane_id,
                "lateral_offset":      lateral_offset,
                "lane_position_norm":  lane_position_norm,
                "road_position_norm":  road_position_norm,
                "distance_to_ego":     distance_to_ego,
                "lanes_total":         lane_info["lanes"],
                "road_type":           lane_info["road_type"],
                "lane_source":         lane_info["source"],
            })

        # --- EGO VEHICLE ------------------------------------------------------
        # The ambulance itself, always present in every exported frame, at the
        # origin of its own relative frame. Position (0,0) and velocity (0,0)
        # are trivially true since every other vehicle's kinematics are
        # measured RELATIVE to the ego. Inserted BEFORE sv.assign() so other
        # vehicles correctly find the ego as a neighbour (preceding/following).
        # track_id = 0 is reserved for the ego (real tracks are >= 1).
        ego = {
            "track_id":            0,
            "type":                "ego",
            "bbox":                None,
            "x_meters":            0.0,
            "y_meters":            0.0,
            "position_reliable":   True,   # ego's own position is exactly known
            "speed_kmh":           0.0,
            "forward_speed_ms":    0.0,
            "lateral_speed_ms":    0.0,
            "acceleration":        0.0,
            "lateral_acceleration": 0.0,
            "jerk":                0.0,
            "ttc_to_ego":          None,
            "lane_id":             h.estimate_lane_id(0.0, lane_info),
            "lateral_offset":      h.estimate_lateral_offset(0.0, lane_info),
            "lane_position_norm":  h.estimate_lane_position_norm(0.0, lane_info),
            "road_position_norm":  h.estimate_road_position_norm(0.0, lane_info),
            "distance_to_ego":     0.0,
            "lanes_total":         lane_info["lanes"],
            "road_type":           lane_info["road_type"],
            "lane_source":         lane_info["source"],
        }
        vehicles.append(ego)

        sv.assign(vehicles, lane_info)

        for v in vehicles:
            if v["track_id"] == 0:
                v["behaviour"] = "ego"   # the ambulance doesn't "yield" to itself
            else:
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

    # CUDA check -- printed first, before any model loads, so it's the very
    # first thing visible in the log regardless of which video/mode runs.
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"CUDA available: YES  [{torch.cuda.get_device_name(0)}]")
    else:
        print("CUDA available: NO  -- running on CPU, this will be SLOW "
              "for 30Hz tracking. Check your torch/CUDA install if a GPU "
              "is expected to be present.")
    print("=" * 60)

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
