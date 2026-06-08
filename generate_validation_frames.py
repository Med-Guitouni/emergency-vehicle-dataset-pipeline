"""
generate_validation_frames.py

Generates 50 annotated validation frames spread evenly across the full video.
Each frame shows every detected vehicle with its full data on the bounding box.
Saves to validation_frames/ replacing all old frames.

Run AFTER main.py has finished (reads the output JSON files, does not rerun
the pipeline - much faster).
"""

import os
import json
import cv2
import numpy as np

from preprocessor import VideoPreprocessor
from detector import VehicleDetector
from tracker import VehicleTracker
from homography import HomographyEstimator
from lane_config import LaneConfig
from annotator import HeuristicAnnotator
from emergency_detector import EmergencyDetector
from surrounding import SurroundingVehicles

# ---- config ----
N_FRAMES      = 50      # how many frames to save
OUTPUT_DIR    = "validation_frames"

# colour per behaviour label
COLOURS = {
    "normal":          (180, 180, 180),   # grey
    "yielded":         (0,   220, 0),     # green
    "braked_abruptly": (0,   140, 255),   # orange
    "failed_to_yield": (0,   0,   255),   # red
}
UNKNOWN_COLOUR = (200, 200, 0)

# ---- setup ----
videos = [f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")]
if not videos:
    raise SystemExit("No video found in videos/")

video_path = videos[0]
video_name = os.path.splitext(os.path.basename(video_path))[0][:30]

# clear old validation frames
if os.path.exists(OUTPUT_DIR):
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".jpg") or f.endswith(".png"):
            os.remove(os.path.join(OUTPUT_DIR, f))
    print(f"Cleared old frames from {OUTPUT_DIR}/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- load pipeline components ----
p  = VideoPreprocessor(video_path)
d  = VehicleDetector()
t  = VehicleTracker()
h  = HomographyEstimator()
a  = HeuristicAnnotator()
ed = EmergencyDetector(video_path)
sv = SurroundingVehicles()
lc = LaneConfig()

frames = p.extract_frames(fps=1)
total  = len(frames)

sample_frames = [frames[i]["frame"] for i in range(min(10, total))]
ed.daytime = ed.detect_daytime(sample_frames)

# pick evenly spaced frame indices across the full video
indices = sorted(set(int(i * (total - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)))

print(f"Generating {len(indices)} annotated frames from {total} total frames...")

prev_vehicles = []

# we process every frame sequentially so Kalman/velocity state is correct,
# but only SAVE the ones at the chosen indices
for idx, item in enumerate(frames):
    timestamp = item["timestamp"]
    frame     = p.spatial_crop(item["frame"])
    fh, fw    = frame.shape[:2]

    ego_H, depth_map = h.process_frame(frame)
    tracked = t.update(d.model, frame, ego_H=ego_H, depth_map=depth_map)

    emergency_active, triggered_by = ed.is_emergency_active(
        timestamp, frame, [], prev_vehicles
    )

    lane_info = lc.get_lane_info(video_name, timestamp)
    vehicles  = []

    for v in tracked:
        tid    = v["track_id"]
        center = v["center"]
        bbox   = v["bbox"]
        bottom_center = [(bbox[0] + bbox[2]) // 2, bbox[3]]

        x_m, y_m = h.get_bev_position(bottom_center, fw, fh)
        fwd, lat, spd = h.estimate_relative_velocity(tid, x_m, y_m)
        acc  = h.estimate_acceleration(tid, spd)
        jrk  = h.estimate_jerk(tid, acc)
        hdg  = h.estimate_heading(tid, center)
        dist = h.estimate_distance_to_ego(x_m, y_m)
        lid  = h.estimate_lane_id(center[0], fw, lane_info)
        loff = h.estimate_lateral_offset(center[0], fw, lane_info)

        vdata = {
            "track_id": tid, "type": v["type"], "bbox": bbox, "center": center,
            "x_meters": x_m, "y_meters": y_m,
            "speed_kmh": spd, "forward_speed_ms": fwd, "lateral_speed_ms": lat,
            "acceleration": acc, "jerk": jrk, "heading_angle": hdg,
            "lane_id": lid, "lateral_offset": loff, "distance_to_ego": dist,
            "lanes_total": lane_info["lanes"], "road_type": lane_info["road_type"]
        }
        vdata["behaviour"] = a.annotate(vdata, emergency_active)
        vehicles.append(vdata)

    sv.assign(vehicles, lane_info)

    # ---- SAVE this frame if it is one of the chosen indices ----
    if idx in indices:
        vis = frame.copy()

        # horizon line (yellow) so you can see where the geometry is anchored
        horizon_y = int(h.horizon_ratio * fh)
        cv2.line(vis, (0, horizon_y), (fw, horizon_y), (0, 255, 255), 1)

        # emergency banner
        if emergency_active:
            cv2.rectangle(vis, (0, 0), (fw, 20), (0, 0, 160), -1)
            cv2.putText(vis, "EMERGENCY ACTIVE", (5, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # timestamp + lane info top-right
        info_str = (f"t={timestamp}s  lanes={lane_info['lanes']}"
                    f"  type={lane_info['road_type']}")
        cv2.putText(vis, info_str, (5, fh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        for v in vehicles:
            bbox = v["bbox"]
            beh  = v["behaviour"]
            col  = COLOURS.get(beh, UNKNOWN_COLOUR)

            # bounding box
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), col, 2)

            # build label lines - put as much as fits
            lines = [
                f"id={v['track_id']} {v['type']}",
                f"fwd={v['y_meters']:.0f}m lat={v['x_meters']:.1f}m",
                f"spd={v['speed_kmh']:.1f}km/h",
                f"fv={v['forward_speed_ms']:.1f} lv={v['lateral_speed_ms']:.1f}m/s",
                f"acc={v['acceleration']:.2f} jrk={v['jerk']:.2f}",
                f"hdg={v['heading_angle']:.1f}  lane={v['lane_id']}",
                f"dist={v['distance_to_ego']:.1f}m",
                beh,
            ]

            # draw lines above the bbox (stack upward from top edge)
            line_h = 11
            for i, line in enumerate(reversed(lines)):
                y_pos = bbox[1] - 4 - i * line_h
                if y_pos < line_h:
                    break  # no room above frame top
                cv2.putText(vis, line, (bbox[0], y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)

        out_path = os.path.join(OUTPUT_DIR, f"t{timestamp:04d}.jpg")
        cv2.imwrite(out_path, vis)

    prev_vehicles = vehicles

saved = len(indices)
print(f"\nDone. {saved} frames saved to {OUTPUT_DIR}/")
print("Colour key: GREY=normal  GREEN=yielded  ORANGE=braked_abruptly  RED=failed_to_yield")
print("Yellow line = assumed horizon (ground-plane geometry anchor)")