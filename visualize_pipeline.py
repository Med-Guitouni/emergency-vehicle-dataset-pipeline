"""
visualize_pipeline.py

Reads the JSON output produced by main.py and the original video,
then renders every second of the video as an annotated frame showing
exactly what the pipeline measured — no recomputation.

Layout:
  LEFT  = cropped video frame with metric grid + vehicle boxes
  RIGHT = info panel listing every vehicle sorted by distance to ego

Box colours:
  GREEN = position_reliable True
  RED   = position_reliable False

Output folder: pipeline_visualization/
One jpg per second covering the full video duration.

python3 visualize_pipeline.py
"""

import os
import json
import cv2
import numpy as np

from preprocessor import VideoPreprocessor
from homography import HomographyEstimator
from lane_config import LaneConfig

# ------------------------------------------------------------------ config ---
OUTPUT_DIR = "pipeline_visualization"
PANEL_W    = 320   # width of the right info panel in pixels

FWD_GRID_M = [5, 10, 20, 30, 50, 75, 100]
LAT_GRID_M = [-10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5]

COL_RELIABLE   = (0, 220, 0)
COL_UNRELIABLE = (0, 0, 255)
COL_YIELDED    = (0, 220, 255)
COL_BRAKED     = (0, 100, 255)
COL_FAILED     = (0, 0, 180)
GRID_COL       = (255, 200, 0)
AXIS_COL       = (0, 255, 255)
PANEL_BG       = (20, 20, 20)
TEXT_COL       = (220, 220, 220)

BEHAVIOUR_COLS = {
    "yielded":         COL_YIELDED,
    "braked_abruptly": COL_BRAKED,
    "failed_to_yield": COL_FAILED,
    "normal":          None,   # use reliability colour
}


def behaviour_colour(v):
    b = v.get("behaviour", "normal")
    override = BEHAVIOUR_COLS.get(b)
    if override:
        return override
    return COL_RELIABLE if v.get("position_reliable", True) else COL_UNRELIABLE


def project_to_pixel(x_m, y_m, f, cx, horizon_row, cam_h):
    if y_m <= 0.01:
        return None
    row = horizon_row + (cam_h * f) / y_m
    col = cx + (x_m * f) / y_m
    return int(round(col)), int(round(row))


def draw_grid(vis, f, cx, fh, horizon_row, cam_h):
    fw = vis.shape[1]
    cv2.line(vis, (0, int(horizon_row)), (fw, int(horizon_row)), AXIS_COL, 1)
    cv2.putText(vis, "horizon", (fw - 70, int(horizon_row) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, AXIS_COL, 1)

    delta_bottom = max(fh - horizon_row, 1.0)
    d_near  = (cam_h * f) / delta_bottom
    vanish  = (int(cx), int(horizon_row))

    for x_m in LAT_GRID_M:
        end = project_to_pixel(x_m, d_near, f, cx, horizon_row, cam_h)
        if end is None:
            continue
        thick = 2 if abs(x_m) < 0.01 else 1
        color = AXIS_COL if abs(x_m) < 0.01 else GRID_COL
        cv2.line(vis, vanish, end, color, thick)
        if end[1] <= fh:
            cv2.putText(vis, f"{x_m:+.1f}",
                        (end[0] - 10, min(end[1] - 2, fh - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, GRID_COL, 1)

    for d in FWD_GRID_M:
        row = horizon_row + (cam_h * f) / d
        if row >= fh or row <= horizon_row:
            continue
        row   = int(round(row))
        left  = project_to_pixel(LAT_GRID_M[0],  d, f, cx, horizon_row, cam_h)
        right = project_to_pixel(LAT_GRID_M[-1], d, f, cx, horizon_row, cam_h)
        if left and right:
            cv2.line(vis, (left[0], row), (right[0], row), GRID_COL, 1)
        cv2.putText(vis, f"{d}m", (6, row - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRID_COL, 1)

    cv2.circle(vis, (int(cx), fh - 4), 5, AXIS_COL, -1)
    cv2.putText(vis, "EGO(0,0)", (int(cx) - 28, fh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, AXIS_COL, 1)


def draw_panel(data, vehicles_sorted, fh):
    panel = np.full((fh, PANEL_W, 3), PANEL_BG, dtype=np.uint8)

    # scene-level info at the top
    ego_spd = data.get("ego_speed_ms")
    ego_str = f"{ego_spd:.1f} m/s ({ego_spd*3.6:.0f} km/h)" if ego_spd else "n/a"
    scene_type = data.get("scenario_type", "?")

    cv2.putText(panel, f"scene: {scene_type}", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 60), 1)
    cv2.putText(panel, f"ego:   {ego_str}", (6, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 60), 1)
    cv2.line(panel, (4, 32), (PANEL_W - 4, 32), (80, 80, 80), 1)

    cv2.putText(panel, "ID   type  x      y     spd    beh",
                (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 60), 1)
    cv2.line(panel, (4, 48), (PANEL_W - 4, 48), (60, 60, 60), 1)

    row_h = 13
    y_cur = 60
    for v in vehicles_sorted:
        if y_cur + row_h > fh:
            break
        col = behaviour_colour(v)
        beh = v.get("behaviour", "normal")[:3]
        line = (f"id{v['id']:<3} {v['type'][:3]:<4}"
                f" {v['x_meters']:+5.1f}"
                f" {v['y_meters']:5.1f}"
                f" {v['speed_kmh']:5.1f}"
                f" {beh}")
        cv2.putText(panel, line, (6, y_cur),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1)
        y_cur += row_h

    # legend at bottom
    legend_y = fh - 50
    cv2.line(panel, (4, legend_y), (PANEL_W - 4, legend_y), (60, 60, 60), 1)
    items = [
        (COL_RELIABLE,   "reliable"),
        (COL_UNRELIABLE, "unreliable"),
        (COL_YIELDED,    "yielded"),
        (COL_BRAKED,     "braked"),
        (COL_FAILED,     "failed"),
    ]
    for i, (c, label) in enumerate(items):
        y = legend_y + 10 + i * 10
        cv2.rectangle(panel, (6, y - 6), (14, y + 2), c, -1)
        cv2.putText(panel, label, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_COL, 1)

    return panel


# ------------------------------------------------------------------ setup ---
videos = [f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")]
if not videos:
    raise SystemExit("No video found in videos/")

video_path = sorted(videos)[0]
video_name = os.path.splitext(os.path.basename(video_path))[0][:30]
json_dir   = os.path.join("output", video_name)

if not os.path.exists(json_dir):
    raise SystemExit(f"No JSON output at {json_dir}/ — run main.py first.")

# count available JSON files to know the full duration
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
total_seconds = len(json_files)
print(f"Video:      {video_path}")
print(f"JSON dir:   {json_dir}/ ({total_seconds} seconds)")
print(f"Output dir: {OUTPUT_DIR}/")

# clean and create output folder
if os.path.exists(OUTPUT_DIR):
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith((".jpg", ".png")):
            os.remove(os.path.join(OUTPUT_DIR, f))
os.makedirs(OUTPUT_DIR, exist_ok=True)

p  = VideoPreprocessor(video_path)
h  = HomographyEstimator()

cam_h      = h.camera_height
foc_factor = h.focal_length_factor
hor_ratio  = h.horizon_ratio
cx_ratio   = h.CX_RATIO

print(f"\nRendering all {total_seconds} seconds...")

saved = 0
for item in p.extract_frames(fps=1):
    timestamp = item["timestamp"]

    json_path = os.path.join(json_dir, f"t{timestamp:04d}.json")
    if not os.path.exists(json_path):
        continue

    with open(json_path) as jf:
        data = json.load(jf)

    frame = p.spatial_crop(item["frame"])
    fh, fw = frame.shape[:2]

    f_px        = fw * foc_factor
    cx          = fw * cx_ratio
    horizon_row = hor_ratio * fh

    vis = frame.copy()
    draw_grid(vis, f_px, cx, fh, horizon_row, cam_h)

    emergency = data.get("emergency_active", False)
    emg_col   = (0, 0, 255) if emergency else (255, 255, 255)
    cv2.putText(vis,
                f"t={timestamp}s  "
                f"emergency={'YES' if emergency else 'no'}  "
                f"[pipeline output]",
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, emg_col, 1)

    vehicles = data.get("vehicles", [])
    vehicles_sorted = sorted(vehicles,
                             key=lambda v: v.get("distance_to_ego", 999))

    for v in vehicles:
        bbox = v.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        col = behaviour_colour(v)

        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)

        # dot at bottom-centre (the projection input pixel)
        bcx = int((x1 + x2) / 2)
        cv2.circle(vis, (bcx, y2), 3, col, -1)

        # ID + behaviour label on box
        beh = v.get("behaviour", "normal")
        label = f"id{v['id']}" if beh == "normal" else f"id{v['id']} {beh[:3]}"
        cv2.putText(vis, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    panel   = draw_panel(data, vehicles_sorted, fh)
    combined = np.hstack([vis, panel])

    out_path = os.path.join(OUTPUT_DIR, f"t{timestamp:04d}.jpg")
    cv2.imwrite(out_path, combined)
    saved += 1

    if saved % 60 == 0:
        print(f"  {saved}/{total_seconds} frames saved...")

print(f"\nDone. {saved} frames saved to {OUTPUT_DIR}/")
print("GREEN=reliable  RED=unreliable  YELLOW=yielded  ORANGE=braked  DARK RED=failed")