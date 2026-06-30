"""
visualize_pipeline.py

Reads the JSON output produced by main.py and the original video,
then renders every second as an annotated frame showing exactly what
the pipeline measured — no recomputation.

Layout:
  LEFT  = cropped video frame with metric grid + vehicle boxes
  RIGHT = info panel listing every vehicle sorted by distance to ego

Box colours:
  GREEN      = yielded
  RED        = any other behaviour (normal / braked_abruptly / failed_to_yield)
  Dim border = position_reliable False (dashed look via thinner line)

TTC displayed on each box when available (vehicles ahead + closing only).

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

# ------------------------------------------------------------------ config ---
OUTPUT_DIR = "pipeline_visualization"
PANEL_W    = 340

FWD_GRID_M = [5, 10, 20, 30, 50, 75, 100]
LAT_GRID_M = [-10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5]

COL_YIELDED    = (0, 220, 0)    # green  — yielded
COL_OTHER      = (0, 0, 220)    # red    — normal / braked / failed
COL_UNRELIABLE = (80, 80, 80)   # dark grey border when position unreliable
GRID_COL       = (255, 200, 0)
AXIS_COL       = (0, 255, 255)
PANEL_BG       = (20, 20, 20)
TEXT_COL       = (220, 220, 220)


def box_colour(v):
    """Green for yielded, red for everything else."""
    if v.get("behaviour") == "yielded":
        return COL_YIELDED
    return COL_OTHER


def box_thickness(v):
    """Thinner border signals an unreliable position measurement."""
    return 1 if not v.get("position_reliable", True) else 2


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


def draw_vehicles(vis, vehicles):
    for v in vehicles:
        bbox = v.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        col   = box_colour(v)
        thick = box_thickness(v)

        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thick)

        # dot at bottom-centre (the projection input pixel)
        bcx = int((x1 + x2) / 2)
        cv2.circle(vis, (bcx, y2), 3, col, -1)

        # top label: ID + behaviour abbreviation
        beh = v.get("behaviour", "normal")
        beh_short = "" if beh == "normal" else f" {beh[:3].upper()}"
        top_label = f"id{v['id']}{beh_short}"
        cv2.putText(vis, top_label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

        # bottom label: TTC when available
        ttc = v.get("ttc_to_ego")
        if ttc is not None:
            ttc_label = f"TTC {ttc:.1f}s"
            cv2.putText(vis, ttc_label, (x1 + 2, y2 + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)


def draw_panel(data, vehicles_sorted, fh):
    panel = np.full((fh, PANEL_W, 3), PANEL_BG, dtype=np.uint8)

    scene_type = data.get("scenario_type", "?")
    cv2.putText(panel, f"scene: {scene_type}", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 60), 1)
    cv2.line(panel, (4, 20), (PANEL_W - 4, 20), (80, 80, 80), 1)

    # column header
    cv2.putText(panel, "ID   type  x      y    spd   TTC    beh",
                (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 160, 60), 1)
    cv2.line(panel, (4, 36), (PANEL_W - 4, 36), (60, 60, 60), 1)

    row_h = 13
    y_cur = 48
    for v in vehicles_sorted:
        if y_cur + row_h > fh - 30:
            break
        col = box_colour(v)
        beh = v.get("behaviour", "normal")[:3]
        ttc = v.get("ttc_to_ego")
        ttc_str = f"{ttc:5.1f}s" if ttc is not None else "  -- "
        line = (f"id{v['id']:<3} {v['type'][:3]:<4}"
                f" {v['x_meters']:+5.1f}"
                f" {v['y_meters']:5.1f}"
                f" {v['speed_kmh']:5.1f}"
                f" {ttc_str}"
                f" {beh}")
        cv2.putText(panel, line, (6, y_cur),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)
        y_cur += row_h

    # legend
    legend_y = fh - 28
    cv2.line(panel, (4, legend_y), (PANEL_W - 4, legend_y), (60, 60, 60), 1)
    items = [
        (COL_YIELDED,    "yielded"),
        (COL_OTHER,      "normal / braked / failed"),
        (COL_UNRELIABLE, "position unreliable (thin border)"),
        ((0, 200, 255),  "TTC = seconds to reach ego"),
    ]
    for i, (c, label) in enumerate(items):
        y = legend_y + 8 + i * 9
        if y > fh - 2:
            break
        cv2.rectangle(panel, (6, y - 5), (13, y + 2), c, -1)
        cv2.putText(panel, label, (17, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, TEXT_COL, 1)

    return panel


# ------------------------------------------------------------------ setup ---
videos = sorted([f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")])
if not videos:
    raise SystemExit("No video found in videos/")

video_path = videos[0]
video_name = os.path.splitext(os.path.basename(video_path))[0][:30]
json_dir   = os.path.join("output", video_name)

if not os.path.exists(json_dir):
    raise SystemExit(f"No JSON output at {json_dir}/ — run main.py first.")

json_files    = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
total_seconds = len(json_files)
print(f"Video:      {video_path}")
print(f"JSON dir:   {json_dir}/ ({total_seconds} seconds)")
print(f"Output dir: {OUTPUT_DIR}/")

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

print(f"\nRendering {total_seconds} seconds...")

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
    draw_vehicles(vis, data.get("vehicles", []))

    emergency = data.get("emergency_active", False)
    emg_col   = (0, 0, 255) if emergency else (255, 255, 255)
    cv2.putText(vis,
                f"t={timestamp}s  emergency={'YES' if emergency else 'no'}",
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, emg_col, 1)

    vehicles_sorted = sorted(data.get("vehicles", []),
                             key=lambda v: v.get("distance_to_ego", 999))
    panel    = draw_panel(data, vehicles_sorted, fh)
    combined = np.hstack([vis, panel])

    out_path = os.path.join(OUTPUT_DIR, f"t{timestamp:04d}.jpg")
    cv2.imwrite(out_path, combined)
    saved += 1

    if saved % 60 == 0:
        print(f"  {saved}/{total_seconds} frames saved...")

print(f"\nDone. {saved} frames saved to {OUTPUT_DIR}/")
print("GREEN = yielded   RED = other   TTC shown in cyan above box bottom")