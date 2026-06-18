"""
generate_validation_frames.py  (GEOMETRY VALIDATION VERSION)

Reads the JSON files produced by main.py and the original video,
then draws the JSON values on each frame alongside a ground-plane
metric reference grid.

Layout:
  LEFT  = cropped video frame with metric grid + boxes labelled by ID only
  RIGHT = black info panel listing every vehicle sorted by distance to ego

Nothing is recomputed. This validates what is already in the JSON.

Box colours:
  GREEN  = position_reliable True
  RED    = position_reliable False

Run: python3 generate_validation_frames.py
Requires: video in videos/, JSON output in output/<video_name>/
"""

import os
import json
import cv2
import numpy as np

from preprocessor import VideoPreprocessor
from homography import HomographyEstimator
from lane_config import LaneConfig

# ---------------- config ----------------
FIRST_N_SECONDS = 60
OUTPUT_DIR      = "validation_frames"
PANEL_W         = 280   # width of the right info panel in pixels

FWD_GRID_M = [5, 10, 20, 30, 50, 75, 100]
LAT_GRID_M = [-10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5]

COL_RELIABLE   = (0, 220, 0)
COL_UNRELIABLE = (0, 0, 255)
GRID_COL       = (255, 200, 0)
AXIS_COL       = (0, 255, 255)
PANEL_BG       = (20, 20, 20)
TEXT_COL       = (220, 220, 220)


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
    d_near = (cam_h * f) / delta_bottom
    vanish = (int(cx), int(horizon_row))

    for x_m in LAT_GRID_M:
        end = project_to_pixel(x_m, d_near, f, cx, horizon_row, cam_h)
        if end is None:
            continue
        thick = 2 if abs(x_m) < 0.01 else 1
        color = AXIS_COL if abs(x_m) < 0.01 else GRID_COL
        cv2.line(vis, vanish, end, color, thick)
        if end[1] <= fh:
            cv2.putText(vis, f"{x_m:+.1f}", (end[0] - 10, min(end[1] - 2, fh - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, GRID_COL, 1)

    for d in FWD_GRID_M:
        row = horizon_row + (cam_h * f) / d
        if row >= fh or row <= horizon_row:
            continue
        row = int(round(row))
        left  = project_to_pixel(LAT_GRID_M[0],  d, f, cx, horizon_row, cam_h)
        right = project_to_pixel(LAT_GRID_M[-1], d, f, cx, horizon_row, cam_h)
        if left and right:
            cv2.line(vis, (left[0], row), (right[0], row), GRID_COL, 1)
        cv2.putText(vis, f"{d}m", (6, row - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRID_COL, 1)

    cv2.circle(vis, (int(cx), fh - 4), 5, AXIS_COL, -1)
    cv2.putText(vis, "EGO(0,0)", (int(cx) - 28, fh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, AXIS_COL, 1)


def draw_panel(vehicles_sorted, fh):
    """Build the right-side info panel as a numpy image."""
    panel = np.full((fh, PANEL_W, 3), PANEL_BG, dtype=np.uint8)

    cv2.putText(panel, "ID  type   x      y    spd   dist",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 60), 1)
    cv2.line(panel, (4, 20), (PANEL_W - 4, 20), (80, 80, 80), 1)

    row_h  = 13
    y_cur  = 34
    for v in vehicles_sorted:
        if y_cur + row_h > fh:
            break

        reliable = v.get("position_reliable", True)
        col = COL_RELIABLE if reliable else COL_UNRELIABLE

        line = (f"id{v['id']:<4} {v['type'][:3]:<4}"
                f" x={v['x_meters']:+5.1f}"
                f" y={v['y_meters']:5.1f}"
                f" {v['speed_kmh']:5.1f}km/h"
                f" {v['distance_to_ego']:5.1f}m")
        cv2.putText(panel, line, (6, y_cur),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1)
        y_cur += row_h

    return panel


# ---------------- setup ----------------
videos = [f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")]
if not videos:
    raise SystemExit("No video found in videos/")

video_path = videos[0]
video_name = os.path.splitext(os.path.basename(video_path))[0][:30]
json_dir   = os.path.join("output", video_name)

if not os.path.exists(json_dir):
    raise SystemExit(f"No JSON output at {json_dir}/ — run main.py first.")

if os.path.exists(OUTPUT_DIR):
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith((".jpg", ".png")):
            os.remove(os.path.join(OUTPUT_DIR, f))
os.makedirs(OUTPUT_DIR, exist_ok=True)

p  = VideoPreprocessor(video_path)
h  = HomographyEstimator()
lc = LaneConfig()

cam_h      = h.camera_height
foc_factor = h.focal_length_factor
hor_ratio  = h.horizon_ratio

print(f"Reading JSON from {json_dir}/")
print(f"Saving t=0 to t={FIRST_N_SECONDS}s → {OUTPUT_DIR}/")

saved = 0
for item in p.extract_frames(fps=1):
    timestamp = item["timestamp"]
    if timestamp > FIRST_N_SECONDS:
        break

    json_path = os.path.join(json_dir, f"t{timestamp:04d}.json")
    if not os.path.exists(json_path):
        print(f"  [skip] no JSON for t={timestamp}")
        continue

    with open(json_path) as jf:
        data = json.load(jf)

    frame = p.spatial_crop(item["frame"])
    fh, fw = frame.shape[:2]

    f           = fw * foc_factor
    cx          = fw / 2.0
    horizon_row = hor_ratio * fh

    vis = frame.copy()
    draw_grid(vis, f, cx, fh, horizon_row, cam_h)

    emergency = data.get("emergency_active", False)
    cv2.putText(vis,
                f"t={timestamp}s  emergency={'YES' if emergency else 'no'}  [FROM JSON]",
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    vehicles = data.get("vehicles", [])
    vehicles_sorted = sorted(vehicles, key=lambda v: v.get("distance_to_ego", 999))

    for v in vehicles:
        bbox = v.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        reliable = v.get("position_reliable", True)
        col = COL_RELIABLE if reliable else COL_UNRELIABLE

        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)

        # dot at bottom-centre (the projection input pixel)
        bcx = int((x1 + x2) / 2)
        cv2.circle(vis, (bcx, y2), 3, col, -1)

        # only ID on the box — everything else is in the panel
        cv2.putText(vis, f"id{v['id']}", (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    panel = draw_panel(vehicles_sorted, fh)
    combined = np.hstack([vis, panel])

    out_path = os.path.join(OUTPUT_DIR, f"t{timestamp:04d}.jpg")
    cv2.imwrite(out_path, combined)
    saved += 1

print(f"\nDone. {saved} frames saved to {OUTPUT_DIR}/")
print("GREEN = position_reliable   RED = unreliable")
print("Panel sorted by distance to ego (closest first)")