"""
visualize_bev.py

Still needs work but good to scrape through json files visually


Usage:  python3 visualize_bev.py
"""

import os
import json
import cv2
import numpy as np

# ---- config ----
OUTPUT_DIR   = "output"
OUTPUT_FILE  = "bev_animation.mp4"
FPS          = 10       # rendered fps
INTERP_STEPS = 10       # frames between each JSON second -> 10fps * 10 = 1 frame per 0.1s

WIDTH        = 900
HEIGHT       = 1000

# how much of the road to show around ego
VIEW_AHEAD   = 80.0     # metres ahead of ego
VIEW_BEHIND  = 15.0     # metres behind ego (so ego is not right at bottom)
VIEW_SIDE    = 35.0     # metres each side

# vehicle persistence: keep vehicle alive this many seconds after last detection
MAX_MISSING_SECONDS = 3

# remove vehicle once ego has passed it by this much
EGO_PASSED_THRESHOLD = -2.0  # metres (y < this = ego has passed)

COLOURS = {
    "normal":          (160, 160, 160),
    "yielded":         (0,   210, 0),
    "braked_abruptly": (0,   140, 255),
    "failed_to_yield": (0,   0,   220),
    "unknown":         (180, 180, 0),
}

VEHICLE_SIZES = {
    "car":        (2.0, 4.5),
    "truck":      (2.5, 8.5),
    "bus":        (2.5, 12.0),
    "motorcycle": (1.0, 2.2),
}

# ---- coordinate helpers ----
def world_to_canvas(x_m, y_m):
    total_y = VIEW_AHEAD + VIEW_BEHIND
    total_x = VIEW_SIDE * 2
    px = int(WIDTH  / 2 + (x_m / total_x) * WIDTH)
    py = int(HEIGHT - (VIEW_BEHIND / total_y) * HEIGHT
             - (y_m / total_y) * HEIGHT)
    return px, py

def m_to_px(metres):
    return max(int(metres / (VIEW_SIDE * 2) * WIDTH), 4)

# ---- drawing helpers ----
def draw_road(canvas, lanes, lane_width_m):
    total_w = m_to_px(lanes * lane_width_m)
    left  = WIDTH // 2 - total_w // 2
    right = WIDTH // 2 + total_w // 2

    # road surface
    cv2.rectangle(canvas, (left, 0), (right, HEIGHT), (48, 48, 48), -1)

    # lane markings
    for i in range(lanes + 1):
        x_m = -(lanes * lane_width_m) / 2 + i * lane_width_m
        px, _ = world_to_canvas(x_m, 0)
        if i == 0 or i == lanes:
            # solid edge lines
            cv2.line(canvas, (px, 0), (px, HEIGHT), (100, 100, 100), 2)
        else:
            # dashed centre lane dividers
            for y in range(0, HEIGHT, 35):
                cv2.line(canvas, (px, y), (px, y + 18), (70, 70, 70), 1)

def draw_vehicle(canvas, x_m, y_m, heading_deg, vtype, behaviour,
                 track_id, speed, fwd_ms, lat_ms, age=0):
    cx, cy = world_to_canvas(x_m, y_m)
    if cx < -60 or cx > WIDTH + 60 or cy < -60 or cy > HEIGHT + 60:
        return

    col = COLOURS.get(behaviour, COLOURS["unknown"])

    # fade colour slightly for vehicles kept alive by dead reckoning
    if age > 0:
        col = tuple(max(0, int(c * (1.0 - age * 0.15))) for c in col)

    vw, vl = VEHICLE_SIZES.get(vtype, (2.0, 4.5))
    pw = max(m_to_px(vw), 10)
    pl = max(m_to_px(vl), 16)

    # heading: 0=straight ahead, +right, -left (our new metric convention)
    angle_rad = np.radians(-heading_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    corners = np.array([
        [-pw/2, -pl/2],
        [ pw/2, -pl/2],
        [ pw/2,  pl/2],
        [-pw/2,  pl/2],
    ], dtype=np.float32)

    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (rot @ corners.T).T + np.array([cx, cy])
    pts = rotated.astype(np.int32)

    cv2.fillPoly(canvas, [pts], col)
    cv2.polylines(canvas, [pts], True, (220, 220, 220), 1)

    # direction arrow pointing forward
    ax = int(cx - sin_a * pl // 2)
    ay = int(cy - cos_a * pl // 2)
    cv2.arrowedLine(canvas, (cx, cy), (ax, ay), (255, 255, 255), 1, tipLength=0.4)

    # label above vehicle
    lbl1 = f"#{track_id} {vtype[:3]}"
    lbl2 = f"{speed:.0f}km/h"
    lbl3 = {"failed_to_yield": "no yield", "braked_abruptly": "braked",
             "yielded": "yielded", "normal": ""}.get(behaviour, "")

    y_lbl = cy - pl // 2 - 3
    cv2.putText(canvas, lbl1, (cx - 18, y_lbl - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.29, col, 1)
    cv2.putText(canvas, lbl2, (cx - 18, y_lbl - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, col, 1)
    if lbl3:
        cv2.putText(canvas, lbl3, (cx - 18, y_lbl + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, col, 1)

def draw_ego(canvas):
    cx, cy = world_to_canvas(0, 0)
    # ambulance body
    cv2.rectangle(canvas, (cx - 12, cy - 20), (cx + 12, cy + 20),
                  (0, 110, 255), -1)
    cv2.rectangle(canvas, (cx - 12, cy - 20), (cx + 12, cy + 20),
                  (255, 255, 255), 2)
    # blue light on roof
    cv2.circle(canvas, (cx, cy - 22), 4, (255, 100, 0), -1)
    cv2.putText(canvas, "AMB", (cx - 13, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

def draw_legend(canvas):
    items = [
        ("normal",    COLOURS["normal"]),
        ("yielded",   COLOURS["yielded"]),
        ("braked",    COLOURS["braked_abruptly"]),
        ("no yield",  COLOURS["failed_to_yield"]),
    ]
    x0 = WIDTH - 90
    for i, (label, col) in enumerate(items):
        y = 34 + i * 18
        cv2.rectangle(canvas, (x0, y - 9), (x0 + 12, y + 3), col, -1)
        cv2.putText(canvas, label, (x0 + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)

def draw_hud(canvas, timestamp, emergency, scenario, lanes, n_vehicles):
    # emergency banner
    if emergency:
        cv2.rectangle(canvas, (0, 0), (WIDTH, 24), (0, 0, 160), -1)
        cv2.putText(canvas, "🚨 EMERGENCY ACTIVE — SIREN DETECTED",
                    (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1)

    # bottom info bar
    cv2.rectangle(canvas, (0, HEIGHT - 22), (WIDTH, HEIGHT), (20, 20, 20), -1)
    mins = timestamp // 60
    secs = timestamp % 60
    info = (f"t={timestamp}s ({mins:02d}:{secs:02d})  |  {scenario}"
            f"  |  {lanes} lanes  |  {n_vehicles} vehicles")
    cv2.putText(canvas, info, (8, HEIGHT - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)

# ---- vehicle state manager (persistence + dead reckoning) ----
class VehicleTracker:
    """
    Keeps vehicles alive between JSON frames.
    Uses last known velocity to extrapolate position (dead reckoning).
    Removes vehicles when ego passes them or they've been missing too long.
    """
    def __init__(self):
        self.states = {}  # track_id -> dict of last known state + missing_count

    def update(self, vehicles_json, dt=1.0):
        """
        Update with new detections from one JSON frame.
        Returns list of vehicle states to render (detected + alive predicted).
        """
        detected_ids = set()

        for v in vehicles_json:
            vid = v["id"]
            detected_ids.add(vid)
            self.states[vid] = {
                "x": v.get("x_meters", 0),
                "y": v.get("y_meters", 0),
                "fwd_ms": v.get("forward_speed_ms", 0),
                "lat_ms": v.get("lateral_speed_ms", 0),
                "speed": v.get("speed_kmh", 0),
                "heading": v.get("heading_angle", 0),
                "type": v.get("type", "car"),
                "behaviour": v.get("behaviour", "normal"),
                "id": vid,
                "missing": 0,
            }

        # increment missing counter for undetected vehicles
        to_remove = []
        for vid, state in self.states.items():
            if vid not in detected_ids:
                # dead reckoning: move by last known velocity
                state["x"] += state["lat_ms"] * dt
                state["y"] += state["fwd_ms"] * dt
                state["missing"] += 1

                # remove if passed by ego or missing too long
                if (state["y"] < EGO_PASSED_THRESHOLD or
                        state["missing"] > MAX_MISSING_SECONDS):
                    to_remove.append(vid)

        for vid in to_remove:
            del self.states[vid]

        return list(self.states.values())

    def interpolate(self, alpha):
        """
        Return interpolated states for sub-second rendering.
        alpha: 0.0 = current frame, 1.0 = next frame
        Uses velocity to project forward by alpha seconds.
        """
        result = []
        for vid, state in self.states.items():
            xi = state["x"] + state["lat_ms"] * alpha
            yi = state["y"] + state["fwd_ms"] * alpha
            result.append({**state, "x": xi, "y": yi})
        return result


# ---- main ----
video_dirs = [d for d in os.listdir(OUTPUT_DIR)
              if os.path.isdir(os.path.join(OUTPUT_DIR, d))
              and d not in ['__pycache__']]
if not video_dirs:
    raise SystemExit("No output folder found. Run main.py first.")

video_dir = os.path.join(OUTPUT_DIR, video_dirs[0])
json_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".json")])
if not json_files:
    raise SystemExit("No JSON files found.")

print(f"Loading {len(json_files)} JSON frames...")
frames_data = []
for jf in json_files:
    with open(os.path.join(video_dir, jf)) as f:
        frames_data.append(json.load(f))

total_frames = len(frames_data) * INTERP_STEPS
duration_s   = total_frames / FPS
print(f"Rendering {total_frames} frames at {FPS}fps = {duration_s:.0f}s "
      f"({duration_s/60:.1f} min) — matches original video duration")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

tracker = VehicleTracker()

for i, fd in enumerate(frames_data):
    timestamp = fd.get("timestamp", i)
    emergency = fd.get("emergency_active", False)
    scenario  = fd.get("scenario_type", "unknown")
    vehicles_json = fd.get("vehicles", [])

    lanes      = vehicles_json[0].get("lanes_total", 3) if vehicles_json else 3
    road_type  = vehicles_json[0].get("road_type", "highway") if vehicles_json else "highway"
    lane_width = 3.75 if road_type == "highway" else 3.00

    # update tracker with this frame's detections
    current_states = tracker.update(vehicles_json)

    # render INTERP_STEPS frames between this and the next JSON second
    for step in range(INTERP_STEPS):
        alpha = step / INTERP_STEPS  # 0.0 to 0.9

        # build canvas
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25)

        draw_road(canvas, lanes, lane_width)

        # draw interpolated vehicle states
        interp_states = tracker.interpolate(alpha)
        for state in interp_states:
            draw_vehicle(
                canvas,
                x_m        = state["x"],
                y_m        = state["y"],
                heading_deg= state["heading"],
                vtype      = state["type"],
                behaviour  = state["behaviour"],
                track_id   = state["id"],
                speed      = state["speed"],
                fwd_ms     = state["fwd_ms"],
                lat_ms     = state["lat_ms"],
                age        = state["missing"]
            )

        draw_ego(canvas)
        draw_legend(canvas)
        draw_hud(canvas, timestamp, emergency, scenario, lanes, len(current_states))

        writer.write(canvas)

    if i % 60 == 0:
        print(f"  Rendered t={timestamp}s ({i}/{len(frames_data)})")

writer.release()
print(f"\nDone. Saved to {OUTPUT_FILE}")
print(f"Duration: {duration_s:.0f}s = {duration_s/60:.1f} min at {FPS}fps")
print(f"Matches original video — pause both at same timestamp to compare.")