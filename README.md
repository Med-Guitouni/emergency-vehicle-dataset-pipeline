# Emergency Vehicle Behaviour Dataset Pipeline

A Python pipeline that processes YouTube dashcam videos from inside German ambulances on emergency runs and produces one JSON file per second describing every nearby vehicle's position, speed, and behaviour. 

---

## Setup

```bash
pip3 install yt-dlp ultralytics opencv-python numpy torch torchvision
brew install ffmpeg
```

**Depth Anything V2** (for EMAP ego-motion compensation):
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
export PYTHONPATH=$PYTHONPATH:~/Desktop/einsatz_pipeline/Depth-Anything-V2
```
Download weights (~98MB) into `models/`: [depth_anything_v2_vits.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true)

**EMAP** (ego-motion aware Kalman Filter):
```bash
git clone https://github.com/noyzzz/EMAP
export PYTHONPATH=$PYTHONPATH:~/Desktop/einsatz_pipeline/EMAP
```
After cloning, open `EMAP/trackers/bytetrack/kalman_filter.py` and delete these four lines (ROS dependencies not needed here):
```python
import rospy
from std_msgs.msg import Float32MultiArray          # delete at top
params_array = Float32MultiArray()                  # delete inside __init__
params_array.data = [self._q1, self._q4, ...]       # delete inside __init__
```
Test: `python3 -c "from EMAP.trackers.bytetrack.kalman_filter import KalmanFilter; print('OK')"`

To make paths permanent:
```bash
echo 'export PYTHONPATH=$PYTHONPATH:~/Desktop/einsatz_pipeline/EMAP' >> ~/.zshrc
echo 'export PYTHONPATH=$PYTHONPATH:~/Desktop/einsatz_pipeline/Depth-Anything-V2' >> ~/.zshrc
source ~/.zshrc
```

---

## Run

```bash
# Download a video
python3 -c "from downloader import VideoDownloader; VideoDownloader().download_single('URL')"

# Run the full pipeline
python3 main.py
```

Output: `output/video_name/t0000.json … t0929.json`

Before running a new video, add it to `video_lanes.json` (see Lane Config below).

---

## Pipeline Architecture

The pipeline runs in two phases with RTS smoothing between them.

**Phase 1 — Collect** (streaming, 1 frame/second):
- Crop sky (top 20%) and dashboard (bottom 15%)
- YOLO detects vehicles; ByteTrack + EMAP assigns stable IDs
- Ground-plane pinhole projection converts each vehicle's pixel position to real-world metres
- Scene classifier and lane config determine road type and emergency state
- Raw positions stored per frame (no metrics yet)

**Between Phases — RTS Smoothing**:
- Each vehicle's full trajectory is smoothed with a Rauch-Tung-Striebel smoother
- Removes detection jitter and pixel quantisation noise before any metric is derived
- Unreliable observations (clipped boxes) are down-weighted so the motion model dominates
- Same post-processing used by highD (Krajewski et al. 2018) and INTERACTION (Zhan et al. 2019)

**Phase 2 — Compute + Export**:
- Split velocity (forward + lateral), acceleration, jerk, heading derived from smoothed positions
- Lane assignment, surrounding vehicle IDs, behaviour labels
- One JSON file per second written to `output/video_name/`

---

## Pipeline Files

| File | What it does |
|---|---|
| `main.py` | Orchestrates both phases and RTS smoothing |
| `downloader.py` | Downloads videos via yt-dlp at best available quality |
| `preprocessor.py` | Extracts 1 frame/sec, crops top 20% and bottom 15% |
| `detector.py` | YOLOv8x detects cars, trucks, buses, motorcycles (conf ≥ 0.25) |
| `tracker.py` | ByteTrack + EMAP-enhanced Kalman Filter for stable IDs across frames |
| `homography.py` | Ground-plane pinhole projection → real-world metres, split velocity, acceleration, jerk, heading, ego motion |
| `smoother.py` | RTS smoother — runs between Phase 1 and Phase 2 |
| `annotator.py` | Labels vehicle behaviour per frame using 5 kinematic rules |
| `scene_classifier.py` | MIT Places365 ResNet18 → highway / urban / intersection / roundabout |
| `lane_config.py` | Reads `video_lanes.json` — manual ground-truth lane count and emergency timing |
| `surrounding.py` | highD-style 6-neighbour IDs from metric positions |
| `exporter.py` | Writes one JSON per second per video |

---

## Calibration Constants

These are set in `homography.py` and measured from the test video using Autobahn lane dash spacing (18m period) and lane width (3.75m) as physical rulers:

| Constant | Value | Notes |
|---|---|---|
| `camera_height` | 1.4 m | Confirmed — mean 1.40m across 6 frames |
| `focal_length_factor` | 0.72 | focal_px = frame_width × 0.72 |
| `horizon_ratio` | 0.60 | Fraction down the cropped frame where horizon sits |
| `CX_RATIO` | 0.47 | Optical centre column  |



---

## Lane Config (`video_lanes.json`)

Each video needs a manual entry before processing. This is the only manual input the pipeline requires.

```json
{
  "video_name": {
    "emergency_start_second": 0,
    "lanes": [
      {
        "from_second": 0,
        "to_second": 930,
        "lanes": 3,
        "road_type": "highway",
        "notes": "A2 Autobahn"
      }
    ]
  }
}
```

`video_name` must match `os.path.splitext(os.path.basename(path))[0][:30]`.

Lane widths are automatic from road type (RASt 06): highway → 3.75m, urban → 3.00m.

Emergency is latched: once `emergency_start_second` is reached it stays active for the rest of the video.

---

## JSON Output

```json
{
  "timestamp":        ,   // seconds from video start
  "video_source":     ,   // video filename (truncated to 30 chars)
  "emergency_active": ,   // true once emergency_start_second is reached
  "scenario_type":    ,   // highway / urban / intersection / roundabout (scene classifier)
  "vehicles": [
    {
      "id":                  ,  // stable ID assigned by ByteTrack
      "type":                ,  // car / truck / bus / motorcycle
      "x_meters":            ,  // lateral position in metres. + = right of ambulance, - = left
      "y_meters":            ,  // forward distance in metres from ambulance
      "position_reliable":   ,  // false when bbox is clipped at frame edge (see below)
      "speed_kmh":           ,  // overall speed magnitude in km/h — RELATIVE to ambulance
      "forward_speed_ms":    ,  // speed along road in m/s. + = moving away, - = ego catching up
      "lateral_speed_ms":    ,  // speed across road in m/s. + = right, - = left. Direct yielding signal
      "acceleration":        ,  // change in forward_speed_ms per second (m/s²). Negative = braking
      "jerk":                ,  // change in acceleration per second (m/s³). High = panic stop
      "lane_id":             ,  // lane number 1 (left) to N (right)
      "lateral_offset":      ,  // distance from lane centre in metres. + = right of centre
      "distance_to_ego":     ,  // straight-line distance to ambulance: sqrt(x² + y²)
      "lanes_total":         ,  // total lanes on this road at this timestamp
      "road_type":           ,  // highway / urban / intersection / roundabout
      "lane_source":         ,  // "config" = from video_lanes.json, "scene_classifier" = fallback
      "preceding_id":        ,  // ID of vehicle directly ahead in same lane
      "following_id":        ,  // ID of vehicle directly behind in same lane
      "left_preceding_id":   ,  // ID of vehicle ahead-left
      "left_following_id":   ,  // ID of vehicle behind-left
      "right_preceding_id":  ,  // ID of vehicle ahead-right
      "right_following_id":  ,  // ID of vehicle behind-right
      "behaviour":           ,  // normal / yielded / braked_abruptly / failed_to_yield
      "bbox":                   // [x1, y1, x2, y2] pixels in the cropped frame
    }
  ]
}
```



---

## Position Reliability

Every observation includes `position_reliable` (true/false). The ground-plane projection uses only the bottom row of the bounding box (where tyres meet road). Three cases break this assumption and are flagged unreliable:

- **Bottom clipped**: tyres below the crop — projection input missing
- **Side clipped**: vehicle half out of frame — lateral centre of visible box is not vehicle centre
- **Lateral clamp fired**: computed position exceeds physical road boundary

Top-clipped boxes (roof out of frame, tyres visible) are **reliable** — the formula only uses the bottom row.

On the test video (930 seconds, 3847 observations):

```
Total observations : 3847
Reliable           : 3315  (86.2%)
Unreliable         : 532   (13.8%)

By distance to ego:
  0–10m    456/1159  (39.3%)  — mainly trucks beside ambulance
  10–20m    66/1175  ( 5.6%)
  20–40m    10/1494  ( 0.7%)
  40–80m     0/19    ( 0.0%)
```

Unreliable rows are kept in the dataset (not deleted). The RTS smoother assigns them 25× lower measurement weight so the motion model interpolates instead.

When a box is clipped and the position is unreliable, the smoother ignores that measurement and fills in the gap using the vehicle's trajectory before and after that frame.

---

## Behaviour Labels

Labels are assigned by `annotator.py` only when `emergency_active = true` and vehicle is within 50m.

| Label | Condition |
|---|---|
| `yielded` | Lateral speed ≥ 0.5 m/s sustained for ≥ 2 consecutive frames, OR heading ≥ 15°, OR cumulative lateral drift ≥ 0.8m over 3s monotonically |
| `braked_abruptly` | Acceleration ≤ −2.5 m/s², OR acceleration ≤ −1.5 m/s² AND jerk ≤ −3.0 m/s³ |
| `failed_to_yield` | Within 20m, ≥ 3 frames of history, nothing else triggered |
| `normal` | None of the above |

Thresholds: Pierson et al. 2019 (lateral speed), Qiu et al. 2025 (heading), Krajewski et al. 2018 (cumulative window), braking literature (acceleration).

---

## Known Limitations

**Speeds are relative only.** All velocities are relative to the ambulance.

**EMAP ego-motion compensation is partial.** EMAP's depth signal (from Depth Anything V2) is relative, not metric. Compensation is approximate. If EMAP is not installed the pipeline falls back to standard ByteTrack silently.









