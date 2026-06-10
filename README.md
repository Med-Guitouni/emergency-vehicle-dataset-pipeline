#  Pipeline Emergency Vehicle Behaviour Dataset 
### 

---

## What This Does

Takes a dashcam video recorded from inside an
ambulance during an emergency run and produces one JSON file per second, 
capturing every surrounding vehicle's position, speed, behaviour, and spatial 
relationships 

---

## Setup

```bash
pip3 install yt-dlp ultralytics opencv-python numpy librosa torch torchvision
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
⚠️ After cloning, open `EMAP/trackers/bytetrack/kalman_filter.py` and delete these four lines — they are ROS dependencies not needed here:
```python
import rospy
from std_msgs.msg import Float32MultiArray          # delete these two at the top
params_array = Float32MultiArray()                  # delete these two inside __init__
params_array.data = [self._q1, self._q4, ...]
```
Test: `python3 -c "from EMAP.trackers.bytetrack.kalman_filter import KalmanFilter; print('OK')"`

To make paths permanent: `echo 'export PYTHONPATH=...' >> ~/.zshrc && source ~/.zshrc`

---

## Run

```bash
# Download a video
python3 -c "from downloader import VideoDownloader; VideoDownloader().download_single('URL')"

# Run the full pipeline
python3 main.py

# Generate 50 annotated validation frames
python3 generate_validation_frames.py
```

Output: `output/video_name/t0000.json … t0929.json`

---

## Pipeline

| File                    |  | What it does                                                            |
|-------------------------|--|-------------------------------------------------------------------------|
| `downloader.py`         |  | Downloads videos via yt-dlp                                             |
| `preprocessor.py`       |  | Extracts 1 frame/sec, crops top 20% and bottom 15%            |
| `detector.py`           |  | YOLOv8x detects cars, trucks, buses, motorcycles (conf ≥ 0.25)          |
| `tracker.py`            |  | ByteTrack + EMAP-enhanced Kalman Filter for stable IDs                  |
| `homography.py`         |  | Ground-plane pinhole projection → real-world metres, split velocity, ego motion |
| `annotator.py`          |  | Labels vehicle behaviour per frame (5 rules, literature-backed)         |
| `emergency_detector.py` |  | FFT siren detection + blue light (night only), latches once confirmed   |
| `scene_classifier.py`   |  | MIT Places365 ResNet18 → highway / urban / intersection / roundabout    |
| `lane_config.py`        |  | Manual ground-truth lane count per video (see `video_lanes.json`)       |
| `surrounding.py`        |  | highD-style 6-neighbour IDs from metric positions                       |
| `exporter.py`           |  | Writes one JSON per second per video                                    |
| `lane_detector.py`      |  | UFLD v2 + YOLOP attempted, both fail under occlusion.   |

---

## JSON Output

```json
{
  "timestamp": 149,
  "video_source": "video_name",
  "emergency_active": true,
  "emergency_triggered_by": ["siren"],
  "scenario_type": "highway",
  "vehicles": [
    {
      "id": 315,
      "type": "car",
      "x_meters": 4.75,
      "y_meters": 24.3,
      "speed_kmh": 3.2,
      "forward_speed_ms": 0.8,
      "lateral_speed_ms": -1.4,
      "acceleration": -0.3,
      "jerk": 0.1,
      "heading_angle": 5.2,
      "lane_id": 2,
      "lateral_offset": 0.3,
      "distance_to_ego": 24.7,
      "lanes_total": 3,
      "road_type": "highway",
      "lane_source": "config",
      "preceding_id": 287,
      "following_id": 301,
      "left_preceding_id": 412,
      "left_following_id": null,
      "right_preceding_id": null,
      "right_following_id": 198,
      "behaviour": "yielded",
      "bbox": [259, 140, 382, 260]
    }
  ]
}
```




## Known Limitations

- **Speeds are relative**, not absolute. The ambulance's own speed is not subtracted. EMAP improves tracking stability but does not give absolute metric ego-speed from an uncalibrated camera.
- **Camera intrinsics are estimated.** Focal length and mount height are reasonable defaults. Calibrate using Autobahn lane dash method (6m line + 12m gap = 18m period) documented in `validate_distances.py`.
- **Lane boundaries are approximate.** Lane ID uses equal-division of frame width. `surrounding.py` uses metric x_meters directly which is more reliable.
- **1Hz sampling.** Fast manoeuvres completed between frames are missed.
- **`failed_to_yield` has false positives** for vehicles legitimately in front of the ambulance.

---









