#  Pipeline Emergency Vehicle Behaviour Dataset 
not updated 
### 

---



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
 After cloning, open `EMAP/trackers/bytetrack/kalman_filter.py` and delete these four lines — they are ROS dependencies not needed here:
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
python3 validate_nuscenes_phaseA.py
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
  "timestamp":              ,   
  "video_source":           ,   
  "emergency_active":       ,   // true once emergency_start_second is reached (from video_lanes.json)
  "scenario_type":          ,   // scene classifier output: highway / urban / intersection / roundabout
  "vehicles": [
    {
      "id":                 ,   //  ID assigned by ByteTrack 

      "type":               ,   // car / truck / bus / motorcycle — from YOLO class

      "x_meters":           ,   // lateral position in metres. + = right of ambulance centre, - = left.
                                // derived from ground-plane pinhole projection (camera height + focal length)

      "y_meters":           ,   // forward distance in metres from the ambulance.
                             

      "position_reliable":  ,   // false when the bounding box is clipped at a frame edge
                                // (vehicle partially outside camera view — tyres not visible,
                                // ground-plane projection unreliable). 

      "speed_kmh":          ,   // overall speed magnitude in km/h — combines forward and lateral motion.
                                // this is RELATIVE to the ambulance, not absolute ground speed.
                                // if ambulance and vehicle travel at same speed, this reads ~0.
                                // (Krajewski et al. 2018 — highD dataset)

      "forward_speed_ms":   ,   // speed component ALONG the road in m/s.
                                //positive moving away from ambulance.//negative  ambulance catching up.
                                // different from speed_kmh which is the total magnitude regardless of direction.
                                // a braking vehicle slows this value. a vehicle being overtaken has negative value.
                                // (Krajewski et al. 2018 — highD dataset, longitudinal/lateral split)

      "lateral_speed_ms":   ,   // speed component ACROSS the road in m/s.
                                // + = moving right. - = moving left.
                                // this is the direct yielding signal: a vehicle pulling aside
                                // shows a clear lateral_speed_ms even when forward_speed_ms is near zero.
                                // (Pierson et al. 2019 — highD analysis; Krajewski et al. 2018)

      "acceleration":       ,   // change in speed_kmh per second, converted to m/s².
                                // negative = decelerating (braking). large negative = hard brake.
                                // threshold -2.5 m/s² used for braked_abruptly label.
                                

      "jerk":               ,   // change in acceleration per second (m/s³).
                                // different from acceleration: acceleration tells you HOW FAST the vehicle
                                // is changing speed; jerk tells you HOW SUDDENLY that change happened.
                                // high jerk = panic stop (acceleration changed very abruptly).
                                // low jerk = smooth braking even if deceleration is large.
                                // (INTERACTION dataset — Zhan et al. 2019)

      "lane_id":            , 

      "lateral_offset":     ,   // distance from the centre of the vehicle's lane in metres.
                                // + = right of lane centre. - = left of lane centre.
                                // used in annotator Rule 3

      "distance_to_ego":    ,   // straight-line distance to the ambulance in metres.
                                // sqrt(x_meters² + y_meters²). used for proximity thresholds in annotator.

      "lanes_total":        ,   // total number of lanes on this road at this timestamp.
                                // manually annotated in video_lanes.json. 

      "road_type":          ,   // highway / urban / intersection / roundabout.
                                // from video_lanes.json (ground truth) or scene classifier fallback.

      "lane_source":        ,   // "config" = from video_lanes.json (ground truth).
                                // "scene_classifier" = automatic fallback (video not yet annotated).
                                

      "preceding_id":       ,   
      "following_id":       ,   
      "left_preceding_id":  ,   // all six computed from x_meters / y_meters. same-lane threshold = half lane width.
      "left_following_id":   ,  // (Krajewski et al. 2018 — highD surrounding vehicle IDs)
       "right_preceding_id"   
        "right_following_id"
         
                                

      "behaviour":          ,   // normal / yielded / braked_abruptly / failed_to_yield.
                                // only assigned when emergency_active = true and vehicle within 50m.
                                // rules based on Pierson et al. 2019, Qiu et al. 2025, braking literature.
                                // see annotator.py for full rule documentation.

      "bbox":               ,   // bounding box [x1, y1, x2, y2] in pixels of the cropped frame.
                                // cropped frame = original with top 20% and bottom 15% removed.
    }
  ]
}
```

## Note for Validation Against other Datasets
The pipeline was calibrated for one specific dashcam. Before validating the measured values against a dataset that
has ground truth, several hard-coded assumptions must be re-calibrated to the new 
camera (distributional shift) . The most important is the camera geometry in homography.py: 
camera_height (currently 1.4m, the ambulance mount height) and focal_length_factor (currently 0.8 × frame width, an estimate)
must be replaced with the real values of the validation dataset's camera.
The horizon_ratio (currently 0.55) must also be re-measured for the new camera because
it encodes the camera's pitch angle (point it at a frame from the new dataset and move the line until
it sits on the true vanishing point of the road).Second, the spatial crop in preprocessor.py 
(top 20%, bottom 15%) is tuned to where the ambulance dashboard and sky 
sit in our frames; a different camera needs different crop fractions or the horizon
geometry breaks. Third, the lane widths in lane_config.py (3.75m highway, 3.00m urban) follow 
the German standard thats why a dataset recorded in another country needs its
national lane-width values, since lane assignment and lateral offset depend on them.
Fourth, the behaviour thresholds in annotator.py
(lateral speed 0.5 m/s, heading 15°, braking −2.5 m/s²) were drawn from 
highD German highway data; they are reasonable defaults but should be re-checked 
if the validation dataset's driving context differs.
Finally, all speeds remain relative to the ego vehicle: if the ground-truth dataset 
reports absolute speeds, the ego vehicle's own speed must be added back before
comparison.
--fix the camera intrinsics and horizon, --confirm distance_to_ego matches ground-truth
## Position Reliability

Every vehicle observation in the JSON includes a `position_reliable` field (true or false).
This flag tells the analysis whether the ground-plane position (x,y) for that vehicle at that second can be trusted.
The position is computed by projecting the bottom of the YOLO bounding box onto the road plane using camera geometry. This works correctly as long as the vehicle's tyres are visible in the frame
 the formula only uses the bottom row of the box, so a vehicle whose roof is cut off at the top of the frame is still measured correctly. 
However, three cases break the assumption: the vehicle's bottom edge is cut off by the crop (tyres not visible), 
the vehicle's sides are clipped (the lateral centre of the visible box is not the centre of the vehicle), 
or the computed lateral position exceeds the physical road boundary and the safety clamp fires. 
All three are flagged position_reliable: false.

The dominant cause in this dataset is large trucks driving directly beside the ambulance during the emergency run 
at 5–10m distance a truck fills most of the camera frame and its sides clip the edges. 
This is a physical constraint of a single fixed camera.

On the test video (930 seconds, 3847 total vehicle-frame observations),
the initial implementation flagged 26.3% of observations as unreliable,
with 77.5% of close-range (0–10m) observations affected. 
After correcting an over-strict rule that was wrongly penalising top-clipped boxes
(whose roof is out of frame but whose tyres are fully visible and whose position
formula is unaffected), the rate dropped to 13.8% overall and 39.3% at 0–10m.
The remaining unreliable observations are genuinely problematic measurements 
the RTS smoother already handles them by assigning them 25× lower measurement weight
so the physics model dominates instead of the bad measurement.

For analysis, filter on position_reliable: true before computing any spatial statistics. 
The unreliable rows are kept in the dataset rather than deleted. 
->RUN count_reliability.py for stats ( here is ex output )

  Total observations : 3847
  Reliable           : 3315  (86.2%)
  Unreliable         : 532  (13.8%)

  Summary: 532/3847 observations flagged unreliable

--- By vehicle type ---
  bus             16/101    ( 15.8%)  ███
  car            121/1581   (  7.7%)  █
  truck          395/2165   ( 18.2%)  ███

--- By distance to ego ---
  0-10m   (very close)    456/1159   ( 39.3%)  ███████
  10-20m  (close)          66/1175   (  5.6%)  █
  20-40m  (mid)            10/1494   (  0.7%)  
  40-80m  (far)             0/19     (  0.0%)  
  80m+    (distant)         0/0      (  0.0%)  



## Known Limitations

to be written
---









