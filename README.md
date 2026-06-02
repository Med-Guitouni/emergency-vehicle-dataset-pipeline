
# Emergency Vehicle Dataset Pipeline

The pipeline is developed and validated on one single video first.
Once all output JSON files are verified to be correct and complete,
the same pipeline runs on additional videos one by one to confirm
consistency. Only after full validation on a small sample will the
complete batch of 32 videos be downloaded and processed in one run.


---


## Setup

```bash
pip3 install yt-dlp ultralytics opencv-python numpy librosa soundfile onnxruntime torch torchvision
brew install ffmpeg
```

---

## Pipeline Classes

**downloader.py** — downloads videos from a YouTube URL or full channel

**preprocessor.py** — extracts 1 frame per second, removes dashboard hood 
and sky by cropping top 20% and bottom 15% of frame

**detector.py** — YOLOv8m detects cars, trucks, buses, motorcycles per frame 
with confidence threshold 0.25

**tracker.py** — ByteTrack assigns persistent IDs across frames and handles 
occlusions. Currently uses standard Kalman Filter without ego motion compensation. 
See homography.py for the planned EMAP upgrade.

**homography.py** — converts pixel positions to real world meters using IPM/BEV 
projection. Computes speed, acceleration, jerk, heading, lane ID, lateral offset 
and distance to ego per vehicle per frame. Currently assumes 3 equal lanes. 
See class docstring for full list of limitations and the ego motion compensation plan.

**annotator.py** — labels each vehicle's behaviour per frame using 5 heuristic 
rules. Labels: normal, yielded, braked_abruptly, failed_to_yield. Only active 
when emergency is detected. Rules 1 to 4 use lateral offset and speed. Rule 5 
uses heading angle change over 3 consecutive frames, added based on Qiu et al. 
2025 lane change detection review.

**emergency_detector.py** — detects when the emergency run is active using 
FFT siren detection on the audio track. Blue light HSV detection is enabled 
at night only to avoid sky false positives. Emergency is confirmed by majority 
vote over 10 frames.

**scene_classifier.py** — classifies road type per frame using MIT Places365 
ResNet18. Labels: highway, urban, intersection, tunnel, unknown. Uses consecutive 
frame confirmation over 3 frames to prevent single bad frame flips. Loaded once 
globally and reset per video.

**lane_detector.py** — INCOMPLETE. Attempted UFLD v2 and YOLOP for lane 
boundary detection. Both fail when vehicles on the shoulder cover lane markings. 
 Connected to nothing yet

**exporter.py** — saves one JSON file per second per video to output/video_name/

**main.py** — 

---

## Usage

**Step 1 - Download one video**
```bash
python3 -c "from downloader import VideoDownloader; VideoDownloader().download_single('URL')"
```

**Step 2 - Run pipeline**
```bash
python3 main.py
```

**Step 3 - Find output**
```
output/
└── video_name/
    ├── t0000.json
    ├── t0001.json
    └── ...
```

---

## JSON Output Format

**Scene level — one per second**
```json
{
  "timestamp": 149,
  "video_source": "video_name",
  "emergency_active": true,
  "emergency_triggered_by": ["siren"],
  "scenario_type": "highway"
}
```

**Vehicle level — one per detected vehicle per second**
```json
{
  "id": "315",
  "type": "car",
  "x_meters": 4.75,
  "y_meters": 12.3,
  "speed_kmh": 0.2,
  "acceleration": -1.2,
  "jerk": 0.3,
  "heading_angle": 5.2,
  "lane_id": 1,
  "lateral_offset": 0.3,
  "distance_to_ego": 12.4,
  "is_emergency": false,
  "behaviour": "yielded",
  "bbox": [259, 140, 282, 160]
}
```

---

## Known Limitations

**Speed values are relative not absolute.** All speeds are measured relative 
to the ambulance camera. Ego motion compensation is planned using Depth Anything 
V2 and EMAP (Mahdian et al. 2024) but not yet implemented. See homography.py 
for the full plan.

**Lane boundaries are approximate.** Lane ID and lateral offset use an equal 
thirds assumption across the frame width. Actual lane marking detection failed 
because yielding vehicles cover the painted lines. See lane_detector.py.

**Camera intrinsics are estimated.** Focal length and mount height are 
reasonable defaults not measured values. Needs calibration from actual dashcam 
specs for accuracy.

**Siren threshold tuned on one video.** The FFT threshold of 0.35 was tuned 
on the test video. May need adjustment per video especially for city driving 
with background noise.

**Failed to yield label has false positives.** Any vehicle within 20 meters 
that does not trigger a yielding rule gets this label. Vehicles legitimately 
in front of the ambulance that cannot move will be mislabelled.

---

## Other  Limitation

Vehicle behaviour is measured as data only and is never used to trigger 
emergency_active status. Using behaviour as a trigger would create a circular 
dependency.

---







