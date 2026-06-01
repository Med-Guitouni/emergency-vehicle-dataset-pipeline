# Emergency Vehicle Dataset Pipeline

The pipeline is developed and validated on one single video first.
Once all output JSON files are verified to be correct and complete,
the same pipeline runs on additional videos one by one to confirm
consistency. Only after full validation on a small sample will the
complete batch of 32 videos be downloaded and processed in one run.
---

## Setup

pip3 install yt-dlp ultralytics opencv-python numpy deep-sort-realtime librosa soundfile
brew install ffmpeg

---

## Pipeline Classes

VidDownloader        - downloads videos from YouTube channel or single URL
Preprocessor         - extracts 1 frame per second, removes dashboard and sky region
VehicleDetector      - YOLOv8m detects cars, trucks, buses, motorcycles per frame
VehicleTracker       - ByteTrack assigns persistent IDs across frames, handles occlusions
Homography           - IPM/BEV projection to real world meters, computes all motion metrics
Heuristicannotator   - labels vehicle behaviour per frame using rule based detection
EmergencyDetector    - detects when emergency run is active using siren and blue light
JSONExporter         - saves one JSON file per second per video

---

## Usage

Step 1 - Download video

Single video:
python3 -c "from downloader import VideoDownloader; VideoDownloader().download_single('URL')"
I am using this for now https://www.youtube.com/watch?v=d57wHsiTS0E&t=60s

Full channel: (for later)
python3 -c "from downloader import VideoDownloader; VideoDownloader().download_channel('URL', max_videos=32)"

Step 2 - Run pipeline:
python3 main.py

Step 3 - Find output:
output/
└── video_name/
    ├── t0000.json
    ├── t0001.json
    └── ...

---

## Output 



---



## What Needs Heavy Work
- Speed is relative to ambulance not absolute - ego motion compensation not yet done
- Camera height and focal length are estimated not measured - needs calibration
- Lateral offset based on estimated lane center not actual road markings
- Vehicles beyond 80m sometimes missed - too small to detect reliably
- Siren threshold tuned on one video only - may need adjustment per video

---

## Known Scientific Limitation

Vehicle behaviour is measured as data only and is never used to trigger
emergency active status. Using behaviour as a trigger would create circular
dependency since behaviour differences between normal and emergency scenarios
are the core subject of analysis in this thesis.

---





