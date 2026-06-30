# Emergency Vehicle Behaviour Dataset Pipeline

A Python pipeline that processes YouTube dashcam videos from inside German
ambulances on emergency runs and produces one JSON file per second
describing every nearby vehicle's position, speed, and behaviour.

---

## Setup

```bash
pip3 install -U yt-dlp ultralytics opencv-python numpy torch torchvision
brew install ffmpeg
```

Confirm Ultralytics is up to date — BoT-SORT's `model: auto` field in
`botsort.yaml` requires a recent version:

```bash
pip install -U ultralytics
```

That's it. No EMAP, no Depth Anything V2, no extra repos to clone. Tracking
is handled entirely by BoT-SORT, which ships with Ultralytics.

---

## Quick Start

```bash
# 1. Download a video (full video)
python3 -c "from downloader import VideoDownloader; VideoDownloader().download_single('URL')"

# OR download only a section of a video (saves bandwidth/disk):
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best" \
  --merge-output-format mp4 \
  --download-sections "*START-END" \
  -o "videos/your_video_name.%(ext)s" \
  "URL"
# example: --download-sections "*04:30-09:00"

# 2. Add the video to video_lanes.json (see Lane Config below) — required
#    before processing, otherwise lane count falls back to a default guess
#    and emergency state is never active.

# 3. Run the pipeline
 python3 main.py
```



**Resuming**: if you stop the run (Ctrl+C, or quitting the review window),
just rerun `caffeinate python3 main.py`. Any video that already has output
in `output/<video_name>/` is skipped automatically — only unprocessed
videos run. If a run was killed mid-processing (not mid-review) for a video,
its `output/` folder won't exist yet, so it correctly reprocesses from
scratch on the next run.

Output: `output/<video_name>/t0000.json … t<N>.json`, one file per exported
second.

---

## What Happens When You Run It

For each video found in `videos/`, in order:

1. **Phase 1 — track & collect.** Frames are read at 5 Hz (5 times per
   second) and fed through YOLOv8x + BoT-SORT for detection and tracking.
   Every 5 Hz observation is converted to a real-world (x, y) position and
   stored. Once per whole second, the cropped frame is also saved to
   `review_data/` for later use in the review step.
2. **RTS smoothing.** Once Phase 1 finishes the whole video, each vehicle's
   complete trajectory (all its 5 Hz observations) is smoothed in one pass.
3. **Phase 2 — compute & export.** Using the smoothed, whole-second
   positions: speed, acceleration, jerk, time-to-collision, lane, and the
   six surrounding-vehicle IDs are computed. Behaviour is labelled. One JSON
   per second is written to `output/<video_name>/`.
4. **Review window opens automatically.** See "Manual Review" below. You go
   through every exported second, optionally correcting labels or deleting
   bad boxes, then it moves to the next video in `videos/`.



---

## Manual Review

After each video's JSON is written, an OpenCV window opens automatically
showing every exported second with the pipeline's boxes and labels drawn on
it, so you can correct mistakes by hand before moving to the next video.

**Controls:**

| Key | Action |
|---|---|
| `ENTER` / `SPACE` | Next frame |
| `B` / Left arrow | Previous frame |
| Click a box | Select it (turns yellow) |
| `D` | Delete the selected box |
| `Y` | Set selected vehicle's behaviour to `yielded` |
| `F` | Set selected vehicle's behaviour to `failed_to_yield` |
| Type digits, then `ENTER`/`B`/`Q` | Change the selected vehicle's ID |
| `ESC` | Deselect |
| `Q` | Quit review for this video (all changes already saved) — moves to next video |

**Colours:** green = yielded, red = failed_to_yield, orange = braked_abruptly,
white = normal, yellow = currently selected.

Every change is saved directly back to the JSON file immediately — there's
no separate "save" step, and no undo. Quitting (`Q`) does not discard
anything; it just closes the window for that video and lets `main.py`
continue to the next one (or finish, if it was the last video).

To re-review a video later without reprocessing it:

```bash
python3 review.py --video video_name
```

(omit `--video` to review the first video found in `output/`)

---

## Pipeline Architecture

```
preprocessor.py  →  detector.py + tracker.py  →  homography.py  →  smoother.py  →  homography.py (again) + surrounding.py + annotator.py  →  exporter.py  →  review.py
   (5 Hz frames)        (YOLO + BoT-SORT)        (pixel → metres)    (RTS smooth)         (metrics, lanes, behaviour)                    (JSON)        (manual QA)
```

| File | What it does |
|---|---|
| `main.py` | Orchestrates Phase 1, smoothing, Phase 2, export, and review for every video |
| `downloader.py` | Downloads videos via yt-dlp at best available quality |
| `preprocessor.py` | Streams frames at a given Hz, crops sky/dashboard, resizes to 1280×720 |
| `detector.py` | Loads YOLOv8x (conf ≥ 0.25 for cars, trucks, buses, motorcycles) |
| `tracker.py` | BoT-SORT (appearance ReID + camera motion compensation) for stable IDs; also detects vehicles partially out of frame on the left/right edges |
| `homography.py` | Ground-plane pinhole projection → metres, split velocity, acceleration, jerk, time-to-collision, lane assignment |
| `smoother.py` | RTS smoother — runs once per video between Phase 1 and Phase 2 |
| `annotator.py` | Labels vehicle behaviour using kinematic rules |
| `scene_classifier.py` | MIT Places365 ResNet18 → highway / urban / intersection / roundabout (fallback only) |
| `lane_config.py` | Reads `video_lanes.json` — manual ground-truth lane count and emergency timing |
| `surrounding.py` | highD-style 6-neighbour IDs from metric positions |
| `exporter.py` | Writes one JSON per second per video |
| `review.py` | Manual QA window — corrects behaviour labels and deletes bad boxes after each video |
| `count_reliability.py` | Standalone report on `position_reliable` flag accuracy across all output |
| `visualize_pipeline.py` | Standalone — renders every second as an annotated debug frame (separate from `review.py`) |
| `test_phase1.py` | Standalone — live tracking-only test on the first 60s of a video, prints unique ID count as a tracking-quality metric |

**Tracker config** (`botsort.yaml`): tuned for 1–5 Hz dashcam footage.
`match_thresh: 0.75` (lenient IoU matching to survive large frame-to-frame
box movement), `with_reid: True` (appearance matching — the main fix for ID
churn), `gmc_method: none` (camera motion compensation is disabled due to a
recurring OpenCV pyramid-size assertion error on this setup; ReID alone
handles tracking quality well).

---

## Calibration Constants

Set in `homography.py`, measured from the test video using Autobahn lane
dash spacing (18 m period) and lane width (3.75 m) as physical rulers:

| Constant | Value | Notes |
|---|---|---|
| `camera_height` | 1.4 m | Mean 1.40 m across 6 frames |
| `focal_length_factor` | 0.72 | focal_px = frame_width × 0.72 |
| `horizon_ratio` | 0.60 | Fraction down the cropped frame where horizon sits |
| `CX_RATIO` | 0.47 | Optical centre column as fraction of frame width |



---

## Lane Config (`video_lanes.json`)

Every video needs a manual entry before processing. This is the only
required manual input.

```json
{
  "20240720_einsatz_1080p": {
    "emergency_start_second": 0,
    "lanes": [
      {
        "from_second": 0,
        "to_second": 930,
        "lanes": 3,
        "road_type": "highway",
        "notes": "A2 Autobahn, 3 lanes"
      }
    ]
  },
  "video1": {
    "emergency_start_second": 0,
    "lanes": [
      {
        "from_second": 0,
        "to_second": 270,
        "lanes": 3,
        "road_type": "highway",
        "notes": "highway, 3 lanes, full clip"
      }
    ]
  }
}
```

All videos live in **one** JSON object — add a new top-level key per video,
don't create separate files or separate `{ }` blocks.

`video_name` (the key) must exactly match
`os.path.splitext(os.path.basename(video_path))[0][:30]` — for
`videos/video1.mp4` that's `"video1"`.

Lane widths are automatic from road type (RASt 06): highway → 3.75 m,
urban → 3.00 m — don't write widths yourself.

Emergency is latched: once `emergency_start_second` is reached, it stays
active for the rest of that video's timeline.

If a video is downloaded as a trimmed section (via `--download-sections`),
its timeline starts at 0 regardless of where the clip began in the original
video — `from_second`/`to_second`/`emergency_start_second` should all be
relative to the trimmed clip, not the original video.

---

## JSON Output

```json
{
  "timestamp":        ,   // seconds from this video's start
  "video_source":     ,   // video filename (truncated to 30 chars)
  "emergency_active": ,   // true once emergency_start_second is reached
  "scenario_type":    ,   // highway / urban / intersection / roundabout (scene classifier)
  "vehicles": [
    {
      "id":                  ,  // stable ID assigned by BoT-SORT
      "type":                ,  // car / truck / bus / motorcycle
      "x_meters":            ,  // lateral position in metres. + = right of ambulance, - = left
      "y_meters":            ,  // forward distance in metres from ambulance
      "position_reliable":   ,  // false when bbox is clipped at frame edge (see below)
      "speed_kmh":           ,  // overall speed magnitude in km/h — RELATIVE to ambulance
      "forward_speed_ms":    ,  // speed along road in m/s. + = moving away, - = ego catching up
      "lateral_speed_ms":    ,  // speed across road in m/s. + = right, - = left. Direct yielding signal
      "acceleration":        ,  // change in forward_speed_ms per second (m/s²). Negative = braking
      "jerk":                ,  // change in acceleration per second (m/s³). High = panic stop
      "ttc_to_ego":          ,  // seconds until this vehicle reaches the ambulance, if ahead and closing. null otherwise
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

Heading is not computed or exported — earlier attempts at a 1 Hz heading
estimate produced unreliable values across several different
implementations and were removed rather than published as a noisy field.

---

## Position Reliability

Every observation includes `position_reliable` (true/false). The
ground-plane projection uses only the bottom row of the bounding box (where
tyres meet road). Three cases break this assumption and are flagged
unreliable:

- **Bottom clipped**: tyres below the crop — projection input missing
- **Side clipped**: vehicle half out of frame — lateral centre of visible
  box is not vehicle centre
- **Lateral clamp fired**: computed position exceeds physical road boundary

Top-clipped boxes (roof out of frame, tyres visible) are **reliable** — the
formula only uses the bottom row.

Unreliable rows are kept in the dataset, not deleted. The RTS smoother
assigns them 25× lower measurement weight so the motion model interpolates
instead of trusting the bad measurement.

Run `python3 count_reliability.py` after processing to get a breakdown by
vehicle type, distance bucket, and track ID, plus a dump of every
unreliable observation's x/y position for manual sanity-checking.

---

## Behaviour Labels

Labels are assigned by `annotator.py` only when `emergency_active = true`
and the vehicle is within 50 m.

| Label | Condition |
|---|---|
| `yielded` | Lateral speed ≥ 0.5 m/s, directed away from the ambulance's path, sustained for ≥ 2 consecutive frames, OR cumulative lateral drift ≥ 0.8 m over 3 s monotonically in one direction |
| `braked_abruptly` | Acceleration ≤ −2.5 m/s², OR acceleration ≤ −1.5 m/s² AND jerk ≤ −3.0 m/s³ |
| `failed_to_yield` | Within 20 m, ≥ 3 frames of history, nothing else triggered |
| `normal` | None of the above |

Thresholds: Pierson et al. 2019 (lateral speed), Krajewski et al. 2018
(cumulative window). A vehicle moving sideways toward the ambulance's path
(rather than away from it) is not counted as yielding, per Cortés &
Stefoni (2023), who found drivers only react when the emergency vehicle is
actually in or near their own path.

Heading-based rules from earlier versions of the annotator were removed —
see "JSON Output" above.

---

## Known Limitations

**Speeds are relative only.** All velocities are relative to the ambulance,
not absolute. A vehicle matching the ambulance's speed reads ~0; one being
overtaken reads negative forward speed.










