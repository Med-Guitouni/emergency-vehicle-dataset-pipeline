# 🚑 Emergency Vehicle Behaviour Dataset Pipeline

Turns dashcam footage filmed inside German ambulances on emergency runs into a
trajectory dataset: for every nearby vehicle, in every frame, where it is in
metres, how fast it is moving relative to the ambulance, which lane it is in,
who its neighbours are, and whether it yielded.

No public dataset records what traffic actually does when an ambulance comes
through. This pipeline builds one from footage anyone can watch.

**In:** an `.mp4` in `videos/`
**Out:** `output/<video_name>/t000000.json …`, one file per exported frame

Tracking runs at 30 Hz. Records are exported at 5 Hz, every sixth tracked
frame.

```
preprocessor → detector + tracker → homography → smoother → homography + surrounding + annotator → exporter → review
 30 Hz frames    YOLOv8x + BoT-SORT   pixels → metres      RTS      metrics, lanes, neighbours, rules      5 Hz JSON    labels
```

---

## ⚙️ Install

```bash
pip3 install -U yt-dlp ultralytics opencv-python numpy torch torchvision pillow
brew install ffmpeg          # or: apt install ffmpeg
```

Keep Ultralytics current: BoT-SORT's `model: auto` in `botsort.yaml` needs a
recent version, and the ReID weights ship with the package. Nothing else to
clone.

A CUDA GPU is strongly recommended. `main.py` prints whether it found one
before loading any model. On CPU, 30 Hz tracking is very slow.

---

## ▶️ Run it

### 1. Get a video

```bash
python3 -c "from downloader import VideoDownloader; VideoDownloader().download_single('URL')"
```

To grab only part of a long video and save bandwidth:

```bash
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best" \
  --merge-output-format mp4 \
  --download-sections "*04:30-09:00" \
  -o "videos/your_video_name.%(ext)s" \
  "URL"
```

A trimmed clip's timeline starts at 0, so every time you write in the next step
is relative to the clip, not the original video.

### 2. Describe the road

Watch the video once and add an entry to `video_lanes.json`: how many lanes,
what kind of road, and when the emergency run starts. This is the only manual
input besides the labels, and it takes a couple of minutes per video. Format
and rules are under [Lane config](#-lane-config-video_lanesjson).

Skipping it does not crash anything. The scene classifier fills in road type
automatically and lane count falls back to three, with every such row stamped
`lane_source: "scene_classifier"` or `"default_highway_3lane"` so you can tell
afterwards which rows were guessed.

### 3. Process

```bash
python3 main.py                                        # every video in videos/
python3 main.py --video video1                         # just one
python3 main.py --video video1 --start 60 --end 180    # a 2-minute window
```

A windowed run writes to `output/video1_t60-180/`, so a test slice can never be
confused with a full run. Lane and emergency lookups still use the real video
name. The tracker starts cold at `--start` with no history, so expect extra ID
churn in the first seconds of a window.

Stopping and resuming: rerun the same command. Any video that already has files
in `output/<video_name>/` is skipped. If a run was killed mid-video, its folder
exists but is incomplete, so delete it before relaunching.

### 4. Label by hand

```bash
python3 review.py --video video1
```

This is the step that produces the real labels. Details below.

---

## 🏷️ Behaviour labels

Four classes: `yielded`, `braked_abruptly`, `failed_to_yield`, `normal`.

The labels `main.py` writes are provisional. They come from the kinematic rules
in `annotator.py`, which are kept as a documented starting point for future
automation but did not produce the released dataset. Two reasons. The
thresholds come from different studies on different road types, sensor setups
and sampling rates, so they are not mutually consistent. And there is no agreed
kinematic definition of yielding to an emergency vehicle, because no prior work
measures it directly. In practice the rules also miss partial manoeuvres,
staged movements, and vehicles boxed in by surrounding traffic.

Braking is the exception and stays rule-derived. You can confirm it by eye from
brake lights, but you cannot measure a deceleration by eye.

So: run `review.py` and label every frame. It draws each tracked vehicle with
its ID and current kinematics, and writes your keystroke straight into the
JSON.

| Key | Action |
|---|---|
| `ENTER` / `SPACE` | Next frame |
| `←` | Previous frame |
| Click a box | Select it, turns yellow |
| `Y` | `yielded` |
| `F` | `failed_to_yield` |
| `K` | `braked_abruptly` |
| `N` | `normal` |
| `D` | Delete the selected box |
| Digits, then `ENTER` | Change the selected vehicle's ID |
| `ESC` | Deselect |
| `Q` | Quit, everything is already saved |

Colours: 🟢 yielded, 🔴 failed_to_yield, 🟠 braked_abruptly, ⚪ normal, 🟡
selected. Every change is written to disk immediately. There is no save step
and no undo. The ego node, id 0, is hidden from the UI and written back
untouched.

<details>
<summary>The rule thresholds, for reference</summary>

| Label | Rule |
|---|---|
| `yielded` | Lateral speed ≥ 0.5 m/s away from the ambulance's path, sustained ≥ 2 s; or cumulative monotonic lateral drift ≥ 0.8 m over 3 s; or contact with the road-boundary clamp, meaning the vehicle left the carriageway |
| `braked_abruptly` | Acceleration ≤ −2.5 m/s²; or ≤ −1.5 m/s² together with jerk ≤ −3.0 m/s³, catching a panic stop split across two frames |
| `failed_to_yield` | Within 20 m, seen for ≥ 3 s, nothing else fired |
| `normal` | None of the above, or further than 50 m away |

Sources: Pierson et al. 2019 for the lateral speed, Krajewski et al. 2018 for
the cumulative window, Cortés and Stefoni 2023 for the directional condition,
who find drivers react only when the emergency vehicle is in their own path.

Heading-based rules were specified and dropped: resolving a few degrees of
steering angle from a moving monocular camera is dominated by noise at this
resolution. A speed-drop rule was dropped too, since it used speed magnitude,
which carries the zero-crossing artifact already fixed for acceleration.
</details>

---

## 🛣️ Lane config (`video_lanes.json`)

One JSON object, one top-level key per video. Don't create separate files or
separate `{ }` blocks.

```json
{
  "video1": {
    "emergency_start_second": 0,
    "lanes": [
      {
        "from_second": 0,
        "to_second": 450,
        "lanes": 3,
        "road_type": "highway",
        "notes": "A2 Autobahn, 3 lanes"
      },
      {
        "from_second": 450,
        "to_second": 930,
        "lanes": 2,
        "road_type": "urban",
        "notes": "city centre after the exit"
      }
    ]
  }
}
```

The key must match the video filename as
`os.path.splitext(os.path.basename(path))[0][:30]`. For `videos/video1.mp4`
that's `"video1"`. Watch the 30-character truncation: long YouTube titles get
cut.

Add one window per change. If the ambulance leaves the Autobahn at 7:30, write
two windows meeting at 450.

Don't write lane widths. They follow from `road_type` per the German
road-design standards: `highway` → 3.75 m, `urban`, `intersection` and
`roundabout` → 3.00 m. Road type and lane count are the only things you
annotate; width is a regulated constant, not something the pipeline estimates
per frame.

Emergency is latched. Once `emergency_start_second` is reached it stays active
for the rest of the clip. A video with no entry defaults to active, since this
footage is curated to be during a run.

If a segment has no manual entry, `scene_classifier.py` fills in the road type
automatically. It is a ResNet18 pretrained on MIT Places365, run once per real
second on the uncropped frame, which votes over its top-5 Places365 categories
and requires three identical predictions in a row before it accepts a change,
so a single misclassified frame cannot flip the label. Those rows are stamped
`lane_source: "scene_classifier"`, and if the classifier has no confirmed
prediction yet they fall back to `"default_highway_3lane"`.

The classifier is a fallback, not the primary source: a manual entry always
wins, and a disagreement between the two is printed but does not override the
config. Lane count has no automatic source at all, since the classifier
predicts a scene category, not how many lanes there are.

Lane-line detection was tried with UFLD v2 and YOLOP and abandoned: in exactly
the dense scenes this dataset targets, large vehicles hide the markings.

---

## 📦 Output format

```json
{
  "timestamp":        ,   // seconds from this video's start
  "video_source":     ,   // video filename, truncated to 30 chars
  "emergency_active": ,   // true once emergency_start_second is reached
  "scenario_type":    ,   // road scene category
  "vehicles": [ ... ]
}
```

Each entry in `vehicles`:

| Field | Meaning |
|---|---|
| `id` | Stable ID from BoT-SORT. 0 is the ego, the ambulance itself |
| `type` | `car`, `truck`, `bus`, `motorcycle`, or `ego` |
| `x_meters` | Lateral position. Positive is right of the ambulance |
| `y_meters` | Forward distance ahead of the ambulance |
| `position_reliable` | `false` when the box was clipped or the lateral clamp fired |
| `speed_kmh` | Overall speed magnitude, relative to the ambulance |
| `forward_speed_ms` | Along the road. Positive is pulling away, negative is the ambulance closing |
| `lateral_speed_ms` | Across the road. Positive is right. The direct yielding signal |
| `acceleration` | Change in `forward_speed_ms` over a 1 s window, m/s². `null` until the window fills |
| `lateral_acceleration` | Same window, lateral component |
| `jerk` | Change in acceleration over the same window, m/s³ |
| `ttc_to_ego` | Seconds to the ambulance if ahead and closing, `null` otherwise, never infinity |
| `lane_id` | 1 leftmost to N rightmost |
| `lateral_offset` | Metres from that lane's centreline. Positive is right of centre |
| `lane_position_norm` | −1 to +1 within its own lane: 0 is centre, ±1 the lane edge |
| `road_position_norm` | −1 to +1 across the whole road: ±1 the road edge or shoulder |
| `distance_to_ego` | √(x² + y²) |
| `lanes_total`, `road_type` | Road layout at this timestamp |
| `lane_source` | `"config"` manual, `"scene_classifier"` CNN fallback, `"default_highway_3lane"` neither |
| `preceding_id`, `following_id` | Nearest vehicle ahead and behind in the same lane |
| `left_*`, `right_*` | Same, in the adjacent lanes, highD convention |
| `behaviour` | The label, or `ego` |
| `bbox` | `[x1, y1, x2, y2]` in the cropped frame, `null` for the ego |

Two things to know before training on this.

All speeds are relative to the ambulance, not absolute. A vehicle travelling at
exactly the ambulance's speed reads about 0, not its road speed.

`lane_position_norm` and `road_position_norm` exist for MTP-GO compatibility.
They mean the same thing whether a lane is 3.00 m or 3.75 m wide, and they
separate "moved onto the shoulder" from "changed lane". The ego node is emitted
in every frame with the same schema so a graph model always has an ego vertex;
its `bbox` and `ttc_to_ego` are structurally null.

Heading is not computed or exported.

---

## 🎯 How much to trust a row

Every observation carries `position_reliable`. The projection uses only the
bottom edge of the bounding box, where the tyres meet the road, so anything
that breaks that gets flagged:

- bottom clipped, tyres below the crop, the input is missing
- side clipped, vehicle half out of frame, so the visible box's centre is not
  the vehicle's centre
- lateral clamp fired, the computed position is off the physical road
- near the horizon with a box under 15 px tall, where even the fallback
  estimator carries about 20% error

Top-clipped boxes, roof cut off but tyres visible, are reliable. The formula
only uses the bottom row.

Unreliable rows are kept, not deleted. The smoother gives them 25× the
measurement variance, so the motion model takes over instead of following a bad
measurement. Filter on the flag if you need precision.

Accuracy falls off with distance, and faster than linearly. Forward-distance
error is under 1 m within 10 m and around 12 m in the 45–60 m band. That is the
geometry, not a bug: the pixel gap to the horizon sits in the denominator, so a
fixed pixel error becomes a metric error scaling roughly with the square of
distance, and every monocular ground-plane pipeline behaves this way.

The practical consequence: this data is dependable for close-range interaction,
which is where yielding happens, and progressively less so with distance. If
you need continuous trajectories, filter on track length. If you need metric
precision, filter on the reliability flag or weight by distance. Both fields
are exported for exactly that.

---

## 📐 Calibration

Four constants in `homography.py`, all measured from the footage rather than
guessed, using two things German regulation fixes as physical rulers: Autobahn
lane width, 3.75 m, and the lane-dash period, 18 m made of a 6 m stripe and a
12 m gap. Raw per-frame measurements are in `calibration_log.json`.

| Constant | Value | Spread | Measured from |
|---|---|---|---|
| `camera_height` | 1.40 m | 1.27–1.72 | Lane width |
| `focal_length_factor` | 0.72 | 0.41–1.18 | Dash spacing, least squares, max residual under 4 px |
| `horizon_ratio` | 0.60 | 0.549–0.619 | Vanishing point |
| `CX_RATIO` | 0.47 | 0.459–0.519 | Vanishing point |

Height and focal length both scale distance linearly, which is why lane width
and dash spacing are two separate measurements. One distance check cannot
separate them.

These are the numbers to re-measure if you process footage from a different
camera. The horizon ratio drifts within a single video in a way that looks like
real road grade rather than measurement error, so a single compromise value is
used and the residual is documented, not removed.

---

## 🗂️ File map

| File | What it does |
|---|---|
| `main.py` | Runs both phases and the export for every video |
| `downloader.py` | yt-dlp wrapper |
| `preprocessor.py` | Streams frames at a given Hz, crops, resizes to 1280×720 |
| `detector.py` | Loads YOLOv8x, picks CPU or CUDA |
| `tracker.py` | BoT-SORT: appearance ReID plus camera-motion compensation |
| `homography.py` | Pixels → metres, plus velocity, distance, TTC, lanes, offsets |
| `smoother.py` | RTS smoother, once per video between the phases |
| `annotator.py` | Provisional kinematic labels |
| `lane_config.py` | Reads `video_lanes.json`, resolves lane width and emergency state |
| `scene_classifier.py` | Places365 ResNet18 road-type fallback for unannotated segments |
| `surrounding.py` | Six neighbour IDs, highD convention |
| `exporter.py` | Writes the JSON |
| `review.py` | Manual labelling window |
| `visualize_pipeline.py` | Renders one annotated debug frame per second |
| `botsort.yaml` | Tracker config, read the header before changing `TRACK_FPS` |
| `calibration_log.json` | Per-frame calibration measurements |
| `validation_vs_nuscenes/` | Per-frame images from the nuScenes validation runs |

---

## 🔬 What actually happens in a run

Phase 1, track. Each frame is cropped to remove sky and dashboard, resized to
1280×720, and passed through YOLOv8x and BoT-SORT at 30 Hz. Every detection is
projected onto the road plane to get an (x, y) position in metres. Every sixth
frame is also kept as an export record.

Between phases, smooth. When the video ends, each vehicle's full 30 Hz
trajectory goes through a Rauch-Tung-Striebel smoother, once, per axis. A plain
Kalman filter only looks backward; RTS adds a backward pass over the finished
trajectory, so every estimate is corrected using what happened after it. This
is the same post-processing step highD and INTERACTION apply before publishing.

Phase 2, measure and export. From the smoothed positions: forward and lateral
speed, acceleration, jerk, time-to-collision, lane, offsets, and the six
neighbour IDs. Acceleration and jerk use a fixed one-second lookback rather
than the gap between two exported frames, which otherwise amplifies a few
centimetres of position noise into impossible values. An ego node is added to
every frame. The rules assign a provisional label, and one JSON per exported
frame is written.

Then you review. `main.py` does not open a review window.

---

## ⚠️ Known limitations

Speeds are relative, not absolute.  A
road-segmentation-gated estimator is the open direction.

The road is assumed flat. The measured horizon ratio is not constant across a
video, and the compromise value leaves a residual error that is documented
rather than removed.

Validation isolates the projection.What it does not measure is detection and tracking error
from the YOLOv8x and BoT-SORT stage, which stays a separate, unquantified
source of error.

Tracks fragment.

---

## 🔧 Troubleshooting

Every row says `lane_source: "default_highway_3lane"`. The video isn't in
`video_lanes.json`, its key doesn't match the filename truncated to 30
characters, or the file isn't valid JSON. `LaneConfig` prints a warning at
startup in the last two cases.

`CUDA available: NO`. Torch was installed without CUDA support. Reinstall from
the PyTorch index for your CUDA version. 30 Hz tracking on CPU is impractical.

OpenCV pyramid-size assertion from the tracker. Camera-motion compensation
needs a constant frame size. `preprocessor.spatial_crop` guarantees that by
resizing every frame to 1280×720, so if you change the crop, keep the fixed
resize.

A video is skipped. It already has files in `output/<video_name>/`. Delete the
folder to reprocess.

IDs churn in the first seconds of a windowed run. Expected. The tracker starts
cold at `--start` with no history, and this doesn't affect full runs.











