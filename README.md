# Emergency Vehicle Behaviour Dataset Pipeline

Turns dashcam footage filmed inside German ambulances on emergency runs into
a trajectory dataset: for every nearby vehicle, in every frame, where it is in
metres, how fast it is moving relative to the ambulance, which lane it is in,
who its neighbours are, and whether it yielded.

Input: an `.mp4` in `videos/`.
Output: `output/<video_name>/t000000.json …`, one file per exported frame.

Tracking runs at 30 Hz. Records are exported at 5 Hz (every 6th tracked frame).



---

## Install

```bash
pip3 install -U yt-dlp ultralytics opencv-python numpy torch torchvision
brew install ffmpeg          # or: apt install ffmpeg
```

Keep Ultralytics current — BoT-SORT's `model: auto` in `botsort.yaml` needs a
recent version, and the ReID weights ship with the package. 

A CUDA GPU is strongly recommended. `main.py` prints whether it found one
before loading any model; on CPU, 30 Hz tracking is very slow.

---

## Run it

### 1. Get a video

```bash
python3 -c "from downloader import VideoDownloader; VideoDownloader().download_single('URL')"
```

To grab only part of a long video (saves bandwidth and disk):

```bash
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best" \
  --merge-output-format mp4 \
  --download-sections "*04:30-09:00" \
  -o "videos/your_video_name.%(ext)s" \
  "URL"
```

A trimmed clip's timeline starts at 0, so all the times you write in the next
step are relative to the clip, not the original video.

### 2. Describe the road

Watch the video once and add an entry to `video_lanes.json`: how many lanes,
what kind of road, and when the emergency run starts. This is the only manual
input besides the labels. Format and rules are in
[Lane config](#lane-config-video_lanesjson) below.

Skipping this step does not crash anything — you fall back to highway with
three lanes, and every exported row is stamped
`lane_source: "default_highway_3lane"` so you can tell afterwards.

### 3. Process

```bash
python3 main.py                                   # every video in videos/
python3 main.py --video video1                    # just one
python3 main.py --video video1 --start 60 --end 180   # a 2-minute window
```

A windowed run writes to `output/video1_t60-180/`, so a test slice can never
be confused with a full run. Lane and emergency lookups still use the real
video name. The tracker starts cold at `--start` with no history, so expect
extra ID churn in the first seconds of a window.

**Stopping and resuming**: rerun the same command. Any video that already has
files in `output/<video_name>/` is skipped. If a run was killed mid-video, its
folder exists but is incomplete — delete it before relaunching.

### 4. Label by hand

```bash
python3 review.py --video video1
```

This is the step that produces the real labels. See
[Behaviour labels](#behaviour-labels) below.

---

## What actually happens

**Phase 1 — track.** Each frame is cropped (sky and dashboard removed),
resized to 1280×720, and passed through YOLOv8x + BoT-SORT at 30 Hz. Every
detection is projected onto the road plane to get an (x, y) position in
metres. Every 6th frame is also kept as an export record.

**Between phases — smooth.** When the video ends, each vehicle's full 30 Hz
trajectory goes through an RTS smoother, once, per axis.

**Phase 2 — measure and export.** From the smoothed positions: forward and
lateral speed, acceleration, jerk, time-to-collision, lane, offsets, and the
six neighbour IDs. An ego node for the ambulance itself is added to every
frame. The kinematic rules assign a provisional behaviour label. One JSON per
exported frame is written.

**Then you review.** `main.py` does not open a review window; run `review.py`
separately when a video finishes.

```
preprocessor → detector + tracker → homography → smoother → homography + surrounding + annotator → exporter → review
 (30 Hz frames)  (YOLOv8x + BoT-SORT)  (pixels → metres)  (RTS)      (metrics, lanes, neighbours, rules)     (5 Hz JSON)  (labels)
```

---

## Behaviour labels

Four classes: `yielded`, `braked_abruptly`, `failed_to_yield`, `normal`.

**The labels `main.py` writes are provisional.** They come from the kinematic
rules in `annotator.py`, which are kept as a documented starting point for
future automation but did not produce the released dataset. Two reasons: the
thresholds come from different studies on different road types, sensor setups
and sampling rates, so they are not mutually consistent; and there is no
agreed kinematic definition of yielding to an emergency vehicle, because no
prior work measures it directly. In practice the rules also miss partial
manoeuvres, staged movements, and vehicles boxed in by surrounding traffic.

Braking is the exception and stays rule-derived — you can confirm it by eye
from brake lights, but you cannot measure a deceleration by eye.

So: run `review.py` and label every frame. It draws each tracked vehicle with
its ID and current kinematics, and writes your keystroke straight into the
JSON.

| Key | Action |
|---|---|
| `ENTER` / `SPACE` | Next frame |
| `←` | Previous frame |
| Click a box | Select it (turns yellow) |
| `Y` | `yielded` |
| `F` | `failed_to_yield` |
| `K` | `braked_abruptly` |
| `N` | `normal` |
| `D` | Delete the selected box |
| Type digits, then `ENTER` | Change the selected vehicle's ID |
| `ESC` | Deselect |
| `Q` | Quit (everything is already saved) |

Colours: green `yielded`, red `failed_to_yield`, orange `braked_abruptly`,
white `normal`, yellow selected. Every change is written to disk immediately
— there is no save step and no undo. The ego node (id 0) is hidden from the
UI and written back untouched.

<details>
<summary>The rule thresholds, for reference</summary>

| Label | Rule |
|---|---|
| `yielded` | Lateral speed ≥ 0.5 m/s away from the ambulance's path, sustained ≥ 2 s; **or** cumulative monotonic lateral drift ≥ 0.8 m over 3 s; **or** contact with the road-boundary clamp (vehicle left the carriageway) |
| `braked_abruptly` | Acceleration ≤ −2.5 m/s²; **or** ≤ −1.5 m/s² together with jerk ≤ −3.0 m/s³ (panic stop split across two frames) |
| `failed_to_yield` | Within 20 m, seen for ≥ 3 s, nothing else fired |
| `normal` | None of the above, or further than 50 m away |

Sources: Pierson et al. 2019 (lateral speed), Krajewski et al. 2018
(cumulative window), Cortés & Stefoni 2023 — drivers react only when the
emergency vehicle is in their own path, which is where the directional
condition comes from.

Heading-based rules were specified and dropped: resolving a few degrees of
steering angle from a moving monocular camera is pure noise at this
resolution. A speed-drop rule was dropped too — it used speed magnitude,
which has the zero-crossing artifact already fixed for acceleration.
</details>

---

## Lane config (`video_lanes.json`)

One JSON object, one top-level key per video. Don't create separate files or
separate `{ }` blocks.

```json
{
  "video1": {
    "emergency_start_second": 0,
    "lanes": [
      {
        "from_second": 0,
        "to_second": 270,
        "lanes": 3,
        "road_type": "highway",
        "notes": "A2 Autobahn, 3 lanes, full clip"
      }
    ]
  }
}
```

- **The key must match the video filename** as
  `os.path.splitext(os.path.basename(path))[0][:30]` — for
  `videos/video1.mp4` that's `"video1"`. Note the 30-character truncation:
  long YouTube titles get cut.
- **Add a window per change.** If the ambulance leaves the Autobahn at 7:30,
  write two windows with `to_second` / `from_second` at 450.
- **Don't write lane widths.** They follow from `road_type` per the German
  standards: `highway` → 3.75 m, `urban` / `intersection` / `roundabout` →
  3.00 m.
- **Emergency is latched.** Once `emergency_start_second` is reached it stays
  active for the rest of the clip. A video with no entry defaults to active,
  since this footage is curated to be during a run.

---

## Output format

```json
{
  "timestamp":        ,   // seconds from this video's start
  "video_source":     ,   // video filename (truncated to 30 chars)
  "emergency_active": ,   // true once emergency_start_second is reached
  "scenario_type":    ,   // road scene category
  "vehicles": [ ... ]
}
```

Each entry in `vehicles`:

| Field | Meaning |
|---|---|
| `id` | Stable ID from BoT-SORT. **0 is the ego**, the ambulance itself |
| `type` | `car` / `truck` / `bus` / `motorcycle`, or `ego` |
| `x_meters` | Lateral position. **+ = right** of the ambulance |
| `y_meters` | Forward distance ahead of the ambulance |
| `position_reliable` | `false` when the box was clipped or the lateral clamp fired — see below |
| `speed_kmh` | Overall speed magnitude, **relative to the ambulance** |
| `forward_speed_ms` | Along the road. + = pulling away, − = ambulance closing |
| `lateral_speed_ms` | Across the road. + = right. **The direct yielding signal** |
| `acceleration` | Change in `forward_speed_ms` over a 1 s window (m/s²). `null` until the window fills |
| `lateral_acceleration` | Same window, lateral component |
| `jerk` | Change in acceleration over the same window (m/s³) |
| `ttc_to_ego` | Seconds to the ambulance if ahead and closing; `null` otherwise (never infinity) |
| `lane_id` | 1 (leftmost) to N (rightmost) |
| `lateral_offset` | Metres from that lane's centreline. + = right of centre |
| `lane_position_norm` | −1…+1 within its own lane: 0 = centre, ±1 = lane edge |
| `road_position_norm` | −1…+1 across the whole road: ±1 = road edge / shoulder |
| `distance_to_ego` | √(x² + y²) |
| `lanes_total`, `road_type` | Road layout at this timestamp |
| `lane_source` | `"config"` = from `video_lanes.json`; `"default_highway_3lane"` = fallback |
| `preceding_id`, `following_id` | Nearest vehicle ahead / behind in the same lane |
| `left_*`, `right_*` | Same, in the adjacent lanes (highD convention) |
| `behaviour` | The label, or `ego` |
| `bbox` | `[x1, y1, x2, y2]` in the cropped frame; `null` for the ego |

Two notes for anyone training on this:

- **All speeds are relative to the ambulance**, not absolute. A vehicle
  travelling at exactly the ambulance's speed reads ≈ 0, not its road speed.
- `lane_position_norm` and `road_position_norm` exist for MTP-GO
  compatibility: they mean the same thing whether a lane is 3.00 m or 3.75 m
  wide, and they separate "moved onto the shoulder" from "changed lane". The
  ego node is emitted in every frame with the same schema so a graph model
  always has an ego vertex; its `bbox` and `ttc_to_ego` are structurally null.

Heading is not computed or exported.

---

## How much to trust a row

Every observation carries `position_reliable`. The projection uses only the
bottom edge of the bounding box, where the tyres meet the road, so anything
that breaks that gets flagged:

- **Bottom clipped** — tyres below the crop, the input is missing
- **Side clipped** — vehicle half out of frame, so the visible box's centre is
  not the vehicle's centre
- **Lateral clamp fired** — the computed position is off the physical road
- **Near the horizon with a tiny box** — under 15 px tall, where even the
  fallback estimator carries ~20% error

Top-clipped boxes (roof cut off, tyres visible) are **reliable** — the formula
only uses the bottom row.

Unreliable rows are kept, not deleted: the smoother gives them 25× the
measurement variance, so the motion model takes over instead of following a
bad measurement. Filter on the flag if you need precision.

Accuracy falls off with distance, and faster than linearly: forward-distance
error is under 1 m within 10 m and around 12 m in the 45–60 m band. That is
the geometry, not a bug — a fixed pixel error becomes a metric error scaling
roughly with the square of distance, and every monocular ground-plane pipeline
behaves this way. The practical consequence: the data is dependable for
close-range interaction, which is where yielding happens, and progressively
less so with distance.

```bash
python3 count_reliability.py    # breakdown by type, distance and track ID
```

---

## Calibration

Four constants in `homography.py`, all measured from the footage rather than
guessed, using two things German regulation fixes as physical rulers:
Autobahn lane width (3.75 m) and the lane-dash period (18 m = 6 m stripe +
12 m gap). Raw per-frame measurements are in `calibration_log.json`.

| Constant | Value | Spread | Measured from |
|---|---|---|---|
| `camera_height` | 1.40 m | 1.27–1.72 | Lane width |
| `focal_length_factor` | 0.72 | 0.41–1.18 | Dash spacing (least squares, max residual < 4 px) |
| `horizon_ratio` | 0.60 | 0.549–0.619 | Vanishing point |
| `CX_RATIO` | 0.47 | 0.459–0.519 | Vanishing point |

Height and focal length both scale distance linearly, which is why lane width
and dash spacing are two separate measurements — one distance check cannot
separate them.

If you process footage from a different camera, these are the numbers to
re-measure. The horizon ratio drifts within a single video in a way that looks
like real road grade rather than measurement error; a single compromise value
is used and the residual is documented, not removed.

---

## File map

| File | What it does |
|---|---|
| `main.py` | Runs both phases and the export for every video |
| `downloader.py` | yt-dlp wrapper |
| `preprocessor.py` | Streams frames at a given Hz, crops, resizes to 1280×720 |
| `detector.py` | Loads YOLOv8x, picks CPU/CUDA |
| `tracker.py` | BoT-SORT: appearance ReID + camera-motion compensation |
| `homography.py` | Pixels → metres, plus velocity, distance, TTC, lanes, offsets |
| `smoother.py` | RTS smoother, once per video between the phases |
| `annotator.py` | Provisional kinematic labels |
| `lane_config.py` | Reads `video_lanes.json` |
| `surrounding.py` | Six neighbour IDs, highD convention |
| `exporter.py` | Writes the JSON |
| `review.py` | Manual labelling window |
| `count_reliability.py` | Reliability report over all output |
| `visualize_pipeline.py` | Renders one annotated debug frame per second |
| `validate_nuscenes_phaseA.py` | Projection accuracy vs nuScenes ground truth |
| `validate_nuscenes_phaseB.py` | Speed accuracy (project → smooth → velocity) vs nuScenes |
| `botsort.yaml` | Tracker config — read the header before changing `TRACK_FPS` |
| `calibration_log.json` | Per-frame calibration measurements |



---

## Known limitations

**Speeds are relative, not absolute.**

**The road is assumed flat.** 

**Tracks fragment.** 

---











