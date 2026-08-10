#!/usr/bin/env python3
"""
review.py -- correction UI for exported JSON, reading frames directly from
the source video (no review_data/ dependency, which main.py no longer writes).

Replaces the old review.py (which needed pre-saved review_data/frame_*.jpg,
no longer produced) and folds in visualize_pipeline.py's box-drawing concept
-- one tool instead of two, since the workflow is watch-and-correct, not
batch-render.

FRAME SOURCING: opens videos/<video_name>.<ext> directly, seeks by FRAME
NUMBER (round(timestamp * native_fps)) rather than by millisecond position --
this is the exact inverse of how preprocessor.py computes timestamps
(frame_count / video_fps), so it's the most consistent choice, not an
independent re-guess. The crop is done via VideoPreprocessor.spatial_crop()
itself (imported, not reimplemented), so box alignment is GUARANTEED
identical to what the pipeline actually used -- no risk of a second,
slightly-different crop implementation drifting out of sync, the same class
of bug that caused problems earlier in this project.

EGO VEHICLE: id=0 (bbox=None, fixed at the origin) is excluded from display,
hit-testing, and correction entirely -- it's structural, not a real
detection, and is written back to the JSON unchanged on save.

JSON DISCOVERY: globs t*.json in the output folder and reads the
"timestamp" field from EACH file's content, rather than reconstructing a
filename from a timestamp -- current exporter.py uses frame-index filenames
(t000000.json), decoupled from timestamp, so guessing filenames from
timestamps breaks silently. This was the exact bug in the old
visualize_pipeline.py.

COLOURS:
  Green       yielded
  Red         failed_to_yield
  Orange      braked_abruptly
  White       normal

CONTROLS:
  ENTER / SPACE   Next frame
  Left arrow      Previous frame
  Click box       Select it (turns yellow)
  D               Delete selected box
  Y               Set behaviour: yielded
  F               Set behaviour: failed_to_yield
  N               Set behaviour: normal
  K               Set behaviour: braked_abruptly
  Type digits     Edit ID of selected box (type new ID, confirm with ENTER)
  ESC             Deselect
  Q               Quit (all changes already saved)

Usage:
    python3 review.py --video video1
    python3 review.py --video video1 --videos-dir videos --output-dir output
"""
import os
import sys
import glob
import json
import argparse
import cv2
import numpy as np

from preprocessor import VideoPreprocessor

OUTPUT_DIR = "output"
VIDEOS_DIR = "videos"
WINDOW = "Review  |  ENTER=next  Left=back  D=del  Y/F/N/K=behaviour  digits=ID  Q=quit"

# colours (BGR)
COL_YIELDED  = (0, 220, 0)      # green
COL_FAILED   = (0, 0, 200)      # red
COL_BRAKED   = (0, 140, 255)    # orange
COL_NORMAL   = (220, 220, 220)  # white
COL_SELECTED = (0, 255, 255)    # yellow

BEHAVIOUR_COLS = {
    "yielded":         COL_YIELDED,
    "failed_to_yield": COL_FAILED,
    "braked_abruptly": COL_BRAKED,
    "normal":          COL_NORMAL,
}

BEHAVIOUR_KEYS = {
    ord('y'): "yielded", ord('Y'): "yielded",
    ord('f'): "failed_to_yield", ord('F'): "failed_to_yield",
    ord('n'): "normal", ord('N'): "normal",
    ord('k'): "braked_abruptly", ord('K'): "braked_abruptly",
}


def find_video_path(video_name, videos_dir):
    exact = os.path.join(videos_dir, f"{video_name}.mp4")
    if os.path.exists(exact):
        return exact
    matches = glob.glob(os.path.join(videos_dir, f"{video_name}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No video found for '{video_name}' in {videos_dir}/")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_export_files(video_name, output_dir):
    """Glob-based discovery, sorted by filename (chronological, since
    exporter.py uses zero-padded increasing frame_index). Never reconstructs
    a filename from a timestamp."""
    json_dir = os.path.join(output_dir, video_name)
    files = sorted(glob.glob(os.path.join(json_dir, "t*.json")))
    return files


def _hit_test(vehicles, x, y, margin=6):
    for i, v in enumerate(vehicles):
        if v["bbox"] is None:
            continue
        x1, y1, x2, y2 = v["bbox"]
        if (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin):
            return i
    return None


def seek_frame(cap, native_fps, timestamp, preprocessor):
    """Seeks by FRAME NUMBER (round(timestamp * native_fps)) -- the exact
    inverse of preprocessor.py's frame_count/video_fps timestamp formula,
    not an independent re-guess. Returns the frame after spatial_crop
    (same 1280x720 crop the pipeline itself used), or None on failure."""
    frame_number = int(round(timestamp * native_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return preprocessor.spatial_crop(frame)


def review_frame(frame, json_data, json_path):
    """Review one frame. Modifies json_data in place and saves to json_path.
    Ego (id=0) is excluded from the editable vehicle list, kept unchanged.
    Returns: "next", "prev", or "quit"
    """
    all_vehicles = [dict(v) for v in json_data.get("vehicles", [])]
    ego_entries      = [v for v in all_vehicles if v.get("id") == 0]
    editable_vehicles = [v for v in all_vehicles if v.get("id") != 0]

    state = {"vehicles": editable_vehicles, "selected": None, "id_input": ""}
    changed = [False]

    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        hit = _hit_test(state["vehicles"], x, y)
        if hit is not None:
            if state["selected"] == hit:
                state["selected"] = None
                state["id_input"] = ""
            else:
                state["selected"] = hit
                state["id_input"] = str(state["vehicles"][hit]["id"])

    cv2.setMouseCallback(WINDOW, mouse_cb)

    result = ["next"]

    while True:
        vis = _render(frame, state,
                      json_data.get("timestamp", 0),
                      json_data.get("emergency_active", False))
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32):  # ENTER or SPACE
            _commit_id(state, changed)
            result[0] = "next"
            break

        if key == 81:  # Left arrow (removed 'B'/'b' -- now reserved by nothing, but keep arrow-only to avoid future collisions)
            _commit_id(state, changed)
            result[0] = "prev"
            break

        if key in (ord('q'), ord('Q')):
            _commit_id(state, changed)
            result[0] = "quit"
            break

        sel = state["selected"]

        if key in range(ord('0'), ord('9') + 1) and sel is not None:
            state["id_input"] += chr(key)
            changed[0] = True
            continue

        if key in (8, 127) and sel is not None:
            state["id_input"] = state["id_input"][:-1]
            continue

        if key in (ord('d'), ord('D')) and sel is not None:
            state["vehicles"].pop(sel)
            state["selected"] = None
            state["id_input"] = ""
            changed[0] = True

        if key in BEHAVIOUR_KEYS and sel is not None:
            state["vehicles"][sel]["behaviour"] = BEHAVIOUR_KEYS[key]
            changed[0] = True

        if key == 27:  # ESC
            state["selected"] = None
            state["id_input"] = ""

    if changed[0]:
        # ego entries are re-attached UNCHANGED -- never editable, never lost
        json_data["vehicles"] = ego_entries + state["vehicles"]
        save_json(json_path, json_data)

    return result[0]


def _commit_id(state, changed):
    sel = state["selected"]
    if sel is not None and state["id_input"]:
        try:
            new_id = int(state["id_input"])
            if new_id != state["vehicles"][sel]["id"]:
                state["vehicles"][sel]["id"] = new_id
                changed[0] = True
        except ValueError:
            pass


def _render(frame, state, timestamp, emergency):
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    emg_col = (0, 0, 255) if emergency else (200, 200, 200)
    cv2.putText(vis,
                f"t={timestamp:.2f}s  {'EMERGENCY' if emergency else 'normal'}  "
                f"|  vehicles:{len(state['vehicles'])}  "
                f"|  ENTER=next  Left=back  D=del  Y/F/N/K=beh  digits=ID  Q=quit",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, emg_col, 1)

    for i, v in enumerate(state["vehicles"]):
        if v["bbox"] is None:
            continue
        x1, y1, x2, y2 = v["bbox"]
        vid = v["id"]
        beh = v.get("behaviour", "normal")

        col = COL_SELECTED if i == state["selected"] else BEHAVIOUR_COLS.get(beh, COL_NORMAL)

        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.circle(vis, ((x1 + x2) // 2, y2), 4, col, -1)

        if i == state["selected"]:
            label = f"id[{state['id_input']}] {v['type']}"
        else:
            label = f"id{vid} {v['type']}"
            if beh != "normal":
                label += f" [{beh[:3]}]"
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    sel = state["selected"]
    if sel is not None and sel < len(state["vehicles"]):
        v = state["vehicles"][sel]
        info = (f"SELECTED id=[{state['id_input']}]  "
                f"type={v['type']}  beh={v.get('behaviour','normal')}  "
                f"dist={v.get('distance_to_ego',0):.1f}m  "
                f"| Y=yielded F=failed N=normal K=braked  D=delete")
        cv2.putText(vis, info, (6, fh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL_SELECTED, 1)

    return vis


def run_review(video_name, videos_dir=VIDEOS_DIR, output_dir=OUTPUT_DIR):
    json_files = get_export_files(video_name, output_dir)
    if not json_files:
        print(f"  [review] no JSON output found for {video_name} in "
              f"{output_dir}/{video_name}/, skipping")
        return

    video_path = find_video_path(video_name, videos_dir)
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        print(f"  [review] could not read a valid fps from {video_path}, aborting")
        return

    p = VideoPreprocessor(video_path)  # only used for its spatial_crop()

    print(f"\nReviewing {len(json_files)} frames for: {video_name}")
    print(f"  source video: {video_path}  (native_fps={native_fps:.3f})")
    print("Green=yielded  Red=failed_to_yield  Orange=braked  White=normal")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1400, 620)

    idx = 0
    while 0 <= idx < len(json_files):
        json_path = json_files[idx]
        json_data = load_json(json_path)
        timestamp = json_data.get("timestamp", 0.0)

        frame = seek_frame(cap, native_fps, timestamp, p)
        if frame is None:
            print(f"  t={timestamp:.2f}s ({json_path}): could not read frame, skipping")
            idx += 1
            continue

        action = review_frame(frame, json_data, json_path)

        if action == "next":
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)
        elif action == "quit":
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Review complete for {video_name}. All changes saved to JSON files.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                         help="Video name (subfolder in output/)")
    parser.add_argument("--videos-dir", type=str, default=VIDEOS_DIR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.video:
        video_name = args.video
    else:
        folders = [f for f in os.listdir(args.output_dir)
                   if os.path.isdir(os.path.join(args.output_dir, f))]
        if not folders:
            raise SystemExit("No output folders found. Run main.py first.")
        video_name = sorted(folders)[0]

    run_review(video_name, args.videos_dir, args.output_dir)


if __name__ == "__main__":
    main()