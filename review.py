"""
review.py

Review UI. Reads the exported JSON files + their saved cropped frames
and lets you correct them in place.

CALLED FROM main.py — automatically launches after each video finishes
processing, via run_review(video_name). Can also be run standalone:

    python3 review.py --video VIDEO_NAME
    python3 review.py   (reviews first video found in output/)

For each second:
  - Shows the saved cropped frame (written to review_data/ during Phase 1
    export, see main.py)
  - Draws boxes coloured by heuristic behaviour label already in the JSON
  - You make corrections → saved directly back to the JSON

COLOURS:
  Green       yielded
  Red         failed_to_yield
  Orange      braked_abruptly
  White       normal

CONTROLS:
  ENTER / SPACE   Next frame
  B / Left arrow  Previous frame
  Click box       Select it (turns yellow)
  D               Delete selected box
  Y               Set behaviour: yielded
  F               Set behaviour: failed_to_yield
  Type digits     Edit ID of selected box (type new ID, confirm with ENTER)
  ESC             Deselect
  Q               Quit (all changes already saved)
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np

REVIEW_DIR = "review_data"
OUTPUT_DIR = "output"
WINDOW     = "Review  |  ENTER=next  B=back  D=del  Y/F=behaviour  digits=ID  Q=quit"

# colours (BGR)
COL_YIELDED   = (0, 220, 0)      # green
COL_FAILED    = (0, 0, 200)      # red
COL_BRAKED    = (0, 140, 255)    # orange
COL_NORMAL    = (220, 220, 220)  # white
COL_SELECTED  = (0, 255, 255)    # yellow

BEHAVIOUR_COLS = {
    "yielded":         COL_YIELDED,
    "failed_to_yield": COL_FAILED,
    "braked_abruptly": COL_BRAKED,
    "normal":          COL_NORMAL,
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_json_path(video_name, timestamp):
    return os.path.join(OUTPUT_DIR, video_name, f"t{timestamp:04d}.json")


def get_frame_path(timestamp):
    return os.path.join(REVIEW_DIR, f"frame_{timestamp:04d}.jpg")


def _hit_test(boxes, x, y, margin=6):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b["bbox"]
        if (x1-margin) <= x <= (x2+margin) and (y1-margin) <= y <= (y2+margin):
            return i
    return None


def review_frame(frame, json_data, json_path):
    """
    Review one frame. Modifies json_data in place and saves to json_path.
    Returns: "next", "prev", or "quit"
    """
    vehicles = [dict(v) for v in json_data.get("vehicles", [])]

    state = {
        "vehicles": vehicles,
        "selected": None,
        "id_input": "",
    }

    changed = [False]

    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        hit = _hit_test([{"bbox": v["bbox"]} for v in state["vehicles"]], x, y)
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

        if key in (ord('b'), ord('B'), 81):  # B or left arrow
            _commit_id(state, changed)
            result[0] = "prev"
            break

        if key in (ord('q'), ord('Q')):
            _commit_id(state, changed)
            result[0] = "quit"
            break

        sel = state["selected"]

        if key in range(ord('0'), ord('9')+1) and sel is not None:
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

        if key in (ord('y'), ord('Y')) and sel is not None:
            state["vehicles"][sel]["behaviour"] = "yielded"
            changed[0] = True
        if key in (ord('f'), ord('F')) and sel is not None:
            state["vehicles"][sel]["behaviour"] = "failed_to_yield"
            changed[0] = True

        if key == 27:
            state["selected"] = None
            state["id_input"] = ""

    if changed[0]:
        json_data["vehicles"] = state["vehicles"]
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
                f"t={timestamp}s  {'EMERGENCY' if emergency else 'normal'}  "
                f"|  vehicles:{len(state['vehicles'])}  "
                f"|  ENTER=next  B=back  D=del  Y/F=beh  digits=ID  Q=quit",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, emg_col, 1)

    for i, v in enumerate(state["vehicles"]):
        x1, y1, x2, y2 = v["bbox"]
        vid = v["id"]
        beh = v.get("behaviour", "normal")

        col = COL_SELECTED if i == state["selected"] else BEHAVIOUR_COLS.get(beh, COL_NORMAL)

        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.circle(vis, ((x1+x2)//2, y2), 4, col, -1)

        if i == state["selected"]:
            label = f"id[{state['id_input']}] {v['type']}"
        else:
            label = f"id{vid} {v['type']}"
            if beh != "normal":
                label += f" [{beh[:3]}]"
        cv2.putText(vis, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    sel = state["selected"]
    if sel is not None and sel < len(state["vehicles"]):
        v   = state["vehicles"][sel]
        vid = v["id"]
        info = (f"SELECTED id=[{state['id_input']}]  "
                f"type={v['type']}  beh={v.get('behaviour','normal')}  "
                f"dist={v.get('distance_to_ego',0):.1f}m  "
                f"| Y=yielded F=failed  D=delete")
        cv2.putText(vis, info, (6, fh-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL_SELECTED, 1)

    return vis


def run_review(video_name):
    """
    Entry point called from main.py right after a video's JSON output is
    saved. Opens the review window for every exported second of that video.

    Requires review_data/frame_<timestamp>.jpg for each second — these are
    written automatically by main.py during Phase 1 export.
    """
    json_dir = os.path.join(OUTPUT_DIR, video_name)
    if not os.path.exists(json_dir):
        print(f"  [review] no JSON output for {video_name}, skipping review")
        return

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    timestamps = [int(f.replace("t", "").replace(".json", "")) for f in json_files]

    if not timestamps:
        print(f"  [review] no JSON files found for {video_name}, skipping review")
        return

    print(f"\nReviewing {len(timestamps)} frames for: {video_name}")
    print("Green=yielded  Red=failed_to_yield  Orange=braked  White=normal")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1400, 620)

    idx = 0
    while 0 <= idx < len(timestamps):
        ts         = timestamps[idx]
        json_path  = get_json_path(video_name, ts)
        frame_path = get_frame_path(ts)

        if not os.path.exists(json_path):
            print(f"  t={ts}s: JSON not found, skipping")
            idx += 1
            continue

        if not os.path.exists(frame_path):
            print(f"  t={ts}s: frame image not found, skipping")
            idx += 1
            continue

        json_data = load_json(json_path)
        frame     = cv2.imread(frame_path)

        if frame is None:
            print(f"  t={ts}s: could not read frame, skipping")
            idx += 1
            continue

        action = review_frame(frame, json_data, json_path)

        if action == "next":
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)
        elif action == "quit":
            break

    cv2.destroyAllWindows()
    print(f"Review complete for {video_name}. All changes saved to JSON files.\n")


def main():
    """Standalone entry point: python3 review.py --video VIDEO_NAME"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Video name (subfolder in output/)")
    args = parser.parse_args()

    if args.video:
        video_name = args.video
    else:
        folders = [f for f in os.listdir(OUTPUT_DIR)
                   if os.path.isdir(os.path.join(OUTPUT_DIR, f))]
        if not folders:
            raise SystemExit("No output folders found. Run main.py first.")
        video_name = sorted(folders)[0]

    run_review(video_name)


if __name__ == "__main__":
    main()