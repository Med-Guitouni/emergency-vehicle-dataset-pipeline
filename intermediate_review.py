"""
intermediate_review.py

An interactive UI to correct raw tracking data *before* smoothing and metric
calculation. This is the primary data correction step in the pipeline.

CALLED FROM main.py — automatically launches after Phase 1 tracking for each
video.

For each 1Hz frame, it allows you to:
  - Add a new vehicle box for a missed detection.
  - Delete an incorrect vehicle box.
  - Change a vehicle's track ID to fix swaps.
  - Resize a bounding box by dragging its corners.

Changes are made directly to the `track_obs` and `records` data structures
that are used by the subsequent smoothing and export phases. This ensures
that all vehicles, including manually added ones, have their metrics
correctly calculated.
"""

import os
import cv2
import json
import argparse

REVIEW_DIR = "review_data"
OUTPUT_DIR = "output"
WINDOW     = "Intermediate Review | Y/F/K/N=Beh A=add T=type D=del ENTER=next Q=quit"

# colours (BGR)
COL_NORMAL    = (220, 220, 220)  # white
COL_SELECTED  = (0, 255, 255)    # yellow
COL_YIELDED   = (0, 220, 0)      # green
COL_FAILED    = (0, 0, 200)      # red
COL_BRAKED    = (0, 140, 255)    # orange

VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle"]

BEHAVIOUR_COLS = {
    "yielded":         COL_YIELDED,
    "failed_to_yield": COL_FAILED,
    "braked_abruptly": COL_BRAKED,
    "normal":          COL_NORMAL,
}

HANDLE_SIZE = 4

def get_handles(bbox):
    x1, y1, x2, y2 = bbox
    return {
        'tl': (x1, y1), 'tr': (x2, y1),
        'bl': (x1, y2), 'br': (x2, y2)
    }

def _hit_test_handles(bbox, x, y):
    handles = get_handles(bbox)
    for name, (hx, hy) in handles.items():
        if abs(x - hx) <= HANDLE_SIZE and abs(y - hy) <= HANDLE_SIZE:
            return name
    return None

def _hit_test(boxes, x, y, margin=6):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b["bbox"]
        if (x1-margin) <= x <= (x2+margin) and (y1-margin) <= y <= (y2+margin):
            return i
    return None


def _commit_id(state, record, track_obs):
    sel = state["selected"]
    if sel is not None and state["id_input"]:
        try:
            new_id = int(state["id_input"])
            v = state["vehicles"][sel]
            old_id = v.get("track_id")

            if new_id != old_id:
                # 1. Update the ID in the current record
                v["track_id"] = new_id

                # 2. Move the observation in track_obs
                ts = record["timestamp"]
                obs_to_move = None
                if old_id in track_obs:
                    for obs in track_obs.get(old_id, []):
                        if obs[0] == ts:
                            obs_to_move = obs
                            break
                    if obs_to_move:
                        track_obs[old_id].remove(obs_to_move)
                        if not track_obs[old_id]:
                            del track_obs[old_id]

                if obs_to_move:
                    track_obs.setdefault(new_id, []).append(obs_to_move)
                    track_obs[new_id].sort(key=lambda o: o[0]) # keep sorted

        except (ValueError, KeyError):
            pass # Ignore invalid ID input

def _render(frame, state, timestamp, emergency):
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    emg_col = (0, 0, 255) if emergency else (200, 200, 200)
    top_text = (f"t={timestamp}s  {'EMERGENCY' if emergency else 'normal'}  "
                f"|  vehicles:{len(state['vehicles'])}")
    cv2.putText(vis, top_text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, emg_col, 1)

    if state["add_mode"]:
        add_text = "ADD MODE: Click and drag to draw a new box. Press A or ESC to cancel."
        cv2.putText(vis, add_text, (6, fh - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL_SELECTED, 1)
        if state["drawing_box"]:
            cv2.rectangle(vis, state["drawing_box"]["start"], state["drawing_box"]["end"], COL_SELECTED, 1)

    for i, v in enumerate(state["vehicles"]):
        x1, y1, x2, y2 = v["bbox"]
        vid = v.get("track_id", "?")
        beh = v.get("behaviour", "normal")

        col = COL_SELECTED if i == state["selected"] else BEHAVIOUR_COLS.get(beh, COL_NORMAL)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.circle(vis, ((x1+x2)//2, y2), 4, col, -1)

        if i == state["selected"]:
            label = f"id[{state['id_input']}] {v.get('type', '?')}"
        else:
            label = f"id{vid} {v.get('type', '?')}"
            if beh != "normal":
                label += f" [{beh[:3]}]"
        cv2.putText(vis, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    sel = state["selected"]
    if sel is not None and sel < len(state["vehicles"]):
        v = state["vehicles"][sel]
        info = (f"SELECTED id=[{state['id_input']}] type={v.get('type','?')} beh={v.get('behaviour','normal')} | "
                f"Y/F/K/N=beh T=type D=del")
        cv2.putText(vis, info, (6, fh-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL_SELECTED, 1)
        
        # Draw resize handles
        handles = get_handles(v['bbox'])
        for name, (hx, hy) in handles.items():
            color = (0, 0, 255) if state.get('hover_handle') == name else COL_SELECTED
            cv2.rectangle(vis, (hx - HANDLE_SIZE, hy - HANDLE_SIZE),
                          (hx + HANDLE_SIZE, hy + HANDLE_SIZE), color, -1)

    return vis


def review_frame_intermediate(frame, record, track_obs, h_estimator):
    """Review one frame, modifying record and track_obs in place."""
    state = {
        "vehicles": record["vehicles_raw"],
        "selected": None,
        "id_input": "",
        "add_mode": False,
        "drawing_box": None,
        "resize_handle": None,
        "hover_handle": None,
    }

    def mouse_cb(event, x, y, flags, param):
        # --- MOUSE MOVE ---
        if event == cv2.EVENT_MOUSEMOVE:
            if state.get('resize_handle') and state['selected'] is not None:
                v = state['vehicles'][state['selected']]
                handle = state['resize_handle']
                if 'l' in handle: v['bbox'][0] = x
                if 'r' in handle: v['bbox'][2] = x
                if 't' in handle: v['bbox'][1] = y
                if 'b' in handle: v['bbox'][3] = y
            elif state['add_mode'] and state['drawing_box']:
                state['drawing_box']['end'] = (x, y)
            elif state['selected'] is not None:
                state['hover_handle'] = _hit_test_handles(state['vehicles'][state['selected']]['bbox'], x, y)

        # --- LEFT BUTTON DOWN ---
        elif event == cv2.EVENT_LBUTTONDOWN:
            if state['add_mode']:
                state['drawing_box'] = {'start': (x, y), 'end': (x, y)}
            elif state['selected'] is not None and state.get('hover_handle'):
                state['resize_handle'] = state['hover_handle']
            else:
                hit = _hit_test([{"bbox": v["bbox"]} for v in state["vehicles"]], x, y)
                if hit is not None:
                    state["selected"] = None if state["selected"] == hit else hit
                    state["id_input"] = str(state["vehicles"][hit].get("track_id", "")) if state["selected"] is not None else ""
                else:
                    state['selected'] = None; state['id_input'] = ""

        # --- LEFT BUTTON UP ---
        elif event == cv2.EVENT_LBUTTONUP:
            if state.get('resize_handle') and state['selected'] is not None:
                v = state['vehicles'][state['selected']]
                x1, y1, x2, y2 = v['bbox']
                v['bbox'] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                state['resize_handle'] = None
            elif state['add_mode'] and state['drawing_box']:
                x1, y1 = min(state["drawing_box"]["start"][0], state["drawing_box"]["end"][0]), min(state["drawing_box"]["start"][1], state["drawing_box"]["end"][1])
                x2, y2 = max(state["drawing_box"]["start"][0], state["drawing_box"]["end"][0]), max(state["drawing_box"]["start"][1], state["drawing_box"]["end"][1])
                state["drawing_box"], state["add_mode"] = None, False
                if x2 - x1 > 5 and y2 - y1 > 5:
                    all_ids = list(track_obs.keys()); new_id = 9800
                    while new_id in all_ids: new_id += 1
                    x_raw, y_raw, reliable = h_estimator.get_vehicle_position([x1, y1, x2, y2], "car", record["frame_width"], frame.shape[0], record["lane_info"])
                    new_vehicle = {"track_id": new_id, "type": "car", "bbox": [x1, y1, x2, y2], "reliable": reliable}
                    state["vehicles"].append(new_vehicle)
                    track_obs[new_id] = [(record["timestamp"], x_raw, y_raw, reliable)]
                    state["selected"], state["id_input"] = len(state["vehicles"]) - 1, str(new_id)

    cv2.setMouseCallback(WINDOW, mouse_cb)

    result = "next"
    while True:
        vis = _render(frame, state, record["timestamp"], record["emergency_active"])
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('a'), ord('A')):
            state["add_mode"] = not state["add_mode"]
            state["selected"], state["id_input"], state["drawing_box"] = None, "", None
            continue

        if key in (13, 32): # ENTER or SPACE
            _commit_id(state, record, track_obs)
            result = "next"; break
        if key in (ord('b'), ord('B'), 81): # B or left arrow
            _commit_id(state, record, track_obs)
            result = "prev"; break
        if key in (ord('q'), ord('Q')):
            _commit_id(state, record, track_obs)
            result = "quit"; break

        sel = state["selected"]
        if sel is None: continue

        if key in range(ord('0'), ord('9')+1): state["id_input"] += chr(key)
        if key in (8, 127): state["id_input"] = state["id_input"][:-1]

        if key in (ord('d'), ord('D')):
            v = state["vehicles"].pop(sel)
            tid, ts = v.get("track_id"), record["timestamp"]
            if tid in track_obs:
                track_obs[tid] = [obs for obs in track_obs[tid] if obs[0] != ts]
                if not track_obs[tid]: del track_obs[tid]
            state["selected"], state["id_input"] = None, ""

        if key in (ord('t'), ord('T')):
            v = state["vehicles"][sel]
            current_type = v.get("type", "car")
            try:
                current_index = VEHICLE_TYPES.index(current_type)
                next_index = (current_index + 1) % len(VEHICLE_TYPES)
                v["type"] = VEHICLE_TYPES[next_index]
            except ValueError:
                v["type"] = VEHICLE_TYPES[0]

        if key in (ord('y'), ord('Y')): state["vehicles"][sel]["behaviour"] = "yielded"
        if key in (ord('f'), ord('F')): state["vehicles"][sel]["behaviour"] = "failed_to_yield"
        if key in (ord('k'), ord('K')): state["vehicles"][sel]["behaviour"] = "braked_abruptly"
        if key in (ord('n'), ord('N')): state["vehicles"][sel]["behaviour"] = "normal"

        if key == 27: # ESC
            state["selected"], state["id_input"], state["add_mode"], state["drawing_box"], state["resize_handle"] = None, "", False, None, None

    return result


def run_intermediate_review(video_name, track_obs, records, h_estimator):
    """
    Main entry point for the intermediate review UI.
    Iterates through 1Hz frames, allowing for corrections.
    Returns the modified track_obs and records.
    """
    if not records:
        print("  [review] No records to review.")
        return track_obs, records

    print(f"  [review] Opening window for {video_name}. Press 'Q' to quit.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1400, 620)

    idx = 0
    while 0 <= idx < len(records):
        record = records[idx]
        timestamp = record["timestamp"]
        frame_path = os.path.join(REVIEW_DIR, video_name, f"frame_{timestamp:04d}.jpg")

        if not os.path.exists(frame_path):
            print(f"  [review] t={timestamp}s: frame image not found, skipping")
            idx += 1; continue

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"  [review] t={timestamp}s: could not read frame, skipping")
            idx += 1; continue

        action = review_frame_intermediate(frame, record, track_obs, h_estimator)

        if action == "next":
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)
        elif action == "quit":
            break

    cv2.destroyAllWindows()
    print(f"  [review] Intermediate review complete for {video_name}.")
    return track_obs, records


# --- Standalone Final Review Functionality ---

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_json_path(video_name, timestamp):
    return os.path.join(OUTPUT_DIR, video_name, f"t{timestamp:04d}.json")

def get_frame_path(video_name, timestamp):
    return os.path.join(REVIEW_DIR, video_name, f"frame_{timestamp:04d}.jpg")

def _render_final(frame, state, timestamp, emergency):
    """Renders a frame for the final review mode."""
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    emg_col = (0, 0, 255) if emergency else (200, 200, 200)
    top_text = (f"t={timestamp}s  {'EMERGENCY' if emergency else 'normal'}  "
                f"|  vehicles:{len(state['vehicles'])}")
    cv2.putText(vis, top_text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, emg_col, 1)

    if state["add_mode"]:
        add_text = "ADD MODE: Click and drag to draw a new box. Press A or ESC to cancel."
        cv2.putText(vis, add_text, (6, fh - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL_SELECTED, 1)
        if state["drawing_box"]:
            cv2.rectangle(vis, state["drawing_box"]["start"], state["drawing_box"]["end"], COL_SELECTED, 1)

    for i, v in enumerate(state["vehicles"]):
        x1, y1, x2, y2 = v["bbox"]
        vid = v.get("id", "?") # Final JSONs use 'id'
        beh = v.get("behaviour", "normal")

        col = COL_SELECTED if i == state["selected"] else BEHAVIOUR_COLS.get(beh, COL_NORMAL)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.circle(vis, ((x1+x2)//2, y2), 4, col, -1)

        if i == state["selected"]:
            label = f"id[{state['id_input']}] {v.get('type', '?')}"
        else:
            label = f"id{vid} {v.get('type', '?')}"
            if beh != "normal":
                label += f" [{beh[:3]}]"
        cv2.putText(vis, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    sel = state["selected"]
    if sel is not None and sel < len(state["vehicles"]):
        v = state["vehicles"][sel]
        dist = v.get('distance_to_ego')
        dist_str = f"{dist:.1f}" if isinstance(dist, (float, int)) else "?"
        info = (f"SELECTED id=[{state['id_input']}] type={v.get('type','?')} beh={v.get('behaviour','normal')} | "
                f"dist={dist_str}m | Y/F/K/N=beh T=type D=del")
        cv2.putText(vis, info, (6, fh-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL_SELECTED, 1)

        # Draw resize handles
        handles = get_handles(v['bbox'])
        for name, (hx, hy) in handles.items():
            color = (0, 0, 255) if state.get('hover_handle') == name else COL_SELECTED
            cv2.rectangle(vis, (hx - HANDLE_SIZE, hy - HANDLE_SIZE),
                          (hx + HANDLE_SIZE, hy + HANDLE_SIZE), color, -1)
    return vis

def review_final_frame(frame, json_data, json_path):
    """UI loop for a single frame in final review mode."""
    vehicles = [dict(v) for v in json_data.get("vehicles", [])]
    state = {
        "vehicles": vehicles, "selected": None, "id_input": "",
        "add_mode": False, "drawing_box": None,
        "resize_handle": None, "hover_handle": None,
    }
    changed = [False]

    def _commit_final_id(state, changed):
        sel = state["selected"]
        if sel is not None and state["id_input"]:
            try:
                new_id = int(state["id_input"])
                if new_id != state["vehicles"][sel].get("id"):
                    state["vehicles"][sel]["id"] = new_id
                    changed[0] = True
            except (ValueError, KeyError): pass

    def mouse_cb(event, x, y, flags, param):
        # --- MOUSE MOVE ---
        if event == cv2.EVENT_MOUSEMOVE:
            if state.get('resize_handle') and state['selected'] is not None:
                v = state['vehicles'][state['selected']]
                handle = state['resize_handle']
                if 'l' in handle: v['bbox'][0] = x
                if 'r' in handle: v['bbox'][2] = x
                if 't' in handle: v['bbox'][1] = y
                if 'b' in handle: v['bbox'][3] = y
                changed[0] = True
            elif state['add_mode'] and state['drawing_box']:
                state['drawing_box']['end'] = (x, y)
            elif state['selected'] is not None:
                state['hover_handle'] = _hit_test_handles(state['vehicles'][state['selected']]['bbox'], x, y)

        # --- LEFT BUTTON DOWN ---
        elif event == cv2.EVENT_LBUTTONDOWN:
            if state['add_mode']:
                state['drawing_box'] = {'start': (x, y), 'end': (x, y)}
            elif state['selected'] is not None and state.get('hover_handle'):
                state['resize_handle'] = state['hover_handle']
            else:
                hit = _hit_test([{"bbox": v["bbox"]} for v in state["vehicles"]], x, y)
                if hit is not None:
                    state["selected"] = None if state["selected"] == hit else hit
                    state["id_input"] = str(state["vehicles"][hit].get("id", "")) if state["selected"] is not None else ""
                else:
                    state['selected'] = None; state['id_input'] = ""

        # --- LEFT BUTTON UP ---
        elif event == cv2.EVENT_LBUTTONUP:
            if state.get('resize_handle') and state['selected'] is not None:
                v = state['vehicles'][state['selected']]
                x1, y1, x2, y2 = v['bbox']
                v['bbox'] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                state['resize_handle'] = None
                changed[0] = True
            elif state['add_mode'] and state['drawing_box']:
                x1, y1 = min(state["drawing_box"]["start"][0], state["drawing_box"]["end"][0]), min(state["drawing_box"]["start"][1], state["drawing_box"]["end"][1])
                x2, y2 = max(state["drawing_box"]["start"][0], state["drawing_box"]["end"][0]), max(state["drawing_box"]["start"][1], state["drawing_box"]["end"][1])
                state["drawing_box"], state["add_mode"] = None, False
                if x2 - x1 > 5 and y2 - y1 > 5:
                    all_ids = [v.get("id", 0) for v in state["vehicles"]]
                    new_id = 9500
                    while new_id in all_ids: new_id += 1
                    new_vehicle = {"id": new_id, "type": "car", "bbox": [x1, y1, x2, y2], "behaviour": "normal"}
                    state["vehicles"].append(new_vehicle)
                    state["selected"] = len(state["vehicles"]) - 1
                    state["id_input"] = str(new_id)
                    changed[0] = True

    cv2.setMouseCallback(WINDOW, mouse_cb)
    result = "next"
    while True:
        vis = _render_final(frame, state, json_data.get("timestamp", 0), json_data.get("emergency_active", False))
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('a'), ord('A')):
            state["add_mode"] = not state["add_mode"]; state["selected"], state["id_input"], state["drawing_box"] = None, "", None; continue
        if key in (13, 32): _commit_final_id(state, changed); result = "next"; break
        if key in (ord('b'), ord('B'), 81): _commit_final_id(state, changed); result = "prev"; break
        if key in (ord('q'), ord('Q')): _commit_final_id(state, changed); result = "quit"; break

        sel = state["selected"]
        if sel is None: continue

        if key in range(ord('0'), ord('9')+1): state["id_input"] += chr(key)
        if key in (8, 127): state["id_input"] = state["id_input"][:-1]
        if key in (ord('d'), ord('D')): state["vehicles"].pop(sel); state["selected"], state["id_input"] = None, ""; changed[0] = True
        if key in (ord('t'), ord('T')):
            v = state["vehicles"][sel]; current_type = v.get("type", "car")
            try: v["type"] = VEHICLE_TYPES[(VEHICLE_TYPES.index(current_type) + 1) % len(VEHICLE_TYPES)]; changed[0] = True
            except ValueError: v["type"] = VEHICLE_TYPES[0]; changed[0] = True
        if key in (ord('y'), ord('Y')): state["vehicles"][sel]["behaviour"] = "yielded"; changed[0] = True
        if key in (ord('f'), ord('F')): state["vehicles"][sel]["behaviour"] = "failed_to_yield"; changed[0] = True
        if key in (ord('k'), ord('K')): state["vehicles"][sel]["behaviour"] = "braked_abruptly"; changed[0] = True
        if key in (ord('n'), ord('N')): state["vehicles"][sel]["behaviour"] = "normal"; changed[0] = True
        if key == 27: state["selected"], state["id_input"], state["add_mode"], state["drawing_box"], state["resize_handle"] = None, "", False, None, None

    if changed[0]:
        json_data["vehicles"] = state["vehicles"]
        save_json(json_path, json_data)
    return result

def run_final_review(video_name):
    """Main loop for standalone final review mode."""
    json_dir = os.path.join(OUTPUT_DIR, video_name)
    if not os.path.exists(json_dir):
        print(f"  [review] no JSON output for {video_name}, skipping review"); return

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    timestamps = [int(f.replace("t", "").replace(".json", "")) for f in json_files]
    if not timestamps:
        print(f"  [review] no JSON files found for {video_name}, skipping review"); return

    print(f"\nReviewing final output for: {video_name}")
    print("Green=yielded  Red=failed_to_yield  Orange=braked  White=normal")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1400, 620)

    idx = 0
    while 0 <= idx < len(timestamps):
        ts = timestamps[idx]
        json_path, frame_path = get_json_path(video_name, ts), get_frame_path(video_name, ts)
        if not os.path.exists(json_path) or not os.path.exists(frame_path):
            idx += 1; continue
        json_data, frame = load_json(json_path), cv2.imread(frame_path)
        if frame is None:
            idx += 1; continue

        action = review_final_frame(frame, json_data, json_path)
        if action == "next": idx += 1
        elif action == "prev": idx = max(0, idx - 1)
        elif action == "quit": break

    cv2.destroyAllWindows()
    print(f"Final review complete for {video_name}. All changes saved.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review final JSON output for a video.")
    parser.add_argument("--video", type=str, required=True, help="Video name (subfolder in output/)")
    args = parser.parse_args()
    if not os.path.isdir(os.path.join(OUTPUT_DIR, args.video)):
        raise SystemExit(f"ERROR: Output directory not found for video '{args.video}'")
    run_final_review(args.video)