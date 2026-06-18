"""
validate_nuscenes_phaseB.py  —  SPEED VALIDATION  (with straight-line filter)

Tests the actual speed path: project -> RTS smooth -> velocity, against
nuScenes ground-truth relative velocity. Calibration injected per frame.
Association via instance_token (best-case, no tracking error).

Speed is computed by differencing adjacent positions (forward = dy/dt,
lateral = dx/dt) — identical to estimate_relative_velocity's math, without
the internal-state hazard when frames are skipped.

 a STRAIGHT-LINE subset isolates pairs where BOTH the object and the ego
have low yaw rate (no turning). This matches highway / Autobahn conditions and
removes the urban-intersection artifacts (turning vehicles swing the near-face
reference; ego turns inject apparent lateral motion).


 python3 validate_nuscenes_phaseB.py
"""

import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from homography import HomographyEstimator
from smoother import RTSSmoother

# ------------------------------------------------------------------ config ---
NUSCENES_ROOT = os.path.expanduser("~/Downloads/v1.0-mini")
NUSCENES_VER  = "v1.0-mini"
CAM           = "CAM_FRONT"
MIN_VISIBILITY = 2
MAX_PAIR_DIST  = 40.0
MAX_DT         = 1.5
STRAIGHT_YAW_RATE_DEG = 2.0   # deg/s; pair is "straight" if obj AND ego below this

SAVE_IMAGES = True
IMG_DIR     = "validation_vs_nuscenes/phaseB"   # speed measured vs ground truth, per image

VEHICLE_CATS = {
    "vehicle.car", "vehicle.truck", "vehicle.bus.rigid",
    "vehicle.bus.bendy", "vehicle.construction", "vehicle.trailer"
}


def horizon_row_from_pitch(R_ego_to_cam, fy, cy):
    d_cam = R_ego_to_cam @ np.array([1.0, 0.0, 0.0])
    if d_cam[2] <= 1e-6:
        return cy
    return cy + fy * (d_cam[1] / d_cam[2])


def ang_diff(a, b):
    """smallest signed difference a-b wrapped to [-pi, pi]"""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ main ----
print(f"Loading nuScenes {NUSCENES_VER} ...")
nusc = NuScenes(version=NUSCENES_VER, dataroot=NUSCENES_ROOT, verbose=False)

h = HomographyEstimator()
if not hasattr(h, "override_focal_px"):
    raise SystemExit("homography.py not patched — see Phase A Edit 1 & 2.")

instances = {}

for scene in nusc.scene:
    stk = scene["first_sample_token"]
    while stk:
        sample = nusc.get("sample", stk)
        if CAM not in sample["data"]:
            stk = sample["next"]; continue

        sd_token = sample["data"][CAM]
        sd = nusc.get("sample_data", sd_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])

        K = np.array(cs["camera_intrinsic"])
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        cam_height = float(cs["translation"][2])
        R_ec = Quaternion(cs["rotation"]).inverse.rotation_matrix

        img_path = os.path.join(NUSCENES_ROOT, sd["filename"])
        img = cv2.imread(img_path)
        if img is None:
            stk = sample["next"]; continue
        ih, iw = img.shape[:2]

        ts_sec   = sample["timestamp"] * 1e-6
        ego_pos  = np.array(ep["translation"])
        ego_yaw  = Quaternion(ep["rotation"]).yaw_pitch_roll[0]

        h.override_focal_px    = fx
        h.override_cx          = cx
        h.override_horizon_row = horizon_row_from_pitch(R_ec, fy, cy)
        h.camera_height        = cam_height

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]
            if not any(cat.startswith(v) for v in VEHICLE_CATS):
                continue
            if int(ann["visibility_token"]) < MIN_VISIBILITY:
                continue
            if ann.get("num_lidar_pts", 0) <= 0:
                continue

            box = nusc.get_box(ann_token)             # global frame
            obj_yaw = box.orientation.yaw_pitch_roll[0]   # global heading

            # global -> ego -> camera
            box.translate(-np.array(ep["translation"]))
            box.rotate(Quaternion(ep["rotation"]).inverse)
            box.translate(-np.array(cs["translation"]))
            box.rotate(Quaternion(cs["rotation"]).inverse)

            gt_z = float(box.center[2])
            gt_x = float(box.center[0])
            if gt_z <= 0 or gt_z > 60:
                continue

            corners = box.corners()
            if np.any(corners[2] <= 0):
                continue

            pts = view_points(corners, K, normalize=True)
            x1 = int(max(0, np.floor(pts[0].min())))
            y1 = int(max(0, np.floor(pts[1].min())))
            x2 = int(min(iw - 1, np.ceil(pts[0].max())))
            y2 = int(min(ih - 1, np.ceil(pts[1].max())))
            if x2 <= x1 or y2 <= y1:
                continue

            est_x, est_y, reliable = h.get_vehicle_position(
                [x1, y1, x2, y2], cat, iw, ih, lane_info=None
            )

            v_glob = nusc.box_velocity(ann_token)
            obj_abs = (float(np.hypot(v_glob[0], v_glob[1]))
                       if not np.any(np.isnan(v_glob[:2])) else np.nan)

            iid = ann["instance_token"]
            instances.setdefault(iid, []).append({
                "ts": ts_sec, "est_x": est_x, "est_y": est_y,
                "reliable": reliable, "gt_x": gt_x, "gt_z": gt_z,
                "ego_pos": ego_pos, "ego_yaw": ego_yaw, "obj_yaw": obj_yaw,
                "obj_abs": obj_abs,
                "img_path": img_path, "bbox": [x1, y1, x2, y2],
                "scene_name": scene["name"], "stamp": sd["timestamp"],
            })

        stk = sample["next"]

def build_pairs_for_rate(instances, decimate):
    """decimate=1 -> native 2Hz; decimate=2 -> every other keyframe (~1Hz).
    Returns list of pair dicts. Smoothing is run per-rate on the decimated
    trajectory so each rate gets its own fair smoothing pass."""
    # decimate per instance
    dec = {}
    for iid, recs in instances.items():
        dec[iid] = recs[::decimate]

    sm = RTSSmoother()
    track_obs = {
        iid: [(r["ts"], r["est_x"], r["est_y"], r["reliable"]) for r in recs]
        for iid, recs in dec.items()
    }
    smoothed = sm.smooth(track_obs)

    out_pairs = []
    for iid, recs in dec.items():
        if len(recs) < 2:
            continue
        sm_map = smoothed.get(iid, {})
        for k in range(1, len(recs)):
            r0, r1 = recs[k - 1], recs[k]
            dt = r1["ts"] - r0["ts"]
            if dt <= 0 or dt > MAX_DT * decimate:
                continue
            if not (r0["reliable"] and r1["reliable"]):
                continue
            if r1["gt_z"] > MAX_PAIR_DIST:
                continue

            gt_fwd = (r1["gt_z"] - r0["gt_z"]) / dt
            gt_lat = (r1["gt_x"] - r0["gt_x"]) / dt
            fwd_raw = (r1["est_y"] - r0["est_y"]) / dt
            lat_raw = (r1["est_x"] - r0["est_x"]) / dt
            xs0, ys0 = sm_map.get(r0["ts"], (r0["est_x"], r0["est_y"]))
            xs1, ys1 = sm_map.get(r1["ts"], (r1["est_x"], r1["est_y"]))
            fwd_smo = (ys1 - ys0) / dt
            lat_smo = (xs1 - xs0) / dt

            obj_rate = abs(np.degrees(ang_diff(r1["obj_yaw"], r0["obj_yaw"])) / dt)
            ego_rate = abs(np.degrees(ang_diff(r1["ego_yaw"], r0["ego_yaw"])) / dt)
            is_straight = (obj_rate < STRAIGHT_YAW_RATE_DEG and
                           ego_rate < STRAIGHT_YAW_RATE_DEG)
            ego_v = np.linalg.norm(r1["ego_pos"][:2] - r0["ego_pos"][:2]) / dt

            out_pairs.append({
                "ef_raw": fwd_raw - gt_fwd, "el_raw": lat_raw - gt_lat,
                "ef_smo": fwd_smo - gt_fwd, "el_smo": lat_smo - gt_lat,
                "gt_z": r1["gt_z"], "straight": is_straight,
                "ego_v": ego_v, "obj_abs": r1["obj_abs"],
            })
    return out_pairs


# ------------------------------------------------------------ RTS smoothing
sm = RTSSmoother()
track_obs = {
    iid: [(r["ts"], r["est_x"], r["est_y"], r["reliable"]) for r in recs]
    for iid, recs in instances.items()
}
smoothed = sm.smooth(track_obs)

# ------------------------------------------------------------ build pairs
pairs = []
draw_by_frame = {}   # (scene_name, stamp) -> {"img": path, "items": [...]}
for iid, recs in instances.items():
    if len(recs) < 2:
        continue
    sm_map = smoothed.get(iid, {})
    for k in range(1, len(recs)):
        r0, r1 = recs[k - 1], recs[k]
        dt = r1["ts"] - r0["ts"]
        if dt <= 0 or dt > MAX_DT:
            continue
        if not (r0["reliable"] and r1["reliable"]):
            continue
        if r1["gt_z"] > MAX_PAIR_DIST:
            continue

        gt_fwd = (r1["gt_z"] - r0["gt_z"]) / dt
        gt_lat = (r1["gt_x"] - r0["gt_x"]) / dt

        # raw est speed = direct difference of adjacent estimated positions
        fwd_raw = (r1["est_y"] - r0["est_y"]) / dt
        lat_raw = (r1["est_x"] - r0["est_x"]) / dt

        xs0, ys0 = sm_map.get(r0["ts"], (r0["est_x"], r0["est_y"]))
        xs1, ys1 = sm_map.get(r1["ts"], (r1["est_x"], r1["est_y"]))
        fwd_smo = (ys1 - ys0) / dt
        lat_smo = (xs1 - xs0) / dt

        obj_rate = abs(np.degrees(ang_diff(r1["obj_yaw"], r0["obj_yaw"])) / dt)
        ego_rate = abs(np.degrees(ang_diff(r1["ego_yaw"], r0["ego_yaw"])) / dt)
        is_straight = (obj_rate < STRAIGHT_YAW_RATE_DEG and
                       ego_rate < STRAIGHT_YAW_RATE_DEG)

        ego_v = np.linalg.norm(r1["ego_pos"][:2] - r0["ego_pos"][:2]) / dt

        pairs.append({
            "ef_raw": fwd_raw - gt_fwd, "el_raw": lat_raw - gt_lat,
            "ef_smo": fwd_smo - gt_fwd, "el_smo": lat_smo - gt_lat,
            "gt_z": r1["gt_z"], "straight": is_straight,
            "ego_v": ego_v, "obj_abs": r1["obj_abs"],
        })

        if SAVE_IMAGES:
            key = (r1["scene_name"], r1["stamp"])
            entry = draw_by_frame.setdefault(
                key, {"img": r1["img_path"], "items": []})
            entry["items"].append({
                "bbox": r1["bbox"],
                "fwd_e": fwd_smo, "fwd_g": gt_fwd,
                "lat_e": lat_smo, "lat_g": gt_lat,
            })

# ------------------------------------------------------------ draw speed images
if SAVE_IMAGES:
    os.makedirs(IMG_DIR, exist_ok=True)
    for f in os.listdir(IMG_DIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(IMG_DIR, f))

    def speed_col(lat_err):
        a = abs(lat_err)
        if a < 0.3: return (0, 200, 0)
        if a < 0.7: return (0, 200, 220)
        return (0, 0, 230)

    for (scene_name, stamp), entry in draw_by_frame.items():
        img = cv2.imread(entry["img"])
        if img is None:
            continue
        for it in entry["items"]:
            x1, y1, x2, y2 = it["bbox"]
            col = speed_col(it["lat_e"] - it["lat_g"])
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            lines = [
                f"vY E{it['fwd_e']:+.1f} G{it['fwd_g']:+.1f}",
                f"vX E{it['lat_e']:+.2f} G{it['lat_g']:+.2f}",
            ]
            for i, ln in enumerate(lines):
                yy = y1 - 4 - (len(lines) - 1 - i) * 13
                if yy < 12:
                    yy = y2 + 14 + i * 13
                cv2.putText(img, ln, (x1, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
        cv2.putText(img, f"{scene_name}  PHASE B: vY=forward vX=lateral  "
                         f"E=estimated G=ground truth (m/s, smoothed)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(IMG_DIR, f"{scene_name}_{stamp}.jpg"), img)


def st(arr):
    a = np.array(arr)
    return a.mean(), np.abs(a).mean(), np.sqrt((a**2).mean())


def pct_within(errors, thresholds):
    a = np.abs(np.asarray(errors))
    n = len(a)
    if n == 0:
        return [(t, float("nan")) for t in thresholds]
    return [(t, 100.0 * np.sum(a <= t) / n) for t in thresholds]


def fmt_pct(pairs):
    return "  ".join(f"<={t:g} {p:5.1f}%" for t, p in pairs)


# thresholds chosen around the actual decision the annotator makes:
# 0.5 m/s is YOUR yielding threshold, so "% within 0.5" answers directly
# "how often does noise alone stay under the line that decides yielded?"
LAT_THRESHOLDS = [0.1, 0.25, 0.5]   # m/s
FWD_THRESHOLDS = [0.5, 1.0, 2.0]    # m/s


def report(subset, label):
    if not subset:
        print(f"\n[{label}] no pairs"); return
    print(f"\n[{label}]  n={len(subset)}")
    for fld, name, thr in [("ef_raw","FWD raw ", FWD_THRESHOLDS),
                            ("ef_smo","FWD smo ", FWD_THRESHOLDS),
                            ("el_raw","LAT raw ", LAT_THRESHOLDS),
                            ("el_smo","LAT smo ", LAT_THRESHOLDS)]:
        vals = [p[fld] for p in subset]
        m, a, r = st(vals)
        print(f"    {name}  mean {m:+7.3f}  MAE {a:6.3f}  RMSE {r:6.3f}  m/s"
              f"   within: {fmt_pct(pct_within(vals, thr))}")


print(f"\n{'='*62}")
print("PHASE B — speed validation (relative to ego)")
print(f"Straight = obj & ego yaw rate < {STRAIGHT_YAW_RATE_DEG} deg/s")
print(f"{'='*62}")

if not pairs:
    raise SystemExit("No valid speed pairs — check dataset path.")

report(pairs, "ALL reliable pairs")
straight = [p for p in pairs if p["straight"]]
report(straight, "STRAIGHT-LINE subset (highway-like)")

print(f"\n  LATERAL-SPEED MAE: all={st([p['el_smo'] for p in pairs])[1]:.3f}"
      f"  straight={st([p['el_smo'] for p in straight])[1]:.3f}  m/s  (smoothed)")
straight_lat = [p["el_smo"] for p in straight]
pw05 = dict(pct_within(straight_lat, [0.5]))[0.5]
print(f"  -> on the straight subset, {pw05:.1f}% of lateral-speed noise stays "
      f"AT OR UNDER the 0.5 m/s yielding threshold")
print(f"     ({100-pw05:.1f}% would exceed 0.5 m/s from noise alone on a "
      f"non-yielding vehicle)")

# distance buckets on the straight subset, smoothed forward
print(f"\n  STRAIGHT-ONLY forward MAE by distance (smoothed, m/s)")
for lo, hi in [(0,10),(10,20),(20,30),(30,40)]:
    b = [p["ef_smo"] for p in straight if lo <= p["gt_z"] < hi]
    if b:
        print(f"    {lo:>2}-{hi:<3}m  n={len(b):>4}  MAE {np.abs(b).mean():.3f}")

es = [p["ego_v"] for p in straight]
oa = [p["obj_abs"] for p in straight if not np.isnan(p["obj_abs"])]
print(f"\n  ABSOLUTE side-check (straight subset)")
if es: print(f"    mean ego speed            : {np.mean(es):5.2f} m/s ({np.mean(es)*3.6:.1f} km/h)")
if oa: print(f"    mean object absolute speed: {np.mean(oa):5.2f} m/s ({np.mean(oa)*3.6:.1f} km/h)")

print(f"\n{'='*62}")
print("1 Hz vs 2 Hz  —  does herz pick  change lateral noise?")
print(f"{'='*62}")
p2 = build_pairs_for_rate(instances, decimate=1)   # native 2Hz
p1 = build_pairs_for_rate(instances, decimate=2)   # ~1Hz
for label, pp in [("2 Hz (0.5s step)", p2), ("1 Hz (1.0s step)", p1)]:
    s = [x for x in pp if x["straight"]]
    if not s:
        continue
    lat = st([x["el_smo"] for x in s])[1]
    fwd = st([x["ef_smo"] for x in s])[1]
    print(f"  {label}:  straight LAT MAE {lat:.3f}   FWD MAE {fwd:.3f}   "
          f"(n={len(s)})  m/s")

print("  If 1Hz LAT MAE is lower, switching to 2Hz would HURT the signal.")


