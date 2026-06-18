"""
validate_nuscenes_phaseA.py  —  POSITION VALIDATION

Tests the ACTUAL HomographyEstimator.get_vehicle_position() from homography.py
against nuScenes ground truth, with ALL calibration replaced by nuScenes'
real per-frame values so the only thing under test is the projection method.



Ground truth target:
  PRIMARY  = near-bottom-centre of the 3D box (what the bottom-pixel method
             physically ranges to)
  SECONDARY= 3D box centre (standard literature comparison)

Requires the two-line patch to homography.py adding:
  self.override_focal_px / self.override_cx / self.override_horizon_row

python3 validate_nuscenes_phaseA.py
"""

import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from homography import HomographyEstimator

# ------------------------------------------------------------------ config ---
NUSCENES_ROOT = os.path.expanduser("~/Downloads/v1.0-mini")
NUSCENES_VER  = "v1.0-mini"
CAM           = "CAM_FRONT"
MAX_GT_DIST   = 60.0
MIN_VISIBILITY = 2     # nuScenes token: 1=0-40%, 2=40-60%, 3=60-80%, 4=80-100%

SAVE_IMAGES = True
IMG_DIR     = "validation_vs_nuscenes/phaseA"   # measured vs ground truth, per image

VEHICLE_CATS = {
    "vehicle.car", "vehicle.truck", "vehicle.bus.rigid",
    "vehicle.bus.bendy", "vehicle.construction", "vehicle.trailer"
}

# ------------------------------------------------------------------ helpers --

def horizon_row_from_pitch(R_ego_to_cam, fy, cy):
    """True image row of the horizon = vanishing row of the ego-forward
    direction projected through the real intrinsics."""
    fwd_ego = np.array([1.0, 0.0, 0.0])     # nuScenes ego x = forward
    d_cam = R_ego_to_cam @ fwd_ego
    if d_cam[2] <= 1e-6:
        return cy
    return cy + fy * (d_cam[1] / d_cam[2])


def near_bottom_centre(corners_cam):
    """corners_cam: 3x8 in camera frame. Returns (x, y_dummy, z) of the
    midpoint of the nearest bottom edge — what the bottom-centre pixel
    physically projects to."""
    x, y, z = corners_cam[0], corners_cam[1], corners_cam[2]
    # camera frame y is DOWN -> bottom corners have the largest y
    bottom_idx = np.argsort(y)[-4:]
    # among the bottom corners, the nearest two have the smallest z (depth)
    nb = bottom_idx[np.argsort(z[bottom_idx])[:2]]
    return float(x[nb].mean()), float(z[nb].mean())


# ------------------------------------------------------------------ main ----
print(f"Loading nuScenes {NUSCENES_VER} ...")
nusc = NuScenes(version=NUSCENES_VER, dataroot=NUSCENES_ROOT, verbose=False)

# real HomographyEstimator — the code actually under test
h = HomographyEstimator()
if not hasattr(h, "override_focal_px"):
    raise SystemExit(
        "homography.py is not patched. Add override_focal_px / override_cx / "
        "override_horizon_row (see Edit 1 & 2)."
    )

rows = []          # one dict per observation
n_clamped = 0
n_total   = 0

if SAVE_IMAGES:
    os.makedirs(IMG_DIR, exist_ok=True)
    for f in os.listdir(IMG_DIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(IMG_DIR, f))


def err_colour(fwd_err):
    """green if |err|<2m, yellow <5m, red otherwise (BGR)."""
    a = abs(fwd_err)
    if a < 2.0:  return (0, 200, 0)
    if a < 5.0:  return (0, 200, 220)
    return (0, 0, 230)


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

        K  = np.array(cs["camera_intrinsic"])
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        cam_height = float(cs["translation"][2])

        R_ec = Quaternion(cs["rotation"]).inverse.rotation_matrix
        horizon_row = horizon_row_from_pitch(R_ec, fy, cy)

        img_path = os.path.join(NUSCENES_ROOT, sd["filename"])
        img = cv2.imread(img_path)
        if img is None:
            stk = sample["next"]; continue
        ih, iw = img.shape[:2]

        # inject real calibration into the real estimator
        h.override_focal_px   = fx
        h.override_cx         = cx
        h.override_horizon_row = horizon_row
        h.camera_height       = cam_height

        vis = img.copy() if SAVE_IMAGES else None
        drew_any = False

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]
            if not any(cat.startswith(v) for v in VEHICLE_CATS):
                continue
            if int(ann["visibility_token"]) < MIN_VISIBILITY:
                continue
            if ann.get("num_lidar_pts", 0) <= 0:
                continue

            box = nusc.get_box(ann_token)
            box.translate(-np.array(ep["translation"]))
            box.rotate(Quaternion(ep["rotation"]).inverse)
            box.translate(-np.array(cs["translation"]))
            box.rotate(Quaternion(cs["rotation"]).inverse)

            gt_fwd_centre = float(box.center[2])  # used only for the early
            gt_lat_centre = float(box.center[0])  # distance-range filter below
            if gt_fwd_centre <= 0 or gt_fwd_centre > MAX_GT_DIST:
                continue

            corners = box.corners()                 # 3x8 in camera frame
            if np.any(corners[2] <= 0):
                continue

            gt_lat_face, gt_fwd_face = near_bottom_centre(corners)

            # 2D bbox from projected corners, clipped to frame like YOLO
            pts = view_points(corners, K, normalize=True)
            x1 = int(max(0, np.floor(pts[0].min())))
            y1 = int(max(0, np.floor(pts[1].min())))
            x2 = int(min(iw - 1, np.ceil(pts[0].max())))
            y2 = int(min(ih - 1, np.ceil(pts[1].max())))
            if x2 <= x1 or y2 <= y1:
                continue

            n_total += 1
            est_x, est_y, reliable = h.get_vehicle_position(
                [x1, y1, x2, y2], cat, iw, ih, lane_info=None
            )
            if not reliable:
                n_clamped += 1

            rows.append({
                "est_x": est_x, "est_y": est_y, "reliable": reliable,
                "gt_fwd_face": gt_fwd_face, "gt_lat_face": gt_lat_face,
                "cat": cat,
            })

            if SAVE_IMAGES:
                drew_any = True
                fwd_err = est_y - gt_fwd_face
                col = err_colour(fwd_err)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                cv2.circle(vis, (int((x1 + x2) / 2), y2), 4, col, -1)
                tag = "" if reliable else " (unrel)"
                lines = [
                    f"E y={est_y:.1f} x={est_x:+.1f}{tag}",
                    f"G y={gt_fwd_face:.1f} x={gt_lat_face:+.1f}",
                    f"err y={fwd_err:+.1f}",
                ]
                for i, ln in enumerate(lines):
                    yy = y1 - 4 - (len(lines) - 1 - i) * 13
                    if yy < 12:
                        yy = y2 + 14 + i * 13
                    cv2.putText(vis, ln, (x1, yy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

        if SAVE_IMAGES and drew_any:
            cv2.putText(vis, f"{scene['name']}  PHASE A: E=estimated  G=ground truth (m)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            out = os.path.join(IMG_DIR, f"{scene['name']}_{sd['timestamp']}.jpg")
            cv2.imwrite(out, vis)

        stk = sample["next"]

# ------------------------------------------------------------------ report --
if not rows:
    raise SystemExit("No observations collected — check the dataset path.")


def pct_within(errors, thresholds):
    """Returns [(threshold, pct_of_observations_within_it), ...]."""
    a = np.abs(np.asarray(errors))
    n = len(a)
    if n == 0:
        return [(t, float("nan")) for t in thresholds]
    return [(t, 100.0 * np.sum(a <= t) / n) for t in thresholds]


def fmt_pct(pairs, unit="m"):
    return "  ".join(f"<={t:g}{unit} {p:5.1f}%" for t, p in pairs)


FORWARD_THRESHOLDS = [1, 2, 5]   # metres
LATERAL_THRESHOLDS = [0.5, 1, 2]  # metres


def report(subset, label):
    if not subset:
        print(f"\n[{label}] no rows"); return
    eff = np.array([r["est_y"] - r["gt_fwd_face"]   for r in subset])
    elf = np.array([r["est_x"] - r["gt_lat_face"]   for r in subset])
    print(f"\n[{label}]  n={len(subset)}")
    print(f"  FORWARD vs near-face   mean {eff.mean():+6.2f}  "
          f"MAE {np.abs(eff).mean():5.2f}  RMSE {np.sqrt((eff**2).mean()):5.2f} m")
    print(f"    within: {fmt_pct(pct_within(eff, FORWARD_THRESHOLDS))}")
    print(f"  LATERAL vs near-face   mean {elf.mean():+6.2f}  "
          f"MAE {np.abs(elf).mean():5.2f}  RMSE {np.sqrt((elf**2).mean()):5.2f} m")
    print(f"    within: {fmt_pct(pct_within(elf, LATERAL_THRESHOLDS))}")


print(f"\n{'='*60}")
print(f"PHASE A — position validation (real get_vehicle_position)")
print(f"nuScenes calibration injected; horizon from real camera pitch")
print(f"{'='*60}")
print(f"\nTotal usable observations: {n_total}")
print(f"Flagged unreliable (clip/clamp): {n_clamped} "
      f"({100*n_clamped/max(n_total,1):.1f}%)")
print("  -> box touches a side/bottom frame edge, or the lateral-clamp "
      "fired (position physically implausible)")

report(rows, "ALL observations")
report([r for r in rows if r["reliable"]], "RELIABLE only")

# forward (near-face) error by distance bucket, reliable only
rel = [r for r in rows if r["reliable"]]
print(f"\nFORWARD (near-face) ERROR BY DISTANCE — reliable only")
print(f"  {'Range':<10} {'N':>4} {'Mean':>8} {'MAE':>7}   {'<=1m':>6} {'<=2m':>6} {'<=5m':>6}")
for lo, hi in [(0,10),(10,20),(20,30),(30,45),(45,60)]:
    b = np.array([r["est_y"] - r["gt_fwd_face"]
                  for r in rel if lo <= r["gt_fwd_face"] < hi])
    if len(b) == 0:
        continue
    pw = dict(pct_within(b, FORWARD_THRESHOLDS))
    print(f"  {lo:>2}-{hi:<3}m   {len(b):>4} {b.mean():>+7.2f}m {np.abs(b).mean():>6.2f}m   "
          f"{pw[1]:>5.1f}% {pw[2]:>5.1f}% {pw[5]:>5.1f}%")


print()