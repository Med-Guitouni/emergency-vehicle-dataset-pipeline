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
import random
import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from homography import HomographyEstimator

# ------------------------------------------------------------------ config ---
KITTI_DIR = "/Users/omeryasar/Downloads/training"
LABEL_DIR = os.path.join(KITTI_DIR, "label_2")
IMAGE_DIR = os.path.join(KITTI_DIR, "image_2")
IMG_DIR     = "validation_vs_nuscenes/phaseA" 
CALIB_DIR = os.path.join(KITTI_DIR, "calib")
OUT_DIR   = "./kitti_eval_results"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Hyperparameters ---
VEHICLE_CATS = ["Car", "Van", "Truck", "Bus"]
MAX_GT_DIST  = 60.0  # Max distance to evaluate (meters)

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

# real HomographyEstimator — the code actually under test
h_estimator = HomographyEstimator()
if not hasattr(h_estimator, "override_focal_px"):
    raise SystemExit(
        "homography.py is not patched. Add override_focal_px / override_cx / "
        "override_horizon_row (see Edit 1 & 2)."
    )


rows = []          # one dict per observation
n_clamped = 0
n_total   = 0

if True:
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

def load_kitti_calib_try(calib_path):
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line: continue
            key, val = line.split(':', 1)
            float_list = []
            for token in val.strip().split():
                try: float_list.append(float(token))
                except ValueError: pass
            calib[key.strip()] = np.array(float_list, dtype=np.float64)
    P2 = calib["P2"][:12].reshape(3, 4)
    return P2

def load_kitti_calib(calib_path):
    """Parses P2 and R0_rect robustly from a standard KITTI calib text file."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line: 
                continue
            
            key, val = line.split(':', 1)
            key = key.strip()
            
            # Extract only valid numeric tokens to prevent inhomogeneous array shapes
            float_list = []
            for token in val.strip().split():
                try:
                    float_list.append(float(token))
                except ValueError:
                    pass  # Safely ignore trailing non-numeric characters
            
            calib[key] = np.array(float_list, dtype=np.float64)
    
    # Slice exactly the required number of elements to guarantee precise shapes
    P2 = calib["P2"][:12].reshape(3, 4)
    R0_rect = calib["R0_rect"][:9].reshape(3, 3)
    
    return P2, R0_rect

def compute_3d_corners(h, w, l, x, y, z, rotation_y):
    """Computes the 8 corners of a 3D bounding box in KITTI Cam2 space."""
    # Rotation matrix around Y-axis
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    # 3D corners template relative to object base-center
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners_3d)
    
    # Shift to actual 3D location
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    return corners_3d # 3x8 matrix

def get_error_color(err_meters):
    """Returns Green for accurate estimations, turning Red as error increases."""
    abs_err = abs(err_meters)
    if abs_err < 1.0:   return (0, 255, 0)    # Green (<1m error)
    if abs_err < 3.0:   return (0, 165, 255)  # Orange (1m - 3m error)
    return (0, 0, 255)                        # Red (>3m error)

def visualize_estimation_vs_gt_2d(kitti_dir, h_estimator, num_samples=3):
    label_dir = os.path.join(kitti_dir, "label_2")
    image_dir = os.path.join(kitti_dir, "image_2")
    calib_dir = os.path.join(kitti_dir, "calib")
    
    all_indices = [f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
    sampled_indices = random.sample(all_indices, min(num_samples, len(all_indices)))
    
    for idx in sampled_indices:
        img = cv2.imread(os.path.join(image_dir, f"{idx}.png"))
        if img is None: continue
        ih, iw = img.shape[:2]
        P2 = load_kitti_calib_try(os.path.join(calib_dir, f"{idx}.txt"))
        
        # Static baseline parameter initialization
        h_estimator.override_focal_px    = P2[0, 0]
        h_estimator.override_cx          = P2[0, 2]
        h_estimator.override_horizon_row = P2[1, 2] 
        h_estimator.camera_height        = 1.65 

        with open(os.path.join(label_dir, f"{idx}.txt"), 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            data = line.strip().split(' ')
            cat = data[0]
            if cat not in ["Car", "Van", "Truck"]: continue
                
            h, w, l = float(data[8]), float(data[9]), float(data[10])
            x, y, z = float(data[11]), float(data[12]), float(data[13])
            rotation_y = float(data[14])
            
            # --- EXTRACTING GROUND TRUTH POSITION (X and Y) ---
            # In KITTI label format: Z is depth forward, X is horizontal displacement
            gt_fwd_face = z - (l / 2)  # Ground Truth Longitudinal (Forward Distance Y)
            gt_lat_face = x            # Ground Truth Lateral (Side Displacement X)
            
            if z <= 0 or z > 50.0: continue
            
            # Generate 2D bounding limits from 3D projections
            corners = compute_3d_corners(h, w, l, x, y, z, rotation_y)
            corners_homo = np.vstack((corners, np.ones((1, 8))))
            pts_2d = np.dot(P2, corners_homo)
            pts_2d[:2, :] /= pts_2d[2, :]
            
            x1 = int(max(0, np.floor(pts_2d[0].min())))
            y1 = int(max(0, np.floor(pts_2d[1].min())))
            x2 = int(min(iw - 1, np.ceil(pts_2d[0].max())))
            y2 = int(min(ih - 1, np.ceil(pts_2d[1].max())))
            if x2 <= x1 or y2 <= y1: continue
            
            # --- RUN ALGORITHM ESTIMATION ---
            # Returns estimated lateral position (est_x) and longitudinal position (est_y)
            est_x, est_y, reliable = h_estimator.get_vehicle_position(
                [x1, y1, x2, y2], cat, iw, ih, lane_info=None
            )
            
            # --- MEASURING METRICS (Focusing solely on Y for error calculations) ---
            err_y = est_y - gt_fwd_face
            color = get_error_color(err_y)
            
            # Draw standard 2D detection box and its base midpoint
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(img, (int((x1 + x2) / 2), y2), 4, (255, 255, 255), -1)
            
            # Display comparison overlays showing both X and Y components
            lines_to_print = [
                f"Est -> X:{est_x:+.1f}m, Y:{est_y:.1f}m",
                f"GT  -> X:{gt_lat_face:+.1f}m, Y:{gt_fwd_face:.1f}m",
                f"Y-Err: {err_y:+.2f}m"
            ]
            
            for i, text_line in enumerate(lines_to_print):
                # Adaptive text positioning to prevent drawing text outside image limits
                y_pos = y1 - 6 - (i * 12) if (y1 - 40) > 15 else y2 + 15 + (i * 12)
                cv2.putText(img, text_line, (x1, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
                
        # Open output frame
        cv2.imshow(f"Estimation vs GT Performance (Frame {idx})", img)
        if cv2.waitKey(0) & 0xFF == ord('q'): 
            break
            
    cv2.destroyAllWindows()
'''visualize_estimation_vs_gt_2d(KITTI_DIR, h_estimator)
exit(0)  # End of script after visualization'''
def visualize_random_kitti_samples(kitti_dir, num_samples=3):
    label_dir = os.path.join(kitti_dir, "label_2")
    image_dir = os.path.join(kitti_dir, "image_2")
    calib_dir = os.path.join(kitti_dir, "calib")
    
    # Randomly select a few distinct frames
    all_indices = [f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
    sampled_indices = random.sample(all_indices, min(num_samples, len(all_indices)))
    
    # Edge sequences to trace individual cuboid faces
    box_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face boundary
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face boundary
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical structural lines
    ]
    
    for idx in sampled_indices:
        img_path = os.path.join(image_dir, f"{idx}.png")
        label_path = os.path.join(label_dir, f"{idx}.txt")
        calib_path = os.path.join(calib_dir, f"{idx}.txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
        P2 = load_kitti_calib(calib_path)
        
        print(f"\n--- Visualizing Frame ID: {idx} ---")
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            data = line.strip().split(' ')
            cat = data[0]
            if cat not in ["Car", "Van", "Truck", "Pedestrian", "Cyclist"]: continue
                
            # 3D Variables
            h, w, l = float(data[8]), float(data[9]), float(data[10])
            x, y, z = float(data[11]), float(data[12]), float(data[13])
            rotation_y = float(data[14])
            
            # Geometry calculations
            gt_fwd_face = z - (l / 2) # Closest front plane surface
            
            # Extract corners and project onto camera vector space
            corners = compute_3d_corners(h, w, l, x, y, z, rotation_y)
            corners_homo = np.vstack((corners, np.ones((1, 8))))
            pts_2d = np.dot(P2, corners_homo)
            pts_2d[:2, :] /= pts_2d[2, :]
            pts_2d = pts_2d[:2, :].astype(int)
            
            # Draw wireframe geometry lines
            for edge in box_edges:
                pt1 = tuple(pts_2d[:, edge[0]])
                pt2 = tuple(pts_2d[:, edge[1]])
                cv2.line(img, pt1, pt2, (0, 255, 0), 2) # Green wireframe
                
            # Mark the ground footprint center point
            center_pixel = np.dot(P2, np.array([x, y, z, 1.0]))
            center_pixel /= center_pixel[2]
            cx_p, cy_p = int(center_pixel[0]), int(center_pixel[1])
            cv2.circle(img, (cx_p, cy_p), 5, (0, 0, 255), -1) # Red marker for GT origin
            
            # Overlay calculated tracking text metadata
            text = f"{cat} Z_center={z:.1f}m | FrontFace={gt_fwd_face:.1f}m"
            cv2.putText(img, text, (cx_p - 40, cy_p - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            
            print(f"[{cat}] Center Coordinates: X={x:+.2f}, Y={y:+.2f}, Z={z:.2f}m | Scaled Front Depth: {gt_fwd_face:.2f}m")
            
        # Display window (Press any key to load next frame)
        cv2.imshow(f"KITTI GT Debug Verification - Frame {idx}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#visualize_random_kitti_samples(KITTI_DIR, num_samples=3)



# --- Main Processing Loop ---
frame_indices = sorted([f.split('.')[0] for f in os.listdir(LABEL_DIR) if f.endswith('.txt')])
rows = []

for idx in frame_indices:
    # 1. Load Calibration
    P2, R0_rect = load_kitti_calib(os.path.join(CALIB_DIR, f"{idx}.txt"))
    
    # Extract Extrinsics & Intrinsics for your homography class
    fx, fy = P2[0, 0], P2[1, 1]
    cx, cy = P2[0, 2], P2[1, 2]
    
    # Static camera height approximation for KITTI platform or calculate pitch
    cam_height = 1.65 
    horizon_row = cy  # Simple center assumption, substitute with pitch calculation if active
    
    # Inject parameters directly into your estimator
    h_estimator.override_focal_px    = fx
    h_estimator.override_cx          = cx
    h_estimator.override_horizon_row = horizon_row
    h_estimator.camera_height        = cam_height
    
    # 2. Load Visual Frame
    img_path = os.path.join(IMAGE_DIR, f"{idx}.png")
    img = cv2.imread(img_path)
    if img is None: continue
    ih, iw = img.shape[:2]
    
    # 3. Process Scene Labels
    with open(os.path.join(LABEL_DIR, f"{idx}.txt"), 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        data = line.strip().split(' ')
        cat = data[0]
        if cat not in VEHICLE_CATS: continue
            
        # Parse Sizes & 3D Center Locations
        h, w, l = float(data[8]), float(data[9]), float(data[10])
        x, y, z = float(data[11]), float(data[12]), float(data[13])
        rotation_y = float(data[14])
        
        # Ground-Truth filter adjustments (KITTI Z coordinate is forward distance)
        gt_fwd_face = z - (l / 2) # Calculate the closest forward-facing edge
        gt_lat_face = x
        
        if z <= 0 or z > MAX_GT_DIST: continue
            
        # Generate 3D box points and project onto pixel array
        corners = compute_3d_corners(h, w, l, x, y, z, rotation_y)
        
        # Project 3D points using Intrinsic P2 Matrix
        corners_homo = np.vstack((corners, np.ones((1, 8))))
        pts_2d = np.dot(P2, corners_homo)
        pts_2d[:2, :] /= pts_2d[2, :] # Normalize by Z
        
        # Create tight 2D box bounds matching detection output
        x1 = int(max(0, np.floor(pts_2d[0].min())))
        y1 = int(max(0, np.floor(pts_2d[1].min())))
        x2 = int(min(iw - 1, np.ceil(pts_2d[0].max())))
        y2 = int(min(ih - 1, np.ceil(pts_2d[1].max())))
        if x2 <= x1 or y2 <= y1: continue
            
        # 4. Generate Homography Distance Estimate
        est_x, est_y, reliable = h_estimator.get_vehicle_position(
            [x1, y1, x2, y2], cat, iw, ih, lane_info=None
        )
        
        rows.append({
            "frame_id": idx, "cat": cat, "reliable": reliable,
            "est_y": est_y, "gt_fwd_face": gt_fwd_face,
            "est_x": est_x, "gt_lat_face": gt_lat_face,
            "error_y": est_y - gt_fwd_face
        })

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