"""
combine_calibration.py

Step 5 of the protocol (over-determination check). Reads calibration_log.json
(populated by calibrate_picker.py, one entry per frame) and reports the
combined estimate + agreement across frames.

Run: python3 combine_calibration.py
"""

import json
import numpy as np

LOG_FILE = "calibration_log.json"

CURRENT_HORIZON_RATIO = 0.55
CURRENT_FOCAL_FACTOR  = 0.8
CURRENT_CAM_HEIGHT    = 1.4

with open(LOG_FILE) as f:
    log = json.load(f)

if len(log) < 2:
    raise SystemExit(f"Only {len(log)} frame(s) logged — need at least 2 to "
                      f"check agreement. Run calibrate_picker.py on another frame.")

print(f"\n{'='*60}")
print(f"COMBINED CALIBRATION — {len(log)} frames")
print(f"{'='*60}")

fields = [
    ("horizon_ratio",     "horizon_ratio",     CURRENT_HORIZON_RATIO, ""),
    ("focal_factor_fit",  "focal_factor",      CURRENT_FOCAL_FACTOR,  ""),
    ("camera_height_fit", "camera_height",     CURRENT_CAM_HEIGHT,    "m"),
]

print(f"\n{'param':<16}{'frame values':<40}{'mean':>8}{'std':>8}{'current':>9}")
results = {}
for key, name, current, unit in fields:
    vals = [e[key] for e in log if e.get(key) is not None]
    if not vals:
        continue
    vals = np.array(vals)
    mean, std = vals.mean(), vals.std()
    results[name] = mean
    pct_std = 100 * std / abs(mean) if mean != 0 else float("nan")
    vals_str = ", ".join(f"{v:.4f}" for v in vals)
    print(f"{name:<16}{vals_str:<40}{mean:>8.4f}{std:>8.4f}{current:>9.4f}")
    flag = "OK" if pct_std < 10 else "DISAGREE across frames - investigate"
    diff_pct = 100 * (mean - current) / current if current != 0 else float("nan")
    print(f"  -> spread {pct_std:.1f}% ({flag})   "
          f"vs current: {diff_pct:+.1f}% difference\n")

print(f"\n{'='*60}")
print("Dash-fit residual check (should be small & frame-independent):")
for e in log:
    print(f"  t={e['timestamp']}s: max residual {e['max_dash_residual_px']:.2f}px"
          f"  (n_dashes={e['n_dashes']})")

print(f"\n{'='*60}")
print("RECOMMENDATION")
print(f"{'='*60}")
if "horizon_ratio" in results:
    print(f"  horizon_ratio  : {results['horizon_ratio']:.4f}  "
          f"(was {CURRENT_HORIZON_RATIO})")
if "focal_factor" in results:
    print(f"  focal_factor   : {results['focal_factor']:.4f}  "
          f"(was {CURRENT_FOCAL_FACTOR})")
if "camera_height" in results:
    print(f"  camera_height  : {results['camera_height']:.3f}m  "
          f"(was {CURRENT_CAM_HEIGHT}m)")
print(f"\nIf spreads above are <10%, these are safe to put in HomographyEstimator's")
print(f"__init__ defaults. If any spread is high, that parameter is unreliable —")
print(f"add 1-2 more frames before trusting it (curved road / incline / bad clicks")
print(f"are the usual causes).")
print(f"{'='*60}\n")