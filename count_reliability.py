"""
count_reliability.py

Goes through every JSON output file and counts how many vehicle-frame
observations have position_reliable=True vs False.

Prints:
- total observations
- reliable vs unreliable counts and percentages
- breakdown by vehicle type (trucks likely worse than cars)
- breakdown by distance bucket (close vehicles likely worse)
- top 10 most unreliable track IDs
- EVERY unreliable observation's x_meters / y_meters, with a flag for
  whether the value is physically plausible or not — so you can judge
  by eye whether position_reliable=False is actually catching bad
  positions or just being overly cautious.

    python3 count_reliability.py
"""

import os
import json
from collections import defaultdict

OUTPUT_DIR = "output"

# Plausibility bounds for a German Autobahn / urban road, used only to
# flag values for visual inspection — NOT used anywhere in the pipeline.
PLAUSIBLE_X_RANGE = (-15.0, 15.0)   # metres either side of ego
PLAUSIBLE_Y_RANGE = (0.0, 250.0)    # metres ahead

total        = 0
reliable     = 0
unreliable   = 0

by_type      = defaultdict(lambda: {"total": 0, "unreliable": 0})
by_distance  = defaultdict(lambda: {"total": 0, "unreliable": 0})
by_track     = defaultdict(lambda: {"total": 0, "unreliable": 0})

# every unreliable observation, kept for the dump at the end
unreliable_observations = []

DISTANCE_BUCKETS = [
    (0,  10,  "0-10m   (very close)"),
    (10, 20,  "10-20m  (close)     "),
    (20, 40,  "20-40m  (mid)       "),
    (40, 80,  "40-80m  (far)       "),
    (80, 999, "80m+    (distant)   "),
]

def get_bucket(dist):
    for lo, hi, label in DISTANCE_BUCKETS:
        if lo <= dist < hi:
            return label
    return "80m+    (distant)   "


def is_physically_plausible(x_m, y_m):
    """Loose sanity check, not the pipeline's own clamp logic — used only
    to flag values worth a second look."""
    x_ok = PLAUSIBLE_X_RANGE[0] <= x_m <= PLAUSIBLE_X_RANGE[1]
    y_ok = PLAUSIBLE_Y_RANGE[0] <= y_m <= PLAUSIBLE_Y_RANGE[1]
    return x_ok and y_ok


# find output folder
video_dirs = [d for d in os.listdir(OUTPUT_DIR)
              if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
if not video_dirs:
    raise SystemExit("No output folder found. Run main.py first.")

for vdir in video_dirs:
    json_files = sorted([f for f in os.listdir(os.path.join(OUTPUT_DIR, vdir))
                         if f.endswith(".json")])
    for jf in json_files:
        path = os.path.join(OUTPUT_DIR, vdir, jf)
        with open(path) as f:
            data = json.load(f)
        timestamp = data.get("timestamp", "?")

        for v in data.get("vehicles", []):
            total += 1
            tid   = v.get("id", 0)
            vtype = v.get("type", "unknown")
            dist  = v.get("distance_to_ego", 0)
            rel   = v.get("position_reliable", True)
            x_m   = v.get("x_meters", 0.0)
            y_m   = v.get("y_meters", 0.0)
            bbox  = v.get("bbox", [])
            bucket = get_bucket(dist)

            by_type[vtype]["total"]     += 1
            by_distance[bucket]["total"] += 1
            by_track[tid]["total"]       += 1

            if not rel:
                unreliable += 1
                by_type[vtype]["unreliable"]     += 1
                by_distance[bucket]["unreliable"] += 1
                by_track[tid]["unreliable"]       += 1

                unreliable_observations.append({
                    "video":     vdir,
                    "timestamp": timestamp,
                    "track_id":  tid,
                    "type":      vtype,
                    "x_meters":  x_m,
                    "y_meters":  y_m,
                    "distance":  dist,
                    "bbox":      bbox,
                    "plausible": is_physically_plausible(x_m, y_m),
                })
            else:
                reliable += 1

# ---- print summary ----
print("=" * 55)
print("  POSITION RELIABILITY REPORT")
print("=" * 55)
print(f"\n  Total observations : {total}")
print(f"  Reliable           : {reliable}  ({reliable/total*100:.1f}%)")
print(f"  Unreliable         : {unreliable}  ({unreliable/total*100:.1f}%)")
print(f"\n  Summary: {unreliable}/{total} observations flagged unreliable")

print("\n--- By vehicle type ---")
for vtype, counts in sorted(by_type.items()):
    t = counts["total"]
    u = counts["unreliable"]
    pct = u / t * 100 if t > 0 else 0
    bar = "█" * int(pct / 5)
    print(f"  {vtype:<12} {u:>5}/{t:<6} ({pct:5.1f}%)  {bar}")

print("\n--- By distance to ego ---")
for lo, hi, label in DISTANCE_BUCKETS:
    counts = by_distance[label]
    t = counts["total"]
    u = counts["unreliable"]
    pct = u / t * 100 if t > 0 else 0
    bar = "█" * int(pct / 5)
    print(f"  {label}  {u:>5}/{t:<6} ({pct:5.1f}%)  {bar}")

print("\n--- Top 10 most unreliable track IDs ---")
top10 = sorted(by_track.items(),
               key=lambda x: x[1]["unreliable"], reverse=True)[:10]
for tid, counts in top10:
    t = counts["total"]
    u = counts["unreliable"]
    pct = u / t * 100 if t > 0 else 0
    print(f"  track #{tid:<6}  {u:>4}/{t:<5} frames unreliable  ({pct:.0f}%)")

# ---- dump every unreliable x/y for visual inspection ----
print("\n" + "=" * 55)
print("  UNRELIABLE POSITIONS — x_meters / y_meters")
print("=" * 55)
print("  plausible = within ±15m lateral and 0-250m forward")
print("  (loose sanity bounds, NOT the pipeline's own clamp logic)\n")

implausible_count = sum(1 for o in unreliable_observations if not o["plausible"])
plausible_count   = len(unreliable_observations) - implausible_count
print(f"  Of {len(unreliable_observations)} unreliable observations:")
print(f"    {plausible_count} have PLAUSIBLE x/y (flag may be overly cautious)")
print(f"    {implausible_count} have IMPLAUSIBLE x/y (flag is catching real errors)")

print(f"\n  {'video':<25} {'t':>5} {'id':>5} {'type':<6} "
      f"{'x_m':>7} {'y_m':>7} {'dist':>6} {'plausible':>10}")
print("  " + "-" * 80)
for o in unreliable_observations:
    flag = "YES" if o["plausible"] else "NO  <-- impossible"
    print(f"  {o['video']:<25} {o['timestamp']:>5} {o['track_id']:>5} "
          f"{o['type']:<6} {o['x_meters']:>7.2f} {o['y_meters']:>7.2f} "
          f"{o['distance']:>6.1f} {flag:>10}")

print("\n" + "=" * 55)
print("  Run again after any fix to compare.")
print("=" * 55)
