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


    python3 count_reliability.py
"""

import os
import json
from collections import defaultdict

OUTPUT_DIR = "output"

total        = 0
reliable     = 0
unreliable   = 0

by_type      = defaultdict(lambda: {"total": 0, "unreliable": 0})
by_distance  = defaultdict(lambda: {"total": 0, "unreliable": 0})
by_track     = defaultdict(lambda: {"total": 0, "unreliable": 0})

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

        for v in data.get("vehicles", []):
            total += 1
            tid   = v.get("id", 0)
            vtype = v.get("type", "unknown")
            dist  = v.get("distance_to_ego", 0)
            rel   = v.get("position_reliable", True)
            bucket = get_bucket(dist)

            by_type[vtype]["total"]     += 1
            by_distance[bucket]["total"] += 1
            by_track[tid]["total"]       += 1

            if not rel:
                unreliable += 1
                by_type[vtype]["unreliable"]     += 1
                by_distance[bucket]["unreliable"] += 1
                by_track[tid]["unreliable"]       += 1
            else:
                reliable += 1

# ---- print results ----
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

print("\n" + "=" * 55)
print("  Run again after any fix to compare.")
print("=" * 55)
