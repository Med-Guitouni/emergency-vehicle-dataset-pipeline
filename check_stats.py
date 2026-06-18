"""
check_stats.py

Reads the pipeline's output JSON and prints:
  - behaviour label distribution (the yielded rate is the headline)
  - heading angle range + distribution (to confirm the heading fix worked)
  - lateral speed distribution (to see how noise sits vs the 0.5 threshold)
  - how many labels sit on position_reliable=False rows

Run: python3 check_stats.py
"""

import os
import json
import numpy as np
from collections import Counter

# find the output folder automatically
videos = [v for v in os.listdir("videos") if v.endswith(".mp4")] if os.path.exists("videos") else []
if not videos:
    raise SystemExit("No video in videos/")
video_name = os.path.splitext(videos[0])[0][:30]
json_dir = os.path.join("output", video_name)
if not os.path.exists(json_dir):
    raise SystemExit(f"No output at {json_dir}/ — run main.py first.")

labels        = Counter()
headings      = []
lateral_spds  = []
emergency_labels = Counter()   # labels only while emergency active
n_total_veh   = 0
n_on_unreliable = 0
labels_on_unreliable = Counter()

files = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))
for fn in files:
    with open(os.path.join(json_dir, fn)) as f:
        data = json.load(f)
    emergency = data.get("emergency_active", False)
    for v in data.get("vehicles", []):
        n_total_veh += 1
        beh = v.get("behaviour", "normal")
        labels[beh] += 1
        if emergency:
            emergency_labels[beh] += 1

        hd = v.get("heading_angle", None)
        if hd is not None:
            headings.append(hd)
        ls = v.get("lateral_speed_ms", None)
        if ls is not None:
            lateral_spds.append(abs(ls))

        if not v.get("position_reliable", True):
            n_on_unreliable += 1
            labels_on_unreliable[beh] += 1

print(f"\n{'='*52}")
print(f"  {len(files)} frames   {n_total_veh} vehicle-observations")
print(f"{'='*52}")

print(f"\nBEHAVIOUR LABELS (all frames)")
for lab, c in labels.most_common():
    print(f"  {lab:<18} {c:>6}  ({100*c/max(n_total_veh,1):5.1f}%)")

if emergency_labels:
    tot_em = sum(emergency_labels.values())
    print(f"\nBEHAVIOUR LABELS (emergency-active only, n={tot_em})")
    for lab, c in emergency_labels.most_common():
        print(f"  {lab:<18} {c:>6}  ({100*c/max(tot_em,1):5.1f}%)")

if headings:
    h = np.array(headings)
    print(f"\nHEADING ANGLE  (should be within +-90, mostly small on highway)")
    print(f"  min {h.min():+.1f}   max {h.max():+.1f}   "
          f"mean|h| {np.abs(h).mean():.1f}")
    print(f"  |h|>90 : {int((np.abs(h)>90).sum())}  (should be 0 after the fix)")
    print(f"  |h|>45 : {int((np.abs(h)>45).sum())}  "
          f"({100*(np.abs(h)>45).mean():.1f}% — high on highway = suspicious)")

if lateral_spds:
    ls = np.array(lateral_spds)
    print(f"\nLATERAL SPEED |m/s|")
    print(f"  mean {ls.mean():.3f}   median {np.median(ls):.3f}   max {ls.max():.3f}")
    print(f"  >= 0.5 (single-frame trigger level): "
          f"{int((ls>=0.5).sum())}  ({100*(ls>=0.5).mean():.1f}%)")

print(f"\nLABELS ON UNRELIABLE POSITIONS")
print(f"  {n_on_unreliable} of {n_total_veh} observations "
      f"({100*n_on_unreliable/max(n_total_veh,1):.1f}%) have position_reliable=False")
for lab, c in labels_on_unreliable.most_common():
    print(f"    {lab:<18} {c:>6}")

print(f"\n{'='*52}\n")