"""
extract_candidate_frames.py


Run: python3 extract_candidate_frames.py
Output: calibration_candidates/  (one jpg per N seconds, labelled with timestamp)
"""

import os
import cv2

from preprocessor import VideoPreprocessor

OUTPUT_DIR = "calibration_candidates"
EVERY_N_SECONDS = 5

videos = [f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")]
if not videos:
    raise SystemExit("No video found in videos/")
video_path = videos[0]

if os.path.exists(OUTPUT_DIR):
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))
os.makedirs(OUTPUT_DIR, exist_ok=True)

p = VideoPreprocessor(video_path)
saved = 0
for item in p.extract_frames(fps=1):
    ts = item["timestamp"]
    if ts % EVERY_N_SECONDS != 0:
        continue
    frame = p.spatial_crop(item["frame"])
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"t{ts:04d}.jpg"), frame)
    saved += 1

print(f"Saved {saved} candidate frames to {OUTPUT_DIR}/")

print("Write down those timestamps (the txxxx in the filename) for Step 2.")