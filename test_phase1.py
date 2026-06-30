"""
test_phase1.py

Runs BoT-SORT tracking on the first MAX_SECONDS of the first video found
in videos/ at TRACKING_HZ, displays bounding boxes + IDs in a window, and
prints a unique-ID count so you can measure tracking quality before running
the full pipeline.

Good result on 60 s of busy highway: < 30 unique IDs total.
Bad result (ID churn): 150+.

Press Q to quit early.
"""

import os
import cv2
from ultralytics import YOLO
from preprocessor import VideoPreprocessor

TRACKING_HZ = 5
MAX_SECONDS  = 60

videos = sorted([f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")])
if not videos:
    raise SystemExit("No .mp4 found in videos/")

video_path = videos[0]
print(f"Video : {video_path}")
print(f"Tracking at {TRACKING_HZ} Hz for first {MAX_SECONDS} s — press Q to quit")

model = YOLO("yolov8x.pt")
p     = VideoPreprocessor(video_path)
cap   = cv2.VideoCapture(video_path)
fps   = cap.get(cv2.CAP_PROP_FPS)
skip  = max(int(round(fps / TRACKING_HZ)), 1)

print(f"Native fps: {fps:.0f}  —  processing every {skip}th frame")

cv2.namedWindow("Phase 1 test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Phase 1 test", 1280, 540)

frame_idx    = 0
all_ids_seen = set()

while True:
    # advance cheaply on skipped frames
    if frame_idx % skip != 0:
        cap.grab()
        frame_idx += 1
        continue

    ret, frame_raw = cap.read()
    if not ret:
        break

    ts = frame_idx / fps
    if ts > MAX_SECONDS:
        break

    frame   = p.spatial_crop(frame_raw)
    results = model.track(frame, tracker="botsort.yaml",
                          persist=True, verbose=False)[0]

    ids = [int(b.id[0]) for b in results.boxes if b.id is not None]
    all_ids_seen.update(ids)
    print(f"t={ts:.2f}s: {ids}  | unique total: {len(all_ids_seen)}")

    # draw boxes + IDs
    vis = frame.copy()
    for box in results.boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tid = int(box.id[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(vis, f"id{tid}", (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

    cv2.putText(vis, f"t={ts:.2f}s  ids:{ids}  unique:{len(all_ids_seen)}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Phase 1 test", vis)

    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal unique IDs over {min(ts, MAX_SECONDS):.0f}s: {len(all_ids_seen)}")
print("Done.")