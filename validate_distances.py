
import os
import cv2

from preprocessor import VideoPreprocessor
from detector import VehicleDetector
from tracker import VehicleTracker
from homography import HomographyEstimator

# how many 1Hz frames to inspect
N_FRAMES = 15
# which frame index to also save as an annotated image for eyeballing
SAVE_EVERY = 3

os.makedirs("validation_frames", exist_ok=True)

videos = [f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")]
if not videos:
    raise SystemExit("No video found in videos/")

p = VideoPreprocessor(videos[0])
d = VehicleDetector()
t = VehicleTracker()
h = HomographyEstimator(horizon_ratio=0.55)
frames = p.extract_frames(fps=1)

print(f"\nInspecting first {N_FRAMES} frames of {os.path.basename(videos[0])}")
print(f"horizon_ratio={h.horizon_ratio}  "
      f"camera_height={h.camera_height}  "
      f"focal_length_factor={h.focal_length_factor}\n")

for item in frames[:N_FRAMES]:
    ts = item["timestamp"]
    frame = p.spatial_crop(item["frame"])
    fh, fw = frame.shape[:2]

    ego_H, depth_map = h.process_frame(frame)
    tracked = t.update(d.model, frame, ego_H=ego_H, depth_map=depth_map)

    annotated = frame.copy()
    # draw the assumed horizon line so you can see where it falls
    horizon_y = int(h.horizon_ratio * fh)
    cv2.line(annotated, (0, horizon_y), (fw, horizon_y), (0, 255, 255), 1)

    for v in tracked:
        bbox = v["bbox"]
        bottom_center = [(bbox[0] + bbox[2]) // 2, bbox[3]]
        x_m, y_m = h.get_bev_position(bottom_center, fw, fh)
        dist = h.estimate_distance_to_ego(x_m, y_m)

        print(f"t={ts:3d}s id={v['track_id']:>3} {v['type']:<10} "
              f"bottom_y={bbox[3]:>4}  forward={y_m:>6.1f}m  "
              f"lateral={x_m:>6.1f}m  dist={dist:>6.1f}m")

        # annotate the saved frame
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 0), 1)
        cv2.putText(annotated, f"{y_m:.0f}m",
                    (bbox[0], bbox[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if ts % SAVE_EVERY == 0:
        out = f"validation_frames/t{ts:04d}.jpg"
        cv2.imwrite(out, annotated)

print("\nAnnotated frames saved to validation_frames/ "
      "(yellow line = assumed horizon, green = vehicles with forward distance)")