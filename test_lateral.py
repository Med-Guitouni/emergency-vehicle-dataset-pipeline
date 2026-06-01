from preprocessor import VideoPreprocessor
from detector import VehicleDetector
from tracker import VehicleTracker
from homography import HomographyEstimator
import os

video_path = 'videos/' + [f for f in os.listdir('videos') if f.endswith('.mp4')][0]

p = VideoPreprocessor(video_path)
d = VehicleDetector()
t = VehicleTracker()
h = HomographyEstimator()

frames = p.extract_frames(fps=1)
prev_vehicles = []

for i in range(0, 25):
    item = frames[i]
    timestamp = item['timestamp']
    frame = p.spatial_crop(item['frame'])
    frame_width = frame.shape[1]

    h.estimate_ego_motion(frame)
    tracked = t.update(d.model, frame)

    vehicles = []
    for v in tracked:
        tid = v['track_id']
        lateral_offset = h.estimate_lateral_offset(v['center'][0], frame_width)

        prev = next((p for p in prev_vehicles if p['track_id'] == tid), None)
        prev_lateral = prev['lateral_offset'] if prev else lateral_offset
        change = abs(lateral_offset - prev_lateral)

        vehicles.append({'track_id': tid, 'lateral_offset': lateral_offset})
        if change >= 0.1:
            print(f't={timestamp}s ID={tid} lateral_change={change:.2f}m')

    prev_vehicles = vehicles
    if not any(True for v in vehicles for pv in prev_vehicles if v['track_id'] == pv['track_id'] and abs(v['lateral_offset'] - pv['lateral_offset']) >= 0.1):
        print(f't={timestamp}s no significant lateral movement')