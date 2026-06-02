from preprocessor import VideoPreprocessor
from detector import VehicleDetector
from tracker import VehicleTracker
from homography import HomographyEstimator
from annotator import HeuristicAnnotator
from emergency_detector import EmergencyDetector
import os

video_path = 'videos/' + [f for f in os.listdir('videos') if f.endswith('.mp4')][0]

p = VideoPreprocessor(video_path)
d = VehicleDetector()
t = VehicleTracker()
h = HomographyEstimator()
a = HeuristicAnnotator()
ed = EmergencyDetector(video_path)

frames = p.extract_frames(fps=1)
sample_frames = [p.spatial_crop(frames[i]['frame']) for i in range(0, min(10, len(frames)))]
ed.daytime = ed.detect_daytime(sample_frames)

first_minute = frames[:60]
prev_vehicles = []

for item in first_minute:
    timestamp = item['timestamp']
    frame = p.spatial_crop(item['frame'])
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    h.estimate_ego_motion(frame)
    tracked = t.update(d.model, frame)

    vehicles = []
    for v in tracked:
        tid = v['track_id']
        center = v['center']
        bbox = v['bbox']
        bottom_center = [(bbox[0] + bbox[2]) // 2, bbox[3]]
        x_m, y_m = h.get_bev_position(bottom_center, frame_width, frame_height)
        speed = h.estimate_speed(tid, center, frame_width)
        acceleration = h.estimate_acceleration(tid, speed)
        lateral_offset = h.estimate_lateral_offset(center[0], frame_width)
        distance_to_ego = h.estimate_distance_to_ego(center, frame_width)
        vehicles.append({
            'track_id': tid,
            'type': v['type'],
            'bbox': bbox,
            'center': center,
            'lateral_offset': lateral_offset,
            'distance_to_ego': distance_to_ego,
            'speed_kmh': speed,
            'acceleration': acceleration,
            'heading_angle': h.estimate_heading(tid, center)
        })

        emergency_active, triggered_by = ed.is_emergency_active(timestamp, frame, vehicles, prev_vehicles)
        prev_vehicles = vehicles

        # annotate each vehicle
        for v in vehicles:
            v['behaviour'] = a.annotate(v, emergency_active)

        # print summary
        yielding = [v for v in vehicles if v['behaviour'] == 'yielded']
        braking = [v for v in vehicles if v['behaviour'] == 'braked_abruptly']
        failed = [v for v in vehicles if v['behaviour'] == 'failed_to_yield']

        if emergency_active:
            print(
                f't={timestamp}s EMERGENCY ON - {len(vehicles)} vehicles - yielded={len(yielding)} braked={len(braking)} failed={len(failed)}')
        else:
            print(f't={timestamp}s normal - {len(vehicles)} vehicles')