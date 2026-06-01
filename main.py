import os
from preprocessor import VideoPreprocessor
from detector import VehicleDetector
from tracker import VehicleTracker
from homography import HomographyEstimator
from exporter import JSONExporter
from annotator import HeuristicAnnotator
from emergency_detector import EmergencyDetector
from scene_classifier import SceneClassifier

"""
For each video, frames are extracted at 1Hz.
The first 10 raw uncropped frames are sampled to determine if the video is
daytime or not - this controls whether blue light detection is used later.
Each frame is then spatially cropped to remove the dashboard and sky.

For every frame, YOLOv8 detects all vehicles and ByteTrack assigns persistent
IDs so the same vehicle keeps the same ID throughout the video. Spatial metrics
are then computed per vehicle - position in meters, speed, acceleration, jerk,
heading angle, lane ID, lateral offset and distance to the ambulance.

Emergency active status is determined per frame using siren audio analysis
for both daytime and nighttime videos. Blue light detection is added as
an extra signal for nighttime only since it causes false positives on
daytime footage due to sky color interference.

Scene type is classified using Places365 ResNet18 with a 20 second
temporal smoothing window to prevent single occluded frames from
flipping the scenario type. The classifier is loaded once and reused
across all videos.

Vehicle behaviour is then labelled per vehicle based on their motion relative
to the ambulance - yielded, braked abruptly, failed to yield or normal.

Results are saved as one JSON file per second inside a subfolder named after
the video.
"""

# load scene classifier once for all videos
sc = SceneClassifier()


def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0][:30]
    print(f"\nProcessing: {video_name}")

    p = VideoPreprocessor(video_path)
    d = VehicleDetector()
    t = VehicleTracker()
    h = HomographyEstimator()
    e = JSONExporter()
    a = HeuristicAnnotator()
    ed = EmergencyDetector(video_path)

    # reset scene classifier smoothing window for each new video
    sc.reset()

    frames = p.extract_frames(fps=1)

    # day/night detection uses original uncropped frames
    # spatial_crop removes the sky region which is needed for brightness analysis
    # everything else in the pipeline uses cropped frames to remove dashboard and sky noise
    sample_frames = [frames[i]['frame'] for i in range(0, min(10, len(frames)))]
    ed.daytime = ed.detect_daytime(sample_frames)

    all_frames_data = []
    prev_vehicles = []

    for item in frames:
        timestamp = item["timestamp"]

        # scene classification uses original uncropped frame
        # needs sky and surroundings to distinguish highway from city
        scenario_type = sc.classify(item["frame"])

        # everything else uses spatially cropped frame
        frame = p.spatial_crop(item["frame"])
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        h.estimate_ego_motion(frame)
        tracked = t.update(d.model, frame)

        vehicles = []
        for v in tracked:
            tid = v["track_id"]
            center = v["center"]
            bbox = v["bbox"]
            bottom_center = [(bbox[0] + bbox[2]) // 2, bbox[3]]

            x_m, y_m = h.get_bev_position(bottom_center, frame_width, frame_height)
            speed = h.estimate_speed(tid, center, frame_width)
            acceleration = h.estimate_acceleration(tid, speed)
            jerk = h.estimate_jerk(tid, acceleration)
            heading = h.estimate_heading(tid, center)
            distance_to_ego = h.estimate_distance_to_ego(center, frame_width)
            lane_id = h.estimate_lane_id(center[0], frame_width)
            lateral_offset = h.estimate_lateral_offset(center[0], frame_width)

            vehicles.append({
                "track_id": tid,
                "type": v["type"],
                "bbox": bbox,
                "center": center,
                "x_meters": x_m,
                "y_meters": y_m,
                "speed_kmh": speed,
                "acceleration": acceleration,
                "jerk": jerk,
                "heading_angle": heading,
                "lane_id": lane_id,
                "lateral_offset": lateral_offset,
                "distance_to_ego": distance_to_ego,
                "is_emergency": False
            })

        emergency_active, triggered_by = ed.is_emergency_active(
            timestamp, frame, vehicles, prev_vehicles
        )

        for v in vehicles:
            v["behaviour"] = a.annotate(v, emergency_active)

        all_frames_data.append({
            "timestamp": timestamp,
            "emergency_active": emergency_active,
            "emergency_triggered_by": triggered_by,
            "scenario_type": scenario_type,
            "vehicles": vehicles
        })

        prev_vehicles = vehicles

        if timestamp % 60 == 0:
            print(f"  t={timestamp}s - {len(vehicles)} vehicles - emergency={emergency_active} scenario={scenario_type}")

    e.save_batch(all_frames_data, video_name)
    return all_frames_data


if __name__ == "__main__":
    videos = [f"videos/{v}" for v in os.listdir("videos") if v.endswith(".mp4")]
    print(f"Found {len(videos)} videos")
    for video in videos:
        process_video(video)
    print("\nDone.")

    # rm - rf output / *  python3  main.py


# Test script - verifies emergency detection across first 60 seconds
# Run: python3 test_emergency.py

# Test script - verifies lateral offset detection across first 25 seconds
# Run: python3 test_lateral.py


