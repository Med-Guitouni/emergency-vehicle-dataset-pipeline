import json
import os


def _json_serialize(obj):
    # numpy floats and ints are not JSON serializable by default
    # this converts them to plain Python types before writing
    import numpy as np
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class JSONExporter:
    """
    just saves data it receives to JSON
    """

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, timestamp, tracked_vehicles, video_name, emergency_active, triggered_by, scenario_type):
        video_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)

        data = {
            "timestamp": timestamp,
            "video_source": video_name,
            "emergency_active": emergency_active,
            "emergency_triggered_by": triggered_by,
            "scenario_type": scenario_type,
            "vehicles": []
        }

        for v in tracked_vehicles:
            data["vehicles"].append({
                "id": v["track_id"],
                "type": v["type"],
                "x_meters": v.get("x_meters", 0.0),
                "y_meters": v.get("y_meters", 0.0),
                "speed_kmh": v.get("speed_kmh", 0.0),

                # split velocity in metres per second
                # forward_speed_ms: + = moving away from ego, - = toward ego
                # lateral_speed_ms: + = moving right, - = moving left
                "forward_speed_ms": v.get("forward_speed_ms", 0.0),
                "lateral_speed_ms": v.get("lateral_speed_ms", 0.0),

                "acceleration": v.get("acceleration", 0.0),
                "jerk": v.get("jerk", 0.0),
                "heading_angle": v.get("heading_angle", 0.0),
                "lane_id": v.get("lane_id", 0),
                "lateral_offset": v.get("lateral_offset", 0.0),
                "distance_to_ego": v.get("distance_to_ego", 0.0),

                # lane count and road type — "config" means manual annotation,
                # "scene_classifier" means automatic fallback (video not in config)
                "lanes_total": v.get("lanes_total", 2),
                "road_type":   v.get("road_type", "unknown"),
                "lane_source": v.get("lane_source", "unknown"),

                # highD-style surrounding vehicle IDs (None if no such neighbour)
                # preceding/following = same lane, ahead/behind
                # left_/right_ = the lane to the left/right of this vehicle
                "preceding_id": v.get("preceding_id"),
                "following_id": v.get("following_id"),
                "left_preceding_id": v.get("left_preceding_id"),
                "left_following_id": v.get("left_following_id"),
                "right_preceding_id": v.get("right_preceding_id"),
                "right_following_id": v.get("right_following_id"),

                "behaviour": v.get("behaviour", "normal"),
                "bbox": v["bbox"]
            })

        filename = os.path.join(video_dir, f"t{timestamp:04d}.json")
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=_json_serialize)

    def save_batch(self, all_frames_data, video_name):
        for frame_data in all_frames_data:
            self.save(
                frame_data["timestamp"],
                frame_data["vehicles"],
                video_name,
                frame_data["emergency_active"],
                frame_data["emergency_triggered_by"],
                frame_data["scenario_type"]
            )