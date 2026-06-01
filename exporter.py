import json
import os

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
                "acceleration": v.get("acceleration", 0.0),
                "jerk": v.get("jerk", 0.0),
                "heading_angle": v.get("heading_angle", 0.0),
                "lane_id": v.get("lane_id", 0),
                "lateral_offset": v.get("lateral_offset", 0.0),
                "distance_to_ego": v.get("distance_to_ego", 0.0),
                "is_emergency": v.get("is_emergency", False),
                "behaviour": v.get("behaviour", "normal"),
                "bbox": v["bbox"]
            })

        filename = os.path.join(video_dir, f"t{timestamp:04d}.json")
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

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

