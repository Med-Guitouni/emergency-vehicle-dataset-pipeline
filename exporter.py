import json
import os


def _json_serialize(obj):
    import numpy as np
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class JSONExporter:
    """
    Writes one JSON file per second to output/<video_name>/.

    save_batch() is the only external entry point. save() is the per-frame
    writer; the output directory is created once in save_batch() rather than
    repeated on every frame.
    """

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, timestamp, tracked_vehicles, video_name,
             emergency_active, scenario_type, video_dir, frame_index=None):
        """
        Write one JSON for a single exported frame.
        video_dir must already exist (created once by save_batch).

        frame_index: unique increasing integer used for the filename.
        Required now that export can run faster than 1Hz (30Hz track+export),
        since multiple frames can share the same whole second and `timestamp`
        can be a float -- neither works as a `:04d`-formatted filename on its
        own. Falls back to `timestamp` (old 1Hz-export behaviour) if not given.
        """
        data = {
            "timestamp":       timestamp,
            "video_source":    video_name,
            "emergency_active": emergency_active,
            "scenario_type":   scenario_type,
            "vehicles":        [],
        }

        for v in tracked_vehicles:
            data["vehicles"].append({
                "id":               v["track_id"],
                "type":             v["type"],
                "x_meters":         v.get("x_meters", 0.0),
                "y_meters":         v.get("y_meters", 0.0),
                # False when bbox was clipped at a frame edge (tyres not
                # visible → ground-plane projection invalid) or the lateral
                # plausibility clamp fired. Filter on this in analysis.
                "position_reliable": v.get("position_reliable", True),
                "speed_kmh":         v.get("speed_kmh", 0.0),
                # + = away from ego / right;  − = toward ego / left
                "forward_speed_ms":  v.get("forward_speed_ms", 0.0),
                "lateral_speed_ms":  v.get("lateral_speed_ms", 0.0),
                "acceleration":      v.get("acceleration"),
                "lateral_acceleration": v.get("lateral_acceleration"),
                "jerk":              v.get("jerk"),
                # seconds until this vehicle reaches ego if it holds current
                # trajectory. None = behind ego or not closing (see homography.py)
                "ttc_to_ego":        v.get("ttc_to_ego"),
                "lane_id":           v.get("lane_id", 0),
                "lateral_offset":    v.get("lateral_offset", 0.0),
                # normalised [-1,+1] position within own lane (0 = lane centre)
                "lane_position_norm": v.get("lane_position_norm"),
                # normalised [-1,+1] position across whole road (0 = road centre)
                "road_position_norm": v.get("road_position_norm"),
                "distance_to_ego":   v.get("distance_to_ego", 0.0),
                "lanes_total":       v.get("lanes_total", 3),
                # "config" = manual annotation;  "scene_classifier" = fallback
                "road_type":         v.get("road_type", "unknown"),
                "lane_source":       v.get("lane_source", "unknown"),
                # highD-style surrounding vehicle IDs (None if no neighbour)
                "preceding_id":       v.get("preceding_id"),
                "following_id":       v.get("following_id"),
                "left_preceding_id":  v.get("left_preceding_id"),
                "left_following_id":  v.get("left_following_id"),
                "right_preceding_id": v.get("right_preceding_id"),
                "right_following_id": v.get("right_following_id"),
                "behaviour":          v.get("behaviour", "normal"),
                "bbox":               v["bbox"],
            })

        file_id = frame_index if frame_index is not None else timestamp
        filename = os.path.join(video_dir, f"t{file_id:06d}.json")
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=_json_serialize)

    def save_batch(self, all_frames_data, video_name):
        """
        Write one JSON per frame for the full video.
        Creates the output directory once here instead of on every save() call.
        """
        video_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)

        for frame_data in all_frames_data:
            self.save(
                frame_data["timestamp"],
                frame_data["vehicles"],
                video_name,
                frame_data["emergency_active"],
                frame_data["scenario_type"],
                video_dir,
                frame_index=frame_data.get("frame_index"),
            )