import json
import os


class LaneConfig:
    """
    Reads video_lanes.json — the single manual annotation file for each video.
    Answers two questions per frame:
        1. How many lanes does this road have and how wide are they?
        2. Is the emergency active at this timestamp?



    LANE WIDTHS ARE AUTOMATIC
    -------------------------
    Lane width is derived from road_type using German standards (RASt 06):
        highway -> 3.75m  (Autobahn standard)
        urban   -> 3.00m  (mid-range of German urban 2.50-3.25m)

    EMERGENCY IS LATCHED
    --------------------
    Once emergency_start_second is reached, is_emergency_active() returns True
    for the rest of the video. An ambulance does not turn its siren off mid-run.

    PRIORITY: manual config wins over scene classifier for road_type.
    If both exist and disagree, config is used and a warning is printed.

    CONFIG FORMAT (video_lanes.json)
    --------------------------------
    {
      "video_name": {
        "emergency_start_second": 22,
        "lanes": [
          {
            "from_second": 0,
            "to_second": 450,
            "lanes": 3,
            "road_type": "highway",
            "notes": "A2 Autobahn"
          },
          {
            "from_second": 450,
            "to_second": 930,
            "lanes": 2,
            "road_type": "urban",
            "notes": "city centre after exit"
          }
        ]
      }
    }

    video_name must match os.path.splitext(os.path.basename(path))[0][:30]

    HOW TO ADD A NEW VIDEO
    ----------------------
    1. Watch the video — note when the siren starts (emergency_start_second)
       and where lane count / road type changes
    2. Add one entry to video_lanes.json
    3. Lane widths are automatic — no need to write them

    FALLBACK
    --------
    If a video is not in the config:
        - lane info falls back to scene classifier output
        - emergency is never active (add the video to the config to fix this)
    """

    LANE_WIDTHS = {
        "highway":      3.75,
        "urban":        3.00,
        "intersection": 3.00,
        "roundabout":   3.00,
        "unknown":      3.00,
    }

    DEFAULT_LANES     = 2
    DEFAULT_ROAD_TYPE = "unknown"

    def __init__(self, config_path="video_lanes.json"):
        self.config = {}
        if not os.path.exists(config_path):
            print(f"WARNING: {config_path} not found - using defaults for all videos")
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            print(f"Video config loaded: {len(self.config)} video(s) annotated")
        except Exception as ex:
            print(f"WARNING: could not read {config_path}: {ex}")

    def get_lane_info(self, video_name, timestamp, scene_type=None):
        """
        Returns lane information for a specific video at a specific timestamp.

        Returns a dict with:
            lanes             - number of lanes (int)
            lane_width_meters - derived from road_type (float)
            road_type         - "highway", "urban", etc (str)
            source            - "config" or "scene_classifier" (str)
        """
        entry = self.config.get(video_name)

        if entry is not None:
            windows = entry.get("lanes", [])
            for window in windows:
                if window["from_second"] <= timestamp < window["to_second"]:
                    road_type = window.get("road_type", self.DEFAULT_ROAD_TYPE)

                    if (scene_type is not None
                            and scene_type != "unknown"
                            and scene_type != road_type):
                        print(f"  NOTE t={timestamp}s: config says '{road_type}' "
                              f"but scene classifier says '{scene_type}' "
                              f"-> using config (ground truth)")

                    return {
                        "lanes":             window["lanes"],
                        "lane_width_meters": self.LANE_WIDTHS.get(road_type, self.LANE_WIDTHS["unknown"]),
                        "road_type":         road_type,
                        "source":            "config"
                    }

        # fallback to scene classifier
        road_type = scene_type if scene_type and scene_type != "unknown" else self.DEFAULT_ROAD_TYPE
        return {
            "lanes":             self.DEFAULT_LANES,
            "lane_width_meters": self.LANE_WIDTHS.get(road_type, self.LANE_WIDTHS["unknown"]),
            "road_type":         road_type,
            "source":            "scene_classifier"
        }

    def is_emergency_active(self, video_name, timestamp):
        """
        Returns (emergency_active, triggered_by) for a given timestamp.

        emergency_active: True if timestamp >= emergency_start_second
        triggered_by:     ["manual"] — distinguishes from old FFT ["siren"] label

        Returns (False, []) if the video is not in the config.
        """
        entry = self.config.get(video_name)
        if entry is None:
            return False, []

        start = entry.get("emergency_start_second")
        if start is None:
            return False, []

        if timestamp >= start:
            return True, ["manual"]
        return False, []