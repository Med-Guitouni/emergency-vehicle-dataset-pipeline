import json
import os


class LaneConfig:
    """
    Reads video_lanes.json and answers one question per frame:
    how wide are our road

    WHY THIS EXISTS

    Lane count cannot be reliably detected from the camera because:
    see lane_detector.py
    - No GPS is embedded in the video files

    The solution: manual ground-truth annotation by watching each video
    and writing down the lane count per time window. This is the most
    correct and reliable approach


    Lane width is derived automatically from road_type:
        highway -> 3.75m per lane (German Autobahn standard, wiki)
        urban   -> 3.00m per lane (mid-range of German urban standard
                   2.50-3.25m, Uncertainty ±0.25m


    CONFIG FORMAT (video_lanes.json)
    {
      "video_name": [
        {
          "from_second": 0,
          "to_second": 450,
          "lanes": 3,
          "road_type": "highway",
          "notes": "A2 Autobahn, 3 lanes"
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

    video_name must match exactly what main.py uses:
        os.path.splitext(os.path.basename(video_path))[0][:30]


    If a video is not in the config, get_lane_info() returns safe defaults:
        lanes=2, road_type="unknown", lane_width=3.0m
    The pipeline never crashes - it just uses conservative estimates.
    """

    # German standard lane widths by road type (RASt 06)
    LANE_WIDTHS = {
        "highway": 3.75,  # Autobahn standard
        "urban": 3.00,  # mid-range of German urban 2.50-3.25m
        "intersection": 3.00,  # treat same as urban
        "roundabout": 3.00,  # treat same as urban
        "unknown": 3.00,  # safe fallback
    }

    DEFAULT_LANES = 2
    DEFAULT_ROAD_TYPE = "unknown"

    def __init__(self, config_path="video_lanes.json"):
        self.config = {}
        if not os.path.exists(config_path):
            print(f"WARNING: {config_path} not found - using scene classifier as road type source")
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            print(f"Lane config loaded: {len(self.config)} video(s) annotated")
        except Exception as ex:
            print(f"WARNING: could not read {config_path}: {ex} - using scene classifier fallback")

    def get_lane_info(self, video_name, timestamp, scene_type=None):
        """
        Return lane information for a specific video at a specific timestamp.

        video_name: truncated video name from main.py
        timestamp:  current frame time in seconds
        scene_type: road type from scene classifier (e.g. "highway", "urban").
                    Used as fallback when video is not in the manual config.
                    If manual config exists and disagrees, config wins and a
                    warning is printed.

        Returns a dict with:
            lanes             - number of lanes (int)
            lane_width_meters - derived automatically from road_type (float)
            road_type         - "highway", "urban", etc (str)
            source            - "config" or "scene_classifier" (str)
        """
        windows = self.config.get(video_name)

        if windows is not None:
            # video is in the manual config - look up the time window
            for window in windows:
                if window["from_second"] <= timestamp < window["to_second"]:
                    road_type = window.get("road_type", self.DEFAULT_ROAD_TYPE)

                    # warn if manual config and scene classifier disagree
                    if (scene_type is not None
                            and scene_type != "unknown"
                            and scene_type != road_type):
                        print(f"  NOTE t={timestamp}s: config says '{road_type}' "
                              f"but scene classifier says '{scene_type}' "
                              f"-> using config (ground truth)")

                    return {
                        "lanes": window["lanes"],
                        "lane_width_meters": self.LANE_WIDTHS.get(road_type, self.LANE_WIDTHS["unknown"]),
                        "road_type": road_type,
                        "source": "config"
                    }

            # video in config but timestamp outside all windows
            # fall through to scene classifier fallback below

        # video not in config or timestamp outside windows
        # use scene classifier as road type source
        road_type = scene_type if scene_type and scene_type != "unknown" else self.DEFAULT_ROAD_TYPE
        return {
            "lanes": self.DEFAULT_LANES,
            "lane_width_meters": self.LANE_WIDTHS.get(road_type, self.LANE_WIDTHS["unknown"]),
            "road_type": road_type,
            "source": "scene_classifier"
        }

    def _default(self):
        return {
            "lanes": self.DEFAULT_LANES,
            "lane_width_meters": self.LANE_WIDTHS[self.DEFAULT_ROAD_TYPE],
            "road_type": self.DEFAULT_ROAD_TYPE,
            "source": "default"
        }
