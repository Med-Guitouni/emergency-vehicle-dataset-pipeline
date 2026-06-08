import json
import os


class LaneConfig:
    """
    Reads video_lanes.json and answers one question per frame:
    "How many lanes does this road have right now, and how wide are they?"

    WHY THIS EXISTS
    ---------------
    Lane count cannot be reliably detected from the camera because:
    - Vision-based models (UFLD v2, YOLOP) fail when vehicles cover markings
      at exactly the moments that matter (see lane_detector.py)
    - Videos mix highway (3 lanes, 3.75m wide) and city (1-2 lanes, 3.25m wide)
      sometimes within the same video
    - No GPS is embedded in the video files to query OpenStreetMap automatically

    The solution: manual ground-truth annotation by the researcher watching
    the first 20 seconds of each video (before the emergency starts, road is
    clear) and writing down the lane count per time window. This is the most
    scientifically defensible approach for a dataset of 32 videos.

    CONFIG FORMAT (video_lanes.json)
    ---------------------------------
    {
      "video_name": [
        {
          "from_second": 0,
          "to_second": 450,
          "lanes": 3,
          "road_type": "highway",
          "lane_width_meters": 3.75,
          "notes": "optional human note"
        },
        {
          "from_second": 450,
          "to_second": 930,
          "lanes": 2,
          "road_type": "urban",
          "lane_width_meters": 3.25,
          "notes": "road narrows after exit"
        }
      ]
    }

    video_name must match exactly what main.py uses:
        os.path.splitext(os.path.basename(video_path))[0][:30]

    HOW TO ADD A NEW VIDEO
    ----------------------
    1. Watch the first 20 seconds (pre-emergency, road is clear)
    2. Note where lane count changes (timestamp + new count)
    3. Add an entry to video_lanes.json
    4. Use lane_width_meters=3.75 for Autobahn, 3.25 for city streets

    FALLBACK
    --------
    If a video is not in the config, or the timestamp falls outside all
    defined windows, get_lane_info() returns safe defaults:
        lanes=2, lane_width_meters=3.5, road_type="unknown"
    This never crashes the pipeline - it just uses a conservative estimate.
    """

    # safe defaults used when no config entry is found for a timestamp
    DEFAULT_LANES = 2
    DEFAULT_LANE_WIDTH = 3.5
    DEFAULT_ROAD_TYPE = "unknown"

    def __init__(self, config_path="video_lanes.json"):
        """
        Load the config file once at startup.
        config_path is relative to the pipeline root folder.
        """
        self.config = {}
        if not os.path.exists(config_path):
            print(f"WARNING: {config_path} not found - using default lane config for all videos")
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            print(f"Lane config loaded: {len(self.config)} video(s) annotated")
        except Exception as ex:
            print(f"WARNING: could not read {config_path}: {ex} - using defaults")

    def get_lane_info(self, video_name, timestamp):
        """
        Return lane information for a specific video at a specific timestamp.

        video_name: the truncated video name used internally by main.py
                    (os.path.splitext(os.path.basename(path))[0][:30])
        timestamp:  current frame time in seconds

        Returns a dict with:
            lanes            - number of lanes (int)
            lane_width_meters - width of one lane in metres (float)
            road_type        - "highway", "urban", or "unknown" (str)
        """
        windows = self.config.get(video_name)

        if windows is None:
            # video not in config at all
            return self._default()

        for window in windows:
            if window["from_second"] <= timestamp < window["to_second"]:
                return {
                    "lanes":             window["lanes"],
                    "lane_width_meters": window.get("lane_width_meters", self.DEFAULT_LANE_WIDTH),
                    "road_type":         window.get("road_type", self.DEFAULT_ROAD_TYPE)
                }

        # timestamp outside all defined windows (e.g. video longer than config)
        return self._default()

    def _default(self):
        return {
            "lanes":             self.DEFAULT_LANES,
            "lane_width_meters": self.DEFAULT_LANE_WIDTH,
            "road_type":         self.DEFAULT_ROAD_TYPE
        }