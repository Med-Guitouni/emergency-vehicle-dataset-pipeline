import cv2
import os


class VideoPreprocessor:
    """
    Extracts frames from the video at a requested Hz and crops each one.

    MEMORY
    ------
    stream_frames() is a generator — it yields one frame at a time and never
    holds the whole video in RAM. main.py's pipeline uses ONLY this generator
    (currently at TRACK_FPS=30), streaming each frame through detection/
    tracking/homography and discarding it immediately after -- memory use
    stays flat regardless of video length or Hz.

    extract_frames() wraps the generator and returns a full LIST instead --
    this holds every sampled frame in RAM simultaneously and its size is
    fps-proportional, since it yields RAW (pre-crop) frames at source
    resolution:
        15 min @ 1 Hz  , 720p  ≈ 2.5 GB   |  1080p ≈ 5.6 GB
        15 min @ 30 Hz , 720p  ≈ 75 GB    |  1080p ≈ 168 GB
    The 1Hz case is what the current sole caller (visualize_pipeline.py) uses
    and is workable but not trivial; the 30Hz case would OOM on virtually any
    machine. NEVER call extract_frames() at TRACK_FPS-scale rates -- use
    stream_frames() directly and process each frame as it arrives, the way
    main.py does, for anything beyond a low, fixed Hz on a short clip.
    """

    def __init__(self, video_path, output_dir="output"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def stream_frames(self, fps=1, start_s=0.0, end_s=None):
        """
        Yields frames at (up to) the requested Hz.

        start_s/end_s: optional real-time window (seconds). Default
        (0.0, None) processes the entire video -- fully backward compatible
        with every existing caller. Frames before start_s are decoded (for
        correct frame-accurate real-time bookkeeping) but not yielded; the
        stream stops as soon as a sampled frame's timestamp exceeds end_s.

        TIMESTAMP CORRECTNESS: timestamp is frame_count / video_fps -- the
        TRUE elapsed real time -- never derived from the requested `fps`.
        An earlier version computed timestamp as `extracted / fps`, which
        silently assumed the requested fps was always achieved. If the
        video's native fps is below the requested fps, `interval` floors at
        1 (every frame read, can't sample faster than the source), but the
        old timestamp label kept advancing at the REQUESTED rate anyway --
        drifting further from real time the longer the stream ran, and
        silently processing MORE real footage than a caller asking for e.g.
        "60 seconds at 30Hz" intended. Fixed here: timestamps always reflect
        true elapsed time regardless of whether the requested fps was
        achievable.
        """
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0:
            cap.release()
            raise RuntimeError(f"Could not read a valid fps from {self.video_path}")

        interval = max(int(round(video_fps / fps)), 1)
        if fps > video_fps:
            print(f"WARNING: requested {fps}Hz exceeds native video fps "
                  f"({video_fps:.3f}) -- cannot sample faster than the "
                  f"source. Effective achieved rate: {video_fps / interval:.3f}Hz.")

        frame_count = 0
        extracted = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                timestamp = round(frame_count / video_fps, 4)  # TRUE elapsed time
                if timestamp < start_s:
                    frame_count += 1
                    continue
                if end_s is not None and timestamp > end_s:
                    break
                yield {"timestamp": timestamp, "frame": frame}
                extracted += 1
            frame_count += 1
        cap.release()
        window_desc = f"{start_s}s-{end_s}s" if end_s is not None else "full video"
        print(f"Streamed {extracted} frames at {fps} Hz ({window_desc})")

    def extract_frames(self, fps=1, start_s=0.0, end_s=None):
        """List-returning wrapper. Casts timestamps to int for scripts that
        use them as JSON filename keys (visualize_pipeline, validators).
        See stream_frames() memory warning above before using this at
        anything beyond a low, fixed Hz."""
        frames = []
        for item in self.stream_frames(fps=fps, start_s=start_s, end_s=end_s):
            frames.append({
                "timestamp": int(round(item["timestamp"])),
                "frame": item["frame"],
            })
        print(f"Extracted {len(frames)} frames at {fps} Hz")
        return frames

    def spatial_crop(self, frame):
        """Remove dashboard and sky — keep rows 20% to 85% of height."""
        h, w = frame.shape[:2]
        cropped = frame[int(h * 0.20):int(h * 0.85), 0:w]
        # fixed output size so BoT-SORT GMC never sees mismatched pyramid levels
        return cv2.resize(cropped, (1280, 720))
