import cv2
import os


class VideoPreprocessor:
    """
    Extracts frames from the video at a requested Hz and crops each one.

    MEMORY
    ------
    stream_frames() is a generator — it yields one frame at a time and never
    holds the whole video in RAM. extract_frames() wraps it and returns a
    list; at 1 Hz a 15-minute video is only ~900 frames, so the list is fine.
    """

    def __init__(self, video_path, output_dir="output"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def stream_frames(self, fps=1):
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(int(round(video_fps / fps)), 1)

        frame_count = 0
        extracted = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                yield {
                    "timestamp": round(extracted / fps, 2),  # float: 0.0, 0.2, 1.0...
                    "frame": frame,
                }
                extracted += 1
            frame_count += 1
        cap.release()
        print(f"Streamed {extracted} frames at {fps} Hz")

    def extract_frames(self, fps=1):
        """List-returning wrapper. Casts timestamps to int for scripts that
        use them as JSON filename keys (visualize_pipeline, validators)."""
        frames = []
        for item in self.stream_frames(fps=fps):
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
