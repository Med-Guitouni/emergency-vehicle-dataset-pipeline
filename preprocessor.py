import cv2
import os


class VideoPreprocessor:
    """
    Extracts frames from the video and crops them.



    MEMORY NOTE
    -----------
    At 5Hz a 15-minute video is ~4600 frames (~12GB if held in RAM at once).
    stream_frames() is therefore a GENERATOR - it yields one frame at a time
    and never holds the whole video in memory. extract_frames() (the old
    list-returning method) is kept for the helper scripts that still use it
    at 1Hz.
    """

    def __init__(self, video_path, output_dir="output"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def sample_frames(self, n=10):
        """
        Quickly grab the first n frames at 1-second spacing.
        Used only for day/night detection before the main loop starts.
        ( before configuring emergency manuaaly - code should be left in case of need)
        """
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(video_fps)  # one frame per second

        samples = []
        frame_count = 0
        while cap.isOpened() and len(samples) < n:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                samples.append(frame)
            frame_count += 1
        cap.release()
        return samples

    def stream_frames(self, fps=5):
        """
        GENERATOR - yields {"timestamp": float_seconds, "frame": frame}
        one at a time at the requested fps. Never holds all frames in memory.

        timestamp is a float (e.g. 12.0, 12.2, 12.4 ...) so main.py can tell
        which frames fall on whole seconds (export frames).
        """
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
                    "timestamp": extracted / fps,
                    "frame": frame
                }
                extracted += 1
            frame_count += 1
        cap.release()
        print(f"Streamed {extracted} frames at {fps}Hz")

    def extract_frames(self, fps=1):
        """
        OLD list-returning method, kept for helper scripts
        (validate_distances.py, generate_validation_frames.py) that run at
        1Hz where memory is not a problem. New code should use stream_frames.
        """
        frames = []
        for item in self.stream_frames(fps=fps):
            frames.append({
                "timestamp": int(round(item["timestamp"])),
                "frame": item["frame"]
            })
        print(f"Extracted {len(frames)} frames at {fps}Hz")
        return frames

    def spatial_crop(self, frame):
        """Remove dashboard and sky - keep rows from 20% to 85% of height."""
        h, w = frame.shape[:2]
        top = int(h * 0.2)
        bottom = int(h * 0.85)
        return frame[top:bottom, 0:w]
