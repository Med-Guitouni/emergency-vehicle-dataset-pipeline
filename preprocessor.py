import cv2
import os


class VideoPreprocessor:

    def __init__(self, video_path, output_dir="output"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_frames(self, fps=1):
        """Extract 1 frame per second from video"""
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)

        frames = []
        frame_count = 0
        timestamp = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frames.append({
                    "timestamp": timestamp,
                    "frame": frame
                })
                timestamp += 1
            frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} frames at {fps}Hz")
        return frames

    def spatial_crop(self, frame):
        """Remove dashboard and sky, keep road area only"""
        h, w = frame.shape[:2]
        # keep middle 60% of frame vertically - removes sky top and dashboard bottom
        top = int(h * 0.2)
        bottom = int(h * 0.85)
        return frame[top:bottom, 0:w]

