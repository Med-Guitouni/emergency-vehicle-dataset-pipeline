import cv2
import numpy as np
from preprocessor import VideoPreprocessor
import os

video = [f for f in os.listdir('videos') if f.endswith('.mp4')][0]
p = VideoPreprocessor(f'videos/{video}')
frames = p.extract_frames(fps=1)

BLUE_LOW = np.array([100, 150, 100])
BLUE_HIGH = np.array([130, 255, 255])

for t in [0, 5, 10, 15, 20, 21, 25, 30]:
    frame = p.spatial_crop(frames[t]['frame'])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    blue_pixels = np.sum(mask > 0)
    print(f't={t}s blue_pixels={blue_pixels}')