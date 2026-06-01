import onnxruntime as ort
import numpy as np
import cv2
import os
from preprocessor import VideoPreprocessor

session = ort.InferenceSession('models/yolop-640-640.onnx')
input_name = session.get_inputs()[0].name

video = [f for f in os.listdir('videos') if f.endswith('.mp4')][0]
p = VideoPreprocessor(f'videos/{video}')
frames = p.extract_frames(fps=1)


def get_lane_boundaries(frame):
    h, w = frame.shape[:2]

    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

    outputs = session.run(None, {input_name: img})

    lane_line = outputs[2][0]
    lane_mask = np.argmax(lane_line, axis=0)
    lane_mask = cv2.resize(lane_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # look at bottom third of frame where boundaries are clearest
    bottom_region = lane_mask[int(h * 0.7):, :]

    # find all x positions where lane lines exist in bottom region
    lane_pixels = np.where(bottom_region == 1)
    if len(lane_pixels[1]) < 10:
        return None, None

    x_positions = lane_pixels[1]
    left_boundary = int(np.percentile(x_positions, 5))
    right_boundary = int(np.percentile(x_positions, 95))

    return left_boundary, right_boundary


test_timestamps = [25, 30, 45, 60, 120, 149]

for t in test_timestamps:
    frame = frames[t]['frame']
    left, right = get_lane_boundaries(frame)
    print(f't={t}s  left={left}  right={right}  frame_width={frame.shape[1]}')