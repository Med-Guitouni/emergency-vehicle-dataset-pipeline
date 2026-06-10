import onnxruntime as ort
import numpy as np
import cv2
import os

class LaneDetector:

    # not used by anything. It exists purely as documentation of what was attempted
    # (UFLD v2, YOLOP, both failed) and as a placeholder
    """
ROAD BOUNDARY DETECTION - INCOMPLETE - NEEDS WORK

What I tried:
I attempted to detect road boundaries (driving lanes + shoulders) to identify
when a vehicle has moved off the road to make space for the ambulance.

Approach 1 - UFLD v2 (Ultra Fast Lane Detection):
I used a pretrained ONNX model to detect lane line positions as pixel coordinates.
Model file: models/ufldv2_tusimple_res18_320x800.onnx
Problem: when vehicles sit on the shoulder they physically cover the lane markings.
The model detects the vehicle as the boundary instead of the actual painted line.
Result: boundaries shift when vehicles yield, making it useless for detecting yielding.

Approach 2 - YOLOP :
I switched to YOLOP which segments the full drivable area and lane lines as pixel masks.
Model file: models/yolop-640-640.onnx
The lane line segmentation (output[2]) worked better than UFLD in occluded frames.
Problem 1: the drivable area mask (output[1]) incorrectly includes the shoulder in green.
Problem 2: lane line detection still shifts when vehicles are on the shoulder because
the vehicles themselves are detected as visual boundaries.
Result: same fundamental failure as UFLD - boundaries are not static.


"""

    # CULane input size expected by the model
    INPUT_WIDTH = 800
    INPUT_HEIGHT = 320

    # CULane row anchors - vertical positions where lane points are predicted
    # 18 anchor rows distributed across the lower portion of the frame
    ROW_ANCHORS = [121, 131, 141, 150, 160, 170, 180, 189, 199,
                   209, 219, 228, 238, 248, 258, 267, 277, 287]

    LANE_WIDTH_METERS = 3.75

    def __init__(self, model_path="models/ufldv2_tusimple_res18_320x800.onnx"):
        if not os.path.exists(model_path):
            print(f"Lane detection model not found at {model_path}")
            print("Lane detection disabled")
            self.session = None
            return

        print("Loading UFLD lane detection model...")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        print("Lane detector ready")

    def preprocess(self, frame):
        """Resize and normalize frame for UFLD input"""
        img = cv2.resize(frame, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        img = img[:, :, ::-1]  # BGR to RGB
        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img[np.newaxis, :].astype(np.float32)
        return img

    def detect(self, frame):
        """
        Detect lane boundaries using UFLD v2 hybrid anchor output.
        Uses existence output to filter low confidence predictions.
        loc_row: (num_grid_row, num_cls_row, num_lane_row)
        exist_row: (2, num_cls_row, num_lane_row) - background/foreground scores
        """
        if self.session is None:
            return []

        h, w = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: input_tensor})

        loc_row = output[0][0]
        exist_row = output[2][0]

        num_grid_row, num_cls_row, num_lane_row = loc_row.shape

        row_anchor = np.linspace(0.42, 1, num_cls_row)

        loc_row_softmax = self._softmax(loc_row, axis=0)
        exist_row_prob = self._softmax(exist_row, axis=0)[1]

        row_positions = np.argmax(loc_row_softmax, axis=0)

        lanes = []

        for lane_idx in range(num_lane_row):
            lane_points = []
            for row_idx in range(num_cls_row):

                # skip low confidence points
                conf = exist_row_prob[row_idx, lane_idx]
                if conf < 0.5:
                    continue

                grid_x = row_positions[row_idx, lane_idx]

                # skip invalid grid positions
                if grid_x >= num_grid_row - 1:
                    continue

                x_pixel = int(grid_x / (num_grid_row - 1) * w)
                y_pixel = int(row_anchor[row_idx] * h)

                lane_points.append((x_pixel, y_pixel))

            if len(lane_points) >= 3:
                lanes.append(lane_points)

        return lanes

    def _softmax(self, x, axis=0):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def get_lane_boundaries(self, frame):
        """
        Returns left and right boundary x positions at bottom of frame.
        Uses inner lane markings (not outermost boundaries) to define
        the actual driving area. Vehicles outside these are on shoulder.
        """
        lanes = self.detect(frame)

        if len(lanes) < 2:
            return None

        w = frame.shape[1]
        center = w // 2

        # get bottom x position of each lane
        left_lanes = [(l, l[-1][0]) for l in lanes if l[-1][0] < center]
        right_lanes = [(l, l[-1][0]) for l in lanes if l[-1][0] >= center]

        if not left_lanes or not right_lanes:
            return None

        # use innermost lanes (closest to center) not outermost
        left_boundary = max(left_lanes, key=lambda x: x[1])[1]
        right_boundary = min(right_lanes, key=lambda x: x[1])[1]

        return left_boundary, right_boundary

    def is_outside_road(self, vehicle_center_x, frame, margin_px=20):
        """
        Check if a vehicle center x position is outside the road boundaries.
        margin_px adds tolerance to account for detection noise.
        """
        boundaries = self.get_lane_boundaries(frame)

        if boundaries is None or boundaries[0] is None or boundaries[1] is None:
            return False  # cannot determine, assume inside

        left_boundary, right_boundary = boundaries
        x = vehicle_center_x

        return x < (left_boundary - margin_px) or x > (right_boundary + margin_px)