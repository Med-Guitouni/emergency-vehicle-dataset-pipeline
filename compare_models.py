import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
import torch
import sys

# --- CONFIGURATION ---

# Directory containing the frames to be sampled for comparison.
# This should point to the specific video's review data folder.
VIDEO_NAME = "20240614_HORROR_STAU_OHNE_RETT"
FRAME_SOURCE_DIR = os.path.join("review_data", VIDEO_NAME)

# The YOLOv8 models to compare.
# Assumes these .pt files are in the project's root directory.
MODEL_PATHS = {
    "YOLOv8x (Official)": "yolov8x.pt",
    "KITTI Finetune": "vehicle_kitti_v0_last.pt"
}

# Number of random frames to process.
NUM_FRAMES_TO_SAMPLE = 150

# Directory where the comparison images will be saved.
OUTPUT_DIR = "model_comparison_results/" + VIDEO_NAME

# --- SCRIPT ---

def get_device():
    """Checks for available hardware backends and returns the best one."""
    if torch.backends.mps.is_available():
        print("  [info] MPS (Apple Silicon GPU) backend is available. Using MPS.")
        return "mps"
    if torch.cuda.is_available():
        print("  [info] CUDA backend is available. Using CUDA.")
        return "cuda"
    print("  [info] No GPU backend found. Using CPU.")
    return "cpu"

def draw_label(image, text, color=(0, 0, 0), bg_color=(255, 255, 255)):
    """Draws a text label with a background on the top-left of an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 10
    rect_start = (0, 0)
    rect_end = (text_width + 2 * padding, text_height + 2 * padding + baseline)
    cv2.rectangle(image, rect_start, rect_end, bg_color, -1)
    cv2.putText(image, text, (padding, text_height + padding), font, font_scale, color, thickness)
    return image

def main():
    """
    Main function to load models, process frames, and save comparison images.
    """
    print("--- YOLOv8 Model Comparison Script ---")

    # 1. Validate paths and setup
    if not os.path.isdir(FRAME_SOURCE_DIR):
        print(f"ERROR: Frame source directory not found at '{FRAME_SOURCE_DIR}'", file=sys.stderr)
        print("Please make sure you have run the main pipeline on the video first to generate review frames.", file=sys.stderr)
        sys.exit(1)

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path) and 'yolov8' not in path:
            print(f"WARNING: Model file not found for '{name}' at '{path}'.", file=sys.stderr)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: '{OUTPUT_DIR}'")

    # 2. Select random frames
    all_frames = [f for f in os.listdir(FRAME_SOURCE_DIR) if f.endswith(".jpg")]
    if not all_frames:
        print(f"ERROR: No frames found in '{FRAME_SOURCE_DIR}'.", file=sys.stderr)
        sys.exit(1)
    
    num_to_sample = min(NUM_FRAMES_TO_SAMPLE, len(all_frames))
    print(f"Found {len(all_frames)} frames. Randomly sampling {num_to_sample}.")
    sampled_frame_names = random.sample(all_frames, num_to_sample)

    # 3. Load models
    device = get_device()
    models = {}
    print("\nLoading models...")
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = YOLO(path)
            print(f"  - Loaded '{name}'")
        except Exception as e:
            print(f"ERROR: Failed to load model '{name}' from '{path}'. Reason: {e}", file=sys.stderr)
            sys.exit(1)
    
    # 4. Process each frame
    print("\nProcessing frames...")
    for i, frame_name in enumerate(sampled_frame_names):
        print(f"  - Processing frame {i+1}/{num_to_sample}: {frame_name}")
        frame_path = os.path.join(FRAME_SOURCE_DIR, frame_name)
        original_frame = cv2.imread(frame_path)
        print(original_frame.shape)
        exit(0)
        if original_frame is None:
            print(f"    WARNING: Could not read frame '{frame_name}'. Skipping.", file=sys.stderr)
            continue

        annotated_frames = []
        for model_name, model in models.items():
            results = model.predict(original_frame, device=device, verbose=False, conf=0.25)
            annotated_frame = results[0].plot()
            annotated_frame_with_label = draw_label(annotated_frame.copy(), model_name)
            annotated_frames.append(annotated_frame_with_label)

        if annotated_frames:
            comparison_image = cv2.hconcat(annotated_frames)
            output_filename = f"comparison_{frame_name}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, comparison_image)

    print(f"\n--- Done! ---\nComparison images saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()