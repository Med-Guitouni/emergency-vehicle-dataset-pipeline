import os
import cv2
from ultralytics import YOLO
import torch
import sys
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_VIDEO = "videos/20240614_HORROR_STAU_OHNE_RETTUNGSGASSE_A3_Einsatzfahrt_Inside_View_cropped_70-223.mp4"
MODEL_PATH = "yolov8x.pt"  # Or "best.pt" if you prefer your custom model

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

def main():
    """
    Main function to load the model, run tracking on a video, and save the annotated video.
    """
    print("--- YOLOv8 Tracking Script ---")

    # 1. Validate paths and setup
    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: Input video not found at '{INPUT_VIDEO}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'", file=sys.stderr)
        sys.exit(1)

    # 2. Load model and select device
    device = get_device()
    try:
        model = YOLO(MODEL_PATH)
        print(f"  - Loaded model '{MODEL_PATH}'")
    except Exception as e:
        print(f"ERROR: Failed to load model. Reason: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Setup video capture and writer
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file '{INPUT_VIDEO}'", file=sys.stderr)
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    TARGET_FPS = 5

    video_dir = os.path.dirname(INPUT_VIDEO) or "."
    base_name = os.path.basename(INPUT_VIDEO)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(video_dir, f"{name}_tracked{ext}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (frame_width, frame_height))
    print(f"Output video will be saved to: '{output_path}'")

    # 4. Process video frame by frame
    print(f"\nProcessing video at {TARGET_FPS} FPS (original: {original_fps:.2f} FPS)...")
    frame_interval = max(1, round(original_fps / TARGET_FPS))
    num_frames_to_process = total_frames // frame_interval

    with tqdm(total=int(num_frames_to_process), desc="Tracking video") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Run tracking. persist=True is essential to maintain track IDs between frames.
                results = model.track(frame, tracker="botsort.yaml", persist=True, device=device, verbose=False)
                annotated_frame = results[0].plot()
                writer.write(annotated_frame)
                pbar.update(1)
            frame_count += 1

    cap.release()
    writer.release()
    print(f"\n--- Done! ---\nTracked video saved to '{output_path}'.")

if __name__ == "__main__":
    main()