import argparse
import subprocess
import os
import sys


def crop_video(video_path, start_seconds, end_seconds, output_path):
    """
    Crops a video to keep the segment from start_seconds to end_seconds using ffmpeg.
    This uses stream copy (-c copy) to avoid re-encoding, which is fast and preserves quality.
    """
    if not os.path.exists(video_path):
        print(f"Error: Input video not found at {video_path}", file=sys.stderr)
        return False

    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', video_path,
        '-ss', str(start_seconds),
        '-to', str(end_seconds),
        '-c', 'copy',
        '-avoid_negative_ts', '1',  # Good practice when cutting
        output_path
    ]

    print(f"Running command: {' '.join(command)}")

    try:
        # Using subprocess.run to execute the command.
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        # ffmpeg often prints progress and other info to stderr.
        if result.stderr:
            print("--- ffmpeg output ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            print("---------------------", file=sys.stderr)
        print(f"\nSuccessfully cropped video and saved to {output_path}")
        return True
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not in your PATH.", file=sys.stderr)
        print("Please install ffmpeg. On macOS with Homebrew: 'brew install ffmpeg'", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print("Error during ffmpeg execution:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Crop a video to a specified time range using ffmpeg. This script KEEPS the specified segment and saves it to a new file with '_cropped_START-END' appended.",
        epilog="Example: python3 crop_video.py --video videos/my_video.mp4 --start 60 --end 120",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--video',
        required=True,
        type=str,
        help="Path to the input video file."
    )
    parser.add_argument(
        '--start',
        required=True,
        type=float,
        help="Start time of the segment to keep, in seconds."
    )
    parser.add_argument(
        '--end',
        required=True,
        type=float,
        help="End time of the segment to keep, in seconds."
    )

    args = parser.parse_args()

    if args.start >= args.end:
        print("Error: Start time must be less than end time.", file=sys.stderr)
        sys.exit(1)
    
    video_dir = os.path.dirname(args.video) or "."
    base_name = os.path.basename(args.video)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(video_dir, f"{name}_cropped_{int(args.start)}-{int(args.end)}{ext}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not crop_video(args.video, args.start, args.end, output_path):
        sys.exit(1)


if __name__ == "__main__":
    main()