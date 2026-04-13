"""
bac-detection-standalone -- Waste bin detection from video files.
No cloud, no infra, just detection + local results.

Usage:
    python main.py --mode video  --video path/to/video.mp4   # batch (recommended)
    python main.py --mode stream --video path/to/video.mp4   # cycle-based stream
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Waste bin detection (standalone)")
    parser.add_argument("--mode", default="video", choices=["video", "stream"],
                        help="video = batch (fast, accurate) | "
                             "stream = cycle-based (IDLE/ACTIVE/END)")
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()

    print("=" * 60)
    print("  bac-detection-standalone  |  Waste Bin Detection")
    print("=" * 60)

    from config import cfg

    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    if args.mode == "video":
        from pipeline.video_mode import run
    else:
        from pipeline.stream_mode import run

    result = run(args.video)
    return result


if __name__ == "__main__":
    main()
