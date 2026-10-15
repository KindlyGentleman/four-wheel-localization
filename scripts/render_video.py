#!/usr/bin/env python3
"""
scripts/render_video.py

High-definition MP4 scientific animation generator.
Renders the complete 1920x1080 30-60 second multi-panel demonstration video:
`videos/localization_demo.mp4` with `videos/localization_demo_preview.png`
and `videos/video_metadata.json`.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_localization.visualization.animation import render_localization_video


def main():
    print("=" * 75)
    print("RENDERING 1920x1080 HIGH-DEFINITION DEMONSTRATION VIDEO")
    print("=" * 75)

    video_dir = PROJECT_ROOT / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results"

    data_file = results_dir / "saved_benchmark_traces.npz"
    if not data_file.exists():
        print(f"Error: Required trace file {data_file} not found. Run scripts/run_quick.py first.")
        sys.exit(1)

    mp4_path = video_dir / "localization_demo.mp4"
    preview_path = video_dir / "localization_demo_preview.png"
    metadata_path = video_dir / "video_metadata.json"

    # Render 30-second animation at 30 FPS = 900 frames
    render_localization_video(
        data_npz_path=data_file,
        output_mp4_path=mp4_path,
        preview_png_path=preview_path,
        metadata_json_path=metadata_path,
        fps=30,
        duration_s=30.0,
        dpi=150 # 12.8 x 7.2 in @ 150 DPI = 1920 x 1080
    )

    print("=" * 75)
    print(f"Video rendering complete: {mp4_path}")
    print(f"Preview image: {preview_path}")
    print(f"Metadata file: {metadata_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
