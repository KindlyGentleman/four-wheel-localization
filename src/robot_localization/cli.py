"""
robot_localization.cli

Command-Line Interface (CLI) entry point for the 4WD mobile robot localization project.
Provides commands to run benchmarks, generate publication figures, render video, and run tests.
"""

import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="4WD Robot Localization with EKF-Based Multi-Sensor Fusion"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("test", help="Run full pytest suite (21 unit tests)")
    subparsers.add_parser("reproduce-reference", help="Reproduce and audit PythonRobotics 2D EKF")
    subparsers.add_parser("run-quick", help="Execute deterministic benchmarks E1-E9 and E12")
    subparsers.add_parser("run-full", help="Execute multi-trial Monte Carlo statistical benchmark")
    subparsers.add_parser("figures", help="Generate all 12 IEEE figures and 6 vector illustrations")
    subparsers.add_parser("video", help="Render 1080p MP4 demonstration video")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    if args.command == "test":
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=project_root))
    elif args.command == "reproduce-reference":
        sys.exit(subprocess.call([sys.executable, "scripts/reproduce_reference.py"], cwd=project_root))
    elif args.command == "run-quick":
        sys.exit(subprocess.call([sys.executable, "scripts/run_quick.py"], cwd=project_root))
    elif args.command == "run-full":
        sys.exit(subprocess.call([sys.executable, "scripts/run_full.py"], cwd=project_root))
    elif args.command == "figures":
        sys.exit(subprocess.call([sys.executable, "scripts/generate_figures.py"], cwd=project_root))
    elif args.command == "video":
        sys.exit(subprocess.call([sys.executable, "scripts/render_video.py"], cwd=project_root))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
