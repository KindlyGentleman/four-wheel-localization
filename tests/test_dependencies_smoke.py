"""
tests.test_dependencies_smoke

Smoke tests for dependencies, packaging, and CLI entry points.
Ensures that the repository installs, imports, and exposes all public interfaces
in clean Python environments without undeclared prerequisites.
"""

import subprocess
import sys
from pathlib import Path


def test_public_imports_smoke():
    """Verify that all core public packages and submodules import without error."""
    import numpy
    import scipy
    import matplotlib
    import pandas
    import yaml
    import cv2
    import PIL
    import rosbags

    import robot_localization
    import robot_localization.models
    import robot_localization.sensors
    import robot_localization.simulation
    import robot_localization.estimation
    import robot_localization.control
    import robot_localization.evaluation
    import robot_localization.visualization

    import realdata
    import reproduce


def test_reproduce_cli_help():
    """Verify that reproduce.py CLI help functions out of the box."""
    root_dir = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root_dir / "reproduce.py"), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root_dir))
    assert res.returncode == 0, f"reproduce.py --help failed: {res.stderr}"
    assert "Reproduction CLI" in res.stdout
    assert "--quick" in res.stdout
    assert "--real-data" in res.stdout
