"""
visualize.py

Top-level convenience facade for plotting, IEEE figures, animations,
and ROS 2-style topic bus and TF tree visualization.
"""

import sys
from pathlib import Path

# Ensure local src/ is accessible
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from robot_localization.visualization.ros2_mock import (
    ROS2MockBus,
    MockTFTree,
    TransformStamped,
    PointStamped
)
from robot_localization.visualization.figures import (
    plot_trajectory_comparison,
    plot_error_analysis,
    plot_nees_consistency
)
from robot_localization.visualization.ieee_style import (
    apply_ieee_style,
    IEEE_SINGLE_COL_WIDTH,
    IEEE_DOUBLE_COL_WIDTH
)

__all__ = [
    "ROS2MockBus",
    "MockTFTree",
    "TransformStamped",
    "PointStamped",
    "plot_trajectory_comparison",
    "plot_error_analysis",
    "plot_nees_consistency",
    "apply_ieee_style",
    "IEEE_SINGLE_COL_WIDTH",
    "IEEE_DOUBLE_COL_WIDTH",
]
