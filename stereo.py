"""
stereo.py

Top-level convenience facade for calibrated stereo vision, depth reconstruction,
3D point-cloud generation, and visual odometry.
"""

import sys
from pathlib import Path

# Ensure local src/ is accessible
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from robot_localization.sensors.stereo import (
    StereoCameraConfig,
    VisualLandmark3D,
    SyntheticStereoArena,
    StereoDepthPipeline,
    StereoDepthResult,
    StereoVisualOdometry,
    VisualOdometryResult,
    ICPRegistrationResult,
    register_point_clouds_icp,
    apply_visual_degradation,
    enhance_stereo_images,
    transform_optical_to_base,
    transform_base_to_optical,
    R_BASE_TO_OPTICAL,
    R_OPTICAL_TO_BASE,
)

# Backwards-compatible aliases
Landmark3D = VisualLandmark3D
DepthResult = StereoDepthResult
VOResult = VisualOdometryResult
enhance_stereo_image = enhance_stereo_images
optical_to_base_link = transform_optical_to_base
base_link_to_optical = transform_base_to_optical

__all__ = [
    "StereoCameraConfig",
    "VisualLandmark3D",
    "Landmark3D",
    "SyntheticStereoArena",
    "StereoDepthPipeline",
    "StereoDepthResult",
    "DepthResult",
    "StereoVisualOdometry",
    "VisualOdometryResult",
    "VOResult",
    "ICPRegistrationResult",
    "register_point_clouds_icp",
    "apply_visual_degradation",
    "enhance_stereo_images",
    "enhance_stereo_image",
    "transform_optical_to_base",
    "transform_base_to_optical",
    "optical_to_base_link",
    "base_link_to_optical",
    "R_BASE_TO_OPTICAL",
    "R_OPTICAL_TO_BASE",
]
