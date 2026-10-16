"""
localization.py

Top-level convenience facade for four-wheel robot state estimation.
Exposes classical, bias-aware, robust, and recovering state estimators.

Available Estimators:
  - DeadReckoning4WD: Open-loop kinematic odometry
  - ExtendedKalmanFilter3D: 3-state [x, y, yaw] standard EKF
  - BiasAwareEKF4D: 4-state [x, y, yaw, gyro_bias] EKF
  - UnscentedKalmanFilter4D: 4-state circular manifold UKF on S^1
  - RobustEKF4D: Chi-square gated Huber robust EKF
  - RecoverableRobustEKF4D: Fault-tolerant EKF with temporal consistency verification
  - ParticleFilter2D: Sampling-importance resampling particle filter
  - AMCLAdaptiveParticleFilter2D: KLD-sampling particle filter with dual-rate recovery
"""

import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Ensure local src/ is accessible
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.estimation.dead_reckoning import DeadReckoning4WD
from robot_localization.estimation.ekf import ExtendedKalmanFilter3D
from robot_localization.estimation.bias_aware_ekf import BiasAwareEKF4D
from robot_localization.estimation.ukf import UnscentedKalmanFilter4D
from robot_localization.estimation.robust_ekf import RobustEKF4D
from robot_localization.estimation.gnss_recovery import (
    RecoverableRobustEKF4D,
    RecoveryStrategy,
    EstimatorHealthState,
)
from robot_localization.estimation.particle_filter import (
    OccupancyMap2D,
    ParticleFilter2D,
    AMCLAdaptiveParticleFilter2D,
)
from robot_localization.estimation.range_assisted_ekf import RangeAssistedEKF

RecoveryState = EstimatorHealthState
RangeAssistedEKF3D = RangeAssistedEKF


def create_estimator(
    method: str = "recoverable_ekf",
    config: Optional[Robot4WDConfig] = None,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
    **kwargs
) -> Any:
    """
    Factory creating a state estimator instance by name.

    Args:
        method: One of ['dead_reckoning', 'ekf3d', 'bias_aware_ekf', 'ukf',
                        'robust_ekf', 'recoverable_ekf', 'particle_filter', 'amcl', 'lidar_ekf']
        config: Rover configuration (Robot4WDConfig)
        recovery_strategy: Recovery strategy for RecoverableRobustEKF4D
        kwargs: Additional arguments passed to the estimator constructor

    Returns:
        State estimator instance
    """
    cfg = config or Robot4WDConfig()
    m = method.lower().strip()

    if m in ["dead_reckoning", "dr", "b0"]:
        return DeadReckoning4WD(cfg, **kwargs)
    elif m in ["ekf3d", "standard_ekf", "b3", "ekf_3d", "ekf"]:
        return ExtendedKalmanFilter3D(**kwargs)
    elif m in ["bias_aware_ekf", "bias_ekf", "ekf4d", "b5"]:
        return BiasAwareEKF4D(**kwargs)
    elif m in ["ukf", "manifold_ukf", "a4", "ukf_4d"]:
        return UnscentedKalmanFilter4D(**kwargs)
    elif m in ["robust_ekf", "gated_ekf", "a5"]:
        return RobustEKF4D(**kwargs)
    elif m in ["recoverable_ekf", "gnss_recovery", "fault_tolerant_ekf", "r1"]:
        return RecoverableRobustEKF4D(strategy=recovery_strategy, **kwargs)
    elif m in ["particle_filter", "pf", "a6"]:
        occ_map = kwargs.pop("occupancy_map", OccupancyMap2D.create_volcanic_testbed())
        return ParticleFilter2D(occupancy_map=occ_map, **kwargs)
    elif m in ["amcl", "adaptive_pf", "a7"]:
        occ_map = kwargs.pop("occupancy_map", OccupancyMap2D.create_volcanic_testbed())
        return AMCLAdaptiveParticleFilter2D(occupancy_map=occ_map, **kwargs)
    elif m in ["lidar_ekf", "range_assisted_ekf", "b6"]:
        return RangeAssistedEKF(**kwargs)
    else:
        raise ValueError(
            f"Unknown estimator '{method}'. Valid options: "
            "['dead_reckoning', 'ekf3d', 'bias_aware_ekf', 'ukf', "
            "'robust_ekf', 'recoverable_ekf', 'particle_filter', 'amcl', 'lidar_ekf']"
        )


build_estimator = create_estimator

__all__ = [
    "Robot4WDConfig",
    "DeadReckoning4WD",
    "ExtendedKalmanFilter3D",
    "BiasAwareEKF4D",
    "UnscentedKalmanFilter4D",
    "RobustEKF4D",
    "RecoverableRobustEKF4D",
    "RecoveryStrategy",
    "EstimatorHealthState",
    "RecoveryState",
    "OccupancyMap2D",
    "ParticleFilter2D",
    "AMCLAdaptiveParticleFilter2D",
    "RangeAssistedEKF",
    "RangeAssistedEKF3D",
    "create_estimator",
    "build_estimator",
]
