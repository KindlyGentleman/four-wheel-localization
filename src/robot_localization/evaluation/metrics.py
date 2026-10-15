"""
robot_localization.evaluation.metrics

Statistical state estimation metrics:
- 2D Position RMSE, MAE, Max Error, Final Error
- Heading RMSE with proper angular wrapping
- Drift percentage of total traveled distance
- Normalized Estimation Error Squared (NEES) for 2D position and 3D pose
- Normalized Innovation Squared (NIS) for measurement updates
- Covariance consistency & 2-sigma / 3-sigma theoretical coverage
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from robot_localization.models.kinematics import wrap_angle


@dataclass
class TrajectoryMetrics:
    """Comprehensive evaluation metrics for an estimated trajectory against ground truth."""
    baseline_name: str
    pos_rmse_m: float
    pos_mae_m: float
    pos_max_err_m: float
    pos_final_err_m: float
    heading_rmse_rad: float
    heading_rmse_deg: float
    heading_max_err_deg: float
    total_distance_traveled_m: float
    drift_percent: float              # Final position error / total distance * 100
    mean_nees_pos: float              # Expected: 2.0 for 2-DOF position
    mean_nees_pose: float             # Expected: 3.0 for 3-DOF pose
    coverage_2sigma: float            # Expected: ~95.4% for chi2(2) bound = 5.991
    coverage_3sigma: float            # Expected: ~99.7% for chi2(2) bound = 11.829
    mean_nis_gnss: Optional[float]    # Expected: 2.0 for 2-DOF GNSS position
    mean_nis_compass: Optional[float] # Expected: 1.0 for 1-DOF heading


def compute_nees_series(
    gt_poses: np.ndarray,
    est_poses: np.ndarray,
    cov_diags: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NEES time series for position (2-DOF) and full pose (3-DOF).

    Note: When full covariance matrices are diagonalized or stored as diags,
    we compute:
        nees_pos = (dx^2 / var_x) + (dy^2 / var_y)
        nees_pose = nees_pos + (dtheta^2 / var_theta)
    """
    dx = gt_poses[:, 0] - est_poses[:, 0]
    dy = gt_poses[:, 1] - est_poses[:, 1]

    dtheta = np.zeros_like(dx)
    for i in range(len(dx)):
        dtheta[i] = wrap_angle(gt_poses[i, 2] - est_poses[i, 2])

    var_x = np.maximum(cov_diags[:, 0], 1e-12)
    var_y = np.maximum(cov_diags[:, 1], 1e-12)
    var_th = np.maximum(cov_diags[:, 2], 1e-12)

    nees_pos = (dx ** 2 / var_x) + (dy ** 2 / var_y)
    nees_pose = nees_pos + (dtheta ** 2 / var_th)
    return nees_pos, nees_pose


def compute_covariance_coverage(
    gt_poses: np.ndarray,
    est_poses: np.ndarray,
    cov_diags: np.ndarray
) -> Tuple[float, float]:
    """
    Compute empirical 2-sigma and 3-sigma error ellipse coverage.
    For 2-DOF position, the 95.45% chi-square threshold is 5.991,
    and 99.73% threshold is 11.829.
    """
    nees_pos, _ = compute_nees_series(gt_poses, est_poses, cov_diags)
    # chi2(2) 95.45% quantile is 5.9915
    cov_2sig = float(np.mean(nees_pos <= 5.9915))
    # chi2(2) 99.73% quantile is 11.829
    cov_3sig = float(np.mean(nees_pos <= 11.829))
    return cov_2sig, cov_3sig


def compute_nis_statistics(nis_records: List[Tuple[float, float]]) -> Optional[float]:
    """Compute average Normalized Innovation Squared from (timestamp, nis) tuples."""
    if not nis_records:
        return None
    values = [val for _, val in nis_records]
    return float(np.mean(values))


def compute_trajectory_metrics(
    baseline_name: str,
    timestamps_gt: np.ndarray,
    gt_poses: np.ndarray,
    timestamps_est: np.ndarray,
    est_poses: np.ndarray,
    cov_diags: np.ndarray,
    nis_gnss: Optional[List[Tuple[float, float]]] = None,
    nis_compass: Optional[List[Tuple[float, float]]] = None
) -> TrajectoryMetrics:
    """
    Compute full suite of benchmark metrics by interpolating ground truth to estimator timestamps.
    """
    # Interpolate GT to estimator timestamps
    gt_x_interp = np.interp(timestamps_est, timestamps_gt, gt_poses[:, 0])
    gt_y_interp = np.interp(timestamps_est, timestamps_gt, gt_poses[:, 1])

    # For heading interpolation, unwrap first to prevent 2*pi phase jumps
    gt_theta_unwrapped = np.unwrap(gt_poses[:, 2])
    gt_theta_interp = wrap_angle_array(np.interp(timestamps_est, timestamps_gt, gt_theta_unwrapped))

    gt_aligned = np.column_stack([gt_x_interp, gt_y_interp, gt_theta_interp])

    # Position errors
    dx = gt_aligned[:, 0] - est_poses[:, 0]
    dy = gt_aligned[:, 1] - est_poses[:, 1]
    pos_err = np.hypot(dx, dy)

    pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))
    pos_mae = float(np.mean(pos_err))
    pos_max = float(np.max(pos_err))
    pos_final = float(pos_err[-1])

    # Heading errors with wrapping
    dtheta = np.zeros(len(timestamps_est))
    for i in range(len(timestamps_est)):
        dtheta[i] = wrap_angle(gt_aligned[i, 2] - est_poses[i, 2])
    heading_rmse = float(np.sqrt(np.mean(dtheta ** 2)))
    heading_max = float(np.max(np.abs(dtheta)))

    # Total distance traveled
    diffs = np.diff(gt_poses[:, :2], axis=0)
    total_dist = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    drift_pct = (pos_final / total_dist * 100.0) if total_dist > 0 else 0.0

    # Consistency metrics
    nees_pos, nees_pose = compute_nees_series(gt_aligned, est_poses, cov_diags)
    mean_nees_p = float(np.mean(nees_pos))
    mean_nees_all = float(np.mean(nees_pose))

    cov_2sig, cov_3sig = compute_covariance_coverage(gt_aligned, est_poses, cov_diags)

    mean_nis_g = compute_nis_statistics(nis_gnss) if nis_gnss else None
    mean_nis_c = compute_nis_statistics(nis_compass) if nis_compass else None

    return TrajectoryMetrics(
        baseline_name=baseline_name,
        pos_rmse_m=pos_rmse,
        pos_mae_m=pos_mae,
        pos_max_err_m=pos_max,
        pos_final_err_m=pos_final,
        heading_rmse_rad=heading_rmse,
        heading_rmse_deg=math.degrees(heading_rmse),
        heading_max_err_deg=math.degrees(heading_max),
        total_distance_traveled_m=total_dist,
        drift_percent=drift_pct,
        mean_nees_pos=mean_nees_p,
        mean_nees_pose=mean_nees_all,
        coverage_2sigma=cov_2sig,
        coverage_3sigma=cov_3sig,
        mean_nis_gnss=mean_nis_g,
        mean_nis_compass=mean_nis_c
    )


def wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angles), np.cos(angles))
