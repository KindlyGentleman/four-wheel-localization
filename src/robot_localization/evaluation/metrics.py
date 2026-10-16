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
    coverage_95: float                # Expected: 95.0% for chi2(2) bound = 5.9915
    coverage_2sigma: float            # Expected: 86.47% for Mahalanobis radius r=2 (r^2=4.0)
    coverage_3sigma: float            # Expected: ~99.73% for chi2(2) bound = 11.829
    mean_nis_gnss: Optional[float]    # Expected: 2.0 for 2-DOF GNSS position
    mean_nis_compass: Optional[float] # Expected: 1.0 for 1-DOF heading


def compute_nees_series(
    gt_poses: np.ndarray,
    est_poses: np.ndarray,
    cov_diags: np.ndarray,
    cov_mats: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NEES time series for position (2-DOF) and full pose (3-DOF).

    When full covariance matrices cov_mats are supplied (M, 2, 2) or (M, 3, 3),
    computes the exact coordinate-invariant Mahalanobis distance squared:
        D^2 = e_xy^T P_xy^{-1} e_xy
    Otherwise, evaluates diagonal approximation:
        nees_pos = (dx^2 / var_x) + (dy^2 / var_y)
    """
    dx = gt_poses[:, 0] - est_poses[:, 0]
    dy = gt_poses[:, 1] - est_poses[:, 1]

    dtheta = np.zeros_like(dx)
    for i in range(len(dx)):
        dtheta[i] = wrap_angle(gt_poses[i, 2] - est_poses[i, 2])

    var_th = np.maximum(cov_diags[:, 2], 1e-12)

    if cov_mats is not None and cov_mats.ndim == 3 and cov_mats.shape[1] >= 2 and cov_mats.shape[2] >= 2:
        nees_pos = np.zeros(len(dx))
        for i in range(len(dx)):
            P_xy = cov_mats[i, :2, :2]
            det = P_xy[0, 0] * P_xy[1, 1] - P_xy[0, 1] * P_xy[1, 0]
            if det > 1e-14:
                # 2x2 analytic inverse
                nees_pos[i] = (P_xy[1, 1] * (dx[i] ** 2) - 2.0 * P_xy[0, 1] * dx[i] * dy[i] + P_xy[0, 0] * (dy[i] ** 2)) / det
            else:
                var_x = max(float(cov_diags[i, 0]), 1e-12)
                var_y = max(float(cov_diags[i, 1]), 1e-12)
                nees_pos[i] = (dx[i] ** 2 / var_x) + (dy[i] ** 2 / var_y)
    else:
        var_x = np.maximum(cov_diags[:, 0], 1e-12)
        var_y = np.maximum(cov_diags[:, 1], 1e-12)
        nees_pos = (dx ** 2 / var_x) + (dy ** 2 / var_y)

    nees_pose = nees_pos + (dtheta ** 2 / var_th)
    return nees_pos, nees_pose


def compute_covariance_coverage(
    gt_poses: np.ndarray,
    est_poses: np.ndarray,
    cov_diags: np.ndarray,
    cov_mats: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """
    Compute empirical coverage of theoretical confidence regions for 2D position:
    1. coverage_95: true chi2(2) 95.00% quantile threshold (s = 5.9915, r = 2.448 sigma)
    2. coverage_2sigma: Mahalanobis radius r=2 (threshold s = 4.0, theoretical 86.47%)
    3. coverage_3sigma: chi2(2) 99.73% quantile threshold (s = 11.829)
    """
    nees_pos, _ = compute_nees_series(gt_poses, est_poses, cov_diags, cov_mats)
    cov_95 = float(np.mean(nees_pos <= 5.9915))
    cov_2sig = float(np.mean(nees_pos <= 4.0))
    cov_3sig = float(np.mean(nees_pos <= 11.829))
    return cov_95, cov_2sig, cov_3sig


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
    nis_compass: Optional[List[Tuple[float, float]]] = None,
    cov_mats: Optional[np.ndarray] = None
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
    nees_pos, nees_pose = compute_nees_series(gt_aligned, est_poses, cov_diags, cov_mats)
    mean_nees_p = float(np.mean(nees_pos))
    mean_nees_all = float(np.mean(nees_pose))

    cov_95, cov_2sig, cov_3sig = compute_covariance_coverage(gt_aligned, est_poses, cov_diags, cov_mats)

    mean_nis_g = compute_nis_statistics(nis_gnss) if (nis_gnss is not None and len(nis_gnss) > 0) else None
    mean_nis_c = compute_nis_statistics(nis_compass) if (nis_compass is not None and len(nis_compass) > 0) else None

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
        coverage_95=cov_95,
        coverage_2sigma=cov_2sig,
        coverage_3sigma=cov_3sig,
        mean_nis_gnss=mean_nis_g,
        mean_nis_compass=mean_nis_c
    )


def compute_interval_metrics(
    baseline_name: str,
    timestamps_gt: np.ndarray,
    gt_poses: np.ndarray,
    timestamps_est: np.ndarray,
    est_poses: np.ndarray,
    cov_diags: np.ndarray,
    t_start: float,
    t_end: float,
    cov_mats: Optional[np.ndarray] = None
) -> TrajectoryMetrics:
    """Evaluate metrics over a specific time window [t_start, t_end]."""
    mask_est = (timestamps_est >= t_start) & (timestamps_est <= t_end)
    mask_gt = (timestamps_gt >= t_start) & (timestamps_gt <= t_end)

    if np.sum(mask_est) < 2 or np.sum(mask_gt) < 2:
        return compute_trajectory_metrics(
            baseline_name=baseline_name,
            timestamps_gt=timestamps_gt,
            gt_poses=gt_poses,
            timestamps_est=timestamps_est,
            est_poses=est_poses,
            cov_diags=cov_diags,
            cov_mats=cov_mats
        )

    t_est_sub = timestamps_est[mask_est]
    est_poses_sub = est_poses[mask_est]
    cov_diags_sub = cov_diags[mask_est]
    cov_mats_sub = cov_mats[mask_est] if cov_mats is not None else None

    return compute_trajectory_metrics(
        baseline_name=f"{baseline_name}_interval_{t_start:.0f}_{t_end:.0f}",
        timestamps_gt=timestamps_gt[mask_gt],
        gt_poses=gt_poses[mask_gt],
        timestamps_est=t_est_sub,
        est_poses=est_poses_sub,
        cov_diags=cov_diags_sub,
        cov_mats=cov_mats_sub
    )


def wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angles), np.cos(angles))
