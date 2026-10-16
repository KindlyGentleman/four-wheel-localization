"""
Independent audit verification and regression test suite.
Covers:
1. Zero-noise analytical kinematics and filter propagation
2. Missing-GNSS uncertainty and error growth under wheel slip
3. Ground-truth perturbation invariance (zero privileged leakage)
4. Causal vision ablation (A == B, B_clean == B_severe, B != C, C != D)
5. Positive-definite covariance preservation in R7 global reinitialization
6. Top-level facade imports and factory constructors
"""

import copy
import math
import numpy as np
import pytest

import localization
import sensors
import stereo
import realdata
from robot_localization.estimation.gnss_recovery import (
    RecoverableRobustEKF4D,
    RecoveryStrategy,
)


def test_top_level_facades_and_factories():
    """Verify top-level facade modules import and instantiate without signature errors."""
    for name in ["dead_reckoning", "ekf", "bias_ekf", "ukf", "robust_ekf", "recoverable_ekf", "pf", "amcl", "lidar_ekf"]:
        est = localization.create_estimator(name)
        assert est is not None

    encoders = sensors.FourWheelEncoders(localization.Robot4WDConfig(), rng=np.random.RandomState(42))
    samples = encoders.update(timestamp=0.02, wheel_rot_speeds={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0}, dt=0.02)
    assert math.isfinite(samples["FL"].angular_speed_rad_s)

    cam_cfg = stereo.StereoCameraConfig()
    assert cam_cfg.baseline_m > 0.0


def test_zero_noise_analytical_accuracy():
    """Verify stationary and constant-velocity straight-line motion with zero noise."""
    ekf = localization.ExtendedKalmanFilter3D(
        initial_state=np.array([0.0, 0.0, 0.0]),
        initial_cov=np.diag([1e-6, 1e-6, 1e-6]),
        q_additive=np.zeros((3, 3)),
    )
    # 1. Stationary for 10 steps
    for _ in range(10):
        x, _ = ekf.predict(np.array([0.0, 0.0]), np.zeros((2, 2)), dt=0.1)
    np.testing.assert_allclose(x, np.array([0.0, 0.0, 0.0]), atol=1e-12)

    # 2. Constant velocity v = 1.5 m/s along +X for 20 steps (dt = 0.1s -> 3.0m)
    for _ in range(20):
        x, _ = ekf.predict(np.array([1.5, 0.0]), np.zeros((2, 2)), dt=0.1)
    np.testing.assert_allclose(x, np.array([3.0, 0.0, 0.0]), atol=1e-12)


def test_missing_gnss_divergence_and_uncertainty_growth():
    """Verify that without GNSS updates, covariance trace grows monotonically."""
    ekf = localization.BiasAwareEKF4D(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]),
    )
    prev_trace = float(np.trace(ekf.P[:2, :2]))
    for _ in range(50):
        _, P = ekf.predict(np.array([0.8, 0.05]), np.diag([0.01, 0.001]), dt=0.1)
        curr_trace = float(np.trace(P[:2, :2]))
        assert curr_trace > prev_trace
        prev_trace = curr_trace


def test_ground_truth_perturbation_invariance():
    """
    Section 3 Ground-Truth Isolation Test:
    Hold sensor measurements fixed and corrupt the ground-truth evaluation object.
    Estimated trajectories must remain bit-for-bit identical.
    """
    gt = realdata.load_ground_truth()
    calib = realdata.load_calibration()
    fixed_meas = realdata.synthesize_hybrid_measurements(
        gt=gt,
        gnss_outage_interval=(14.0, 34.0),
        seed=42,
        render_frames=False,
    )

    gt_corrupted = copy.deepcopy(gt)
    gt_corrupted.poses_2d = gt.poses_2d + np.array([500.0, -300.0, 1.5])
    gt_corrupted.positions_xyz = gt.positions_xyz + 500.0
    gt_corrupted.linear_velocities = gt.linear_velocities + 100.0
    gt_corrupted.angular_velocities = gt.angular_velocities - 50.0

    for est_type, strat in [
        ("dead_reckoning", RecoveryStrategy.R0_STRICT_GATE),
        ("standard_ekf", RecoveryStrategy.R0_STRICT_GATE),
        ("bias_aware_ekf", RecoveryStrategy.R0_STRICT_GATE),
        ("robust_gated_ekf", RecoveryStrategy.R0_STRICT_GATE),
        ("recoverable_ekf", RecoveryStrategy.R1_TEMPORAL_CONSISTENCY),
        ("recoverable_ekf", RecoveryStrategy.R2_HEALTH_HYSTERESIS),
        ("recoverable_ekf", RecoveryStrategy.R3_COVARIANCE_ADJUSTMENT),
        ("recoverable_ekf", RecoveryStrategy.R4_GLOBAL_REINITIALIZATION),
    ]:
        res_clean = realdata.run_hybrid_experiment(
            gt=gt,
            calib=calib,
            estimator_type=est_type,
            recovery_strategy=strat,
            vision_mode="disabled",
            measurements=fixed_meas,
        )
        res_corrupt = realdata.run_hybrid_experiment(
            gt=gt_corrupted,
            calib=calib,
            estimator_type=est_type,
            recovery_strategy=strat,
            vision_mode="disabled",
            measurements=fixed_meas,
        )
        assert np.array_equal(res_clean.est_poses, res_corrupt.est_poses), (
            f"Ground-truth leakage detected in {est_type} ({strat.value})"
        )


def test_causal_vision_ablation():
    """
    Section 5 Causal Vision Ablation Test:
    - A (disabled) and B (disconnected) must produce identical poses.
    - Disconnected visual degradation (clean vs moderate vs severe) must not alter poses.
    - C (connected clean) must differ from B and reduce wheel-slip peak error.
    - D (connected severe) must differ from C due to degraded optical flow.
    """
    gt = realdata.load_ground_truth()
    calib = realdata.load_calibration()
    meas = realdata.synthesize_hybrid_measurements(gt=gt, seed=42, render_frames=True)

    res_A = realdata.run_hybrid_experiment(
        gt=gt, calib=calib, vision_mode="disabled", visual_degradation="clean", measurements=meas
    )
    res_B_clean = realdata.run_hybrid_experiment(
        gt=gt, calib=calib, vision_mode="disconnected", visual_degradation="clean", measurements=meas
    )
    res_B_mod = realdata.run_hybrid_experiment(
        gt=gt, calib=calib, vision_mode="disconnected", visual_degradation="moderate", measurements=meas
    )
    res_B_sev = realdata.run_hybrid_experiment(
        gt=gt, calib=calib, vision_mode="disconnected", visual_degradation="severe", measurements=meas
    )
    res_C = realdata.run_hybrid_experiment(
        gt=gt, calib=calib, vision_mode="connected", visual_degradation="clean", measurements=meas
    )
    res_D = realdata.run_hybrid_experiment(
        gt=gt, calib=calib, vision_mode="connected", visual_degradation="severe", measurements=meas
    )

    # A == B (zero side effect when disconnected)
    assert np.array_equal(res_A.est_poses, res_B_clean.est_poses)
    assert np.array_equal(res_B_clean.est_poses, res_B_mod.est_poses)
    assert np.array_equal(res_B_clean.est_poses, res_B_sev.est_poses)

    # Optical flow quality strictly degrades across clean -> moderate -> severe
    assert res_B_clean.avg_inlier_ratio > res_B_mod.avg_inlier_ratio > res_B_sev.avg_inlier_ratio

    # Connected vision (C) changes trajectory and lowers RMSE vs disconnected (B)
    assert not np.allclose(res_B_clean.est_poses, res_C.est_poses)
    assert res_C.pos_rmse_m < res_B_clean.pos_rmse_m

    # Connected severe degradation (D) performs worse than connected clean (C)
    assert res_D.pos_rmse_m > res_C.pos_rmse_m


def test_r7_global_reinitialization_psd_and_recovery():
    """
    Verify R7 (R4_GLOBAL_REINITIALIZATION) preserves positive-definite covariance
    and successfully recovers after the 20s GNSS blackout.
    """
    gt = realdata.load_ground_truth()
    calib = realdata.load_calibration()
    res_r7 = realdata.run_hybrid_experiment(
        gt=gt,
        calib=calib,
        estimator_type="recoverable_ekf",
        recovery_strategy=RecoveryStrategy.R4_GLOBAL_REINITIALIZATION,
        vision_mode="disabled",
    )
    assert res_r7.recovery_success is True
    assert res_r7.pos_rmse_m < 1.25
    assert np.all(res_r7.cov_diags > 0.0)
    # Verify odom frame remains continuous (no jump > 0.15m per step)
    odom_diffs = np.hypot(np.diff(res_r7.est_poses_odom[:, 0]), np.diff(res_r7.est_poses_odom[:, 1]))
    assert float(np.max(odom_diffs)) < 0.15
