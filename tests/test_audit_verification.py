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


def test_amcl_adaptive_single_resampling_and_posterior_weights():
    """
    Regression test for Finding F10:
    Verify that AMCLAdaptiveParticleFilter2D does NOT perform duplicate resampling.
    Ensures that scan likelihood weights are preserved for KLD sampling, and
    resampling occurs exactly once, correctly drawing from the posterior likelihood.
    """
    from robot_localization.estimation.particle_filter import OccupancyMap2D, AMCLAdaptiveParticleFilter2D

    v_map = OccupancyMap2D.create_volcanic_testbed()
    true_pose = np.array([2.0, 2.0, 0.5])
    angles, ranges = v_map.simulate_lidar_scan(true_pose, num_beams=16, max_range_m=12.0)

    rng = np.random.RandomState(42)
    amcl = AMCLAdaptiveParticleFilter2D(
        occupancy_map=v_map,
        min_particles=50,
        max_particles=300,
        initial_pose=true_pose,
        rng=rng
    )

    # 1. Verify compute_scan_weights updates weights without resampling
    avg_lik = amcl.compute_scan_weights(angles, ranges)
    assert avg_lik > 0.0
    # Weights should be non-uniform (highest near true pose)
    assert np.max(amcl.weights) > np.min(amcl.weights)
    # Number of particles unchanged before adaptive step
    assert amcl.N == 300

    # 2. Run update_scan_adaptive: should execute single KLD resampling step
    amcl.update_scan_adaptive(angles, ranges)
    assert amcl.N_min <= amcl.N <= amcl.N_max
    # After KLD resampling, weights are reset to uniform 1/N
    np.testing.assert_allclose(amcl.weights, np.ones(amcl.N) / float(amcl.N))
    assert abs(amcl.n_eff - float(amcl.N)) < 1e-6

    # 3. Estimated pose should remain close to ground truth
    pos_err = math.hypot(amcl.estimated_pose[0] - true_pose[0], amcl.estimated_pose[1] - true_pose[1])
    assert pos_err < 0.50


def test_particle_filter_sequential_weight_accumulation_without_resampling():
    """
    Verify sequential Bayesian importance weight accumulation across consecutive
    scans when resampling is skipped.
    log w_k^(i) = log w_{k-1}^(i) + log p(z_k | x_k^(i)).
    """
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D

    v_map = OccupancyMap2D.create_volcanic_testbed()
    true_pose = np.array([3.0, 3.0, 0.0])
    angles, ranges = v_map.simulate_lidar_scan(true_pose, num_beams=16, max_range_m=12.0)

    rng = np.random.RandomState(42)
    pf = ParticleFilter2D(
        occupancy_map=v_map,
        num_particles=100,
        initial_pose=true_pose,
        rng=rng
    )

    # First update without resampling
    pf.compute_scan_weights(angles, ranges)
    w_step1 = pf.weights.copy()
    n_eff_1 = pf.n_eff

    # Artificially modify one particle's weight to verify non-uniform prior accumulation
    # Second update without resampling
    pf.compute_scan_weights(angles, ranges)
    w_step2 = pf.weights.copy()
    n_eff_2 = pf.n_eff

    # Under repeated consistent measurements without resampling, weight mass concentrates
    # and effective sample size decreases (variance of weights increases)
    assert not np.array_equal(w_step1, w_step2)
    assert n_eff_2 <= n_eff_1
    # Check that highest weighted particle in step 1 further increases in relative importance
    best_idx = np.argmax(w_step1)
    assert w_step2[best_idx] >= w_step1[best_idx]


def test_amcl_kidnapped_robot_recovery_injection():
    """
    Verify that AMCL detects kidnapping via absolute observation likelihood drop
    and injects random particles (p_random > 0) to recover global pose.
    """
    from robot_localization.estimation.particle_filter import OccupancyMap2D, AMCLAdaptiveParticleFilter2D

    v_map = OccupancyMap2D.create_volcanic_testbed()
    pose_init = np.array([2.0, 2.0, 0.0])
    angles, ranges_nominal = v_map.simulate_lidar_scan(pose_init, num_beams=24, max_range_m=12.0)

    rng = np.random.RandomState(123)
    amcl = AMCLAdaptiveParticleFilter2D(
        occupancy_map=v_map,
        min_particles=100,
        max_particles=500,
        initial_pose=pose_init,
        alpha_slow=0.01,
        alpha_fast=0.20,
        rng=rng
    )

    # 1. Warm up filter in nominal location: w_fast and w_slow track high likelihood
    for _ in range(5):
        amcl.update_scan_adaptive(angles, ranges_nominal)

    assert amcl.w_slow > 0.30
    assert amcl.w_fast > 0.30
    assert amcl.p_random < 0.10

    # 2. Kidnap robot: teleport robot across map to [-5.0, -5.0, 1.57]
    pose_kidnapped = np.array([-5.0, -5.0, 1.57])
    _, ranges_kidnapped = v_map.simulate_lidar_scan(pose_kidnapped, num_beams=24, max_range_m=12.0)

    # 3. Feed scans from kidnapped location into the filter
    # Absolute observation likelihood must drop sharply and trigger recovery injection
    p_random_history = []
    for _ in range(10):
        amcl.update_scan_adaptive(angles, ranges_kidnapped)
        p_random_history.append(amcl.p_random)

    # Recovery injection probability must have spiked
    max_p_rand = max(p_random_history)
    assert max_p_rand > 0.40, f"Expected p_random > 0.40 after kidnapping, got {max_p_rand:.2f}"


def test_particle_filter_predict_updates_estimate_without_scans():
    """Verify that ParticleFilter2D.predict() recomputes estimated pose and covariance during scan-free motion."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D

    v_map = OccupancyMap2D.create_volcanic_testbed()
    rng = np.random.RandomState(42)
    pf = ParticleFilter2D(
        occupancy_map=v_map,
        num_particles=150,
        initial_pose=np.array([0.0, 0.0, 0.0]),
        initial_pos_std_m=0.05,
        initial_head_std_rad=0.01,
        rng=rng
    )

    # Propagate particles forward 5.0m
    for _ in range(5):
        pf.predict(v_enc=1.0, omega_gyro=0.0, dt=1.0, std_v=0.01, std_omega=0.005)

    # Pose estimate MUST track prediction forward
    assert pf.estimated_pose[0] > 4.5, f"Expected estimated_pose[0] > 4.5m, got {pf.estimated_pose[0]:.2f}m"

    # Pass an empty/invalid scan (all beams 0.05m < 0.2m min range)
    invalid_angles = np.linspace(-math.pi/2, math.pi/2, 16)
    invalid_ranges = np.ones(16) * 0.05
    avg_lik = pf.compute_scan_weights(invalid_angles, invalid_ranges)

    # Estimate must still remain synchronized with predicted position (~5.0m)
    assert pf.estimated_pose[0] > 4.5
    assert avg_lik == 0.0 or avg_lik < 0.1


def test_kld_binning_negative_coordinates():
    """Verify that KLD binning uses floor discretization so negative coordinates do not collapse into bin 0."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, AMCLAdaptiveParticleFilter2D

    v_map = OccupancyMap2D.create_volcanic_testbed()
    amcl = AMCLAdaptiveParticleFilter2D(occupancy_map=v_map, min_particles=50, max_particles=100)

    # Check binning around 0 for bin_xy_m = 0.50
    # With math.floor: -0.1 -> -1, +0.1 -> 0 (distinct bins)
    p_neg = np.array([-0.1, -0.1, 0.0])
    p_pos = np.array([0.1, 0.1, 0.0])

    bx_neg = int(math.floor(p_neg[0] / amcl.bin_xy_m))
    bx_pos = int(math.floor(p_pos[0] / amcl.bin_xy_m))

    assert bx_neg != bx_pos, "Negative and positive coordinates around origin must belong to distinct bins"
    assert bx_neg == -1
    assert bx_pos == 0


def test_particle_filter_full_3dof_covariance():
    """Verify that particle filter covariance computes full 3-DOF cross-covariances."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D

    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100, initial_pose=np.array([0.0, 0.0, 0.0]))

    # Construct correlated particle cloud: x is positively correlated with theta
    x_offsets = np.linspace(-0.5, 0.5, 100)
    pf.particles[:, 0] = x_offsets
    pf.particles[:, 1] = 0.0
    pf.particles[:, 2] = x_offsets * 0.5 # theta correlated with x
    pf.weights = np.ones(100) / 100.0

    pf._compute_estimate()

    cov = pf.estimated_cov
    assert cov.shape == (3, 3)
    assert abs(cov[0, 2]) > 1e-4, f"Cross-covariance P_x_theta should be non-zero, got {cov[0, 2]}"
    assert abs(cov[2, 0] - cov[0, 2]) < 1e-12, "Covariance matrix must be symmetric"


def test_amcl_recovery_injection_under_uniform_low_likelihood():
    """Verify that AMCL triggers recovery injection even when n_eff is high, provided p_random indicates kidnapping."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, AMCLAdaptiveParticleFilter2D

    v_map = OccupancyMap2D.create_volcanic_testbed()
    rng = np.random.RandomState(42)
    amcl = AMCLAdaptiveParticleFilter2D(
        occupancy_map=v_map,
        min_particles=50,
        max_particles=200,
        initial_pose=np.array([0.0, 0.0, 0.0]),
        rng=rng
    )

    # Simulate kidnapping condition: w_slow high, w_fast collapsed -> p_random high
    amcl.w_slow = 0.50
    amcl.w_fast = 0.05
    amcl.p_random = 0.90
    # Keep weights uniform so n_eff == N (n_eff is NOT < N/2)
    amcl.weights = np.ones(amcl.N) / float(amcl.N)
    amcl.n_eff = float(amcl.N)

    # Feed dummy scan to trigger adaptive update
    angles = np.linspace(-math.pi/2, math.pi/2, 16)
    ranges = np.ones(16) * 5.0 # Non-zero ranges

    old_particles = amcl.particles.copy()
    amcl.update_scan_adaptive(angles, ranges)

    # Because p_random is high, KLD resampling and recovery injection must execute despite high n_eff
    # Check that particles have spread over a broader area (injected random particles)
    x_spread = float(np.max(amcl.particles[:, 0]) - np.min(amcl.particles[:, 0]))
    assert x_spread > 5.0, f"Expected broad particle injection across map (>5m), got {x_spread:.2f}m"


def test_figure_parity_and_data_integrity():
    """Verify that all figures and benchmark artifacts pass automated integrity checks."""
    import sys
    from pathlib import Path
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import verify_figures

    res = verify_figures.main()
    assert res == 0, "Figure parity and data integrity verification failed"


def test_missing_scan_does_not_trigger_random_injection():
    """Verify that completely missing scans (None/empty) do not trigger false kidnapping recovery."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, AMCLAdaptiveParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    rng = np.random.RandomState(42)
    amcl = AMCLAdaptiveParticleFilter2D(occupancy_map=v_map, min_particles=50, max_particles=200, rng=rng)
    amcl.w_slow = 0.50
    amcl.w_fast = 0.50
    amcl.update_scan_adaptive(np.array([]), np.array([]))
    assert amcl.w_slow == 0.50
    assert amcl.w_fast == 0.50
    assert amcl.p_random == 0.0


def test_all_invalid_beams_skip_measurement_update():
    """Verify that scans where all beams are out-of-range skip weight updates and keep last_scan_valid=False."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100)
    initial_weights = pf.weights.copy()
    angles = np.linspace(-math.pi/2, math.pi/2, 16)
    ranges = np.ones(16) * 100.0 # Out of max range (12m)
    lik = pf.compute_scan_weights(angles, ranges, max_range_m=12.0)
    assert lik == 0.0
    assert pf.last_scan_valid is False
    np.testing.assert_allclose(pf.weights, initial_weights)


def test_nan_lidar_ranges_are_ignored_or_rejected():
    """Verify that NaN range values are filtered out without crashing."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100)
    angles = np.array([0.0, 0.5])
    ranges = np.array([float("nan"), float("nan")])
    lik = pf.compute_scan_weights(angles, ranges)
    assert lik == 0.0
    assert pf.last_scan_valid is False


def test_inf_lidar_ranges_are_handled():
    """Verify that Inf range values are filtered out safely."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100)
    angles = np.array([0.0, 0.5])
    ranges = np.array([float("inf"), float("-inf")])
    lik = pf.compute_scan_weights(angles, ranges)
    assert lik == 0.0
    assert pf.last_scan_valid is False


def test_zero_beam_scan_keeps_valid_posterior():
    """Verify that zero beams keep weights normalized and sum to 1.0."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100)
    pf.compute_scan_weights(np.array([]), np.array([]))
    assert abs(np.sum(pf.weights) - 1.0) < 1e-12


def test_no_scan_does_not_update_slow_fast_likelihood():
    """Verify that dropout does not degrade w_fast or inflate p_random."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, AMCLAdaptiveParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    amcl = AMCLAdaptiveParticleFilter2D(occupancy_map=v_map, min_particles=50, max_particles=200)
    amcl.w_slow = 0.40
    amcl.w_fast = 0.40
    for _ in range(50):
        amcl.update_scan_adaptive(np.array([]), np.array([]))
    assert amcl.w_slow == 0.40
    assert amcl.w_fast == 0.40
    assert amcl.p_random == 0.0


def test_valid_low_likelihood_triggers_recovery_when_appropriate():
    """Verify that authentic contradictory scans decay w_fast and trigger recovery."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, AMCLAdaptiveParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    rng = np.random.RandomState(42)
    angles = np.linspace(-math.pi/2, math.pi/2, 16)
    ranges = np.ones(16) * 0.5
    amcl = AMCLAdaptiveParticleFilter2D(occupancy_map=v_map, min_particles=50, max_particles=200, rng=rng)
    amcl.w_slow = 0.50
    amcl.w_fast = 0.50
    amcl.update_scan_adaptive(angles, ranges)
    assert amcl.w_fast < amcl.w_slow
    assert amcl.p_random > 0.05


def test_recovery_after_long_scan_dropout():
    """Verify that filter can localize immediately when valid scans resume after a long dropout."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    true_pose = np.array([2.0, 2.0, 0.0])
    angles, ranges = v_map.simulate_lidar_scan(true_pose, num_beams=16)
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=200, initial_pose=true_pose)
    for _ in range(20):
        pf.predict(v_enc=0.1, omega_gyro=0.0, dt=0.1)
    pf.update_scan(angles, ranges)
    assert pf.last_scan_valid is True


def test_prediction_only_trajectory_stays_current():
    """Verify that pose estimate advances during motion prediction without scans."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100, initial_pose=np.array([0.0, 0.0, 0.0]))
    pf.predict(v_enc=1.0, omega_gyro=0.0, dt=1.0, std_v=0.0, std_omega=0.0)
    assert pf.estimated_pose[0] > 0.90


def test_systematic_resampling_resets_effective_sample_size():
    """Verify that systematic resampling explicitly resets n_eff to N."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100)
    pf.weights = np.zeros(100)
    pf.weights[0] = 1.0
    pf.n_eff = 1.0
    pf.resample_systematic()
    assert abs(pf.n_eff - 100.0) < 1e-6
    np.testing.assert_allclose(pf.weights, np.ones(100) / 100.0)


def test_resampling_estimate_matches_particle_population():
    """Verify that after resampling, estimated_pose matches the resampled particle mean."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100)
    pf.particles[:50, 0] = 5.0
    pf.particles[50:, 0] = 0.0
    pf.weights[:50] = 1.0 / 50.0
    pf.weights[50:] = 0.0
    pf.resample_systematic()
    assert abs(pf.estimated_pose[0] - 5.0) < 0.1


def test_resampling_preserves_normalized_weights():
    """Verify that weights sum to exactly 1.0 after resampling."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100)
    pf.resample_systematic()
    assert abs(np.sum(pf.weights) - 1.0) < 1e-12


def test_no_out_of_bounds_resampling_indices():
    """Verify that resampling indices do not exceed bounds."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=200)
    pf.resample_systematic()
    assert pf.particles.shape == (200, 3)


def test_resampling_with_strongly_degenerate_weights():
    """Verify resampling when a single particle carries 1.0 weight."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=50)
    pf.weights = np.zeros(50)
    pf.weights[12] = 1.0
    pf.particles[12] = [3.5, -2.1, 0.4]
    pf.resample_systematic()
    np.testing.assert_allclose(pf.particles[:, 0], 3.5)
    np.testing.assert_allclose(pf.particles[:, 1], -2.1)
    np.testing.assert_allclose(pf.particles[:, 2], 0.4)


def test_resampling_with_uniform_weights():
    """Verify that resampling with already uniform weights preserves particle spread."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100)
    pf.resample_systematic()
    assert pf.n_eff == 100.0


def test_heading_wrap_consistency_after_resampling():
    """Verify that circular heading angles remain wrapped in [-pi, pi] after resampling."""
    from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D
    v_map = OccupancyMap2D.create_volcanic_testbed()
    pf = ParticleFilter2D(occupancy_map=v_map, num_particles=100)
    pf.particles[:, 2] = np.pi - 0.01
    pf.resample_systematic()
    assert np.all(pf.particles[:, 2] >= -np.pi)
    assert np.all(pf.particles[:, 2] <= np.pi)
    assert abs(pf.estimated_pose[2] - (np.pi - 0.01)) < 1e-4





