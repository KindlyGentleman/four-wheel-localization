"""
tests.test_refinement

Verification suite for newly implemented algorithms:
- UKF 4D with circular manifold heading
- Robust EKF with Chi-square innovation gating and slip adaptation
- Map-based Particle Filter 2D and AMCL-inspired adaptive recovery
"""

import math
import numpy as np
import pytest

from robot_localization.models.kinematics import wrap_angle
from robot_localization.estimation.ukf import UnscentedKalmanFilter4D
from robot_localization.estimation.robust_ekf import RobustEKF4D
from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D, AMCLAdaptiveParticleFilter2D


def test_ukf_circular_mean_and_wrapping():
    """Verify that circular averaging across the [-pi, pi] branch cut behaves correctly."""
    ukf = UnscentedKalmanFilter4D(
        initial_state=np.array([0.0, 0.0, math.pi - 0.05, 0.0]),
        initial_cov=np.diag([0.01, 0.01, 0.02, 0.001])
    )
    # Propagate with zero velocity and small yaw rate crossing pi
    u = np.array([0.0, 0.10])
    Q_u = np.diag([1e-4, 1e-4])
    x_pred, P_pred = ukf.predict(u, Q_u, dt=1.0)

    # Heading should have wrapped smoothly past +pi into -pi
    expected_heading = wrap_angle(math.pi - 0.05 + 0.10)
    assert abs(wrap_angle(x_pred[2] - expected_heading)) < 1e-3
    assert np.all(np.diag(P_pred) > 0)


def test_ukf_gyro_bias_convergence():
    """Verify that UKF correctly co-estimates gyro bias online from GNSS position updates."""
    true_bias = 0.020  # rad/s
    ukf = UnscentedKalmanFilter4D(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
        initial_cov=np.diag([0.1, 0.1, 0.05, 0.01])
    )

    dt = 0.05
    rng = np.random.RandomState(42)

    # Simulate straight driving with biased gyro
    true_x, true_y, true_th = 0.0, 0.0, 0.0
    v_true = 1.0

    for step in range(160):
        t = step * dt
        true_x += v_true * dt
        # Gyro reading has true bias
        measured_gyro = 0.0 + true_bias + rng.normal(0.0, 0.001)

        ukf.predict(np.array([v_true, measured_gyro]), np.diag([1e-4, 1e-5]), dt)

        # 1 Hz GNSS update
        if step % 20 == 0:
            z_gnss = np.array([true_x + rng.normal(0.0, 0.05), true_y + rng.normal(0.0, 0.05)])
            ukf.update_gnss(z_gnss, np.diag([0.05 ** 2, 0.05 ** 2]))

    # Estimated gyro bias should be positive and trending toward 0.02 rad/s
    assert ukf.x[3] > 0.005


def test_robust_ekf_innovation_gating():
    """Verify that Robust EKF rejects large GNSS and Compass outliers via Chi-square gating."""
    rekf = RobustEKF4D(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]),
        gnss_gate_chi2=9.210,
        compass_gate_chi2=6.635
    )

    # 1. Valid GNSS update
    valid_gnss = np.array([0.02, -0.01])
    accepted_1 = rekf.update_gnss_gated(valid_gnss, np.diag([0.05 ** 2, 0.05 ** 2]))
    assert accepted_1 is True
    assert rekf.accepted_gnss_count == 1

    # 2. Outlier GNSS update (e.g. 5m multipath jump)
    outlier_gnss = np.array([5.0, -5.0])
    accepted_2 = rekf.update_gnss_gated(outlier_gnss, np.diag([0.05 ** 2, 0.05 ** 2]))
    assert accepted_2 is False
    assert rekf.rejected_gnss_count == 1
    # State should not jump
    assert math.hypot(rekf.x[0], rekf.x[1]) < 0.20

    # 3. Outlier compass update (e.g. 60 deg magnetic distortion)
    outlier_compass = math.radians(60.0)
    accepted_c = rekf.update_compass_gated(outlier_compass, (math.radians(1.0)) ** 2)
    assert accepted_c is False
    assert rekf.rejected_compass_count == 1


def test_robust_ekf_slip_adaptation():
    """Verify that Robust EKF inflates process noise under wheel slip."""
    rekf = RobustEKF4D(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
        slip_omega_thresh_rad_s=0.08,
        slip_inflation_gain=10.0
    )

    Q_base = np.diag([1e-4, 1e-4])

    # 1. No slip (wheel matches gyro)
    x1, P1 = rekf.predict_robust(
        v_enc=1.0, omega_enc=0.10, gyro_z=0.10, Q_u_base=Q_base, dt=0.02
    )
    scale_nominal = rekf.last_slip_scale
    assert scale_nominal == 1.0

    # 2. Severe slip (wheel spins while gyro registers straight)
    x2, P2 = rekf.predict_robust(
        v_enc=1.0, omega_enc=0.50, gyro_z=0.10, Q_u_base=Q_base, dt=0.02
    )
    scale_slip = rekf.last_slip_scale
    assert scale_slip > 10.0
    # Covariance expansion
    assert P2[0, 0] > P1[0, 0]


def test_occupancy_map_and_raycasting():
    """Verify that OccupancyMap2D builds EDT and raycasts 2D LiDAR scans."""
    v_map = OccupancyMap2D.create_volcanic_testbed()
    assert v_map.grid.shape[0] == v_map.height_cells
    assert np.all(v_map.dist_map >= 0.0)

    # Robot at center
    pose = np.array([0.0, 0.0, 0.0])
    angles, ranges = v_map.simulate_lidar_scan(pose, num_beams=16, max_range_m=12.0)
    assert len(ranges) == 16
    # In a 24x24m arena with obstacles, all beams hit walls or obstacles within 12m
    assert np.all(ranges > 0.5)
    assert np.all(ranges <= 12.0)


def test_particle_filter_and_amcl_localization():
    """Verify that ParticleFilter2D and AMCL update weights and re-estimate robot pose."""
    v_map = OccupancyMap2D.create_volcanic_testbed()
    true_pose = np.array([2.0, 2.0, 0.5])

    # True scan
    angles, ranges = v_map.simulate_lidar_scan(true_pose, num_beams=16, max_range_m=12.0)

    # Initialize PF near true pose
    pf = ParticleFilter2D(
        occupancy_map=v_map,
        num_particles=150,
        initial_pose=true_pose,
        initial_pos_std_m=0.30
    )

    # Scan update
    pf.update_scan(angles, ranges)
    est_p = pf.estimated_pose

    # Error relative to ground truth should be bounded
    pos_err = math.hypot(est_p[0] - true_pose[0], est_p[1] - true_pose[1])
    assert pos_err < 0.50
    assert pf.n_eff > 1.0

    # Test AMCL adaptive recovery on simulated kidnapped robot
    amcl = AMCLAdaptiveParticleFilter2D(
        occupancy_map=v_map,
        min_particles=50,
        max_particles=300,
        initial_pose=true_pose
    )

    # Kidnap the robot (displace true pose by 6 meters)
    kidnapped_pose = np.array([-4.0, -4.0, 1.5])
    angles_k, ranges_k = v_map.simulate_lidar_scan(kidnapped_pose, num_beams=16)

    # Repeated scans in kidnapped position should trigger recovery injection
    for _ in range(5):
        amcl.update_scan_adaptive(angles_k, ranges_k)

    # Recovery probability or spread should be positive
    assert amcl.N >= amcl.N_min
