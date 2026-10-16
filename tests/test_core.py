"""
tests/test_core.py

Core verification test suite for the four-wheel-localization repository.
Tests kinematics, sensor simulation, EKF, UKF, Robust EKF, Particle Filter, and AMCL.
"""

import math
import numpy as np
import pytest

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.models.kinematics import wrap_angle, wrap_angle_array
from robot_localization.sensors.gnss import GNSSReceiver
from robot_localization.sensors.cmps12 import CMPS12Sensor
from robot_localization.sensors.mpu9250 import MPU9250Sensor
from robot_localization.sensors.encoder import FourWheelEncoders, EncoderChannel
from robot_localization.simulation.robot_simulator import RobotSimulator
from robot_localization.simulation.world import TrajectoryGenerator
from robot_localization.estimation.ekf import ExtendedKalmanFilter3D
from robot_localization.estimation.bias_aware_ekf import BiasAwareEKF4D
from robot_localization.estimation.ukf import UnscentedKalmanFilter4D
from robot_localization.estimation.robust_ekf import RobustEKF4D
from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D, AMCLAdaptiveParticleFilter2D
from robot_localization.evaluation.metrics import compute_nees_series, compute_covariance_coverage


def test_angle_wrapping():
    """Verify angular normalization to [-pi, pi]."""
    assert math.isclose(wrap_angle(0.0), 0.0)
    assert math.isclose(wrap_angle(3.0 * math.pi), math.pi, abs_tol=1e-6)
    assert math.isclose(wrap_angle(-3.0 * math.pi), -math.pi, abs_tol=1e-6)
    assert math.isclose(wrap_angle(2.0 * math.pi), 0.0, abs_tol=1e-6)


def test_robot_kinematics_and_encoders():
    """Verify 4WD geometry, effective track width, and encoder ticks."""
    cfg = Robot4WDConfig()
    assert cfg.wheel_base_m > 0.0
    assert cfg.track_width_m > 0.0
    assert cfg.effective_track_width_m >= cfg.track_width_m

    encoders = FourWheelEncoders(cfg)
    channel = encoders.channels["FL"]
    sample = channel.update(timestamp=0.01, rotational_speed_rad_s=10.0, dt=0.01)
    assert sample.is_valid
    assert sample.linear_speed_m_s > 0.0


def test_ekf_prediction_and_update():
    """Verify 3D and 4D EKF predict and update cycles."""
    ekf = BiasAwareEKF4D()
    assert np.allclose(ekf.x, [0.0, 0.0, 0.0, 0.0])

    # Predict with forward velocity v = 1.0 m/s for 10 steps (0.2s)
    u = np.array([1.0, 0.0])
    Q_u = np.diag([1e-4, 1e-4])
    for _ in range(10):
        ekf.predict(u, Q_u, dt=0.02)

    assert math.isclose(ekf.x[0], 0.20, abs_tol=0.01)
    assert math.isclose(ekf.x[1], 0.0, abs_tol=0.01)

    # GNSS update at (0.22, 0.01)
    accepted = ekf.update_gnss(np.array([0.22, 0.01]), np.diag([0.05**2, 0.05**2]))
    assert accepted is True
    assert math.isclose(ekf.x[0], 0.21, abs_tol=0.02)


def test_ukf_circular_manifold():
    """Verify UKF sigma point propagation and circular mean wrapping."""
    ukf = UnscentedKalmanFilter4D(
        initial_state=np.array([0.0, 0.0, math.pi - 0.05, 0.0]),
        initial_cov=np.diag([0.01, 0.01, 0.02, 0.001])
    )
    u = np.array([0.0, 0.10])
    Q_u = np.diag([1e-4, 1e-4])
    x_pred, P_pred = ukf.predict(u, Q_u, dt=1.0)

    # Angle wraps across branch cut
    expected_heading = wrap_angle(math.pi - 0.05 + 0.10)
    assert abs(wrap_angle(x_pred[2] - expected_heading)) < 1e-3
    assert np.all(np.diag(P_pred) > 0)


def test_robust_ekf_outlier_gating():
    """Verify that RobustEKF4D rejects large GNSS jumps via Chi-squared gating."""
    rekf = RobustEKF4D(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]),
        gnss_gate_chi2=9.210
    )

    # Valid GNSS update
    accepted_1 = rekf.update_gnss_gated(np.array([0.02, -0.01]), np.diag([0.05**2, 0.05**2]))
    assert accepted_1 is True
    assert rekf.accepted_gnss_count == 1

    # Injected 5-meter multipath outlier
    accepted_2 = rekf.update_gnss_gated(np.array([5.0, -5.0]), np.diag([0.05**2, 0.05**2]))
    assert accepted_2 is False
    assert rekf.rejected_gnss_count == 1
    # State should not jump
    assert math.hypot(rekf.x[0], rekf.x[1]) < 0.20


def test_particle_filter_and_amcl():
    """Verify 2D occupancy map raycasting, PF weight update, and AMCL particle adaptation."""
    v_map = OccupancyMap2D.create_volcanic_testbed()
    true_pose = np.array([2.0, 2.0, 0.5])

    # Raycasting / simulate scan
    angles, ranges = v_map.simulate_lidar_scan(true_pose, num_beams=16, max_range_m=12.0)
    assert len(ranges) == 16
    assert np.all(ranges > 0.5)

    # Particle Filter 2D
    pf = ParticleFilter2D(
        occupancy_map=v_map,
        num_particles=150,
        initial_pose=true_pose,
        initial_pos_std_m=0.30
    )
    pf.update_scan(angles, ranges)
    est_pose = pf.estimated_pose
    pos_err = math.hypot(est_pose[0] - true_pose[0], est_pose[1] - true_pose[1])
    assert pos_err < 0.60

    # AMCL adaptive filter
    amcl = AMCLAdaptiveParticleFilter2D(
        occupancy_map=v_map,
        min_particles=50,
        max_particles=300,
        initial_pose=true_pose
    )
    assert 50 <= amcl.N <= 300
    amcl.update_scan_adaptive(angles, ranges)
    assert amcl.N >= amcl.N_min


def test_mahalanobis_consistency_evaluation():
    """Verify coordinate-invariant Mahalanobis distance and 95% chi2 evaluation."""
    gt = np.array([[1.0, 2.0, 0.0]])
    est = np.array([[1.05, 2.02, 0.0]])
    cov_diag = np.array([[0.01, 0.01, 0.001]])
    cov_mat = np.array([[[0.01, 0.002, 0.0], [0.002, 0.01, 0.0], [0.0, 0.0, 0.001]]])

    nees_pos, nees_pose = compute_nees_series(gt, est, cov_diag, cov_mats=cov_mat)
    assert nees_pos[0] < 5.9915

    cov_95, cov_2sig, cov_3sig = compute_covariance_coverage(gt, est, cov_diag, cov_mats=cov_mat)
    assert cov_95 == 1.0
    assert cov_2sig == 1.0
