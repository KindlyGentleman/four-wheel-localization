"""
tests/test_p2_range_sensors.py

Unit and validation tests for Phase P2:
- Known-landmark LiDAR range-bearing simulation
- Analytical LiDAR observation Jacobians verified against numerical finite differences
- 3-transducer ultrasonic array ray casting against planar walls
- Range-Assisted EKF (Baseline B6) state updates
- Slip-Aware Adaptive EKF (Baseline B7) process noise scaling
"""

import math
import numpy as np
import pytest

from robot_localization.models.kinematics import wrap_angle
from robot_localization.simulation.world import LandmarkMap
from robot_localization.sensors.lidar import KnownLandmarkLiDAR, LandmarkObservation
from robot_localization.sensors.ultrasonic import UltrasonicArray
from robot_localization.estimation.range_assisted_ekf import RangeAssistedEKF
from robot_localization.estimation.adaptive_ekf import SlipAwareAdaptiveEKF


def test_lidar_observation_jacobians_finite_difference():
    """Verify analytical LiDAR measurement Jacobian H (2x3) matches numerical finite differences (<1e-5)."""
    mounting_offset = (0.25, -0.05)
    landmark = np.array([5.0, 4.0])

    test_poses = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, -1.0, 0.75]),
        np.array([-3.0, 2.5, -1.8]),
    ]

    eps = 1e-6
    for pose in test_poses:
        z_ana, H_ana = KnownLandmarkLiDAR.compute_observation_jacobian(
            robot_state=pose,
            landmark_pos=landmark,
            mounting_offset=mounting_offset
        )

        H_num = np.zeros((2, 3))
        for j in range(3):
            dp = np.zeros(3)
            dp[j] = eps
            z_plus, _ = KnownLandmarkLiDAR.compute_observation_jacobian(pose + dp, landmark, mounting_offset)
            z_minus, _ = KnownLandmarkLiDAR.compute_observation_jacobian(pose - dp, landmark, mounting_offset)
            diff = z_plus - z_minus
            diff[1] = wrap_angle(diff[1])
            H_num[:, j] = diff / (2.0 * eps)

        err_H = np.max(np.abs(H_ana - H_num))
        assert err_H < 1e-5, f"LiDAR Jacobian error too high: {err_H}"


def test_lidar_range_limits_and_scan():
    """Verify LiDAR respects min/max detection range and identifies visible landmarks."""
    lidar = KnownLandmarkLiDAR(
        max_range_m=10.0,
        min_range_m=0.5,
        rng=np.random.RandomState(42)
    )
    landmarks = {
        1: np.array([3.0, 0.0]),    # In range (3m)
        2: np.array([15.0, 0.0]),   # Out of range (15m > 10m)
        3: np.array([0.1, 0.0]),    # Below min range (0.1m < 0.5m)
    }

    pose = np.array([0.0, 0.0, 0.0])
    scan = lidar.observe_landmarks(timestamp=0.0, robot_pose=pose, landmarks=landmarks)

    observed_ids = [obs.landmark_id for obs in scan.observations]
    assert 1 in observed_ids
    assert 2 not in observed_ids
    assert 3 not in observed_ids


def test_ultrasonic_wall_raycasting():
    """Verify ultrasonic transducers compute correct geometric distance to perpendicular wall."""
    us = UltrasonicArray(rng=np.random.RandomState(42))
    # Vertical wall at x = 2.0 m: from (2, -5) to (2, 5)
    walls = [(np.array([2.0, -5.0]), np.array([2.0, 5.0]))]

    # Robot facing wall at (0, 0, theta=0)
    # Front-center transducer is at x = 0.275 m, so distance to wall = 2.0 - 0.275 = 1.725 m
    pose = np.array([0.0, 0.0, 0.0])
    echos = us.measure_walls(timestamp=0.0, robot_pose=pose, walls=walls)

    assert "front_center" in echos
    echo_fc = echos["front_center"]
    assert echo_fc.is_valid is True
    assert pytest.approx(echo_fc.expected_range_m, rel=1e-3) == (2.0 - 0.275)
    assert abs(echo_fc.range_m - echo_fc.expected_range_m) < 0.10


def test_range_assisted_ekf_updates():
    """Verify RangeAssistedEKF reduces uncertainty when landmark updates arrive."""
    ekf = RangeAssistedEKF(
        initial_state=np.array([0.0, 0.0, 0.0]),
        initial_cov=np.diag([1.0, 1.0, 0.2])
    )

    init_tr = np.trace(ekf.P)

    # Fake landmark observation
    obs = LandmarkObservation(
        landmark_id=1,
        range_m=4.8,
        bearing_rad=0.0,
        landmark_pos_world=np.array([5.0, 0.0]),
        cov_rb=np.diag([0.03 ** 2, 0.015 ** 2])
    )

    accepted = ekf.update_landmark_observation(obs)
    assert accepted is True
    post_tr = np.trace(ekf.P)
    assert post_tr < init_tr, "Covariance trace must decrease after landmark update"


def test_slip_aware_adaptive_ekf():
    """Verify adaptive EKF scales process noise when wheel-gyro discrepancy occurs."""
    a_ekf = SlipAwareAdaptiveEKF(
        omega_threshold_rad_s=0.05,
        slip_alpha=20.0
    )

    Q_base = np.diag([0.01, 0.001])

    # Case 1: Wheel and gyro agree (no slip)
    a_ekf.predict_adaptive(v_wheel=1.0, omega_wheel=0.10, omega_gyro=0.10, Q_u_base=Q_base, dt=0.02)
    assert a_ekf.last_slip_scale == 1.0

    # Case 2: Severe slip -> wheel reports 0.5 rad/s, gyro reports 0.1 rad/s
    a_ekf.predict_adaptive(v_wheel=1.0, omega_wheel=0.50, omega_gyro=0.10, Q_u_base=Q_base, dt=0.02)
    assert a_ekf.last_slip_scale > 1.0
