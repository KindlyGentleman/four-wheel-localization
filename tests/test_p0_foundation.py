"""
tests/test_p0_foundation.py

Unit and validation tests for Phase P0:
- Robot kinematics & numerical integration
- Four independent wheel encoder channels
- 4WD skid-steer kinematics and inverse kinematics
- Longitudinal slip ground velocity derivation
- Wheel disagreement indicators
- Dead-reckoning odometry & covariance propagation
- EKF Jacobians finite-difference verification
- EKF measurement updates & Joseph covariance form
"""

import math
import numpy as np
import pytest

from robot_localization.models.robot_4wd import Robot4WDConfig, WheelMountConfig
from robot_localization.models.kinematics import (
    wrap_angle,
    integrate_pose_rk2,
    integrate_pose_euler,
    body_to_world,
    world_to_body,
)
from robot_localization.models.skid_steer import SkidSteerKinematics, WheelSpeeds, ChassisVelocity
from robot_localization.sensors.encoder import EncoderChannel, FourWheelEncoders
from robot_localization.estimation.dead_reckoning import DeadReckoning4WD
from robot_localization.estimation.ekf import ExtendedKalmanFilter3D


def test_angle_wrapping():
    """Verify wrap_angle maps arbitrarily large or negative angles to [-pi, pi]."""
    assert pytest.approx(wrap_angle(0.0)) == 0.0
    assert pytest.approx(wrap_angle(math.pi)) == math.pi
    assert pytest.approx(wrap_angle(-math.pi)) == -math.pi
    assert pytest.approx(wrap_angle(3 * math.pi)) == math.pi
    assert pytest.approx(wrap_angle(-3 * math.pi)) == -math.pi
    assert pytest.approx(wrap_angle(2 * math.pi + 0.5)) == 0.5
    assert pytest.approx(wrap_angle(-2 * math.pi - 0.5)) == -0.5


def test_coordinate_transforms():
    """Verify body_to_world and world_to_body are exact inverses."""
    pose = np.array([5.0, 10.0, math.pi / 4.0])
    p_body = np.array([0.5, -0.2])

    p_world = body_to_world(p_body, pose)
    p_body_rec = world_to_body(p_world, pose)

    assert np.allclose(p_body, p_body_rec, atol=1e-12)


def test_skid_steer_forward_and_inverse_kinematics():
    """Verify forward and inverse kinematics consistency on 4WD platform."""
    cfg = Robot4WDConfig()
    kin = SkidSteerKinematics(cfg)

    # Test straight motion: v = 1.0 m/s, omega = 0.0 rad/s
    cmd_straight = ChassisVelocity(v=1.0, omega=0.0)
    w_speeds = kin.chassis_to_wheel_speeds(cmd_straight)
    # Left and right wheel speeds should be identical
    assert pytest.approx(w_speeds.fl) == w_speeds.fr
    assert pytest.approx(w_speeds.rl) == w_speeds.rr
    assert pytest.approx(w_speeds.fl * cfg.wheel_radius_m) == 1.0

    # Recover chassis speed
    rec_straight = kin.wheel_speeds_to_chassis(w_speeds)
    assert pytest.approx(rec_straight.v) == 1.0
    assert pytest.approx(rec_straight.omega) == 0.0

    # Test point turn: v = 0.0 m/s, omega = 0.5 rad/s
    cmd_spin = ChassisVelocity(v=0.0, omega=0.5)
    w_spin = kin.chassis_to_wheel_speeds(cmd_spin)
    # Left wheels spin backward, right wheels spin forward
    assert w_spin.fl < 0
    assert w_spin.fr > 0
    rec_spin = kin.wheel_speeds_to_chassis(w_spin)
    assert pytest.approx(rec_spin.v) == 0.0
    assert pytest.approx(rec_spin.omega) == 0.5


def test_wheel_slip_ground_velocity():
    """Verify ground chassis speed differs from wheel rotation speed under slip."""
    cfg = Robot4WDConfig()
    kin = SkidSteerKinematics(cfg)

    # Commanded 1.0 m/s straight
    w_speeds = kin.chassis_to_wheel_speeds(ChassisVelocity(v=1.0, omega=0.0))

    # Case 1: No slip
    no_slip = {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}
    vg_no_slip = kin.compute_ground_chassis_velocity(w_speeds, no_slip)
    assert pytest.approx(vg_no_slip.v) == 1.0
    assert pytest.approx(vg_no_slip.omega) == 0.0

    # Case 2: Symmetric 30% slip (wheels spin 30% faster than ground motion)
    sym_slip = {"FL": 0.3, "FR": 0.3, "RL": 0.3, "RR": 0.3}
    vg_sym = kin.compute_ground_chassis_velocity(w_speeds, sym_slip)
    assert pytest.approx(vg_sym.v) == 0.70  # (1 - 0.3)*1.0
    assert pytest.approx(vg_sym.omega) == 0.0

    # Case 3: Asymmetric slip (left side slips 50%, right side 0%)
    asym_slip = {"FL": 0.5, "FR": 0.0, "RL": 0.5, "RR": 0.0}
    vg_asym = kin.compute_ground_chassis_velocity(w_speeds, asym_slip)
    # vg_left = 0.5 m/s, vg_right = 1.0 m/s -> induces turning motion!
    assert pytest.approx(vg_asym.v) == 0.75
    # omega = (1.0 - 0.5) / b_eff > 0 (turns left counter-clockwise)
    assert vg_asym.omega > 0.0


def test_wheel_disagreement_detection():
    """Verify front/rear disagreement and limitation under simultaneous side slip."""
    cfg = Robot4WDConfig()
    kin = SkidSteerKinematics(cfg)

    # Isolated front-left slip
    speeds = {"FL": 1.5, "FR": 1.0, "RL": 1.0, "RR": 1.0}
    diag = kin.compute_wheel_disagreement(speeds)
    assert pytest.approx(diag["d_left"]) == 0.5
    assert pytest.approx(diag["d_right"]) == 0.0

    # Symmetrical side slip (both FL and RL slip equally)
    speeds_both = {"FL": 1.5, "FR": 1.0, "RL": 1.5, "RR": 1.0}
    diag_both = kin.compute_wheel_disagreement(speeds_both)
    # Limitation documented: front/rear difference cannot see uniform side slip!
    assert pytest.approx(diag_both["d_left"]) == 0.0
    # But variance across all 4 wheels is non-zero
    assert diag_both["speed_variance"] > 0.0


def test_four_encoders_quantization_and_ticks():
    """Verify 4 distinct encoder streams, integer counts, and quantization error bounds."""
    cfg = Robot4WDConfig()
    rng = np.random.RandomState(42)
    encoders = FourWheelEncoders(
        cfg,
        quantization_enabled=True,
        jitter_std_ticks=0.0,
        missed_count_prob=0.0,
        rng=rng
    )

    # Drive wheels at constant speed 10 rad/s for 1 second
    dt = 0.02
    speeds = {"FL": 10.0, "FR": 10.0, "RL": 10.0, "RR": 10.0}
    samples = None
    for _ in range(50):
        samples = encoders.update(timestamp=0.0, wheel_rot_speeds=speeds, dt=dt)

    assert len(samples) == 4
    for w_id in ["FL", "FR", "RL", "RR"]:
        sample = samples[w_id]
        # In 1 second, angle = 10 rad. Expected ticks = 10 * 2048 / (2*pi) ~= 3259.49 -> 3259 or 3260
        assert isinstance(sample.cumulative_ticks, int)
        assert abs(sample.cumulative_ticks - 3259) <= 2
        # Angular speed should approximate 10 rad/s
        assert pytest.approx(sample.angular_speed_rad_s, rel=1e-2) == 10.0


def test_ekf_jacobians_finite_difference():
    """Verify analytical Jacobians F, G, H_gnss match finite difference approximations (<1e-5)."""
    ekf = ExtendedKalmanFilter3D(
        initial_state=np.array([2.5, -1.8, 1.25]),
        gnss_lever_arm_m=(0.15, -0.08)
    )

    test_states = [
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, -3.0, math.pi / 3.0]),
        np.array([-10.0, 8.0, -2.5]),
    ]
    test_inputs = [
        np.array([1.0, 0.1]),
        np.array([0.5, -0.4]),
        np.array([2.0, 0.0]),
    ]

    for x in test_states:
        for u in test_inputs:
            errs = ekf.verify_jacobians_finite_difference(x_test=x, u_test=u, dt=0.02, eps=1e-6)
            assert errs["error_F"] < 1e-5, f"State Jacobian error too high: {errs['error_F']}"
            assert errs["error_G"] < 1e-5, f"Input Jacobian error too high: {errs['error_G']}"
            assert errs["error_H_gnss"] < 1e-5, f"GNSS Jacobian error too high: {errs['error_H_gnss']}"


def test_ekf_measurement_updates_and_joseph_symmetry():
    """Verify GNSS and compass measurement updates, Joseph covariance update, and positive-definiteness."""
    ekf = ExtendedKalmanFilter3D(
        initial_state=np.array([0.0, 0.0, 0.0]),
        initial_cov=np.diag([1.0, 1.0, 0.1])
    )

    # 1. Predict
    u = np.array([1.0, 0.0])
    Q_u = np.diag([0.01, 0.001])
    ekf.predict(u, Q_u, dt=0.1)

    # 2. GNSS Update
    z_gnss = np.array([0.105, 0.02])
    R_gnss = np.diag([0.25, 0.25])
    accepted_gnss = ekf.update_gnss(z_gnss, R_gnss)
    assert accepted_gnss is True

    # Verify P is symmetric and positive definite
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(ekf.P)
    assert np.all(eigvals > 0), "Covariance matrix must remain positive-definite"

    # 3. Compass Update with wraparound
    z_compass = wrap_angle(math.pi * 1.95)  # -0.05 rad
    R_compass = 0.02 ** 2
    accepted_compass = ekf.update_compass(z_compass, R_compass)
    assert accepted_compass is True

    # Covariance check after compass
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(ekf.P)
    assert np.all(eigvals > 0), "Covariance matrix must remain positive-definite"


def test_dead_reckoning_odometry():
    """Verify 4-wheel dead reckoning odometry integration and covariance expansion."""
    cfg = Robot4WDConfig()
    dr = DeadReckoning4WD(cfg)

    # Fake samples for 50 steps
    dt = 0.02
    from robot_localization.sensors.encoder import EncoderSample
    for _ in range(50):
        # 1 m/s straight
        samples = {
            w_id: EncoderSample(
                timestamp=0.0,
                wheel_id=w_id,
                cumulative_ticks=100,
                delta_ticks=5,
                dt=dt,
                delta_angle_rad=0.2,
                angular_speed_rad_s=10.0,
                linear_travel_m=0.02,
                linear_speed_m_s=1.0,
                is_valid=True,
                variance_speed=1e-4
            )
            for w_id in ["FL", "FR", "RL", "RR"]
        }
        pose, cov = dr.update_from_encoders(samples, dt=dt)

    # Travel should be ~1.0 meter along X
    assert pytest.approx(pose[0], rel=1e-2) == 1.0
    assert pytest.approx(pose[1], abs=1e-3) == 0.0
    assert pytest.approx(pose[2], abs=1e-3) == 0.0
    # Covariance must have grown
    assert cov[0, 0] > 1e-4
    assert np.allclose(cov, cov.T, atol=1e-12)
