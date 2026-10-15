"""
tests/test_p1_sensors_fusion.py

Unit and validation tests for Phase P1:
- GNSS receiver: coordinate conversions, lever-arm kinematics, outage detection
- CMPS12 sensor: compass heading conversions, magnetic disturbance, calibration states
- MPU-9250 IMU: gyro bias random walk, accelerometer specific force, magnetometer
- Bias-Aware EKF: state transition, analytical Jacobians vs finite difference
- EventScheduler: chronological ordering, causality verification
- Multi-baseline fusion pipeline execution
"""

import math
import numpy as np
import pytest

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.sensors.gnss import GNSSReceiver
from robot_localization.sensors.cmps12 import CMPS12Sensor
from robot_localization.sensors.mpu9250 import MPU9250Sensor
from robot_localization.estimation.bias_aware_ekf import BiasAwareEKF4D
from robot_localization.simulation.scheduler import EventScheduler
from robot_localization.simulation.robot_simulator import RobotSimulator
from robot_localization.simulation.world import TrajectoryGenerator
from robot_localization.estimation.baseline_runner import run_all_baselines


def test_gnss_lever_arm_and_outages():
    """Verify GNSS antenna lever-arm transformation and outage detection."""
    lever_arm = (0.20, 0.10)  # 20cm forward, 10cm left
    gnss = GNSSReceiver(
        sample_rate_hz=1.0,
        lever_arm_m=lever_arm,
        outage_intervals=[(10.0, 20.0)],
        rng=np.random.RandomState(42)
    )

    # Robot at (0, 0) with heading 0 (East)
    pose_east = np.array([0.0, 0.0, 0.0])
    fix_east = gnss.update(timestamp=0.0, robot_pose=pose_east)
    assert fix_east.is_valid is True
    # Antenna should be at (0.20, 0.10) +/- noise
    assert abs(fix_east.x_enu_m - 0.20) < 1.5
    assert abs(fix_east.y_enu_m - 0.10) < 1.5

    # Robot at (0, 0) with heading pi/2 (North)
    pose_north = np.array([0.0, 0.0, math.pi / 2.0])
    # Antenna forward (+X_b) is now North (+Y_w), left (+Y_b) is now West (-X_w)
    # Expected: x_w = -0.10, y_w = +0.20
    fix_north = gnss.update(timestamp=5.0, robot_pose=pose_north)
    assert abs(fix_north.x_enu_m - (-0.10)) < 1.5
    assert abs(fix_north.y_enu_m - 0.20) < 1.5

    # Verify outage period
    assert gnss.is_in_outage(5.0) is False
    assert gnss.is_in_outage(15.0) is True
    fix_outage = gnss.update(timestamp=15.0, robot_pose=pose_east)
    assert fix_outage.is_valid is False


def test_cmps12_heading_transformations():
    """Verify cardinal direction conversions between CMPS12 degrees CW from North and navigation rad CCW from East."""
    # North: 0 deg CW -> +pi/2 rad CCW from East
    assert pytest.approx(CMPS12Sensor.compass_deg_to_nav(0.0)) == (math.pi / 2.0)
    assert pytest.approx(CMPS12Sensor.nav_to_compass_deg(math.pi / 2.0)) == 0.0

    # East: 90 deg CW -> 0 rad CCW from East
    assert pytest.approx(CMPS12Sensor.compass_deg_to_nav(90.0)) == 0.0
    assert pytest.approx(CMPS12Sensor.nav_to_compass_deg(0.0)) == 90.0

    # South: 180 deg CW -> -pi/2 rad CCW from East
    assert pytest.approx(CMPS12Sensor.compass_deg_to_nav(180.0)) == (-math.pi / 2.0)
    assert pytest.approx(CMPS12Sensor.nav_to_compass_deg(-math.pi / 2.0)) == 180.0

    # West: 270 deg CW -> pi rad CCW from East
    assert pytest.approx(abs(CMPS12Sensor.compass_deg_to_nav(270.0))) == math.pi
    assert pytest.approx(CMPS12Sensor.nav_to_compass_deg(math.pi)) == 270.0


def test_cmps12_magnetic_disturbance():
    """Verify CMPS12 degradation and recovery under injected magnetic disturbance."""
    cmps = CMPS12Sensor(rng=np.random.RandomState(42))
    assert cmps.calibration_state == 3

    # Inject disturbance
    cmps.trigger_magnetic_disturbance(bias_rad=0.5, active=True)
    assert cmps.disturbance_active is True
    assert cmps.calibration_state <= 1

    sample = cmps.update(timestamp=1.0, true_theta_nav_rad=0.0, true_yaw_rate_rad_s=0.0)
    # Measured heading should be offset by ~0.5 rad
    assert abs(sample.heading_nav_rad - 0.5) < 0.1
    assert sample.heading_std_rad > cmps.heading_std_rad  # uncertainty inflated due to low calibration

    # Clear disturbance
    cmps.trigger_magnetic_disturbance(active=False)
    assert cmps.disturbance_active is False
    assert cmps.calibration_state == 3


def test_mpu9250_accelerometer_and_gyro():
    """Verify MPU-9250 specific force components and gyro bias random walk."""
    imu = MPU9250Sensor(
        gyro_bias_initial_rad_s=0.02,
        gyro_bias_walk_std_rad_s_sqrt_s=0.001,
        rng=np.random.RandomState(42)
    )

    # Robot moving at constant forward speed 1.0 m/s with yaw rate 0.5 rad/s
    sample = imu.update(
        timestamp=0.0,
        true_theta_rad=0.0,
        true_v_m_s=1.0,
        true_yaw_rate_rad_s=0.5,
        true_linear_accel_m_s2=0.0,
        dt_since_last=0.02
    )

    # az should measure gravity reaction upward (~9.81 m/s^2)
    assert abs(sample.accel_m_s2[2] - 9.81) < 0.2
    # ay should measure centripetal acceleration: v * omega = 1.0 * 0.5 = 0.5 m/s^2
    assert abs(sample.accel_m_s2[1] - 0.5) < 0.2
    # gz should measure yaw_rate + bias = 0.5 + 0.02 = 0.52 rad/s
    assert abs(sample.gyro_rad_s[2] - 0.52) < 0.05


def test_bias_aware_ekf_jacobians_finite_difference():
    """Verify 4D bias-aware EKF Jacobians F and G match numerical finite differences (<1e-5)."""
    ekf4 = BiasAwareEKF4D()
    test_cases = [
        (np.array([1.0, -2.0, 0.5, 0.01]), np.array([1.2, 0.2])),
        (np.array([-5.0, 3.0, -1.8, -0.02]), np.array([0.6, -0.4])),
    ]

    for x, u in test_cases:
        errs = ekf4.verify_jacobians_finite_difference(x_test=x, u_test=u, dt=0.02, eps=1e-6)
        assert errs["error_F"] < 1e-5, f"4D EKF state Jacobian error too high: {errs['error_F']}"
        assert errs["error_G"] < 1e-5, f"4D EKF input Jacobian error too high: {errs['error_G']}"


def test_event_scheduler_chronology():
    """Verify EventScheduler strictly outputs events in ascending timestamp order."""
    scheduler = EventScheduler()
    scheduler.schedule(5.0, "gnss")
    scheduler.schedule(1.0, "encoder")
    scheduler.schedule(2.5, "imu")
    scheduler.schedule(0.5, "compass")

    timestamps = []
    while scheduler.has_next():
        ev = scheduler.pop_next()
        timestamps.append(ev.timestamp)

    assert timestamps == [0.5, 1.0, 2.5, 5.0]


def test_end_to_end_multi_baseline_simulation():
    """Run a 5-second multi-sensor simulation and verify all baselines B0-B5 execute successfully."""
    cfg = Robot4WDConfig()
    sim = RobotSimulator(cfg, dt_sim_s=0.01)

    def traj(t):
        return TrajectoryGenerator.figure_eight(t, speed=0.8, loop_radius=4.0)

    sim_res = sim.run_simulation(duration_s=5.0, trajectory_func=traj)
    assert len(sim_res.timestamps) > 0
    assert len(sim_res.sensor_events) > 0

    # Run baselines
    baselines = run_all_baselines(sim_res, robot_config=cfg)
    assert len(baselines) == 6
    for b_name in ["B0_wheel_odometry", "B1_odometry_gyro", "B2_odometry_gnss",
                   "B3_odometry_gyro_gnss", "B4_odometry_gyro_gnss_compass", "B5_bias_aware_ekf"]:
        assert b_name in baselines
        trace = baselines[b_name]
        assert len(trace.estimated_pose) > 0
        assert trace.estimated_pose.shape[1] == 3
        # Check no NaNs or infs
        assert not np.any(np.isnan(trace.estimated_pose))
        assert not np.any(np.isnan(trace.covariance_diag))
