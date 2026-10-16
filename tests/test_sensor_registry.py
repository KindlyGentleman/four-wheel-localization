"""
tests.test_sensor_registry

Unit tests for SensorRegistry, SensorConfig, observation structures, and health monitor.
"""

import pytest
import numpy as np

from robot_localization.sensors.registry import (
    SensorRegistry,
    SensorType,
    SensorHealthState,
    SensorObservation,
    SensorConfig
)
from robot_localization.sensors.health import (
    SensorHealthMonitor,
    LocalizationConfidence
)


def test_sensor_registry_initialization_and_defaults():
    registry = SensorRegistry()
    enabled = registry.get_enabled_sensors()

    # Wheel encoders, IMU, GNSS are enabled by default
    assert "wheel_encoders" in enabled
    assert "imu" in enabled
    assert "gnss" in enabled

    # Stereo, LiDAR, Compass, Ultrasonic disabled by default
    assert "stereo" not in enabled
    assert "lidar" not in enabled
    assert "compass" not in enabled


def test_sensor_registry_enable_disable():
    registry = SensorRegistry()

    registry.disable_sensor("gnss")
    assert not registry.is_enabled("gnss")
    assert "gnss" not in registry.get_enabled_sensors()
    assert registry.get_health_state("gnss") == SensorHealthState.INACTIVE

    registry.enable_sensor("stereo")
    assert registry.is_enabled("stereo")
    assert "stereo" in registry.get_enabled_sensors()
    assert registry.get_health_state("stereo") == SensorHealthState.HEALTHY


def test_sensor_registry_fault_injection():
    registry = SensorRegistry()
    registry.inject_fault("imu", "gyro_bias_jump", bias_rad_s=0.20)

    cfg = registry.get_config("imu")
    assert "gyro_bias_jump" in cfg.active_faults
    assert cfg.active_faults["gyro_bias_jump"]["bias_rad_s"] == 0.20
    assert registry.get_health_state("imu") == SensorHealthState.FAULTY

    registry.clear_faults("imu")
    cfg_cleared = registry.get_config("imu")
    assert len(cfg_cleared.active_faults) == 0
    assert registry.get_health_state("imu") == SensorHealthState.HEALTHY


def test_sensor_registry_validation_rejection():
    registry = SensorRegistry()

    # AMCL without prior map must fail
    ok, msg = registry.validate_for_estimator("amcl", has_prior_map=False)
    assert not ok
    assert "occupancy map" in msg.lower()

    # AMCL with map but without LiDAR/stereo range sensor must fail
    ok, msg = registry.validate_for_estimator("amcl", has_prior_map=True)
    assert not ok
    assert "range observations" in msg.lower()

    # Now enable lidar and check AMCL passes
    registry.enable_sensor("lidar")
    ok, msg = registry.validate_for_estimator("amcl", has_prior_map=True)
    assert ok

    # Disabling wheels without stereo must fail standard EKF
    registry.disable_sensor("wheel_encoders")
    ok, msg = registry.validate_for_estimator("bias_aware_ekf")
    assert not ok


def test_sensor_registry_yaml_roundtrip():
    registry = SensorRegistry()
    registry.enable_sensor("stereo")
    registry.configure_sensor("stereo", rate_hz=15.0, params={"disparity_method": "sgbm"})

    yaml_str = registry.to_yaml()
    assert "stereo" in yaml_str
    assert "sgbm" in yaml_str

    reloaded = SensorRegistry.from_yaml(yaml_str)
    assert reloaded.is_enabled("stereo")
    assert reloaded.get_config("stereo").rate_hz == 15.0
    assert reloaded.get_config("stereo").params.get("disparity_method") == "sgbm"


def test_sensor_health_monitor_slip_and_outage_detection():
    monitor = SensorHealthMonitor(outage_timeout_s=1.0)
    monitor.register_sensor("gnss", nominal_rate_hz=1.0)
    monitor.register_sensor("wheel_encoders", nominal_rate_hz=50.0)
    monitor.register_sensor("imu", nominal_rate_hz=50.0)

    # Initial state
    monitor.update_wheel_odometry(t=0.0, v_forward=0.5, omega_wheel=0.0)
    monitor.update_imu(t=0.0, gyro_z=0.0, estimated_bias=0.0)
    monitor.record_innovation("gnss", t=0.0, nis=1.5, accepted=True)

    conf, diag = monitor.evaluate_health(current_time=0.0)
    assert conf == LocalizationConfidence.HEALTHY

    # Induce wheel slip: wheel yaw rate differs by 0.15 rad/s from gyro
    monitor.update_wheel_odometry(t=0.5, v_forward=0.5, omega_wheel=0.20)
    monitor.update_imu(t=0.5, gyro_z=0.02, estimated_bias=0.0)
    conf, diag = monitor.evaluate_health(current_time=0.5)
    assert diag["flags"]["wheel_slip_suspected"] is True

    # Induce prolonged GNSS outage > 1.0s while slip is active -> LOST
    conf, diag = monitor.evaluate_health(current_time=2.0)
    assert conf == LocalizationConfidence.LOST
