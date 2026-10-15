"""
robot_localization.sensors.mpu9250

Simulation of an MPU-9250 9-DOF Inertial Measurement Unit (IMU).
Features:
- 3-axis gyroscope with white noise and random-walk bias drift on Z (yaw rate)
- 3-axis accelerometer modeling body accelerations and vertical gravity reaction
- 3-axis magnetometer with hard-iron offset and soft-iron distortion
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from robot_localization.models.kinematics import wrap_angle


@dataclass
class MPU9250Sample:
    """A discrete 9-DOF IMU observation packet."""
    timestamp: float
    gyro_rad_s: np.ndarray          # [gx, gy, gz]
    accel_m_s2: np.ndarray          # [ax, ay, az]
    mag_uT: np.ndarray              # [mx, my, mz] in microteslas
    gyro_bias_z_true: float         # Diagnostic ground-truth bias
    is_valid: bool


class MPU9250Sensor:
    """
    Simulated MPU-9250 IMU.
    """

    def __init__(
        self,
        sample_rate_hz: float = 50.0,
        gyro_noise_std_rad_s: float = 0.005,           # ~0.28 deg/s
        gyro_bias_initial_rad_s: float = 0.015,        # ~0.86 deg/s
        gyro_bias_walk_std_rad_s_sqrt_s: float = 0.0001,
        accel_noise_std_m_s2: float = 0.05,
        mag_noise_std_uT: float = 0.8,
        hard_iron_bias_uT: Tuple[float, float, float] = (2.5, -4.0, 1.0),
        rng: Optional[np.random.RandomState] = None
    ):
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.gyro_noise_std = gyro_noise_std_rad_s
        self.gyro_bias_z = gyro_bias_initial_rad_s
        self.gyro_bias_walk_std = gyro_bias_walk_std_rad_s_sqrt_s
        self.accel_noise_std = accel_noise_std_m_s2
        self.mag_noise_std = mag_noise_std_uT
        self.hard_iron = np.array(hard_iron_bias_uT, dtype=float)

        self.rng = rng if rng is not None else np.random.RandomState()

        # Local Earth magnetic field vector in NED (North, East, Down in uT)
        # Approximate mid-latitude values: [22.0, 0.0, 42.0]
        self.b_earth_ned = np.array([22.0, 0.0, 42.0])

    def update(
        self,
        timestamp: float,
        true_theta_rad: float,
        true_v_m_s: float,
        true_yaw_rate_rad_s: float,
        true_linear_accel_m_s2: float = 0.0,
        dt_since_last: Optional[float] = None
    ) -> MPU9250Sample:
        """
        Generate IMU measurement packet from ground truth kinematics.
        """
        dt = dt_since_last if dt_since_last is not None else self.dt

        # 1. Update gyro random-walk bias
        if dt > 0 and self.gyro_bias_walk_std > 0:
            step = self.rng.normal(0.0, self.gyro_bias_walk_std * math.sqrt(dt))
            self.gyro_bias_z += step

        # Gyroscope measurement
        # Roll and pitch rates are near zero for planar ground robot
        gx = self.rng.normal(0.0, self.gyro_noise_std)
        gy = self.rng.normal(0.0, self.gyro_noise_std)
        gz = true_yaw_rate_rad_s + self.gyro_bias_z + self.rng.normal(0.0, self.gyro_noise_std)
        gyro = np.array([gx, gy, gz])

        # 2. Accelerometer
        # In body frame:
        # ax = forward acceleration
        # ay = centripetal acceleration: v * omega
        # az = reaction force against gravity (+9.81 m/s^2 upward)
        ax = true_linear_accel_m_s2 + self.rng.normal(0.0, self.accel_noise_std)
        ay = (true_v_m_s * true_yaw_rate_rad_s) + self.rng.normal(0.0, self.accel_noise_std)
        az = 9.80665 + self.rng.normal(0.0, self.accel_noise_std)
        accel = np.array([ax, ay, az])

        # 3. Magnetometer
        # Rotate Earth NED field into robot body frame
        # theta is CCW from East (+X_w), so heading relative to North is psi = pi/2 - theta
        psi = (math.pi / 2.0) - true_theta_rad
        # In body frame (X forward, Y left):
        # mx = B_north * cos(psi) + B_east * sin(psi)
        # my = -B_north * sin(psi) + B_east * cos(psi)
        # mz = B_down
        mx_ideal = self.b_earth_ned[0] * math.cos(psi) + self.b_earth_ned[1] * math.sin(psi)
        my_ideal = -self.b_earth_ned[0] * math.sin(psi) + self.b_earth_ned[1] * math.cos(psi)
        mz_ideal = self.b_earth_ned[2]

        mag_meas = np.array([
            mx_ideal + self.hard_iron[0] + self.rng.normal(0.0, self.mag_noise_std),
            my_ideal + self.hard_iron[1] + self.rng.normal(0.0, self.mag_noise_std),
            mz_ideal + self.hard_iron[2] + self.rng.normal(0.0, self.mag_noise_std)
        ])

        return MPU9250Sample(
            timestamp=timestamp,
            gyro_rad_s=gyro,
            accel_m_s2=accel,
            mag_uT=mag_meas,
            gyro_bias_z_true=self.gyro_bias_z,
            is_valid=True
        )
