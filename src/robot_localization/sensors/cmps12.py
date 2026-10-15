"""
robot_localization.sensors.cmps12

Simulation of a CMPS12-style tilt-compensated magnetic compass and integrated gyro.
Features:
- Compass heading in degrees clockwise from magnetic north
- Exact transformation to/from filter navigation frame (radians CCW from East)
- Gyroscope Z angular rate output
- Calibration state simulation (0=uncalibrated to 3=fully calibrated)
- Local magnetic disturbance injection and declination compensation
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from robot_localization.models.kinematics import wrap_angle


@dataclass
class CMPS12Sample:
    """A discrete observation packet from the CMPS12 module."""
    timestamp: float
    heading_compass_deg: float      # Raw sensor output: 0 to 359.9 deg CW from North
    heading_nav_rad: float          # Converted to filter frame: [-pi, pi] rad CCW from East
    gyro_z_rad_s: float             # Yaw rate in rad/s (positive CCW)
    pitch_deg: float                # Simulated planar pitch (approx 0)
    roll_deg: float                 # Simulated planar roll (approx 0)
    calibration_state: int          # 0=uncalibrated, 1=minimal, 2=good, 3=fully calibrated
    heading_std_rad: float          # Nominal heading uncertainty
    is_valid: bool                  # False if severely disturbed or degraded
    magnetic_disturbance: bool      # Flag for active magnetic distortion


class CMPS12Sensor:
    """
    Simulated CMPS12 module.
    """

    def __init__(
        self,
        sample_rate_hz: float = 10.0,
        heading_std_rad: float = 0.0175,       # ~1.0 degree
        gyro_noise_std_rad_s: float = 0.005,   # ~0.28 deg/s
        gyro_bias_rad_s: float = 0.002,
        magnetic_declination_rad: float = 0.0,
        initial_calibration_state: int = 3,
        rng: Optional[np.random.RandomState] = None
    ):
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.heading_std_rad = heading_std_rad
        self.gyro_noise_std_rad_s = gyro_noise_std_rad_s
        self.gyro_bias_rad_s = gyro_bias_rad_s
        self.declination_rad = magnetic_declination_rad
        self.calibration_state = initial_calibration_state

        self.rng = rng if rng is not None else np.random.RandomState()

        # Disturbance states
        self.disturbance_bias_rad = 0.0
        self.disturbance_active = False

    def trigger_magnetic_disturbance(self, bias_rad: float = math.radians(25.0), active: bool = True):
        """Simulate a local magnetic distortion (e.g. driving near ferromagnetic structure)."""
        self.disturbance_active = active
        self.disturbance_bias_rad = bias_rad if active else 0.0
        if active:
            # Degrade calibration state under severe distortion
            self.calibration_state = min(self.calibration_state, 1)
        else:
            self.calibration_state = 3

    @staticmethod
    def nav_to_compass_deg(theta_nav_rad: float, declination_rad: float = 0.0) -> float:
        """
        Convert filter heading (radians CCW from East) to compass reading (degrees CW from North).
        """
        # East = 0 rad -> 90 deg
        # North = pi/2 rad -> 0 deg
        # West = pi rad -> 270 deg
        # South = -pi/2 rad -> 180 deg
        theta_mag = wrap_angle(theta_nav_rad - declination_rad)
        heading_cw_north = (math.pi / 2.0) - theta_mag
        heading_deg = math.degrees(heading_cw_north) % 360.0
        return heading_deg

    @staticmethod
    def compass_deg_to_nav(heading_deg: float, declination_rad: float = 0.0) -> float:
        """
        Convert compass reading (degrees CW from North) to filter heading (radians CCW from East).
        """
        heading_cw_rad = math.radians(heading_deg)
        theta_mag = (math.pi / 2.0) - heading_cw_rad
        theta_nav = wrap_angle(theta_mag + declination_rad)
        return theta_nav

    def update(
        self,
        timestamp: float,
        true_theta_nav_rad: float,
        true_yaw_rate_rad_s: float
    ) -> CMPS12Sample:
        """
        Generate CMPS12 sample from ground truth pose heading and yaw rate.
        """
        # Add heading noise and magnetic disturbance
        noise_h = self.rng.normal(0.0, self.heading_std_rad)
        measured_theta_nav = wrap_angle(
            true_theta_nav_rad + self.disturbance_bias_rad + noise_h
        )

        # Convert to raw compass output
        compass_deg = self.nav_to_compass_deg(measured_theta_nav, self.declination_rad)

        # Gyro output
        noise_g = self.rng.normal(0.0, self.gyro_noise_std_rad_s)
        gyro_z = true_yaw_rate_rad_s + self.gyro_bias_rad_s + noise_g

        # Adjust uncertainty based on calibration state
        cal_multiplier = {3: 1.0, 2: 1.5, 1: 3.0, 0: 6.0}.get(self.calibration_state, 1.0)
        eff_std = self.heading_std_rad * cal_multiplier

        is_valid = (self.calibration_state >= 1)

        return CMPS12Sample(
            timestamp=timestamp,
            heading_compass_deg=compass_deg,
            heading_nav_rad=measured_theta_nav,
            gyro_z_rad_s=gyro_z,
            pitch_deg=0.0,
            roll_deg=0.0,
            calibration_state=self.calibration_state,
            heading_std_rad=eff_std,
            is_valid=is_valid,
            magnetic_disturbance=self.disturbance_active
        )
