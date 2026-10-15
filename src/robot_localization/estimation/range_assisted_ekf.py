"""
robot_localization.estimation.range_assisted_ekf

Range-Assisted Extended Kalman Filter (Baseline B6).
Incorporates LiDAR range-bearing updates to known mapped landmarks during GNSS denial.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from robot_localization.models.kinematics import wrap_angle
from robot_localization.estimation.ekf import ExtendedKalmanFilter3D
from robot_localization.sensors.lidar import KnownLandmarkLiDAR, LandmarkObservation


class RangeAssistedEKF(ExtendedKalmanFilter3D):
    """
    EKF with range-bearing observation model to known landmarks.
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        lidar_mounting_offset_m: Tuple[float, float] = (0.20, 0.0),
        gnss_lever_arm_m: Tuple[float, float] = (0.05, 0.0),
    ):
        super().__init__(
            initial_state=initial_state,
            initial_cov=initial_cov,
            gnss_lever_arm_m=gnss_lever_arm_m
        )
        self.lidar_offset = lidar_mounting_offset_m

    def update_landmark_observation(
        self,
        obs: LandmarkObservation,
        gating_chi2: Optional[float] = 9.21 # 99% 2-DOF
    ) -> bool:
        """
        Update state using a range-bearing observation to a known landmark.
        """
        z_meas = np.array([obs.range_m, obs.bearing_rad])
        R_mat = obs.cov_rb

        # Predicted observation and Jacobian
        z_pred, H = KnownLandmarkLiDAR.compute_observation_jacobian(
            self.x,
            obs.landmark_pos_world,
            mounting_offset=self.lidar_offset
        )

        # Innovation: range is linear, bearing must be wrapped!
        y = np.array([
            z_meas[0] - z_pred[0],
            wrap_angle(z_meas[1] - z_pred[1])
        ])

        S = H @ self.P @ H.T + R_mat
        S_inv = np.linalg.inv(S)

        nis = float(y.T @ S_inv @ y)
        if gating_chi2 is not None and nis > gating_chi2:
            return False

        K = self.P @ H.T @ S_inv

        # State update
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])

        # Joseph update
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_mat @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True
