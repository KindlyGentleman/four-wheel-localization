"""
robot_localization.estimation.adaptive_ekf

Slip-Aware Adaptive Process Noise Covariance EKF (Baseline B7).
Monitors:
1. Discrepancy between wheel-derived yaw rate and gyroscope angular rate:
   delta_omega = |omega_wheel - omega_gyro|
2. Wheel disagreement across front/rear pairs:
   d_left = |v_FL - v_RL|, d_right = |v_FR - v_RR|

When slip is detected, process noise Q_u is adaptively scaled to prevent
filter overconfidence and divergence.
"""

import math
from typing import Optional, Tuple, Dict
import numpy as np

from robot_localization.estimation.ekf import ExtendedKalmanFilter3D


class SlipAwareAdaptiveEKF(ExtendedKalmanFilter3D):
    """
    Adaptive EKF that inflates process covariance when slip indicators trigger.
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        omega_threshold_rad_s: float = 0.08, # ~4.5 deg/s
        slip_alpha: float = 15.0,            # Scaling gain
        gnss_lever_arm_m: Tuple[float, float] = (0.05, 0.0),
    ):
        super().__init__(
            initial_state=initial_state,
            initial_cov=initial_cov,
            gnss_lever_arm_m=gnss_lever_arm_m
        )
        self.omega_thresh = omega_threshold_rad_s
        self.alpha = slip_alpha
        self.last_slip_scale = 1.0

    def predict_adaptive(
        self,
        v_wheel: float,
        omega_wheel: float,
        omega_gyro: float,
        Q_u_base: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptively scale process covariance Q_u before prediction.
        """
        # Discrepancy between wheel and gyro
        delta_omega = abs(omega_wheel - omega_gyro)

        # Scale factor
        if delta_omega > self.omega_thresh:
            excess = delta_omega - self.omega_thresh
            scale = 1.0 + self.alpha * (excess ** 2)
        else:
            scale = 1.0

        self.last_slip_scale = scale
        Q_u_adaptive = Q_u_base * scale

        # Prediction uses gyro for rotation and wheel for forward speed
        u = np.array([v_wheel, omega_gyro])
        return self.predict(u, Q_u_adaptive, dt)
