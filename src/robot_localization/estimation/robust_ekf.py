"""
robot_localization.estimation.robust_ekf

Robust Bias-Aware 4D Extended Kalman Filter (A5).
Combines:
1. Online estimation of gyroscope bias b_g: x = [p_x, p_y, theta, b_g]^T
2. Chi-square (NIS) innovation gating on GNSS (2-DOF) and Compass (1-DOF)
3. Dynamic process noise scaling based on wheel-gyro discrepancy and wheel disagreement
4. Outage-aware covariance expansion during sustained GNSS blackouts
"""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np

from robot_localization.models.kinematics import wrap_angle
from robot_localization.estimation.bias_aware_ekf import BiasAwareEKF4D


class RobustEKF4D(BiasAwareEKF4D):
    """
    Robust 4D EKF incorporating statistical innovation gating, slip-adaptive process
    noise, and blackout-aware covariance inflation.
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        gyro_bias_walk_var: float = 1e-6,
        q_pos_add: float = 1e-4,
        q_theta_add: float = 1e-4,
        gnss_lever_arm_m: Tuple[float, float] = (0.05, 0.0),
        gnss_gate_chi2: float = 9.210,       # 99% quantile for 2-DOF
        compass_gate_chi2: float = 6.635,    # 99% quantile for 1-DOF
        slip_omega_thresh_rad_s: float = 0.08, # ~4.5 deg/s
        slip_v_thresh_m_s: float = 0.05,
        slip_inflation_gain: float = 12.0,
        outage_drift_rate_m_s: float = 0.05  # Accumulated odometric drift rate during outage
    ):
        super().__init__(
            initial_state=initial_state,
            initial_cov=initial_cov,
            gyro_bias_walk_var=gyro_bias_walk_var,
            q_pos_add=q_pos_add,
            q_theta_add=q_theta_add,
            gnss_lever_arm_m=gnss_lever_arm_m
        )
        self.gnss_gate = gnss_gate_chi2
        self.compass_gate = compass_gate_chi2
        self.omega_thresh = slip_omega_thresh_rad_s
        self.v_thresh = slip_v_thresh_m_s
        self.slip_gain = slip_inflation_gain
        self.outage_drift_rate = outage_drift_rate_m_s

        # Tracking state
        self.time_since_gnss = 0.0
        self.last_slip_scale = 1.0
        self.rejected_gnss_count = 0
        self.accepted_gnss_count = 0
        self.rejected_compass_count = 0
        self.accepted_compass_count = 0

    def predict_robust(
        self,
        v_enc: float,
        omega_enc: float,
        gyro_z: float,
        Q_u_base: np.ndarray,
        dt: float,
        wheel_disagreement: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive prediction step accounting for wheel slip and GNSS blackout duration.
        """
        self.time_since_gnss += dt

        # 1. Slip detection via gyro-wheel discrepancy
        unbiased_gyro = gyro_z - self.x[3]
        delta_omega = abs(omega_enc - unbiased_gyro)

        slip_detected = (delta_omega > self.omega_thresh) or (wheel_disagreement > self.v_thresh)

        if slip_detected:
            excess = max(delta_omega - self.omega_thresh, 0.0) / self.omega_thresh
            excess_v = max(wheel_disagreement - self.v_thresh, 0.0) / self.v_thresh
            scale = 1.0 + self.slip_gain * (excess + excess_v)
        else:
            scale = 1.0

        self.last_slip_scale = scale
        Q_u_scaled = Q_u_base.copy()
        Q_u_scaled[0, 0] *= scale  # Inflate forward speed uncertainty
        Q_u_scaled[1, 1] *= scale  # Inflate yaw rate uncertainty

        # 2. Blackout-aware covariance expansion
        orig_q_pos = self.q_pos_add
        if self.time_since_gnss > 2.0:
            drift_var = (self.outage_drift_rate ** 2)
            self.q_pos_add += drift_var

        # 3. Standard prediction with modified covariances
        u = np.array([v_enc, gyro_z])
        x_pred, P_pred = super().predict(u, Q_u_scaled, dt)
        self.q_pos_add = orig_q_pos

        return x_pred, P_pred

    def update_gnss_gated(self, z_gnss: np.ndarray, R_gnss: np.ndarray) -> bool:
        """Apply GNSS update with Chi-square innovation gating."""
        accepted = super().update_gnss(z_gnss, R_gnss, gating_chi2=self.gnss_gate)
        if accepted:
            self.accepted_gnss_count += 1
            self.time_since_gnss = 0.0
        else:
            self.rejected_gnss_count += 1
        return accepted

    def update_compass_gated(self, z_heading_rad: float, R_heading_rad2: float) -> bool:
        """Apply compass update with Chi-square innovation gating."""
        accepted = super().update_compass(z_heading_rad, R_heading_rad2, gating_chi2=self.compass_gate)
        if accepted:
            self.accepted_compass_count += 1
        else:
            self.rejected_compass_count += 1
        return accepted
