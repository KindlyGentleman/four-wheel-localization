"""
robot_localization.estimation.dead_reckoning

Independent four-wheel odometry integrator with analytical Jacobian-based
covariance propagation from individual wheel encoder uncertainties.
"""

import math
from typing import Dict, Tuple, Optional
import numpy as np

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.models.skid_steer import SkidSteerKinematics
from robot_localization.models.kinematics import wrap_angle, integrate_pose_rk2
from robot_localization.sensors.encoder import EncoderSample


class DeadReckoning4WD:
    """
    Four-wheel skid-steer dead reckoning state estimator.
    Maintains pose [x, y, theta]^T and pose covariance P (3x3).
    """

    def __init__(
        self,
        robot_config: Robot4WDConfig,
        initial_pose: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None
    ):
        self.config = robot_config
        self.kinematics = SkidSteerKinematics(robot_config)

        # Pose state [x, y, theta]
        self.pose = np.zeros(3) if initial_pose is None else np.array(initial_pose, dtype=float).copy()
        # Covariance (3x3)
        self.cov = np.eye(3) * 1e-4 if initial_cov is None else np.array(initial_cov, dtype=float).copy()

        # Cached telemetry
        self.last_v = 0.0
        self.last_omega = 0.0
        self.last_wheel_disagreement: Dict[str, float] = {}
        self.last_input_cov = np.zeros((2, 2))

    def update_from_encoders(
        self,
        samples: Dict[str, EncoderSample],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update dead reckoning using samples from FL, FR, RL, RR encoders.

        Returns:
            pose: np.ndarray shape (3,)
            cov: np.ndarray shape (3, 3)
        """
        # Linear wheel speeds v_i = r_i * omega_i
        v_fl = samples["FL"].linear_speed_m_s
        v_fr = samples["FR"].linear_speed_m_s
        v_rl = samples["RL"].linear_speed_m_s
        v_rr = samples["RR"].linear_speed_m_s

        v_wheels = np.array([v_fl, v_fr, v_rl, v_rr])
        speeds_dict = {"FL": v_fl, "FR": v_fr, "RL": v_rl, "RR": v_rr}

        # Individual encoder speed variances
        var_fl = samples["FL"].variance_speed
        var_fr = samples["FR"].variance_speed
        var_rl = samples["RL"].variance_speed
        var_rr = samples["RR"].variance_speed
        sigma_wheel = np.diag([var_fl, var_fr, var_rl, var_rr])

        # Kinematic aggregation
        v_left = 0.5 * (v_fl + v_rl)
        v_right = 0.5 * (v_fr + v_rr)
        v = 0.5 * (v_left + v_right)
        omega = (v_right - v_left) / self.config.effective_track_width_m

        self.last_v = v
        self.last_omega = omega
        self.last_wheel_disagreement = self.kinematics.compute_wheel_disagreement(speeds_dict)

        # Jacobian J_kin: (2x4) mapping wheel speeds to [v, omega]
        J_kin = self.kinematics.get_kinematic_jacobian()

        # Input covariance Q_u = J_kin * Sigma_wheel * J_kin^T (2x2)
        Q_u = J_kin @ sigma_wheel @ J_kin.T
        self.last_input_cov = Q_u

        # Numerical integration of pose
        old_theta = self.pose[2]
        self.pose = integrate_pose_rk2(self.pose, v, omega, dt)

        # Covariance propagation: P_k = F_p * P_{k-1} * F_p^T + G_u * Q_u * G_u^T
        # Midpoint angle used in RK2
        theta_mid = old_theta + 0.5 * omega * dt

        # State transition Jacobian F_p = d(pose_k) / d(pose_{k-1})
        F_p = np.array([
            [1.0, 0.0, -v * math.sin(theta_mid) * dt],
            [0.0, 1.0,  v * math.cos(theta_mid) * dt],
            [0.0, 0.0,  1.0]
        ])

        # Input Jacobian G_u = d(pose_k) / d([v, omega])
        G_u = np.array([
            [math.cos(theta_mid) * dt, -0.5 * v * math.sin(theta_mid) * (dt ** 2)],
            [math.sin(theta_mid) * dt,  0.5 * v * math.cos(theta_mid) * (dt ** 2)],
            [0.0,                      dt]
        ])

        self.cov = F_p @ self.cov @ F_p.T + G_u @ Q_u @ G_u.T
        # Symmetrize
        self.cov = 0.5 * (self.cov + self.cov.T)

        return self.pose.copy(), self.cov.copy()
