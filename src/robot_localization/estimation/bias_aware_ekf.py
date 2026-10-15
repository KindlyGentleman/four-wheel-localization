"""
robot_localization.estimation.bias_aware_ekf

Four-state Extended Kalman Filter estimating:
    x = [p_x, p_y, theta, b_g]^T
where b_g is the continuous gyroscope Z bias.
"""

import math
from typing import Optional, Tuple, Dict
import numpy as np

from robot_localization.models.kinematics import wrap_angle


class BiasAwareEKF4D:
    """
    Four-state EKF with online gyro bias estimation.
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        gyro_bias_walk_var: float = 1e-6,
        q_pos_add: float = 1e-4,
        q_theta_add: float = 1e-4,
        gnss_lever_arm_m: Tuple[float, float] = (0.05, 0.0),
    ):
        # State: [x, y, theta, b_g]
        self.x = np.zeros(4) if initial_state is None else np.array(initial_state, dtype=float).copy()
        # Covariance: (4, 4)
        if initial_cov is None:
            self.P = np.diag([1e-3, 1e-3, 1e-3, 1e-3])
        else:
            self.P = np.array(initial_cov, dtype=float).copy()

        self.gyro_bias_walk_var = gyro_bias_walk_var
        self.q_pos_add = q_pos_add
        self.q_theta_add = q_theta_add
        self.lever_arm = np.array(gnss_lever_arm_m, dtype=float)

        # Telemetry
        self.last_nis_gnss: Optional[float] = None
        self.last_nis_compass: Optional[float] = None

    @staticmethod
    def state_transition(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate 4D state:
            x = [px, py, theta, b_g]
            u = [v, omega_gyro]
        """
        px, py, theta, bg = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        v, omega_gyro = float(u[0]), float(u[1])

        omega_eff = omega_gyro - bg
        theta_mid = theta + 0.5 * omega_eff * dt

        px_next = px + v * math.cos(theta_mid) * dt
        py_next = py + v * math.sin(theta_mid) * dt
        theta_next = wrap_angle(theta + omega_eff * dt)
        bg_next = bg  # Random walk process

        return np.array([px_next, py_next, theta_next, bg_next])

    @staticmethod
    def state_jacobian(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Analytical Jacobian F = df/dx (4x4).
        """
        theta, bg = float(x[2]), float(x[3])
        v, omega_gyro = float(u[0]), float(u[1])
        omega_eff = omega_gyro - bg
        theta_mid = theta + 0.5 * omega_eff * dt

        c_mid = math.cos(theta_mid)
        s_mid = math.sin(theta_mid)

        # d(theta_mid)/d(bg) = -0.5 * dt
        # d(px)/d(bg) = v * dt * (-s_mid) * (-0.5*dt) = 0.5 * v * dt^2 * s_mid
        # d(py)/d(bg) = v * dt * (c_mid) * (-0.5*dt) = -0.5 * v * dt^2 * c_mid

        F = np.array([
            [1.0, 0.0, -v * s_mid * dt,  0.5 * v * s_mid * (dt ** 2)],
            [0.0, 1.0,  v * c_mid * dt, -0.5 * v * c_mid * (dt ** 2)],
            [0.0, 0.0,  1.0,           -dt],
            [0.0, 0.0,  0.0,            1.0]
        ])
        return F

    @staticmethod
    def input_jacobian(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Analytical Jacobian G = df/du (4x2) where u = [v, omega_gyro]^T.
        """
        theta, bg = float(x[2]), float(x[3])
        v, omega_gyro = float(u[0]), float(u[1])
        omega_eff = omega_gyro - bg
        theta_mid = theta + 0.5 * omega_eff * dt

        c_mid = math.cos(theta_mid)
        s_mid = math.sin(theta_mid)

        G = np.array([
            [c_mid * dt, -0.5 * v * s_mid * (dt ** 2)],
            [s_mid * dt,  0.5 * v * c_mid * (dt ** 2)],
            [0.0,         dt],
            [0.0,         0.0]
        ])
        return G

    def predict(self, u: np.ndarray, Q_u: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF prediction step for 4D bias-aware model.
        """
        F = self.state_jacobian(self.x, u, dt)
        G = self.input_jacobian(self.x, u, dt)

        # Propagate state
        self.x = self.state_transition(self.x, u, dt)

        # Additive process noise (position drift + gyro random walk)
        Q_add = np.diag([
            self.q_pos_add * dt,
            self.q_pos_add * dt,
            self.q_theta_add * dt,
            self.gyro_bias_walk_var * dt
        ])

        Q_step = G @ Q_u @ G.T + Q_add
        self.P = F @ self.P @ F.T + Q_step
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy(), self.P.copy()

    def update_gnss(
        self,
        z_gnss: np.ndarray,
        R_gnss: np.ndarray,
        gating_chi2: Optional[float] = None
    ) -> bool:
        """Update 4D state with GNSS 2D position observation."""
        px, py, theta = self.x[0], self.x[1], self.x[2]
        lx, ly = self.lever_arm[0], self.lever_arm[1]

        c = math.cos(theta)
        s = math.sin(theta)
        z_pred = np.array([
            px + lx * c - ly * s,
            py + lx * s + ly * c
        ])

        # H is 2x4
        H = np.array([
            [1.0, 0.0, -lx * s - ly * c, 0.0],
            [0.0, 1.0,  lx * c - ly * s, 0.0]
        ])

        y = z_gnss - z_pred
        S = H @ self.P @ H.T + R_gnss
        S_inv = np.linalg.inv(S)

        nis = float(y.T @ S_inv @ y)
        self.last_nis_gnss = nis

        if gating_chi2 is not None and nis > gating_chi2:
            return False

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])

        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gnss @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_compass(
        self,
        z_heading: float,
        R_heading: float,
        gating_chi2: Optional[float] = None
    ) -> bool:
        """Update 4D state with compass heading observation."""
        z_pred = self.x[2]
        H = np.array([[0.0, 0.0, 1.0, 0.0]])

        y_val = wrap_angle(z_heading - z_pred)
        S_val = float((H @ self.P @ H.T).item()) + float(R_heading)

        nis = (y_val ** 2) / S_val
        self.last_nis_compass = nis

        if gating_chi2 is not None and nis > gating_chi2:
            return False

        K = (self.P @ H.T) / S_val
        self.x = self.x + (K.flatten() * y_val)
        self.x[2] = wrap_angle(self.x[2])

        I_KH = np.eye(4) - K @ H
        R_mat = np.array([[R_heading]])
        self.P = I_KH @ self.P @ I_KH.T + K @ R_mat @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def verify_jacobians_finite_difference(
        self,
        x_test: Optional[np.ndarray] = None,
        u_test: Optional[np.ndarray] = None,
        dt: float = 0.02,
        eps: float = 1e-6
    ) -> Dict[str, float]:
        """Verify analytical Jacobians against numerical finite differences."""
        x = np.array([2.0, -1.5, 0.85, 0.025]) if x_test is None else x_test.copy()
        u = np.array([1.1, 0.3]) if u_test is None else u_test.copy()

        F_ana = self.state_jacobian(x, u, dt)
        F_num = np.zeros_like(F_ana)
        for j in range(4):
            dx = np.zeros(4)
            dx[j] = eps
            x_plus = self.state_transition(x + dx, u, dt)
            x_minus = self.state_transition(x - dx, u, dt)
            diff = x_plus - x_minus
            diff[2] = wrap_angle(diff[2])
            F_num[:, j] = diff / (2.0 * eps)
        err_F = float(np.max(np.abs(F_ana - F_num)))

        G_ana = self.input_jacobian(x, u, dt)
        G_num = np.zeros_like(G_ana)
        for j in range(2):
            du = np.zeros(2)
            du[j] = eps
            x_plus = self.state_transition(x, u + du, dt)
            x_minus = self.state_transition(x, u - du, dt)
            diff = x_plus - x_minus
            diff[2] = wrap_angle(diff[2])
            G_num[:, j] = diff / (2.0 * eps)
        err_G = float(np.max(np.abs(G_ana - G_num)))

        return {"error_F": err_F, "error_G": err_G}
