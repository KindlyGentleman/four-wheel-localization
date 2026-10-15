"""
robot_localization.estimation.ekf

Three-state Extended Kalman Filter (EKF) estimating pose x = [p_x, p_y, theta]^T.
Features:
- Second-order Runge-Kutta state propagation
- Analytical state and input transition Jacobians
- Joseph-stabilized covariance updates
- Wrapped angular innovations
- Configurable GNSS lever-arm transformation and Jacobians
- Normalized Innovation Squared (NIS) calculation
- Numerical finite-difference Jacobian verification
"""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np

from robot_localization.models.kinematics import wrap_angle


class ExtendedKalmanFilter3D:
    """
    Planar EKF for mobile robot state estimation:
        State: x = [p_x, p_y, theta]^T
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        q_additive: Optional[np.ndarray] = None,
        gnss_lever_arm_m: Tuple[float, float] = (0.0, 0.0),
    ):
        # State: [x, y, theta]
        self.x = np.zeros(3) if initial_state is None else np.array(initial_state, dtype=float).copy()
        # Covariance: (3, 3)
        self.P = np.eye(3) * 1e-3 if initial_cov is None else np.array(initial_cov, dtype=float).copy()
        # Additive process noise covariance (3, 3)
        self.Q_add = np.diag([1e-4, 1e-4, 1e-4]) if q_additive is None else np.array(q_additive, dtype=float).copy()

        # Antenna lever-arm in robot body frame: [l_x, l_y]
        self.lever_arm = np.array(gnss_lever_arm_m, dtype=float)

        # Telemetry
        self.last_nis_gnss: Optional[float] = None
        self.last_nis_compass: Optional[float] = None
        self.last_innovation_gnss: Optional[np.ndarray] = None
        self.last_innovation_compass: Optional[float] = None

    @staticmethod
    def state_transition(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Nonlinear discrete state propagation using midpoint (RK2) integration.

        Args:
            x: Current state [p_x, p_y, theta]
            u: Control input [v, omega]
            dt: Time interval in seconds

        Returns:
            x_next: Propagated state
        """
        px, py, theta = float(x[0]), float(x[1]), float(x[2])
        v, omega = float(u[0]), float(u[1])

        theta_mid = theta + 0.5 * omega * dt
        px_next = px + v * math.cos(theta_mid) * dt
        py_next = py + v * math.sin(theta_mid) * dt
        theta_next = wrap_angle(theta + omega * dt)

        return np.array([px_next, py_next, theta_next])

    @staticmethod
    def state_jacobian(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Analytical Jacobian of state transition w.r.t state: F = df/dx (3x3).
        """
        theta = float(x[2])
        v, omega = float(u[0]), float(u[1])
        theta_mid = theta + 0.5 * omega * dt

        F = np.array([
            [1.0, 0.0, -v * math.sin(theta_mid) * dt],
            [0.0, 1.0,  v * math.cos(theta_mid) * dt],
            [0.0, 0.0,  1.0]
        ])
        return F

    @staticmethod
    def input_jacobian(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Analytical Jacobian of state transition w.r.t control input: G = df/du (3x2).
        """
        theta = float(x[2])
        v, omega = float(u[0]), float(u[1])
        theta_mid = theta + 0.5 * omega * dt

        G = np.array([
            [math.cos(theta_mid) * dt, -0.5 * v * math.sin(theta_mid) * (dt ** 2)],
            [math.sin(theta_mid) * dt,  0.5 * v * math.cos(theta_mid) * (dt ** 2)],
            [0.0,                      dt]
        ])
        return G

    def predict(self, u: np.ndarray, Q_u: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF prediction step.

        Args:
            u: Measured control input [v, omega]
            Q_u: Covariance of control input (2x2)
            dt: Time step in seconds
        """
        F = self.state_jacobian(self.x, u, dt)
        G = self.input_jacobian(self.x, u, dt)

        # Propagate state
        self.x = self.state_transition(self.x, u, dt)

        # Propagate covariance with process noise
        Q_step = G @ Q_u @ G.T + self.Q_add * dt
        self.P = F @ self.P @ F.T + Q_step
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy(), self.P.copy()

    def update_gnss(
        self,
        z_gnss: np.ndarray,
        R_gnss: np.ndarray,
        gating_chi2: Optional[float] = None
    ) -> bool:
        """
        Update state using GNSS 2D position observation [x, y], accounting for lever arm.

        Args:
            z_gnss: Observed [x_gnss, y_gnss] (shape 2,)
            R_gnss: Measurement covariance (2x2)
            gating_chi2: Optional threshold on NIS (e.g. 9.21 for 99% 2-DOF)

        Returns:
            accepted: True if update was performed
        """
        px, py, theta = self.x[0], self.x[1], self.x[2]
        lx, ly = self.lever_arm[0], self.lever_arm[1]

        # Predicted antenna position
        c = math.cos(theta)
        s = math.sin(theta)
        z_pred = np.array([
            px + lx * c - ly * s,
            py + lx * s + ly * c
        ])

        # Measurement Jacobian H (2x3)
        H = np.array([
            [1.0, 0.0, -lx * s - ly * c],
            [0.0, 1.0,  lx * c - ly * s]
        ])

        # Innovation
        y = z_gnss - z_pred
        S = H @ self.P @ H.T + R_gnss
        S_inv = np.linalg.inv(S)

        # NIS
        nis = float(y.T @ S_inv @ y)
        self.last_nis_gnss = nis
        self.last_innovation_gnss = y.copy()

        # Chi-square innovation gating
        if gating_chi2 is not None and nis > gating_chi2:
            return False

        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ H.T @ S_inv

        # State update
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])

        # Joseph-form stabilized covariance update
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gnss @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def update_compass(
        self,
        z_heading: float,
        R_heading: float,
        gating_chi2: Optional[float] = None
    ) -> bool:
        """
        Update state using compass absolute heading observation.

        Args:
            z_heading: Heading angle in radians CCW from +X (shape 1,)
            R_heading: Heading variance (scalar)
            gating_chi2: Optional threshold on NIS (e.g. 6.63 for 99% 1-DOF)

        Returns:
            accepted: True if update was accepted
        """
        # Predicted observation: h(x) = theta
        z_pred = self.x[2]
        H = np.array([[0.0, 0.0, 1.0]])

        # Angular innovation with mandatory wrapping
        y_val = wrap_angle(z_heading - z_pred)
        S_val = float((H @ self.P @ H.T).item()) + float(R_heading)

        nis = (y_val ** 2) / S_val
        self.last_nis_compass = nis
        self.last_innovation_compass = y_val

        if gating_chi2 is not None and nis > gating_chi2:
            return False

        # Gain (3x1)
        K = (self.P @ H.T) / S_val

        # Update
        self.x = self.x + (K.flatten() * y_val)
        self.x[2] = wrap_angle(self.x[2])

        # Joseph update
        I_KH = np.eye(3) - K @ H
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
        """
        Compute finite-difference numerical Jacobians and compare with analytical Jacobians.

        Returns:
            Dict of max absolute errors for F, G, H_gnss.
        """
        x = np.array([1.5, -2.0, 0.785]) if x_test is None else x_test.copy()
        u = np.array([1.2, 0.35]) if u_test is None else u_test.copy()

        # --- 1. State Jacobian F ---
        F_ana = self.state_jacobian(x, u, dt)
        F_num = np.zeros_like(F_ana)
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = eps
            x_plus = self.state_transition(x + dx, u, dt)
            x_minus = self.state_transition(x - dx, u, dt)
            # Unwrap angle difference for theta row
            diff = x_plus - x_minus
            diff[2] = wrap_angle(diff[2])
            F_num[:, j] = diff / (2.0 * eps)
        err_F = float(np.max(np.abs(F_ana - F_num)))

        # --- 2. Input Jacobian G ---
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

        # --- 3. GNSS Lever-Arm Jacobian H ---
        lx, ly = self.lever_arm[0], self.lever_arm[1]
        def h_gnss(s):
            px, py, th = s[0], s[1], s[2]
            return np.array([
                px + lx * math.cos(th) - ly * math.sin(th),
                py + lx * math.sin(th) + ly * math.cos(th)
            ])

        H_ana = np.array([
            [1.0, 0.0, -lx * math.sin(x[2]) - ly * math.cos(x[2])],
            [0.0, 1.0,  lx * math.cos(x[2]) - ly * math.sin(x[2])]
        ])
        H_num = np.zeros_like(H_ana)
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = eps
            H_num[:, j] = (h_gnss(x + dx) - h_gnss(x - dx)) / (2.0 * eps)
        err_H = float(np.max(np.abs(H_ana - H_num)))

        return {"error_F": err_F, "error_G": err_G, "error_H_gnss": err_H}
