"""
robot_localization.estimation.ukf

Four-state Unscented Kalman Filter (UKF) on SE(2) x R:
    State: x = [p_x, p_y, theta, b_g]^T

Features:
- Full Riemannian / circular manifold handling on S^1 for heading:
  * Circular mean reconstruction via vector trigonometry: atan2(sum W_i sin theta_i, sum W_i cos theta_i)
  * Wrapped angular differences in covariance computation
  * Wrapped measurement innovations for compass heading
- Merwe scaled unscented transform with parameters alpha, beta, kappa
- Numerical positive-definite regularization with jitter fallback
- Matched 4D state and sensor models corresponding to BiasAwareEKF4D
"""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np

from robot_localization.models.kinematics import wrap_angle


class UnscentedKalmanFilter4D:
    """
    4D Unscented Kalman Filter estimating planar pose and gyroscope bias:
        x = [p_x, p_y, theta, b_g]^T
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        q_additive: Optional[np.ndarray] = None,
        gnss_lever_arm_m: Tuple[float, float] = (0.05, 0.0),
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        self.L = 4  # State dimension
        self.x = np.zeros(self.L) if initial_state is None else np.array(initial_state, dtype=float).copy()
        self.P = np.diag([1e-2, 1e-2, 5e-3, 1e-3]) if initial_cov is None else np.array(initial_cov, dtype=float).copy()
        self.Q_add = np.diag([1e-4, 1e-4, 1e-4, 1e-6]) if q_additive is None else np.array(q_additive, dtype=float).copy()

        self.lever_arm = np.array(gnss_lever_arm_m, dtype=float)

        # UKF Scaling parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_param = (alpha ** 2) * (self.L + kappa) - self.L
        self.gamma = math.sqrt(self.L + self.lambda_param)

        # Compute weights
        self.wm = np.zeros(2 * self.L + 1)
        self.wc = np.zeros(2 * self.L + 1)

        self.wm[0] = self.lambda_param / (self.L + self.lambda_param)
        self.wc[0] = self.wm[0] + (1.0 - alpha ** 2 + beta)

        for i in range(1, 2 * self.L + 1):
            w = 1.0 / (2.0 * (self.L + self.lambda_param))
            self.wm[i] = w
            self.wc[i] = w

        # Telemetry
        self.last_nis_gnss: Optional[float] = None
        self.last_nis_compass: Optional[float] = None
        self.last_innovation_gnss: Optional[np.ndarray] = None
        self.last_innovation_compass: Optional[float] = None

    def _compute_sigma_points(self) -> np.ndarray:
        """
        Generate 2L + 1 sigma points around current state x and covariance P.
        Returns:
            sigmas: (2L + 1, L)
        """
        P_sym = 0.5 * (self.P + self.P.T)
        # Numerical safeguard: ensure positive definiteness
        min_eig = np.min(np.real(np.linalg.eigvals(P_sym)))
        if min_eig < 1e-10:
            P_sym += np.eye(self.L) * (1e-10 - min_eig)

        try:
            L_mat = np.linalg.cholesky(P_sym)
        except np.linalg.LinAlgError:
            # Fallback to SVD
            u, s, vt = np.linalg.svd(P_sym)
            s_pos = np.maximum(s, 1e-12)
            L_mat = u @ np.diag(np.sqrt(s_pos))

        scaled_L = self.gamma * L_mat  # (L, L)

        sigmas = np.zeros((2 * self.L + 1, self.L))
        sigmas[0] = self.x.copy()

        for i in range(self.L):
            sigmas[i + 1] = self.x + scaled_L[:, i]
            sigmas[i + 1, 2] = wrap_angle(sigmas[i + 1, 2])

            sigmas[i + 1 + self.L] = self.x - scaled_L[:, i]
            sigmas[i + 1 + self.L, 2] = wrap_angle(sigmas[i + 1 + self.L, 2])

        return sigmas

    def predict(self, u: np.ndarray, Q_u: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF prediction step through nonlinear 4WD skid-steer kinematics.

        Args:
            u: [v_enc, gyro_z]
            Q_u: (2x2) input covariance [var_v, 0; 0, var_gyro]
            dt: integration time step
        """
        sigmas = self._compute_sigma_points()
        v_cmd = float(u[0])
        gyro_z = float(u[1])

        # 1. Propagate each sigma point through kinematics
        sigmas_pred = np.zeros_like(sigmas)
        for i in range(2 * self.L + 1):
            px = sigmas[i, 0]
            py = sigmas[i, 1]
            th = sigmas[i, 2]
            bg = sigmas[i, 3]

            # Unbiased yaw rate
            omega_unbiased = gyro_z - bg
            th_mid = th + 0.5 * omega_unbiased * dt

            px_next = px + v_cmd * math.cos(th_mid) * dt
            py_next = py + v_cmd * math.sin(th_mid) * dt
            th_next = wrap_angle(th + omega_unbiased * dt)
            bg_next = bg  # Random walk

            sigmas_pred[i] = [px_next, py_next, th_next, bg_next]

        # 2. Compute predicted mean
        # Linear states: x, y, b_g
        x_pred = np.zeros(self.L)
        x_pred[0] = np.sum(self.wm * sigmas_pred[:, 0])
        x_pred[1] = np.sum(self.wm * sigmas_pred[:, 1])
        x_pred[3] = np.sum(self.wm * sigmas_pred[:, 3])

        # Circular state: theta on S^1
        sin_sum = np.sum(self.wm * np.sin(sigmas_pred[:, 2]))
        cos_sum = np.sum(self.wm * np.cos(sigmas_pred[:, 2]))
        x_pred[2] = math.atan2(sin_sum, cos_sum)

        # 3. Compute predicted covariance
        P_pred = np.zeros((self.L, self.L))
        for i in range(2 * self.L + 1):
            dx = np.zeros(self.L)
            dx[0] = sigmas_pred[i, 0] - x_pred[0]
            dx[1] = sigmas_pred[i, 1] - x_pred[1]
            dx[2] = wrap_angle(sigmas_pred[i, 2] - x_pred[2])
            dx[3] = sigmas_pred[i, 3] - x_pred[3]

            P_pred += self.wc[i] * np.outer(dx, dx)

        # Add process noise
        th_mean = x_pred[2]
        c = math.cos(th_mean)
        s = math.sin(th_mean)
        G = np.array([
            [c * dt, -0.5 * v_cmd * s * (dt ** 2)],
            [s * dt,  0.5 * v_cmd * c * (dt ** 2)],
            [0.0,     dt],
            [0.0,     0.0]
        ])
        Q_step = G @ Q_u @ G.T + self.Q_add * dt
        P_pred += Q_step
        P_pred = 0.5 * (P_pred + P_pred.T)

        self.x = x_pred
        self.P = P_pred
        return self.x.copy(), self.P.copy()

    def update_gnss(
        self,
        z_gnss: np.ndarray,
        R_gnss: np.ndarray,
        gating_chi2: Optional[float] = None
    ) -> bool:
        """
        Update state using GNSS 2D position observation [x, y], accounting for lever arm.
        """
        sigmas = self._compute_sigma_points()
        lx, ly = self.lever_arm[0], self.lever_arm[1]

        # 1. Measurement function on sigma points
        gamma_pts = np.zeros((2 * self.L + 1, 2))
        for i in range(2 * self.L + 1):
            px = sigmas[i, 0]
            py = sigmas[i, 1]
            th = sigmas[i, 2]
            c = math.cos(th)
            s = math.sin(th)
            gamma_pts[i, 0] = px + lx * c - ly * s
            gamma_pts[i, 1] = py + lx * s + ly * c

        # 2. Predicted measurement mean
        z_pred = np.sum(self.wm[:, None] * gamma_pts, axis=0)

        # 3. Innovation and cross covariances
        P_zz = np.zeros((2, 2))
        P_xz = np.zeros((self.L, 2))

        for i in range(2 * self.L + 1):
            dz = gamma_pts[i] - z_pred
            dx = np.zeros(self.L)
            dx[0] = sigmas[i, 0] - self.x[0]
            dx[1] = sigmas[i, 1] - self.x[1]
            dx[2] = wrap_angle(sigmas[i, 2] - self.x[2])
            dx[3] = sigmas[i, 3] - self.x[3]

            P_zz += self.wc[i] * np.outer(dz, dz)
            P_xz += self.wc[i] * np.outer(dx, dz)

        P_zz += R_gnss
        P_zz = 0.5 * (P_zz + P_zz.T)

        # Innovation
        y = z_gnss - z_pred
        try:
            S_inv = np.linalg.inv(P_zz)
        except np.linalg.LinAlgError:
            return False

        nis = float(y.T @ S_inv @ y)
        self.last_nis_gnss = nis
        self.last_innovation_gnss = y.copy()

        if gating_chi2 is not None and nis > gating_chi2:
            return False

        # Kalman gain
        K = P_xz @ S_inv

        # State update
        dx_corr = K @ y
        self.x[0] += dx_corr[0]
        self.x[1] += dx_corr[1]
        self.x[2] = wrap_angle(self.x[2] + dx_corr[2])
        self.x[3] += dx_corr[3]

        # Covariance update
        self.P -= K @ P_zz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_compass(
        self,
        z_heading_rad: float,
        R_heading_rad2: float,
        gating_chi2: Optional[float] = None
    ) -> bool:
        """
        Update state using direct heading observation (CMPS12 compass) on S^1.
        """
        sigmas = self._compute_sigma_points()

        # Measurement on sigma points
        gamma_pts = sigmas[:, 2].copy()  # headings

        # Circular mean of measurement
        sin_sum = np.sum(self.wm * np.sin(gamma_pts))
        cos_sum = np.sum(self.wm * np.cos(gamma_pts))
        z_pred = math.atan2(sin_sum, cos_sum)

        # Covariances
        P_zz = 0.0
        P_xz = np.zeros(self.L)

        for i in range(2 * self.L + 1):
            dz = wrap_angle(gamma_pts[i] - z_pred)
            dx = np.zeros(self.L)
            dx[0] = sigmas[i, 0] - self.x[0]
            dx[1] = sigmas[i, 1] - self.x[1]
            dx[2] = wrap_angle(sigmas[i, 2] - self.x[2])
            dx[3] = sigmas[i, 3] - self.x[3]

            P_zz += self.wc[i] * (dz ** 2)
            P_xz += self.wc[i] * dx * dz

        P_zz += float(R_heading_rad2)
        P_zz = max(P_zz, 1e-12)

        # Innovation
        y = wrap_angle(z_heading_rad - z_pred)
        nis = float((y ** 2) / P_zz)
        self.last_nis_compass = nis
        self.last_innovation_compass = y

        if gating_chi2 is not None and nis > gating_chi2:
            return False

        # Kalman gain
        K = P_xz / P_zz

        # State update
        dx_corr = K * y
        self.x[0] += dx_corr[0]
        self.x[1] += dx_corr[1]
        self.x[2] = wrap_angle(self.x[2] + dx_corr[2])
        self.x[3] += dx_corr[3]

        # Covariance update
        self.P -= np.outer(K, K) * P_zz
        self.P = 0.5 * (self.P + self.P.T)
        return True
