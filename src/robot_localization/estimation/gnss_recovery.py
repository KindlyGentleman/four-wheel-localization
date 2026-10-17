"""
robot_localization.estimation.gnss_recovery

Rigorous recovery mechanisms for the GNSS gate-lockout failure mode.
When a robot experiences wheel slip during a prolonged GNSS blackout, open-loop
drift causes the innovation residual upon GNSS resumption to exceed standard
Chi-square gates (chi^2_2(0.99) = 9.210), permanently locking the estimator out.

Implements five distinct, testable recovery strategies:
- R0: Strict baseline gate (current behavior, reproduces lockout)
- R1: Temporal measurement consistency buffer (validates consecutive returning fixes)
- R2: Health-state hysteresis machine (TRACKING -> DEGRADED -> RECOVERY_PENDING -> RELOCALIZING -> TRACKING_RECOVERED)
- R3: Physically justified distance-scaled process covariance growth during outage
- R4: Controlled global pose re-anchoring with continuous local odometry frame preservation
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from robot_localization.models.kinematics import wrap_angle
from robot_localization.estimation.robust_ekf import RobustEKF4D


class RecoveryStrategy(str, Enum):
    R0_STRICT_GATE = "R0_strict_gate"
    R1_TEMPORAL_CONSISTENCY = "R1_temporal_consistency"
    R2_HEALTH_HYSTERESIS = "R2_health_hysteresis"
    R3_COVARIANCE_ADJUSTMENT = "R3_covariance_adjustment"
    R4_GLOBAL_REINITIALIZATION = "R4_global_reinitialization"


class EstimatorHealthState(str, Enum):
    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    RECOVERY_PENDING = "RECOVERY_PENDING"
    RELOCALIZING = "RELOCALIZING"
    TRACKING_RECOVERED = "TRACKING_RECOVERED"


@dataclass
class BufferedGNSSFix:
    timestamp: float
    position_enu: np.ndarray        # [x, y]
    covariance_enu: np.ndarray      # 2x2
    nis_against_prior: float
    integrated_dr_distance: float
    dr_position: Optional[np.ndarray] = None


class RecoverableRobustEKF4D(RobustEKF4D):
    """
    4D Robust EKF equipped with selectable fault recovery strategies (R0 - R4).
    """

    def __init__(
        self,
        strategy: RecoveryStrategy = RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        gyro_bias_walk_var: float = 1e-6,
        q_pos_add: float = 1e-4,
        q_theta_add: float = 1e-4,
        gnss_lever_arm_m: Tuple[float, float] = (0.05, 0.0),
        gnss_gate_chi2: float = 9.210,
        compass_gate_chi2: float = 6.635,
        slip_omega_thresh_rad_s: float = 0.08,
        slip_v_thresh_m_s: float = 0.05,
        slip_inflation_gain: float = 12.0,
        outage_drift_rate_m_s: float = 0.05,
        # Recovery specific parameters
        consistency_window_size: int = 2,
        max_physical_speed_m_s: float = 1.5,
        relative_disp_tol_m: float = 0.80,
        distance_drift_fraction: float = 0.12 # 12% odometric drift per meter traveled
    ):
        super().__init__(
            initial_state=initial_state,
            initial_cov=initial_cov,
            gyro_bias_walk_var=gyro_bias_walk_var,
            q_pos_add=q_pos_add,
            q_theta_add=q_theta_add,
            gnss_lever_arm_m=gnss_lever_arm_m,
            gnss_gate_chi2=gnss_gate_chi2,
            compass_gate_chi2=compass_gate_chi2,
            slip_omega_thresh_rad_s=slip_omega_thresh_rad_s,
            slip_v_thresh_m_s=slip_v_thresh_m_s,
            slip_inflation_gain=slip_inflation_gain,
            outage_drift_rate_m_s=outage_drift_rate_m_s
        )
        self.strategy = strategy
        self.state_machine = EstimatorHealthState.TRACKING
        self.consistency_window = consistency_window_size
        self.max_physical_speed = max_physical_speed_m_s
        self.relative_disp_tol = relative_disp_tol_m
        self.distance_drift_fraction = distance_drift_fraction

        # Diagnostics and recovery buffers
        self.gnss_buffer: List[BufferedGNSSFix] = []
        self.total_dr_distance = 0.0
        self.distance_since_gnss = 0.0
        self.last_predicted_pos = self.x[:2].copy()

        # REP-105 Coordinate Frame tracking:
        # odom_pose: continuous, smooth dead-reckoning trajectory (never jumps)
        # map_to_odom: global correction transform (absorbs relocalization jumps)
        # map_pose: x[:3] (global pose in map frame)
        self.odom_pose: np.ndarray = self.x[:3].copy()
        self.map_to_odom: np.ndarray = np.zeros(3, dtype=np.float64)
        self.relocalization_events: List[Dict[str, Any]] = []

        # Telemetry
        self.recovery_events: List[Dict[str, Any]] = []
        self.health_history: List[Tuple[float, EstimatorHealthState]] = []
        self.false_recoveries = 0
        self.successful_recoveries = 0

    def predict_recoverable(
        self,
        v_enc: float,
        omega_enc: float,
        gyro_z: float,
        Q_u_base: np.ndarray,
        dt: float,
        wheel_disagreement: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step with strategy-specific uncertainty expansion and continuous odom tracking.
        """
        # Distance accumulation
        ds = abs(v_enc) * dt
        self.total_dr_distance += ds
        self.distance_since_gnss += ds

        # Continuous odom-frame integration (REP-105: odom -> base_link never jumps)
        from robot_localization.models.kinematics import integrate_pose_rk2
        self.odom_pose = integrate_pose_rk2(self.odom_pose, v_enc, omega_enc, dt)

        # Strategy R3: Physically justified distance-and-time-scaled process noise
        orig_q_pos = self.q_pos_add
        if self.strategy == RecoveryStrategy.R3_COVARIANCE_ADJUSTMENT:
            if self.time_since_gnss > 1.5:
                # Account for longitudinal odometry drift and unmodeled lateral slip during blackout
                drift_rate = self.distance_drift_fraction * max(abs(v_enc), 0.25) + 0.05 * abs(omega_enc)
                self.q_pos_add += float(drift_rate ** 2)

        # Base prediction
        x_pred, P_pred = super().predict_robust(
            v_enc=v_enc,
            omega_enc=omega_enc,
            gyro_z=gyro_z,
            Q_u_base=Q_u_base,
            dt=dt,
            wheel_disagreement=wheel_disagreement
        )
        self.q_pos_add = orig_q_pos
        self.last_predicted_pos = self.x[:2].copy()

        # Health state progression during blackout
        if self.time_since_gnss > 2.0:
            if self.state_machine == EstimatorHealthState.TRACKING:
                self.state_machine = EstimatorHealthState.DEGRADED

        return x_pred, P_pred

    def update_gnss_recoverable(
        self,
        t: float,
        z_gnss: np.ndarray,
        R_gnss: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Process GNSS update under the configured recovery strategy.
        Returns: (accepted, reason_or_state)
        """
        px, py, theta = float(self.x[0]), float(self.x[1]), float(self.x[2])
        lx, ly = float(self.lever_arm[0]), float(self.lever_arm[1])
        c = math.cos(theta)
        s = math.sin(theta)
        z_pred = np.array([px + lx * c - ly * s, py + lx * s + ly * c])

        H = np.array([
            [1.0, 0.0, -lx * s - ly * c, 0.0],
            [0.0, 1.0,  lx * c - ly * s, 0.0]
        ])
        y = z_gnss - z_pred
        S = H @ self.P @ H.T + R_gnss
        S_inv = np.linalg.inv(S)
        nis = float(y.T @ S_inv @ y)
        self.last_nis_gnss = nis

        # -------------------------------------------------------------
        # Strategy R0: Strict Gate (Lockout baseline)
        # -------------------------------------------------------------
        if self.strategy == RecoveryStrategy.R0_STRICT_GATE:
            if nis <= self.gnss_gate:
                self._apply_standard_update(y, H, S_inv, R_gnss)
                self.time_since_gnss = 0.0
                self.distance_since_gnss = 0.0
                self.accepted_gnss_count += 1
                return True, "Accepted (R0 nominal)"
            else:
                self.rejected_gnss_count += 1
                return False, "Rejected (R0 gate lockout)"

        # -------------------------------------------------------------
        # Strategy R3: Gating with covariance-adjusted innovation
        # -------------------------------------------------------------
        if self.strategy == RecoveryStrategy.R3_COVARIANCE_ADJUSTMENT:
            sigma_drift = (
                self.distance_drift_fraction * self.distance_since_gnss
                + 0.04 * self.time_since_gnss
            ) if self.time_since_gnss > 1.5 else 0.0
            S_adj = S + (sigma_drift ** 2) * np.eye(2)
            nis_adj = float(y.T @ np.linalg.inv(S_adj) @ y)
            self.last_nis_gnss = nis_adj
            if nis_adj <= self.gnss_gate:
                if self.time_since_gnss > 1.5:
                    self.P[0:2, 0:2] += (sigma_drift ** 2) * np.eye(2)
                    self.P[0:2, 2:] = 0.0
                    self.P[2:, 0:2] = 0.0
                    H_pos = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
                    S_pos = H_pos @ self.P @ H_pos.T + R_gnss
                    self._apply_standard_update(y, H_pos, np.linalg.inv(S_pos), R_gnss)
                    self.successful_recoveries += 1
                else:
                    self._apply_standard_update(y, H, S_inv, R_gnss)
                self.time_since_gnss = 0.0
                self.distance_since_gnss = 0.0
                self.accepted_gnss_count += 1
                self.state_machine = EstimatorHealthState.TRACKING
                return True, "Accepted (R3 covariance adjusted)"
            else:
                self.rejected_gnss_count += 1
                return False, "Rejected (R3 outlier)"

        # -------------------------------------------------------------
        # Nominal Gate Check (for R1, R2, R4 when tracking normally)
        # -------------------------------------------------------------
        if nis <= self.gnss_gate and self.state_machine in (EstimatorHealthState.TRACKING, EstimatorHealthState.TRACKING_RECOVERED):
            self._apply_standard_update(y, H, S_inv, R_gnss)
            self.time_since_gnss = 0.0
            self.distance_since_gnss = 0.0
            self.accepted_gnss_count += 1
            self.gnss_buffer.clear()
            self.state_machine = EstimatorHealthState.TRACKING
            return True, "Accepted (Nominal)"

        # -------------------------------------------------------------
        # Fix failed nominal gate! Buffer fix for temporal validation
        # -------------------------------------------------------------
        buf_item = BufferedGNSSFix(
            timestamp=t,
            position_enu=z_gnss.copy(),
            covariance_enu=R_gnss.copy(),
            nis_against_prior=nis,
            integrated_dr_distance=self.total_dr_distance,
            dr_position=self.odom_pose[:2].copy()
        )
        self.gnss_buffer.append(buf_item)
        if len(self.gnss_buffer) > 10:
            self.gnss_buffer.pop(0)

        # -------------------------------------------------------------
        # Strategy R1: Temporal Measurement Consistency Check
        # -------------------------------------------------------------
        if self.strategy == RecoveryStrategy.R1_TEMPORAL_CONSISTENCY:
            consistent, reason = self._check_buffer_consistency()
            if consistent:
                # Valid returning GNSS confirmed by temporal consistency!
                # Apply recovery update with temporarily expanded Kalman gain
                self._apply_recovery_update(z_gnss, R_gnss)
                self.time_since_gnss = 0.0
                self.distance_since_gnss = 0.0
                self.accepted_gnss_count += 1
                self.successful_recoveries += 1
                self.gnss_buffer.clear()
                self.state_machine = EstimatorHealthState.TRACKING_RECOVERED
                self.recovery_events.append({"time": t, "strategy": "R1", "reason": reason})
                return True, f"Accepted via Recovery ({reason})"
            else:
                self.rejected_gnss_count += 1
                return False, f"Quarantined in consistency buffer ({reason})"

        # -------------------------------------------------------------
        # Strategy R2: Health-State Hysteresis Machine
        # -------------------------------------------------------------
        elif self.strategy == RecoveryStrategy.R2_HEALTH_HYSTERESIS:
            if self.state_machine == EstimatorHealthState.TRACKING_RECOVERED and nis <= 2.5 * self.gnss_gate:
                # Hysteresis band: wider acceptance gate during post-recovery settling
                self._apply_standard_update(y, H, S_inv, R_gnss)
                self.time_since_gnss = 0.0
                self.distance_since_gnss = 0.0
                self.accepted_gnss_count += 1
                self.gnss_buffer.clear()
                self.state_machine = EstimatorHealthState.TRACKING
                return True, "Accepted within R2 hysteresis settling band"

            if self.state_machine in (
                EstimatorHealthState.DEGRADED,
                EstimatorHealthState.RECOVERY_PENDING,
                EstimatorHealthState.TRACKING_RECOVERED,
                EstimatorHealthState.TRACKING,
            ):
                self.state_machine = EstimatorHealthState.RECOVERY_PENDING
                consistent, reason = self._check_buffer_consistency()
                if consistent:
                    self.state_machine = EstimatorHealthState.RELOCALIZING
                    self._apply_recovery_update(z_gnss, R_gnss, infl_factor=0.60)
                    self.state_machine = EstimatorHealthState.TRACKING_RECOVERED
                    self.time_since_gnss = 0.0
                    self.distance_since_gnss = 0.0
                    self.accepted_gnss_count += 1
                    self.successful_recoveries += 1
                    self.gnss_buffer.clear()
                    self.recovery_events.append({"time": t, "strategy": "R2", "reason": reason})
                    return True, "Recovered via Hysteresis (RELOCALIZING -> TRACKING_RECOVERED)"
                else:
                    self.rejected_gnss_count += 1
                    return False, "Pending consistency verification in RECOVERY_PENDING"
            else:
                self.rejected_gnss_count += 1
                return False, "Rejected (R2 unexpected state)"

        # -------------------------------------------------------------
        # Strategy R4: Controlled Global Reinitialization with Local Continuity
        # -------------------------------------------------------------
        elif self.strategy == RecoveryStrategy.R4_GLOBAL_REINITIALIZATION:
            consistent, reason = self._check_buffer_consistency()
            if consistent:
                # Re-anchor global position while keeping heading and gyro bias intact
                pre_pos = self.x[:2].copy()
                self.x[0] = float(z_gnss[0] - (lx * c - ly * s))
                self.x[1] = float(z_gnss[1] - (lx * s + ly * c))
                post_pos = self.x[:2].copy()
                jump = post_pos - pre_pos
                jump_dist = float(np.linalg.norm(jump))
                if jump_dist > 0.05:
                    from robot_localization.models.kinematics import compute_map_to_odom_se2
                    self.map_to_odom = compute_map_to_odom_se2(self.x[:3], self.odom_pose)
                    self.relocalization_events.append({
                        "time": t,
                        "pre_pos": pre_pos.copy(),
                        "post_pos": post_pos.copy(),
                        "jump_dist": jump_dist,
                        "map_to_odom": self.map_to_odom.copy()
                    })

                # Reset position covariance and zero cross-terms to preserve positive-definiteness
                self.P[0:2, 0:2] = R_gnss.copy() * 1.5
                self.P[0:2, 2:] = 0.0
                self.P[2:, 0:2] = 0.0
                self.time_since_gnss = 0.0
                self.distance_since_gnss = 0.0
                self.accepted_gnss_count += 1
                self.successful_recoveries += 1
                self.gnss_buffer.clear()
                self.state_machine = EstimatorHealthState.TRACKING_RECOVERED
                self.recovery_events.append({"time": t, "strategy": "R4", "reason": "Global re-anchored"})
                return True, "Accepted via Controlled Reinitialization (R4)"
            else:
                self.rejected_gnss_count += 1
                return False, "Quarantined in R4 buffer"

        self.rejected_gnss_count += 1
        return False, "Rejected (Default)"

    def _check_buffer_consistency(self) -> Tuple[bool, str]:
        """
        Verify whether buffered GNSS fixes are mutually consistent over time
        and dynamically plausible given vehicle kinematics.
        Enforces that at least self.consistency_window (>=2) fixes are validated.
        """
        if len(self.gnss_buffer) < self.consistency_window:
            return False, f"Need at least {self.consistency_window} consecutive fixes (have {len(self.gnss_buffer)})"

        # Check all consecutive pairs in the buffer
        apparent_speed = 0.0
        for idx in range(1, len(self.gnss_buffer)):
            fix_prev = self.gnss_buffer[idx - 1]
            fix_curr = self.gnss_buffer[idx]

            dt = fix_curr.timestamp - fix_prev.timestamp
            if dt <= 1e-4:
                return False, "Zero or negative dt"

            # 1. Kinematic speed test
            delta_gnss = fix_curr.position_enu - fix_prev.position_enu
            disp_gnss = float(np.linalg.norm(delta_gnss))
            apparent_speed = disp_gnss / dt

            if apparent_speed > self.max_physical_speed:
                return False, f"Implausible speed {apparent_speed:.2f} m/s > {self.max_physical_speed} m/s"

            # 2. Relative displacement consistency with dead reckoning (vector check)
            if fix_curr.dr_position is not None and fix_prev.dr_position is not None:
                delta_dr = fix_curr.dr_position - fix_prev.dr_position
                discrepancy = float(np.linalg.norm(delta_gnss - delta_dr))
            else:
                dr_dist = abs(fix_curr.integrated_dr_distance - fix_prev.integrated_dr_distance)
                discrepancy = abs(disp_gnss - dr_dist)

            thresh = self.relative_disp_tol + 2.0 * float(np.sqrt(np.trace(fix_curr.covariance_enu)))
            if discrepancy > thresh:
                return False, f"Displacement discrepancy {discrepancy:.2f} m > threshold {thresh:.2f} m"

        return True, f"Consistent across {len(self.gnss_buffer)} fixes (speed {apparent_speed:.2f} m/s)"

    def _apply_standard_update(self, y: np.ndarray, H: np.ndarray, S_inv: np.ndarray, R: np.ndarray):
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])

        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def _apply_recovery_update(self, z_gnss: np.ndarray, R_gnss: np.ndarray, infl_factor: float = 0.85):
        """
        Apply recovery update by inflating P_prior for this step to allow convergence
        without risking numerical singularity or unbounded covariance collapse.
        Records discrete relocalization jump in map_to_odom.
        """
        px, py, theta = float(self.x[0]), float(self.x[1]), float(self.x[2])
        lx, ly = float(self.lever_arm[0]), float(self.lever_arm[1])
        c = math.cos(theta)
        s = math.sin(theta)
        z_pred = np.array([px + lx * c - ly * s, py + lx * s + ly * c])

        # Translational recovery Jacobian (decoupled from heading during coarse jump)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        y = z_gnss - z_pred

        # Temporary recovery variance inflation with cross-covariance decoupled
        P_temp = self.P.copy()
        err_norm = float(np.linalg.norm(y))
        P_temp[0, 0] = max(P_temp[0, 0], (err_norm * infl_factor) ** 2)
        P_temp[1, 1] = max(P_temp[1, 1], (err_norm * infl_factor) ** 2)
        P_temp[0:2, 2:] = 0.0
        P_temp[2:, 0:2] = 0.0

        S = H @ P_temp @ H.T + R_gnss
        S_inv = np.linalg.inv(S)
        K = P_temp @ H.T @ S_inv

        pre_pos = self.x[:2].copy()
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        post_pos = self.x[:2].copy()

        jump = post_pos - pre_pos
        jump_dist = float(np.linalg.norm(jump))
        if jump_dist > 0.05:
            from robot_localization.models.kinematics import compute_map_to_odom_se2
            self.map_to_odom = compute_map_to_odom_se2(self.x[:3], self.odom_pose)
            self.relocalization_events.append({
                "pre_pos": pre_pos.copy(),
                "post_pos": post_pos.copy(),
                "jump_dist": jump_dist,
                "map_to_odom": self.map_to_odom.copy()
            })

        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ P_temp @ I_KH.T + K @ R_gnss @ K.T
        self.P = 0.5 * (self.P + self.P.T)
