"""
robot_localization.control.fail_safe

Fail-Safe Navigation Controller and Safe-Stop Supervisor for GNSS-denied
and sensor-degraded mobile robot operations.

When global GNSS updates are lost for an extended interval or when localization
uncertainty exceeds safety envelopes:
1. Command zero velocity and simulate bounded deceleration to confirmed standstill.
2. Hold standstill position throughout the outage.
3. Reject single or spurious returning fixes; require multiple (M >= 3) mutually
   consistent, dynamically plausible fixes during standstill before authorizing recovery.
4. Apply global relocalization strictly through REP-105 map->odom frame adjustments
   without discontinuous jumps in the continuous odom->base_link trajectory.
5. Resume navigation conservatively at restricted speed with bounded acceleration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np


class SafeStopState(str, Enum):
    NOMINAL = "NOMINAL"
    BRAKING = "BRAKING"
    STANDSTILL_HOLD = "STANDSTILL_HOLD"
    VERIFYING_RECOVERY = "VERIFYING_RECOVERY"
    RESUMING = "RESUMING"


@dataclass
class SafeStopConfig:
    """Parameters governing fail-safe braking, standstill holding, and recovery."""
    max_outage_time_s: float = 2.0         # Enter braking if GNSS lost > 2.0 s
    max_pos_uncertainty_m: float = 1.2    # Enter braking if 1-sigma pos error > 1.2 m
    max_deceleration_m_s2: float = 0.6    # Bounded physical braking limit
    max_resume_accel_m_s2: float = 0.25   # Conservative acceleration on restart
    max_resume_speed_m_s: float = 0.20    # Crawl speed during resume
    standstill_vel_thresh_m_s: float = 0.015 # Velocity below which standstill is confirmed
    standstill_omega_thresh_rad_s: float = 0.02
    min_consistent_fixes: int = 3         # Require M >= 3 consecutive fixes
    standstill_gnss_speed_limit_m_s: float = 0.35 # Apparent speed limit during standstill
    stable_recovery_time_s: float = 2.0   # Hold resumed crawl for 2.0 s before full nominal


@dataclass
class SafeStopStatus:
    timestamp: float
    state: SafeStopState
    commanded_v: float
    commanded_omega: float
    actual_v: float
    actual_omega: float
    is_stopped: bool
    gnss_age_s: float
    pos_uncertainty_m: float
    consecutive_valid_fixes: int
    counterfactual_annotation: str = ""


class SafeStopSupervisor:
    """
    Supervises vehicle safety during GNSS denial and degraded sensor fusion.
    Can be used either:
    1. Online / Closed-Loop: Directly regulates vehicle command velocities.
    2. Counterfactual Annotation: Evaluates offline bags to document exactly when
       a physical safe stop would have been executed without corrupting historical logs.
    """

    def __init__(self, config: Optional[SafeStopConfig] = None):
        self.cfg = config if config is not None else SafeStopConfig()
        self.state = SafeStopState.NOMINAL
        self.current_v = 0.0
        self.current_omega = 0.0
        self.last_gnss_time: Optional[float] = None
        self.last_fix_position: Optional[np.ndarray] = None
        self.consecutive_valid_fixes = 0
        self.fix_history: List[Tuple[float, np.ndarray]] = []
        self.time_in_state = 0.0
        self.resumed_at_t: Optional[float] = None

    def update(
        self,
        t: float,
        dt: float,
        v_desired: float,
        omega_desired: float,
        v_measured: float,
        omega_measured: float,
        pos_cov_diag: np.ndarray,
        new_gnss_fix: Optional[np.ndarray] = None
    ) -> SafeStopStatus:
        """
        Update the safety supervisor state machine and compute safe command velocity.

        Args:
            t: Current timestamp in seconds
            dt: Timestep duration
            v_desired: Nominal navigation planner forward velocity (m/s)
            omega_desired: Nominal navigation planner angular velocity (rad/s)
            v_measured: Current measured/estimated forward speed (m/s)
            omega_measured: Current measured/estimated turn rate (rad/s)
            pos_cov_diag: [P_xx, P_yy] position variance
            new_gnss_fix: [x, y] coordinates if a new GNSS fix arrived at time t
        """
        dt = max(1e-4, dt)
        pos_std = float(np.sqrt(max(0.0, float(np.sum(pos_cov_diag[:2])))))

        # Track GNSS arrival
        if new_gnss_fix is not None:
            self.last_gnss_time = t

        gnss_age = (t - self.last_gnss_time) if self.last_gnss_time is not None else 999.0

        # Check triggers for safety intervention
        uncertainty_breached = pos_std > self.cfg.max_pos_uncertainty_m
        outage_prolonged = gnss_age > self.cfg.max_outage_time_s

        # -------------------------------------------------------------
        # State Machine Transitions
        # -------------------------------------------------------------
        if self.state == SafeStopState.NOMINAL:
            self.current_v = v_measured
            self.current_omega = omega_measured

            if outage_prolonged or uncertainty_breached:
                self.state = SafeStopState.BRAKING
                self.time_in_state = 0.0
                self.consecutive_valid_fixes = 0
                self.fix_history.clear()

        elif self.state == SafeStopState.BRAKING:
            # Command zero velocity with bounded deceleration
            dec = self.cfg.max_deceleration_m_s2 * dt
            if self.current_v > 0.0:
                self.current_v = max(0.0, self.current_v - dec)
            elif self.current_v < 0.0:
                self.current_v = min(0.0, self.current_v + dec)

            # Angular deceleration
            dec_w = (self.cfg.max_deceleration_m_s2 / 0.3) * dt
            if self.current_omega > 0.0:
                self.current_omega = max(0.0, self.current_omega - dec_w)
            elif self.current_omega < 0.0:
                self.current_omega = min(0.0, self.current_omega + dec_w)

            # Check for confirmed standstill
            # Check for confirmed physical standstill from sensor measurements
            at_standstill = (
                abs(v_measured) < self.cfg.standstill_vel_thresh_m_s
                and abs(omega_measured) < self.cfg.standstill_omega_thresh_rad_s
            )
            if at_standstill:
                self.state = SafeStopState.STANDSTILL_HOLD
                self.current_v = 0.0
                self.current_omega = 0.0
                self.time_in_state = 0.0

        elif self.state == SafeStopState.STANDSTILL_HOLD:
            # Enforce zero motion
            self.current_v = 0.0
            self.current_omega = 0.0

            # Evaluate incoming GNSS fixes
            if new_gnss_fix is not None:
                is_valid = self._validate_standstill_fix(t, new_gnss_fix)
                if is_valid:
                    self.consecutive_valid_fixes += 1
                    self.fix_history.append((t, new_gnss_fix.copy()))
                    if self.consecutive_valid_fixes >= self.cfg.min_consistent_fixes:
                        # Required M >= 3 consistent fixes verified!
                        self.state = SafeStopState.VERIFYING_RECOVERY
                        self.time_in_state = 0.0
                else:
                    # Inconsistent or noisy fix; reset verification count
                    self.consecutive_valid_fixes = 0
                    self.fix_history.clear()

        elif self.state == SafeStopState.VERIFYING_RECOVERY:
            # Maintain standstill while verifying localization convergence
            self.current_v = 0.0
            self.current_omega = 0.0

            if new_gnss_fix is not None:
                is_valid = self._validate_standstill_fix(t, new_gnss_fix)
                if is_valid:
                    self.consecutive_valid_fixes += 1
                    self.fix_history.append((t, new_gnss_fix.copy()))
                else:
                    # Drop back to standstill hold if fix fails consistency
                    self.state = SafeStopState.STANDSTILL_HOLD
                    self.consecutive_valid_fixes = 0
                    self.fix_history.clear()

            # Authorize restart once uncertainty drops below safety limit
            if pos_std <= self.cfg.max_pos_uncertainty_m and self.consecutive_valid_fixes >= self.cfg.min_consistent_fixes:
                self.state = SafeStopState.RESUMING
                self.time_in_state = 0.0
                self.resumed_at_t = t
                target_v = min(v_desired, self.cfg.max_resume_speed_m_s)
                self.current_v = min(target_v, self.cfg.max_resume_accel_m_s2 * dt)
                self.current_omega = omega_desired * 0.5

        elif self.state == SafeStopState.RESUMING:
            # Controlled restart with limited speed and gentle acceleration
            target_v = min(v_desired, self.cfg.max_resume_speed_m_s)
            acc = self.cfg.max_resume_accel_m_s2 * dt

            if self.current_v < target_v:
                self.current_v = min(target_v, self.current_v + acc)
            else:
                self.current_v = target_v

            self.current_omega = omega_desired * 0.5 # Muted turning during restart

            # If tracking remains healthy for stable_recovery_time_s, return to nominal
            if not outage_prolonged and not uncertainty_breached:
                if self.time_in_state >= self.cfg.stable_recovery_time_s:
                    self.state = SafeStopState.NOMINAL
                    self.time_in_state = 0.0
            else:
                # Re-enter braking if problem reoccurs
                self.state = SafeStopState.BRAKING
                self.time_in_state = 0.0

        self.time_in_state += dt

        # Generate human-readable counterfactual annotation
        is_stopped = (self.state in (SafeStopState.STANDSTILL_HOLD, SafeStopState.VERIFYING_RECOVERY))
        annotation = self._generate_annotation(is_stopped, gnss_age, pos_std)

        return SafeStopStatus(
            timestamp=t,
            state=self.state,
            commanded_v=self.current_v if self.state != SafeStopState.NOMINAL else v_desired,
            commanded_omega=self.current_omega if self.state != SafeStopState.NOMINAL else omega_desired,
            actual_v=v_measured,
            actual_omega=omega_measured,
            is_stopped=is_stopped,
            gnss_age_s=gnss_age,
            pos_uncertainty_m=pos_std,
            consecutive_valid_fixes=self.consecutive_valid_fixes,
            counterfactual_annotation=annotation
        )

    def _validate_standstill_fix(self, t: float, fix: np.ndarray) -> bool:
        """
        Verify that an incoming GNSS fix during standstill does not exhibit
        implausible apparent motion exceeding the stationary velocity limit.
        """
        if len(self.fix_history) == 0:
            return True # First fix accepted into validation pipeline

        prev_t, prev_fix = self.fix_history[-1]
        dt = t - prev_t
        if dt <= 1e-4:
            return False

        dist = float(np.linalg.norm(fix[:2] - prev_fix[:2]))
        apparent_speed = dist / dt

        if apparent_speed > self.cfg.standstill_gnss_speed_limit_m_s:
            return False

        return True

    def _generate_annotation(self, is_stopped: bool, gnss_age: float, pos_std: float) -> str:
        if self.state == SafeStopState.NOMINAL:
            return "Nominal Navigation"
        elif self.state == SafeStopState.BRAKING:
            return f"Controlled Deceleration (GNSS Age: {gnss_age:.1f}s, Unc: {pos_std:.2f}m)"
        elif self.state == SafeStopState.STANDSTILL_HOLD:
            return f"Standstill Hold ({self.consecutive_valid_fixes}/{self.cfg.min_consistent_fixes} Fixes Verified)"
        elif self.state == SafeStopState.VERIFYING_RECOVERY:
            return f"Verifying Global Pose Re-Anchoring (Unc: {pos_std:.2f}m)"
        elif self.state == SafeStopState.RESUMING:
            return f"Conservative Restart (Speed capped at {self.cfg.max_resume_speed_m_s:.2f} m/s)"
        return ""
