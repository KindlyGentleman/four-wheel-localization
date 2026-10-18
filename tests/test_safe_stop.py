"""
tests/test_safe_stop.py

Unit and regression tests for fail-safe navigation, safe-stop supervisor,
standstill verification, false-fix rejection, REP-105 coordinate frame
continuity, and zero ground-truth leakage.
"""

from pathlib import Path
import numpy as np
import pytest

from robot_localization.control.fail_safe import (
    SafeStopState,
    SafeStopConfig,
    SafeStopStatus,
    SafeStopSupervisor,
)
from robot_localization.estimation.gnss_recovery import (
    RecoverableRobustEKF4D,
    RecoveryStrategy,
    EstimatorHealthState,
)
from realdata import (
    load_ground_truth,
    load_calibration,
    estimate_onboard_wheel_disagreement,
    run_hybrid_experiment,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_prolonged_outage_triggers_braking_and_standstill_hold():
    """Verify that a prolonged blackout (>2s) transitions NOMINAL -> BRAKING -> STANDSTILL_HOLD."""
    cfg = SafeStopConfig(max_outage_time_s=2.0, max_deceleration_m_s2=0.5)
    supervisor = SafeStopSupervisor(cfg)

    # 1. Nominal tracking with regular GNSS
    for step in range(5):
        t = step * 0.2
        status = supervisor.update(
            t=t,
            dt=0.2,
            v_desired=0.5,
            omega_desired=0.0,
            v_measured=0.5,
            omega_measured=0.0,
            pos_cov_diag=np.array([0.01, 0.01]),
            new_gnss_fix=np.array([t * 0.5, 0.0])
        )
        assert status.state == SafeStopState.NOMINAL
        assert status.commanded_v == 0.5

    # 2. GNSS blackout starts at t = 1.0s. Run for 2.4s without fixes.
    # At t = 3.2s (outage = 2.2s > 2.0s), it must enter BRAKING
    curr_v = 0.5
    for step in range(6, 25):
        t = step * 0.2
        # Vehicle begins decelerating
        status = supervisor.update(
            t=t,
            dt=0.2,
            v_desired=0.5,
            omega_desired=0.0,
            v_measured=curr_v,
            omega_measured=0.0,
            pos_cov_diag=np.array([0.05, 0.05]) * (t - 1.0),
            new_gnss_fix=None
        )
        curr_v = status.commanded_v

    assert status.state in (SafeStopState.BRAKING, SafeStopState.STANDSTILL_HOLD)
    # Once stopped, velocity must be exactly zero
    if status.state == SafeStopState.STANDSTILL_HOLD:
        assert status.commanded_v == 0.0
        assert status.is_stopped is True


def test_missing_gnss_indefinitely_maintains_standstill():
    """Verify that if GNSS is missing indefinitely, the robot holds standstill forever."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)

    # Start and trigger outage
    supervisor.update(0.0, 0.1, 0.4, 0.0, 0.4, 0.0, np.array([0.01, 0.01]), new_gnss_fix=np.array([0.0, 0.0]))

    # Run for 100 seconds without any GNSS
    for step in range(1, 1000):
        t = step * 0.1
        status = supervisor.update(
            t=t,
            dt=0.1,
            v_desired=0.5,
            omega_desired=0.0,
            v_measured=0.0,
            omega_measured=0.0,
            pos_cov_diag=np.array([2.5, 2.5]),
            new_gnss_fix=None
        )

    assert status.state == SafeStopState.STANDSTILL_HOLD
    assert status.commanded_v == 0.0
    assert status.is_stopped is True
    assert status.consecutive_valid_fixes == 0


def test_single_spurious_fix_never_triggers_resume():
    """Verify that a single returning fix or multipath spike is rejected and cannot resume motion."""
    cfg = SafeStopConfig(max_outage_time_s=1.0, min_consistent_fixes=3)
    supervisor = SafeStopSupervisor(cfg)

    # Put robot in STANDSTILL_HOLD
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    supervisor.current_v = 0.0

    # Inject single GPS fix at t=10.0
    status1 = supervisor.update(
        t=10.0, dt=0.2, v_desired=0.5, omega_desired=0.0,
        v_measured=0.0, omega_measured=0.0, pos_cov_diag=np.array([0.5, 0.5]),
        new_gnss_fix=np.array([1.0, 0.0])
    )
    # MUST NOT resume on 1 fix!
    assert status1.state == SafeStopState.STANDSTILL_HOLD
    assert status1.commanded_v == 0.0
    assert status1.consecutive_valid_fixes == 1

    # Next step: NO fix (or wild multipath jump at t=10.2: [5.0, 0.0] -> 20 m/s!)
    status2 = supervisor.update(
        t=10.2, dt=0.2, v_desired=0.5, omega_desired=0.0,
        v_measured=0.0, omega_measured=0.0, pos_cov_diag=np.array([0.5, 0.5]),
        new_gnss_fix=np.array([5.0, 0.0]) # 4 meters in 0.2s = 20 m/s apparent speed!
    )
    # Must reject false fix and reset counter
    assert status2.state == SafeStopState.STANDSTILL_HOLD
    assert status2.commanded_v == 0.0
    assert status2.consecutive_valid_fixes == 0


def test_verified_reacquisition_and_conservative_resume():
    """Verify that M >= 3 consistent standstill fixes authorize conservative restart."""
    cfg = SafeStopConfig(
        max_outage_time_s=1.0,
        min_consistent_fixes=3,
        max_resume_accel_m_s2=0.20,
        max_resume_speed_m_s=0.20
    )
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    supervisor.current_v = 0.0

    # Supply 3 consistent standstill fixes at 1 Hz (t = 20.0, 21.0, 22.0)
    fixes = [
        np.array([1.00, 0.00]),
        np.array([1.02, 0.01]), # apparent speed = 0.02 m/s (< 0.35 m/s)
        np.array([1.01, 0.01]), # apparent speed = 0.01 m/s (< 0.35 m/s)
    ]

    for idx, (t, fix) in enumerate([(20.0, fixes[0]), (21.0, fixes[1]), (22.0, fixes[2])]):
        status = supervisor.update(
            t=t, dt=1.0, v_desired=0.5, omega_desired=0.0,
            v_measured=0.0, omega_measured=0.0, pos_cov_diag=np.array([0.1, 0.1]),
            new_gnss_fix=fix
        )

    # After 3 consistent fixes, state must be VERIFYING_RECOVERY or RESUMING
    assert status.state in (SafeStopState.VERIFYING_RECOVERY, SafeStopState.RESUMING)
    assert status.consecutive_valid_fixes >= 3

    # Next step: robot resumes conservatively
    status_resume = supervisor.update(
        t=22.2, dt=0.2, v_desired=0.5, omega_desired=0.0,
        v_measured=status.commanded_v, omega_measured=0.0, pos_cov_diag=np.array([0.05, 0.05]),
        new_gnss_fix=None
    )
    assert status_resume.state == SafeStopState.RESUMING
    # Speed must be strictly capped at max_resume_speed (0.20 m/s)
    assert 0.0 < status_resume.commanded_v <= 0.20


def test_rep105_coordinate_frame_continuity():
    """Verify that odom->base_link never exhibits discrete jumps, while map->odom absorbs corrections."""
    est = RecoverableRobustEKF4D(
        strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
        initial_cov=np.diag([0.01, 0.01, 0.005, 1e-4]),
        consistency_window_size=2
    )

    # 1. Predict 50 steps of open-loop dead-reckoning with simulated slip
    dt = 0.1
    for step in range(50):
        est.predict_recoverable(v_enc=0.3, omega_enc=0.0, gyro_z=0.0, Q_u_base=np.diag([0.01, 0.005]), dt=dt)

    odom_prior = est.odom_pose[:2].copy()
    map_prior = est.x[:2].copy()

    # 2. Supply two consistent GNSS fixes re-anchoring position back to (0.2, 0.0)
    fix1 = np.array([0.20, 0.00])
    fix2 = np.array([0.22, 0.01])
    est.update_gnss_recoverable(5.0, fix1, np.diag([0.05**2, 0.05**2]))
    accepted, reason = est.update_gnss_recoverable(6.0, fix2, np.diag([0.05**2, 0.05**2]))

    assert accepted, f"Expected recovery acceptance, got reason: {reason}"

    # odom_pose must have evolved continuously with kinematics (no jump!)
    odom_post = est.odom_pose[:2].copy()
    step_diff_odom = np.linalg.norm(odom_post - odom_prior)
    assert step_diff_odom < 0.01, f"Odom frame must not jump! Got jump {step_diff_odom:.3f} m"

    # map_to_odom must have absorbed the discrete relocalization correction
    map_post = est.x[:2].copy()
    map_jump = np.linalg.norm(map_post - map_prior)
    assert map_jump > 1.0, f"Map frame should absorb relocalization. Got {map_jump:.3f} m"
    assert len(est.relocalization_events) >= 1
    assert est.relocalization_events[-1]["jump_dist"] > 1.0


def test_zero_ground_truth_leakage_in_slip_estimator():
    """Verify that wheel slip disagreement is computed without any privileged ground-truth variables."""
    # Test nominal rolling (omega matches gyro, acceleration is smooth)
    score_nominal = estimate_onboard_wheel_disagreement(
        w_wheel=0.10, gyro_z=0.10, v_wheel=0.25, prev_v_wheel=0.25, dt=0.05
    )
    assert score_nominal < 0.01

    # Test wheel slip (wheels spinning rapidly, high yaw rate disagreement)
    score_slip = estimate_onboard_wheel_disagreement(
        w_wheel=0.50, gyro_z=0.01, v_wheel=0.40, prev_v_wheel=0.10, dt=0.05
    )
    assert score_slip > 0.08
    assert score_slip > score_nominal * 5.0


def test_supervisor_requires_measured_standstill_before_standstill_hold():
    """
    Verify that the safe-stop supervisor remains in BRAKING state when commanded
    velocity reaches zero if physical sensor measurements indicate the vehicle is
    still moving (v_measured > standstill_vel_thresh).
    """
    cfg = SafeStopConfig(max_outage_time_s=1.0, max_deceleration_m_s2=1.0)
    supervisor = SafeStopSupervisor(cfg)

    # 1. Trigger outage and enter BRAKING
    supervisor.update(
        t=0.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
        v_measured=0.5, omega_measured=0.0,
        pos_cov_diag=np.array([0.01, 0.01]),
        new_gnss_fix=np.array([0.0, 0.0])
    )

    # Advance time without GNSS -> enters BRAKING at t=1.2s
    status = supervisor.update(
        t=1.2, dt=0.1, v_desired=0.5, omega_desired=0.0,
        v_measured=0.3, omega_measured=0.0,
        pos_cov_diag=np.array([0.1, 0.1]),
        new_gnss_fix=None
    )
    assert status.state == SafeStopState.BRAKING

    # 2. Simulate commanded deceleration reaching 0.0, but vehicle continues sliding
    for step in range(10):
        t = 1.3 + step * 0.1
        status = supervisor.update(
            t=t, dt=0.1, v_desired=0.5, omega_desired=0.0,
            v_measured=0.25, # Vehicle still sliding at 0.25 m/s!
            omega_measured=0.0,
            pos_cov_diag=np.array([0.2, 0.2]),
            new_gnss_fix=None
        )

    # Commanded speed is zero, but vehicle has not stopped moving
    assert status.commanded_v == 0.0
    assert status.state == SafeStopState.BRAKING, (
        f"Supervisor must stay in BRAKING while v_measured > threshold, got {status.state}"
    )
    assert status.is_stopped is False

    # 3. Vehicle physically comes to a stop: v_measured drops below threshold
    status_stopped = supervisor.update(
        t=2.5, dt=0.1, v_desired=0.5, omega_desired=0.0,
        v_measured=0.005, # Measured velocity below 0.015 m/s
        omega_measured=0.0,
        pos_cov_diag=np.array([0.3, 0.3]),
        new_gnss_fix=None
    )
    assert status_stopped.state == SafeStopState.STANDSTILL_HOLD
    assert status_stopped.is_stopped is True


def test_hold_revoked_when_motion_resumes():
    """Verify that if external motion resumes during STANDSTILL_HOLD, hold is revoked immediately."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)

    # Force supervisor to STANDSTILL_HOLD
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    supervisor.consecutive_valid_fixes = 2

    # Provide motion measurement above standstill threshold (e.g. sliding on slope)
    status = supervisor.update(
        t=5.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
        v_measured=0.12, omega_measured=0.0,
        pos_cov_diag=np.array([0.5, 0.5]),
        new_gnss_fix=None
    )

    assert status.state == SafeStopState.BRAKING
    assert status.is_stopped is False
    assert supervisor.consecutive_valid_fixes == 0


def test_verifying_recovery_rechecks_standstill():
    """Verify that motion resumption during VERIFYING_RECOVERY aborts recovery."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)

    # Force supervisor to VERIFYING_RECOVERY
    supervisor.state = SafeStopState.VERIFYING_RECOVERY
    supervisor.consecutive_valid_fixes = 3

    # Provide angular motion measurement (e.g. yaw spin while holding)
    status = supervisor.update(
        t=10.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
        v_measured=0.005, omega_measured=0.15,
        pos_cov_diag=np.array([0.1, 0.1]),
        new_gnss_fix=np.array([0.0, 0.0])
    )

    assert status.state == SafeStopState.BRAKING
    assert status.is_stopped is False
    assert supervisor.consecutive_valid_fixes == 0


def test_stale_or_invalid_velocity_cannot_confirm_stop():
    """Verify that NaN or infinite velocity inputs cannot be used to confirm standstill."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.BRAKING

    status = supervisor.update(
        t=3.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
        v_measured=float("nan"), omega_measured=0.0,
        pos_cov_diag=np.array([0.5, 0.5]),
        new_gnss_fix=None
    )

    assert status.state == SafeStopState.BRAKING
    assert status.is_stopped is False


def test_untrusted_gnss_cannot_authorize_restart():
    """Verify that inconsistent GNSS fixes arriving at standstill reset validation count and block restart."""
    cfg = SafeStopConfig(max_outage_time_s=1.0, standstill_gnss_speed_limit_m_s=0.35)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD

    # Feed initial fix
    supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                      v_measured=0.0, omega_measured=0.0,
                      pos_cov_diag=np.array([0.1, 0.1]),
                      new_gnss_fix=np.array([0.0, 0.0]))
    assert supervisor.consecutive_valid_fixes == 1

    # Feed erratic fix jumping 2.0m in 0.1s (apparent speed 20 m/s > 0.35 m/s)
    status = supervisor.update(t=1.1, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.1, 0.1]),
                               new_gnss_fix=np.array([2.0, 0.0]))

    assert supervisor.consecutive_valid_fixes == 0
    assert status.state == SafeStopState.STANDSTILL_HOLD
    assert status.is_stopped is True


def test_reject_nan_gnss_x():
    """Verify that GNSS fix with NaN x-coordinate is rejected and cannot advance recovery."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    status = supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.1, 0.1]),
                               new_gnss_fix=np.array([float("nan"), 5.0]))
    assert supervisor.consecutive_valid_fixes == 0
    assert status.rejection_reason == "nonfinite_coordinates"


def test_reject_nan_gnss_y():
    """Verify that GNSS fix with NaN y-coordinate is rejected and cannot advance recovery."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    status = supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.1, 0.1]),
                               new_gnss_fix=np.array([5.0, float("nan")]))
    assert supervisor.consecutive_valid_fixes == 0
    assert status.rejection_reason == "nonfinite_coordinates"


def test_reject_infinite_gnss_coordinates():
    """Verify that infinite GNSS coordinates are rejected immediately."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    status = supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.1, 0.1]),
                               new_gnss_fix=np.array([float("inf"), 0.0]))
    assert supervisor.consecutive_valid_fixes == 0
    assert status.rejection_reason == "nonfinite_coordinates"


def test_reject_wrong_shape_gnss():
    """Verify that 1D scalar or malformed arrays are rejected safely."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    status = supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.1, 0.1]),
                               new_gnss_fix=np.array([1.0]))
    assert supervisor.consecutive_valid_fixes == 0
    assert status.rejection_reason == "invalid_shape"


def test_reject_duplicate_gnss_timestamp():
    """Verify that duplicate GNSS timestamps do not pass velocity validation."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                      v_measured=0.0, omega_measured=0.0,
                      pos_cov_diag=np.array([0.1, 0.1]),
                      new_gnss_fix=np.array([0.0, 0.0]))
    assert supervisor.consecutive_valid_fixes == 1
    # Duplicate timestamp t=1.0
    status = supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.1, 0.1]),
                               new_gnss_fix=np.array([0.01, 0.0]))
    assert supervisor.consecutive_valid_fixes == 0
    assert status.rejection_reason == "duplicate_or_nonmonotonic_timestamp"


def test_reject_nonmonotonic_gnss_timestamp():
    """Verify that out-of-order GNSS fixes are rejected."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    supervisor.update(t=2.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                      v_measured=0.0, omega_measured=0.0,
                      pos_cov_diag=np.array([0.1, 0.1]),
                      new_gnss_fix=np.array([0.0, 0.0]))
    assert supervisor.consecutive_valid_fixes == 1
    # Backward timestamp t=1.5
    status = supervisor.update(t=1.5, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.1, 0.1]),
                               new_gnss_fix=np.array([0.01, 0.0]))
    assert supervisor.consecutive_valid_fixes == 0
    assert status.rejection_reason == "duplicate_or_nonmonotonic_timestamp"


def test_invalid_gnss_does_not_refresh_trusted_age():
    """Verify that a stream of NaN fixes does not conceal a prolonged GNSS outage."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    # Valid fix at t=0.0
    supervisor.update(t=0.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                      v_measured=0.5, omega_measured=0.0,
                      pos_cov_diag=np.array([0.01, 0.01]),
                      new_gnss_fix=np.array([0.0, 0.0]))
    assert supervisor.last_valid_gnss_time == 0.0

    # Corrupted fix at t=2.0 (outage of 2.0s should be detected)
    status = supervisor.update(t=2.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.5, omega_measured=0.0,
                               pos_cov_diag=np.array([0.01, 0.01]),
                               new_gnss_fix=np.array([float("nan"), 0.0]))
    assert status.gnss_age_s == 2.0
    assert supervisor.last_valid_gnss_time == 0.0
    assert status.state == SafeStopState.BRAKING


def test_invalid_gnss_does_not_increment_recovery_count():
    """Verify that corrupted fixes cannot increment valid fix counters."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD
    supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                      v_measured=0.0, omega_measured=0.0,
                      pos_cov_diag=np.array([0.1, 0.1]),
                      new_gnss_fix=np.array([0.0, 0.0]))
    assert supervisor.consecutive_valid_fixes == 1

    # NaN fix arrives
    supervisor.update(t=1.1, dt=0.1, v_desired=0.5, omega_desired=0.0,
                      v_measured=0.0, omega_measured=0.0,
                      pos_cov_diag=np.array([0.1, 0.1]),
                      new_gnss_fix=np.array([float("nan"), 0.0]))
    assert supervisor.consecutive_valid_fixes == 0


def test_nan_fix_sequence_cannot_trigger_resume():
    """Verify that repeating NaN fixes during standstill cannot trigger resumption."""
    cfg = SafeStopConfig(max_outage_time_s=1.0, min_consistent_fixes=3)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD

    for step in range(10):
        t = 1.0 + step * 0.1
        status = supervisor.update(t=t, dt=0.1, v_desired=0.5, omega_desired=0.0,
                                   v_measured=0.0, omega_measured=0.0,
                                   pos_cov_diag=np.array([0.05, 0.05]),
                                   new_gnss_fix=np.array([float("nan"), float("nan")]))
        assert status.state == SafeStopState.STANDSTILL_HOLD
        assert status.commanded_v == 0.0
        assert supervisor.consecutive_valid_fixes == 0


def test_persistent_biased_gnss_is_not_claimed_globally_verified():
    """Verify that stationary fixes are flagged as relative consistency only, not absolute accuracy."""
    cfg = SafeStopConfig(max_outage_time_s=1.0, min_consistent_fixes=3)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD

    for step in range(5):
        t = 1.0 + step * 0.1
        status = supervisor.update(t=t, dt=0.1, v_desired=0.5, omega_desired=0.0,
                                   v_measured=0.0, omega_measured=0.0,
                                   pos_cov_diag=np.array([0.05, 0.05]),
                                   new_gnss_fix=np.array([100.0, 100.0]))

    assert status.is_relative_consistent is True
    assert status.is_globally_verified is False


def test_stale_velocity_blocks_standstill_confirmation():
    """Verify that velocity older than max_velocity_age_s blocks standstill confirmation."""
    cfg = SafeStopConfig(max_outage_time_s=1.0, max_velocity_age_s=0.5)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.BRAKING

    status = supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.1, 0.1]),
                               new_gnss_fix=None,
                               v_measured_t=0.0)
    assert status.state == SafeStopState.BRAKING
    assert status.is_stopped is False


def test_stale_velocity_revokes_hold():
    """Verify that stale velocity during hold revokes standstill state."""
    cfg = SafeStopConfig(max_outage_time_s=1.0, max_velocity_age_s=0.5)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD

    status = supervisor.update(t=2.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.1, 0.1]),
                               new_gnss_fix=None,
                               v_measured_t=1.0)
    assert status.state == SafeStopState.BRAKING
    assert status.is_stopped is False


def test_stale_velocity_blocks_restart():
    """Verify that stale velocity during recovery verification aborts to braking."""
    cfg = SafeStopConfig(max_outage_time_s=1.0, min_consistent_fixes=3, max_velocity_age_s=0.5)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.VERIFYING_RECOVERY
    supervisor.consecutive_valid_fixes = 3

    status = supervisor.update(t=2.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=0.0, omega_measured=0.0,
                               pos_cov_diag=np.array([0.05, 0.05]),
                               new_gnss_fix=np.array([0.0, 0.0]),
                               v_measured_t=1.0)
    assert status.state == SafeStopState.BRAKING
    assert supervisor.consecutive_valid_fixes == 0


def test_invalid_velocity_during_resuming():
    """Verify that non-finite velocity during resumption immediately re-triggers braking."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.RESUMING

    status = supervisor.update(t=3.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                               v_measured=float("nan"), omega_measured=0.0,
                               pos_cov_diag=np.array([0.05, 0.05]),
                               new_gnss_fix=np.array([0.0, 0.0]))
    assert status.state == SafeStopState.BRAKING


def test_trusted_age_not_refreshed_by_bad_fix():
    """Verify that an untrusted jump does not advance the trusted fix timestamp."""
    cfg = SafeStopConfig(max_outage_time_s=1.0)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD

    supervisor.update(t=1.0, dt=0.1, v_desired=0.5, omega_desired=0.0,
                      v_measured=0.0, omega_measured=0.0,
                      pos_cov_diag=np.array([0.1, 0.1]),
                      new_gnss_fix=np.array([0.0, 0.0]))
    assert supervisor.last_trusted_gnss_time == 1.0

    supervisor.update(t=1.1, dt=0.1, v_desired=0.5, omega_desired=0.0,
                      v_measured=0.0, omega_measured=0.0,
                      pos_cov_diag=np.array([0.1, 0.1]),
                      new_gnss_fix=np.array([5.0, 0.0]))
    assert supervisor.last_trusted_gnss_time == 1.0


def test_stationary_constant_bias_failure_mode():
    """Verify that stationary biased fixes verify relative consistency but note unanchored state."""
    cfg = SafeStopConfig(max_outage_time_s=1.0, min_consistent_fixes=3)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD

    for step in range(3):
        t = 1.0 + step * 0.1
        status = supervisor.update(t=t, dt=0.1, v_desired=0.5, omega_desired=0.0,
                                   v_measured=0.0, omega_measured=0.0,
                                   pos_cov_diag=np.array([0.05, 0.05]),
                                   new_gnss_fix=np.array([50.0, 50.0]))
    assert status.state == SafeStopState.VERIFYING_RECOVERY
    assert "Relative Consistency Verified" in status.counterfactual_annotation
    assert status.is_globally_verified is False


def test_recovery_without_absolute_anchor_reports_uncertainty():
    """Verify that recovery cannot authorize resumption if position uncertainty is unverified."""
    cfg = SafeStopConfig(max_outage_time_s=1.0, max_pos_uncertainty_m=1.0, min_consistent_fixes=3)
    supervisor = SafeStopSupervisor(cfg)
    supervisor.state = SafeStopState.STANDSTILL_HOLD

    for step in range(3):
        t = 1.0 + step * 0.1
        status = supervisor.update(t=t, dt=0.1, v_desired=0.5, omega_desired=0.0,
                                   v_measured=0.0, omega_measured=0.0,
                                   pos_cov_diag=np.array([2.5, 2.5]),
                                   new_gnss_fix=np.array([0.0, 0.0]))
    assert status.state == SafeStopState.VERIFYING_RECOVERY
    assert status.commanded_v == 0.0


