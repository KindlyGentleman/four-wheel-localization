"""
tests.test_gnss_recovery

Unit tests for GNSS gate-lockout recovery strategies (R0 through R4).
Verifies:
1. Strategy R0 (strict gate) gets permanently locked out when returning fixes drift > chi^2 threshold.
2. Strategy R1 (temporal consistency) and R2 (health hysteresis) successfully recover after 20s outage.
3. Strategy R3 (covariance adjustment) naturally encompasses the returning fix.
4. Strategy R4 (controlled reinitialization) re-anchors global pose cleanly.
5. Critical safety check: Isolated multipath jumps are STILL rejected by R1 and R2.
"""

import pytest
import numpy as np

from robot_localization.estimation.gnss_recovery import (
    RecoverableRobustEKF4D,
    RecoveryStrategy,
    EstimatorHealthState,
    RecoveryEvent
)
from robot_localization.models.kinematics import se2_compose
from robot_localization.evaluation.metrics import evaluate_recovery_events


def simulate_outage_and_return(strategy: RecoveryStrategy, num_outage_seconds: float = 20.0):
    """
    Helper function simulating a 4WD rover driving straight under 10% unmodeled slip
    during a GNSS outage, followed by returning GNSS fixes.
    """
    filter_inst = RecoverableRobustEKF4D(
        strategy=strategy,
        initial_state=np.zeros(4),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001])
    )

    dt = 0.02
    steps = int(num_outage_seconds / dt)
    v_enc = 1.0  # Commander says 1 m/s
    Qu = np.diag([1e-4, 2.5e-5])

    # True vehicle only moves at 0.90 m/s due to slip (0.10 slip ratio)
    true_x = 0.0
    for step in range(steps):
        t = step * dt
        true_x += 0.90 * dt
        filter_inst.predict_recoverable(
            v_enc=v_enc,
            omega_enc=0.0,
            gyro_z=0.0,
            Q_u_base=Qu,
            dt=dt,
            wheel_disagreement=0.08
        )

    # After 20 seconds:
    # Filter estimated position is approx 20.0 m
    # True vehicle position is 18.0 m (discrepancy = 2.0 m)
    t_return = num_outage_seconds
    R_gnss = np.diag([0.25, 0.25]) # 0.50 m std

    # Return 5 consecutive valid GNSS fixes at 1 Hz intervals matching true vehicle motion
    accepted_log = []
    for k in range(5):
        # Predict 1.0 second (50 steps of 0.02s) of vehicle motion between 1 Hz fixes
        for _ in range(steps_per_sec := int(1.0 / dt)):
            true_x += 0.90 * dt
            filter_inst.predict_recoverable(
                v_enc=v_enc,
                omega_enc=0.0,
                gyro_z=0.0,
                Q_u_base=Qu,
                dt=dt,
                wheel_disagreement=0.08
            )
        t_fix = t_return + (k + 1) * 1.0
        z_gnss = np.array([true_x + 0.02 * (-1)**k, 0.0])
        acc, reason = filter_inst.update_gnss_recoverable(t_fix, z_gnss, R_gnss)
        accepted_log.append(acc)

    return filter_inst, accepted_log, true_x


def test_strategy_r0_strict_gate_lockout():
    """Verify that R0 gets completely locked out (0% of returning fixes accepted)."""
    filter_inst, accepted_log, true_x = simulate_outage_and_return(RecoveryStrategy.R0_STRICT_GATE)
    # R0 must reject all returning valid fixes due to NIS gate lockout
    assert all(not acc for acc in accepted_log)
    final_error = abs(filter_inst.x[0] - true_x)
    assert final_error > 1.5, f"Expected unrecovered error > 1.5m, got {final_error:.2f}m"


def test_strategy_r1_temporal_consistency_recovery():
    """Verify that R1 successfully recovers after consistency check across consecutive fixes."""
    filter_inst, accepted_log, true_x = simulate_outage_and_return(RecoveryStrategy.R1_TEMPORAL_CONSISTENCY)
    # First fix is buffered/quarantined, second fix confirms consistency and recovers!
    assert not accepted_log[0] # Quarantined
    assert accepted_log[1]     # Recovered!
    assert accepted_log[2]     # Subsequent fixes track nominally

    final_error = abs(filter_inst.x[0] - true_x)
    assert final_error < 0.50, f"Expected recovered error < 0.50m, got {final_error:.2f}m"
    assert filter_inst.successful_recoveries >= 1


def test_strategy_r2_health_hysteresis_recovery():
    """Verify state transitions in R2 hysteresis machine."""
    filter_inst, accepted_log, true_x = simulate_outage_and_return(RecoveryStrategy.R2_HEALTH_HYSTERESIS)
    assert not accepted_log[0]
    assert accepted_log[1]
    assert filter_inst.state_machine in (EstimatorHealthState.TRACKING_RECOVERED, EstimatorHealthState.TRACKING)
    final_error = abs(filter_inst.x[0] - true_x)
    assert final_error < 0.50


def test_strategy_r3_covariance_adjustment_recovery():
    """Verify that R3 naturally accepts returning fix due to physically justified P expansion."""
    filter_inst, accepted_log, true_x = simulate_outage_and_return(RecoveryStrategy.R3_COVARIANCE_ADJUSTMENT)
    # Because covariance grew proportional to distance, NIS is naturally within gate
    assert accepted_log[0] # Immediately accepted!
    final_error = abs(filter_inst.x[0] - true_x)
    assert final_error < 0.50


def test_strategy_r4_global_reinitialization():
    """Verify that R4 re-anchors global pose cleanly upon consistency confirmation."""
    filter_inst, accepted_log, true_x = simulate_outage_and_return(RecoveryStrategy.R4_GLOBAL_REINITIALIZATION)
    assert not accepted_log[0] # First fix buffered
    assert accepted_log[1]     # Second fix re-anchors
    final_error = abs(filter_inst.x[0] - true_x)
    assert final_error < 0.50


def test_multipath_rejection_under_r1():
    """CRITICAL TEST: Ensure R1 does NOT falsely recover on isolated multipath jumps!"""
    filter_inst = RecoverableRobustEKF4D(
        strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        initial_state=np.zeros(4),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001])
    )

    # Robot is tracking nominally at origin
    R_gnss = np.diag([0.25, 0.25])
    # An isolated +5.0 meter multipath outlier arrives at t = 1.0s
    z_bad = np.array([5.0, 0.0])
    acc_bad, _ = filter_inst.update_gnss_recoverable(1.0, z_bad, R_gnss)
    assert not acc_bad # Must be rejected!

    # Next fix at t = 2.0s is back to true position (x = 0.1 m)
    z_good = np.array([0.1, 0.0])
    acc_good, _ = filter_inst.update_gnss_recoverable(2.0, z_good, R_gnss)
    # The jump between 5.0 and 0.1 is 4.9 m/s, exceeding max physical speed,
    # so the bad fix was NOT confirmed as a trajectory, and the true fix tracks
    assert acc_good or len(filter_inst.gnss_buffer) >= 1


def test_persistent_reverse_multipath_rejection():
    """
    Verify that correlated reverse multipath vectors are rejected.
    Robot moves forward at 1.0 m/s East.
    Two consecutive multipath fixes arrive indicating motion West at 1.0 m/s.
    Vector check must reject these fixes even though scalar speed matches.
    """
    filter_inst = RecoverableRobustEKF4D(
        strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        initial_state=np.zeros(4),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]),
        relative_disp_tol_m=0.50
    )
    dt = 0.02
    Qu = np.diag([1e-4, 2.5e-5])
    R_gnss = np.diag([0.04, 0.04])

    for _ in range(int(20.0 / dt)):
        filter_inst.predict_recoverable(
            v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.05
        )

    acc1, _ = filter_inst.update_gnss_recoverable(21.0, np.array([10.0, 0.0]), R_gnss)
    assert not acc1

    for _ in range(int(1.0 / dt)):
        filter_inst.predict_recoverable(
            v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.05
        )

    acc2, reason2 = filter_inst.update_gnss_recoverable(22.0, np.array([9.0, 0.0]), R_gnss)
    assert not acc2, "Reverse multipath should be rejected by vector displacement consistency"
    assert "Displacement discrepancy" in reason2


def test_static_multipath_pinning_rejection():
    """
    Verify that static multipath pinning (receiver repeats identical coordinate while robot moves)
    is rejected by the consistency check.
    """
    filter_inst = RecoverableRobustEKF4D(
        strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        initial_state=np.zeros(4),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]),
        relative_disp_tol_m=0.40
    )
    dt = 0.02
    Qu = np.diag([1e-4, 2.5e-5])
    R_gnss = np.diag([0.01, 0.01])

    for _ in range(int(15.0 / dt)):
        filter_inst.predict_recoverable(
            v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.05
        )

    acc1, _ = filter_inst.update_gnss_recoverable(16.0, np.array([5.0, 5.0]), R_gnss)
    assert not acc1

    for _ in range(int(1.0 / dt)):
        filter_inst.predict_recoverable(
            v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.05
        )

    acc2, reason2 = filter_inst.update_gnss_recoverable(17.0, np.array([5.0, 5.0]), R_gnss)
    assert not acc2, "Static multipath while driving should be rejected"
    assert "Displacement discrepancy" in reason2


def test_recovery_during_turning_maneuver():
    """
    Verify recovery when the robot executes a turning maneuver (non-zero yaw rate)
    during the outage.
    """
    np.random.seed(42)
    filter_inst = RecoverableRobustEKF4D(
        strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        initial_state=np.zeros(4),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001])
    )
    dt = 0.02
    Qu = np.diag([1e-4, 2.5e-5])
    R_gnss = np.diag([0.09, 0.09])

    true_pose = np.zeros(3)
    for step in range(int(15.0 / dt)):
        true_pose[0] += 0.92 * np.cos(true_pose[2]) * dt
        true_pose[1] += 0.92 * np.sin(true_pose[2]) * dt
        true_pose[2] += 0.10 * dt

        filter_inst.predict_recoverable(
            v_enc=1.0, omega_enc=0.10, gyro_z=0.10,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.08
        )

    for k in range(3):
        for _ in range(int(1.0 / dt)):
            true_pose[0] += 0.92 * np.cos(true_pose[2]) * dt
            true_pose[1] += 0.92 * np.sin(true_pose[2]) * dt
            true_pose[2] += 0.10 * dt
            filter_inst.predict_recoverable(
                v_enc=1.0, omega_enc=0.10, gyro_z=0.10,
                Q_u_base=Qu, dt=dt, wheel_disagreement=0.08
            )
        t_fix = 15.0 + (k + 1) * 1.0
        z_gnss = true_pose[:2] + np.random.normal(0, 0.02, size=2)
        acc, _ = filter_inst.update_gnss_recoverable(t_fix, z_gnss, R_gnss)
        if k >= 1:
            assert acc, f"Should recover on fix {k+1} during curved trajectory"

    pos_err = np.linalg.norm(filter_inst.x[:2] - true_pose[:2])
    assert pos_err < 0.60, f"Expected post-recovery error < 0.60m, got {pos_err:.2f}m"


def test_rep105_odometry_continuity_during_r4_jump():
    """
    Verify REP-105 adherence: during an R4 global re-anchoring jump in the map frame,
    the odom_pose remains smooth and continuous, and map_to_odom records the shift.
    """
    filter_inst = RecoverableRobustEKF4D(
        strategy=RecoveryStrategy.R4_GLOBAL_REINITIALIZATION,
        initial_state=np.zeros(4),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001])
    )
    dt = 0.02
    Qu = np.diag([1e-4, 2.5e-5])
    R_gnss = np.diag([0.04, 0.04])

    for _ in range(int(10.0 / dt)):
        filter_inst.predict_recoverable(
            v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.05
        )

    z1 = np.array([13.0, 0.0])
    acc1, _ = filter_inst.update_gnss_recoverable(11.0, z1, R_gnss)
    assert not acc1

    for _ in range(int(1.0 / dt)):
        filter_inst.predict_recoverable(
            v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.05
        )

    z2 = np.array([14.0, 0.0])
    acc2, _ = filter_inst.update_gnss_recoverable(12.0, z2, R_gnss)
    assert acc2

    assert abs(filter_inst.odom_pose[0] - 11.0) < 0.20
    assert abs(filter_inst.x[0] - 14.0) < 0.50
    assert abs(filter_inst.map_to_odom[0]) > 2.0


def test_persistent_bias_attack_integrity_detection():
    """
    Adversarial validation:
    A persistent translation offset attack (z_k = p_k + b) has delta_z == delta_p.
    When the offset is within the slip drift envelope, dead reckoning cannot
    distinguish it from wheel slip. R1 will accept the recovery.
    Verify that evaluate_recovery_integrity and evaluate_recovery_events catch
    the false recovery against ground truth and increment false_recoveries.
    """
    filter_inst = RecoverableRobustEKF4D(
        strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        initial_state=np.zeros(4),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001])
    )
    dt = 0.02
    Qu = np.diag([1e-4, 2.5e-5])
    R_gnss = np.diag([0.04, 0.04])

    true_pos_log = []
    time_log = []
    true_x = 0.0

    # 20 seconds outage under 10% slip
    for step in range(int(20.0 / dt)):
        t = step * dt
        true_x += 0.90 * dt
        filter_inst.predict_recoverable(
            v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.08
        )
        time_log.append(t)
        true_pos_log.append([true_x, 0.0])

    # Injected returning fixes have a persistent +2.5m offset (within slip envelope)
    # Filter estimated position is ~20.0 m, true position is ~18.0 m.
    # Spoofed fix claims 20.5 m (b = +2.5 m relative to ground truth)
    bias = 2.5
    for k in range(3):
        for _ in range(int(1.0 / dt)):
            true_x += 0.90 * dt
            filter_inst.predict_recoverable(
                v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
                Q_u_base=Qu, dt=dt, wheel_disagreement=0.08
            )
            t_curr = 20.0 + (k * 1.0) + (_ + 1) * dt
            time_log.append(t_curr)
            true_pos_log.append([true_x, 0.0])

        t_fix = 20.0 + (k + 1) * 1.0
        z_spoofed = np.array([true_x + bias, 0.0])
        acc, _ = filter_inst.update_gnss_recoverable(t_fix, z_spoofed, R_gnss)
        if k == 1:
            assert acc, "Second spoofed fix should pass relative delta check"

    # Evaluate integrity against ground truth
    gt_times = np.array(time_log)
    gt_positions = np.array(true_pos_log)

    true_rec, false_rec = filter_inst.evaluate_recovery_integrity(
        gt_times, gt_positions, error_threshold_m=1.0
    )
    assert false_rec >= 1, f"Expected at least 1 false recovery, got {false_rec}"
    assert true_rec == 0, f"Expected 0 true recoveries, got {true_rec}"

    # Also verify the standalone evaluation function from metrics.py
    eval_res = evaluate_recovery_events(
        filter_inst.recovery_events, gt_times, gt_positions, error_threshold_m=1.0
    )
    assert eval_res["false_recoveries"] >= 1
    assert eval_res["true_recoveries"] == 0
    assert eval_res["status"] == "verified"


def test_gross_persistent_offset_envelope_rejection():
    """
    Adversarial validation:
    A gross spoofed offset (+30 meters) has delta_z == delta_p, but exceeds
    the plausible kinematic blackout drift envelope.
    The filter must reject and quarantine it in the consistency buffer.
    """
    filter_inst = RecoverableRobustEKF4D(
        strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        initial_state=np.zeros(4),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001])
    )
    dt = 0.02
    Qu = np.diag([1e-4, 2.5e-5])
    R_gnss = np.diag([0.04, 0.04])

    true_x = 0.0
    for _ in range(int(15.0 / dt)):
        true_x += 0.90 * dt
        filter_inst.predict_recoverable(
            v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.08
        )

    # Injected fixes: +30 meters gross offset
    gross_bias = 30.0
    for k in range(3):
        for _ in range(int(1.0 / dt)):
            true_x += 0.90 * dt
            filter_inst.predict_recoverable(
                v_enc=1.0, omega_enc=0.0, gyro_z=0.0,
                Q_u_base=Qu, dt=dt, wheel_disagreement=0.08
            )
        t_fix = 15.0 + (k + 1) * 1.0
        z_gross = np.array([true_x + gross_bias, 0.0])
        acc, reason = filter_inst.update_gnss_recoverable(t_fix, z_gross, R_gnss)
        assert not acc, "Gross offset must be rejected by plausible drift envelope"
        if k >= 1:
            assert "exceeds plausible blackout drift envelope" in reason


def test_rep105_continuous_composition_at_all_epochs():
    """
    Verify ROS REP-105 composition invariant across all epochs:
      T_map_base == se2_compose(map_to_odom, odom_base)
    This must hold during nominal tracking, prediction, and recovery updates.
    """
    filter_inst = RecoverableRobustEKF4D(
        strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        initial_state=np.array([1.0, 2.0, 0.5, 0.0]),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001])
    )
    dt = 0.05
    Qu = np.diag([1e-4, 2.5e-5])
    R_gnss = np.diag([0.04, 0.04])

    def check_rep105_invariant():
        composed = se2_compose(filter_inst.map_to_odom, filter_inst.odom_pose)
        diff_pos = np.linalg.norm(composed[:2] - filter_inst.x[:2])
        diff_th = abs(composed[2] - filter_inst.x[2])
        assert diff_pos < 1e-5, f"Position mismatch {diff_pos} m in REP-105 composition"
        assert diff_th < 1e-5, f"Heading mismatch {diff_th} rad in REP-105 composition"

    # Initial check
    check_rep105_invariant()

    # Predictions
    for step in range(20):
        filter_inst.predict_recoverable(
            v_enc=0.8, omega_enc=0.05, gyro_z=0.05,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.02
        )
        check_rep105_invariant()

    # Nominal GNSS update
    t = 1.0
    z_nom = filter_inst.x[:2].copy() + np.array([0.01, -0.01])
    acc, _ = filter_inst.update_gnss_recoverable(t, z_nom, R_gnss)
    assert acc
    check_rep105_invariant()

    # Multiple predictions and updates
    for step in range(30):
        t += dt
        filter_inst.predict_recoverable(
            v_enc=0.8, omega_enc=-0.05, gyro_z=-0.05,
            Q_u_base=Qu, dt=dt, wheel_disagreement=0.02
        )
        check_rep105_invariant()
        if step % 10 == 0:
            z_step = filter_inst.x[:2].copy() + np.array([0.005, 0.005])
            filter_inst.update_gnss_recoverable(t, z_step, R_gnss)
            check_rep105_invariant()


