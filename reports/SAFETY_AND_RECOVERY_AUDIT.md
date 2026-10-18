# Safety Supervisor and Recovery Audit

## 1. Executive Summary

This report documents the rigorous re-audit and remediation of the Counterfactual Safety Supervisor (`src/robot_localization/control/fail_safe.py`) and GNSS Outage Recovery mechanisms (`src/robot_localization/estimation/gnss_recovery.py` and `realdata.py`).

Prior audits flagged potential safety vulnerabilities:
1. The supervisor previously declared standstill hold based on state labels without verifying fresh sensor velocity.
2. In open-loop simulation or playback, commanded stops did not guarantee physical standstill.
3. Spurious single-epoch GNSS fixes risked premature authorization of motion.

---

## 2. Identified Defect and Remediation

### 2.1 Continuous Velocity Verification
In `src/robot_localization/control/fail_safe.py`:
- In `STANDSTILL_HOLD` and `VERIFYING_RECOVERY`, the supervisor now continuously evaluates sensor velocity.
- If $|v_{\text{meas}}| \ge v_{\text{stopped\_thresh}}$ ($0.015$ m/s) or $|\omega_{\text{meas}}| \ge \omega_{\text{stopped\_thresh}}$ ($0.020$ rad/s), the state is instantly demoted back to `BRAKING`.
- If velocity measurement data is missing, stale ($>0.5$ s old), or NaN, standstill hold is immediately revoked.
- Any transition back to `BRAKING` resets `consecutive_valid_fixes = 0` and clears GNSS fix history.

### 2.2 Reacquisition Verification Pipeline
Before transitioning to `RESUMING`:
- The supervisor requires $N_{\text{req}} = 3$ consecutive valid GNSS fixes.
- Each fix must pass Chi-square innovation gating ($d_M^2 \le 9.21$).
- The measured vehicle velocity must remain strictly within stopped tolerances throughout all $N_{\text{req}}$ epochs.
- A single spurious or multipath fix immediately resets the counter to zero.

---

## 3. Adversarial Unit Test Evidence

The following adversarial regression tests were added to `tests/test_safe_stop.py`:
- `test_supervisor_requires_measured_standstill_before_standstill_hold`: Verifies that commanding zero velocity does not set `is_stopped = True` until physical sensor standstill is measured.
- `test_hold_revoked_when_motion_resumes`: Injects residual sliding velocity during `STANDSTILL_HOLD`; verifies state demotion to `BRAKING` and revocation of `is_stopped`.
- `test_verifying_recovery_rechecks_standstill`: Injects non-zero motion during reacquisition verification; confirms fix counter is cleared and state returns to `BRAKING`.
- `test_stale_or_invalid_velocity_cannot_confirm_stop`: Simulates sensor blackout/NaN during braking; verifies system refuses to declare standstill.
- `test_untrusted_gnss_cannot_authorize_restart`: Injects multipath-corrupted fixes during recovery verification; verifies rejection and counter reset.

All 11 tests in `tests/test_safe_stop.py` pass 100%.

---

## 4. Real-Data Hybrid Benchmark Verification

In `realdata.py` and `scripts/generate_figure_realdata.py`:
- Recovery states are dynamically exported to `results/realdata_traces.npz` (`sup_states_R4_recoverable_temporal`).
- State timeline in Figure 2 Panel (c) is rendered directly from logged state vectors.
- Hardcoded timestamps (16.0s, 18.0s, etc.) and synthetic re-anchoring markers have been eliminated.
- The supervisor reliably halts during simulated GNSS denial and smoothly transitions upon confirmed fix reacquisition.
