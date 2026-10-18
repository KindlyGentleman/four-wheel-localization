# GNSS Safety and Measurement Freshness Validation Report

## 1. Executive Summary

This report documents the forensic investigation and implementation of safeguards for GNSS input validation, measurement freshness verification, and recovery anti-poisoning in `src/robot_localization/control/fail_safe.py`.

Findings F01, F03, and F04 have been resolved:
- **F01 (P0):** Non-finite and malformed GNSS coordinates are strictly rejected before arithmetic operations.
- **F03 (P1):** Measurement freshness tracking was introduced (`v_measured_t` and `max_velocity_age_s = 0.5s`), revoking standstill holds when velocity measurements are stale or missing.
- **F04 (P1):** Documentation claims were harmonized: the supervisor's standstill check was clarified as a relative kinematic apparent-speed consistency check ($v_{\text{apparent}} \le 0.35$ m/s across consecutive fixes during standstill), distinct from the EKF's chi-square innovation gating.

---

## 2. Technical Remediation Details

### 2.1 Non-finite GNSS Coordinate and Malformed Input Rejection
In `SafeStopSupervisor`:
- Added `_validate_gnss_input(t, fix)`:
  - Validates `t` is finite.
  - Validates `fix` is not None, is a list/tuple/ndarray, converts to float array with shape `(>=2,)`.
  - Verifies `np.all(np.isfinite(fix[:2]))`.
  - Rejects NaN, Inf, incorrect dimensions, and non-numeric payloads.
- Rejection reasons are recorded in `status.rejection_reason`.
- Non-finite or malformed fixes do NOT refresh `last_valid_gnss_time` and do NOT suppress outage detection.
- In `STANDSTILL_HOLD` or `VERIFYING_RECOVERY`, any malformed fix immediately resets `consecutive_valid_fixes = 0`, clears `fix_history`, and aborts recovery.

### 2.2 Velocity Freshness Checking
- `SafeStopConfig` now includes `max_velocity_age_s: float = 0.5`.
- `update()` accepts optional `v_measured_t: Optional[float] = None`.
- If `v_measured_t` is provided, freshness is verified: `(t - v_measured_t) <= max_velocity_age_s` and `v_measured_t <= t + 1e-3`.
- Stale or non-finite velocity measurements invalidate `valid_motion_measurement`, preventing standstill declaration, revoking standstill holds back to active `BRAKING`, and aborting `RESUMING`.

### 2.3 Clarification of Standstill Consistency vs Global Integrity
- Kinematic fix consistency during standstill checks temporal displacement: $\Delta d / \Delta t \le 0.35$ m/s.
- `SafeStopStatus` now explicitly outputs:
  - `is_relative_consistent = (consecutive_valid_fixes >= min_consistent_fixes)`
  - `is_globally_verified = False` (explicitly disclosing that stationary consistency does not verify absolute ground-truth accuracy without an independent anchor).

---

## 3. Test Evidence

The following 17 targeted regression tests were implemented in `tests/test_safe_stop.py`:
1. `test_reject_nan_gnss_x`: Verified rejection of NaN in x coordinate.
2. `test_reject_nan_gnss_y`: Verified rejection of NaN in y coordinate.
3. `test_reject_infinite_gnss_coordinates`: Verified rejection of +/-Inf.
4. `test_reject_wrong_shape_gnss`: Verified rejection of 1D scalar or malformed arrays.
5. `test_reject_duplicate_gnss_timestamp`: Verified rejection of dt <= 1e-4.
6. `test_reject_nonmonotonic_gnss_timestamp`: Verified rejection of backward timestamps.
7. `test_invalid_gnss_does_not_refresh_trusted_age`: Verified corrupted streams do not suppress outage detection.
8. `test_invalid_gnss_does_not_increment_recovery_count`: Verified corrupted fixes do not increment valid fix counters.
9. `test_nan_fix_sequence_cannot_trigger_resume`: Verified repeated NaN fixes cannot trigger restart.
10. `test_persistent_biased_gnss_is_not_claimed_globally_verified`: Verified `is_globally_verified` remains False.
11. `test_stale_velocity_blocks_standstill_confirmation`: Verified stale velocity blocks stop confirmation.
12. `test_stale_velocity_revokes_hold`: Verified stale velocity during hold demotes state to braking.
13. `test_stale_velocity_blocks_restart`: Verified stale velocity aborts verifying recovery.
14. `test_invalid_velocity_during_resuming`: Verified non-finite velocity during resumption re-triggers braking.
15. `test_trusted_age_not_refreshed_by_bad_fix`: Verified untrusted jumps do not advance trusted fix timestamp.
16. `test_stationary_constant_bias_failure_mode`: Verified stationary biased fixes document relative consistency without false global claims.
17. `test_recovery_without_absolute_anchor_reports_uncertainty`: Verified restart is blocked when position uncertainty is unresolved.

All 28 tests in `tests/test_safe_stop.py` pass 100%.
