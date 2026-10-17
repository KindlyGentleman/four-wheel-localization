# GNSS Recovery Validation and Multipath Immunity Report

## 1. Executive Summary

During prolonged GNSS outages where a mobile robot experiences unmodeled wheel slip or odometry drift, the innovation residual upon GNSS resumption exceeds the classical Chi-square gate:
$$\gamma = (\mathbf{z} - \hat{\mathbf{z}})^T \mathbf{S}^{-1} (\mathbf{z} - \hat{\mathbf{z}}) > \chi^2_m(1 - \alpha)$$
In classical gating implementations (such as R0 / R3 in this repository), every returning GNSS fix is permanently rejected, stranding the estimator in open-loop dead reckoning.

This audit addressed two vulnerabilities in the recovery mechanisms:
1. **Finding F09**: Inconsistency between the documented consistency window ($M=3$ fixes) in `README.md` and the implementation parameter default (`consistency_window_size=2`) in `src/robot_localization/estimation/gnss_recovery.py`.
2. **Finding F11**: Vulnerability in `_check_buffer_consistency` where displacement consistency was evaluated using scalar distance differences instead of 2D vector displacement discrepancy, leaving the estimator vulnerable to reverse-moving multipath and heading-inverted outliers.

Both issues have been remediated, verified mathematically, and proven empirically through expanded adversarial unit tests and hybrid dataset benchmarks.

---

## 2. Audit of Finding F09: Consistency Window Parameter Alignment

### 2.1 Diagnosis
In `README.md`, section "Temporal Recovery Buffer" documented:
> "Buffers $M=3$ consecutive incoming GNSS fixes before accepting recovery update."

However, `RecoverableRobustEKF4D.__init__` specified:
```python
consistency_window_size: int = 2
```
Empirical evaluation demonstrated that with $M=2$ consecutive fixes (defining 1 discrete displacement vector between epochs $k-1$ and $k$), the hybrid realdata benchmark produces:
- Recovery latency: 1.86 s (first fix quarantined, second fix confirmed at 1.86 s after outage)
- Fix acceptance rate: 96.3% (52 accepted out of 54 received)
- Trajectory RMSE: 1.094 m (matching reported literature metrics)

Evaluating $M=3$ on the same dataset required 2 full intervals after blackout (2.85 s latency), reduced acceptance to 92.6%, and yielded an RMSE of 1.126 m.

### 2.2 Resolution
The implementation parameter `consistency_window_size=2` represents the minimal, dynamically sound window for verifying a 2D velocity vector ($\mathbf{v} = \Delta \mathbf{p} / \Delta t$). The documentation in `README.md` was updated to accurately specify $M=2$ consecutive fixes (1 verified displacement vector).

---

## 3. Audit of Finding F11: Vector Displacement Consistency vs Scalar Distance

### 3.1 Vulnerability in Original Implementation
The original consistency check in `src/robot_localization/estimation/gnss_recovery.py` calculated:
```python
disp_gnss = float(np.linalg.norm(fix_curr.position_enu - fix_prev.position_enu))
dr_dist = abs(fix_curr.integrated_dr_distance - fix_prev.integrated_dr_distance)
discrepancy = abs(disp_gnss - dr_dist)
```
This checked only the scalar arc length traveled by odometry against the scalar Euclidean distance between GNSS coordinates.

**Failure Mode (Reverse Multipath):**
Consider a robot moving East along the x-axis at $1.0\text{ m/s}$ ($\Delta \mathbf{p}_{\mathrm{dr}} = [1.0, 0.0]^T$).
Suppose multipath reflection causes two consecutive GNSS fixes to drift West at $1.0\text{ m/s}$ ($\Delta \mathbf{p}_{\mathrm{gnss}} = [-1.0, 0.0]^T$).
Under the scalar check:
$$\operatorname{disp}_{\mathrm{gnss}} = \|[-1.0, 0.0]\| = 1.0\text{ m}$$
$$\operatorname{dr\_dist} = 1.0\text{ m}$$
$$\text{discrepancy} = |1.0 - 1.0| = 0.0\text{ m}$$
The scalar check evaluated the discrepancy as $0.0\text{ m}$, erroneously accepting the inverted multipath vector as valid vehicle motion.

### 3.2 Remediated Vector Consistency Formulation
The check was upgraded to evaluate the full 2D vector displacement discrepancy against integrated local odometry in the global navigation frame:
$$\Delta \mathbf{p}_{\mathrm{gnss}} = \mathbf{p}_{\mathrm{gnss}, k} - \mathbf{p}_{\mathrm{gnss}, k-1}$$
$$\Delta \mathbf{p}_{\mathrm{dr}} = \mathbf{p}_{\mathrm{dr}, k} - \mathbf{p}_{\mathrm{dr}, k-1}$$
$$\epsilon_{\mathrm{vec}} = \|\Delta \mathbf{p}_{\mathrm{gnss}} - \Delta \mathbf{p}_{\mathrm{dr}}\|$$
$$\tau_{\mathrm{thresh}} = \delta_{\mathrm{rel}} + 2 \sqrt{\operatorname{tr}(\mathbf{R}_{\mathrm{gnss}})}$$

Under this formulation for the reverse multipath scenario:
$$\Delta \mathbf{p}_{\mathrm{gnss}} - \Delta \mathbf{p}_{\mathrm{dr}} = [-1.0, 0.0]^T - [1.0, 0.0]^T = [-2.0, 0.0]^T$$
$$\epsilon_{\mathrm{vec}} = 2.0\text{ m}$$
With $\delta_{\mathrm{rel}} = 0.50\text{ m}$ and GNSS standard deviation $\sigma = 0.20\text{ m}$ ($\tau_{\mathrm{thresh}} = 1.07\text{ m}$), the test yields:
$$2.0\text{ m} > 1.07\text{ m} \implies \text{REJECTED}$$
The erroneous fix is quarantined and rejected.

---

## 4. Test Verification Matrix

Ten dedicated tests in `tests/test_gnss_recovery.py` now execute and pass across nominal and adversarial operating regimes:

| Test Identifier | Category | Verification Objective | Result |
|---|---|---|---|
| `test_strategy_r0_strict_gate_lockout` | Baseline Gating | Proves R0 permanently rejects returning GNSS fixes after 20s slip | PASSED |
| `test_strategy_r1_temporal_consistency_recovery` | Recovery (R1) | Quarantines fix 1, recovers on fix 2, error < 0.50 m | PASSED |
| `test_strategy_r2_health_hysteresis_recovery` | Recovery (R2) | Validates state machine: TRACKING -> DEGRADED -> RECOVERY_PENDING -> RELOCALIZING -> TRACKING_RECOVERED | PASSED |
| `test_strategy_r3_covariance_adjustment_recovery` | Recovery (R3) | Outage distance-scaled covariance encompasses returning fix | PASSED |
| `test_strategy_r4_global_reinitialization` | Recovery (R4) | Re-anchors global pose cleanly upon consistency verification | PASSED |
| `test_multipath_rejection_under_r1` | Robustness | Isolated 5.0 m multipath spike is quarantined and rejected | PASSED |
| `test_persistent_reverse_multipath_rejection` | Adversarial (F11) | Rejects consecutive reverse multipath vectors matching vehicle speed | PASSED |
| `test_static_multipath_pinning_rejection` | Adversarial (F11) | Rejects receiver coordinate freeze while vehicle is in motion | PASSED |
| `test_recovery_during_turning_maneuver` | Kinematic (F11) | Verifies recovery during coordinated turning motion ($\omega \neq 0$) | PASSED |
| `test_rep105_odometry_continuity_during_r4_jump` | Coordinate Frames (F17) | Confirms `odom_pose` does not jump while `map_to_odom` absorbs offset | PASSED |

---

## 5. Hybrid Experimental Verification

Execution of `python realdata.py` on the M2DGR-plus hybrid dataset confirmed:
- Baseline Strict Gate (R3): RMSE 1.362 m, 0 recoveries, 55.6% acceptance (permanent lockout after outage).
- Temporal Recovery (R4): RMSE 1.094 m, recovered in 1.86 s, 96.3% acceptance.
- Covariance Adjustment (R6): RMSE 1.056 m, recovered in 0.86 s, 100% acceptance.
- Vision Connected (Clean): RMSE 0.837 m, recovered in 1.86 s, 96.3% acceptance.
- Vision Connected (Moderate Degradation): RMSE 0.839 m, recovered in 1.86 s, 84.9% inliers.

Findings F09 and F11 are resolved.
