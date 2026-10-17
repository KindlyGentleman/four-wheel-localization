# Baseline Audit and Reality Rebase Report

**Date:** 2026-10-17  
**Auditor:** Senior Robotics Research Scientist & Independent Scientific Auditor  
**Repository Working Directory:** `d:\03_Proyek\Kalman_Filter`  
**Current Branch:** `remediation-adversarial-audit`  
**Baseline Commit SHA:** `047430c946311095ba4bf212e625b95eed7230fd`  
**Working Tree Status:** Clean  
**Runtime Environment:** Windows 11 AMD64, Python 3.13.7, CPU-only (NumPy 2.2.6, SciPy 1.15.2, OpenCV 4.12.0)  

---

## 1. Test Suite Baseline Execution

Executed command:
```bash
python -m pytest -q
```
**Result:** 83 passed in 26.52s across 15 test files:
- `tests/test_audit_verification.py` (7 tests)
- `tests/test_core.py` (7 tests)
- `tests/test_dependencies_smoke.py` (2 tests)
- `tests/test_gnss_recovery.py` (10 tests)
- `tests/test_p0_foundation.py` (9 tests)
- `tests/test_p1_sensors_fusion.py` (7 tests)
- `tests/test_p2_range_sensors.py` (5 tests)
- `tests/test_realdata.py` (5 tests)
- `tests/test_refinement.py` (6 tests)
- `tests/test_ros2_mock.py` (4 tests)
- `tests/test_safe_stop.py` (6 tests)
- `tests/test_sensor_registry.py` (6 tests)
- `tests/test_stereo.py` (9 tests)

---

## 2. Forensic Investigation of Newly Identified Defects

While the previous audit resolved structural packaging defects (F01), fabricated ICP fallbacks (F02/F03), and figure-plotting inconsistencies (F04-F06), deep mathematical and algorithmic inspection confirms 10 critical remaining defects:

### Defect 1: Unmeasured `false_recoveries` Metric (Priority P0)
- **Location:** `src/robot_localization/estimation/gnss_recovery.py` line 117 and `realdata.py` lines 1033, 1159.
- **Defect:** `self.false_recoveries` is initialized to 0 in `RecoverableRobustEKF4D.__init__`, but is never incremented anywhere in the codebase.
- **Impact:** `realdata.py` extracts `est.false_recoveries` and writes `false_recoveries: 0` to CSV and summary tables. Reporting an unmeasured safety metric as zero is a serious scientific integrity violation.

### Defect 2: Adversarial Persistent GNSS Coordinate Bias (Priority P1)
- **Location:** `src/robot_localization/estimation/gnss_recovery.py` `_check_buffer_consistency()`.
- **Defect:** The vector displacement check evaluates $\Delta \mathbf{p}_{\mathrm{gnss}} - \Delta \mathbf{p}_{\mathrm{dr}}$. For a persistent constant bias $\mathbf{z}_k = \mathbf{p}_k + \mathbf{b}$, $\Delta \mathbf{p}_{\mathrm{gnss}} \approx \Delta \mathbf{p}_{\mathrm{dr}}$.
- **Impact:** The relative consistency buffer accepts persistent multipath offsets and re-anchors the robot to an arbitrary false global position.

### Defect 3: Sequential Bayesian Particle Weight Erasure (Priority P1)
- **Location:** `src/robot_localization/estimation/particle_filter.py` `compute_scan_weights()`.
- **Defect:** Lines 310-318 overwrite `self.weights = weights_unnorm / total_w`.
- **Impact:** In sequential Monte Carlo filtering without resampling, weights must accumulate: $w_k^{(i)} \propto w_{k-1}^{(i)} p(z_k \mid x_k^{(i)})$. Overwriting weights erases all prior posterior information whenever resampling is skipped.

### Defect 4: AMCL Recovery Likelihood Distortion via Relative Normalization (Priority P1)
- **Location:** `src/robot_localization/estimation/particle_filter.py` lines 312, 440-445.
- **Defect:** `avg_likelihood` is computed from `weights_unnorm = np.exp(log_weights - max_log)`.
- **Impact:** Because $\max(\text{weights\_unnorm}) \equiv 1.0$, `avg_likelihood` is always normalized relative to the scan's best particle. Under severe kidnapping or sensor failure where all particles have poor absolute likelihood, `w_fast` and `w_slow` track identical relative values, preventing random particle recovery injection.

### Defect 5: Full-Pose NEES Neglects Position-Heading Covariance (Priority P1/P2)
- **Location:** `src/robot_localization/evaluation/metrics.py` line 85.
- **Defect:** `nees_pose = nees_pos + (dtheta ** 2 / var_th)`.
- **Impact:** Ignores off-diagonal cross-covariances between position and orientation ($\mathbf{P}_{x\theta}, \mathbf{P}_{y\theta}$). Full-state NEES must be $\mathbf{e}_k^T \mathbf{P}_k^{-1} \mathbf{e}_k$.

### Defect 6: Stereo Reprojection Matrix Sign and Ignored Mounting RPY (Priority P2)
- **Location:** `src/robot_localization/sensors/stereo.py` lines 69, 86-101.
- **Defect:** `q_matrix` has $Q_{3,2} = -1.0 / B$, which produces negative depth $Z = -f B / d$ for positive disparities. Furthermore, `mounting_rpy_rad` is declared on `StereoCameraConfig` but ignored by `transform_optical_to_base` and `transform_base_to_optical`.

### Defect 7: Incomplete ROS REP-105 Transform Continuity (Priority P1)
- **Location:** `src/robot_localization/estimation/gnss_recovery.py` lines 242-250.
- **Defect:** `self.map_to_odom` is only computed during large recovery jumps. During nominal GNSS updates, `self.x` updates via Kalman gain while `odom_pose` and `map_to_odom` remain unchanged.
- **Impact:** Violates the REP-105 invariant $T_{\mathrm{map}}^{\mathrm{base}} = T_{\mathrm{map}}^{\mathrm{odom}} \cdot T_{\mathrm{odom}}^{\mathrm{base}}$ during nominal tracking.

### Defect 8: Safety Supervisor Premature Standstill Declaration (Priority P1)
- **Location:** `src/robot_localization/control/fail_safe.py` lines 150-155.
- **Defect:** Transition to `STANDSTILL_HOLD` checks `if at_standstill or abs(self.current_v) < 1e-3:`.
- **Impact:** When the commanded braking ramp reaches zero (`self.current_v < 1e-3`), the supervisor declares `STANDSTILL_HOLD` even if measured velocity `v_measured` is still non-zero (e.g. vehicle sliding or decelerating late).

### Defect 9: Overstated Visual-Fusion Methodology (Priority P1)
- **Location:** `realdata.py` lines 882-890.
- **Defect:** Vision-assisted localization in `realdata.py` is implemented as an empirical wheel-speed dampener `v_eff = (1 - alpha) * v_wheel + alpha * vo_speed`, not a Kalman measurement update with an innovation gate.
- **Impact:** Calling this an "innovation gate" in documentation is factually inaccurate.

### Defect 10: GNSS Update Cadence Discrepancy (Priority P2)
- **Location:** `realdata.py` line 919 and `README.md`.
- **Defect:** Main hybrid experiment samples GNSS via `if i % 5 == 0:`. Given 5.02 Hz native data, GNSS cadence is ~1.00 Hz, while `README.md` claimed 2.5 Hz.

---

## 3. Prior Issue Status Matrix

| Initial ID | Description | Current Status in `047430c` | Needs Further Remediation? |
|:---|:---|:---|:---|
| F01 | `rosbags` packaging dependency | Verified fixed | No |
| F02 | ICP fallback fabricated metrics | Verified fixed | No |
| F03 | Open3D optional decoupling | Verified fixed | No |
| F04 | Inlier ratio percentage axis scaling | Verified fixed | No |
| F05 | Moderate degradation benchmark execution | Verified fixed | No |
| F06 | Hardcoded CSV fallback elimination | Verified fixed | No |
| F07 | Covariance overconfidence diagnosis | Documented limitation | Yes (metrics & filter models) |
| F08 | Cadence documentation vs code | Partially fixed | Yes (GNSS cadence 1 Hz vs 2.5 Hz) |
| F09 | Window size default vs README | Verified fixed | No |
| F10 | AMCL duplicate resampling | Partially fixed | Yes (Bayesian weight accumulation & likelihood) |
| F11 | Adversarial GNSS multipath tests | Partially fixed | Yes (persistent bias failure mode) |
| F12 | Counterfactual safe stop | Partially fixed | Yes (supervisor standstill semantics) |
| F13 | 2D camera motion assumption | Documented limitation | No |
| F14 | Documentation harmony | Partially fixed | Yes (align with new findings) |
| F15 | Dataset attribution | Verified fixed | No |
| F16 | Duration and coverage metrics | Verified fixed | No |
| F17 | SE(2) composition in recovery | Partially fixed | Yes (nominal updates violate invariant) |
| F18 | Regression test coverage | Verified fixed (83 tests) | Yes (add tests for new findings) |

---

## 4. Execution Plan & Priorities

We proceed through the phases in strict dependency order:
1. **Phase 1 (P0):** Fix false-recovery metric integrity in `gnss_recovery.py` and `realdata.py`.
2. **Phase 2 (P1):** Adversarial GNSS recovery validation (persistent bias detection / explicit unobservability).
3. **Phase 3 (P1):** Repair sequential Bayesian particle weight accumulation in `particle_filter.py`.
4. **Phase 4 (P1):** Repair AMCL recovery likelihood and KLD sampling.
5. **Phase 5 (P1/P2):** Fix full-pose NEES Mahalanobis computation in `metrics.py`.
6. **Phase 6 (P1/P2):** Repair stereo reprojection matrix sign and mounting RPY transforms in `stereo.py`.
7. **Phase 7 (P1):** Enforce ROS REP-105 composition invariant across all epochs.
8. **Phase 8 (P1):** Enforce measured-standstill semantics in `fail_safe.py`.
9. **Phase 9 (P1):** Reconcile visual-odometry coupling terminology and experimental claims.
10. **Phase 10 (P1/P2):** Fix sensor cadence manifests and documentation.
11. **Phases 11-17:** Re-benchmark, re-generate figures, expand regression tests, and perform red-team re-audit.
