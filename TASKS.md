# Engineering Remediation & Verification Task List

## Active Workspace & Branch
- **Working Directory:** `d:\03_Proyek\Kalman_Filter`
- **Branch:** `reaudit-figure-redesign`
- **Baseline Commit SHA:** `fda5d09dce97ae438a75b968680155d56dd49dc4`
- **Audit Date:** 2026-10-18

---

## Phases and Progress

- [x] **Phase 0 - Baseline Inspection & Audit Inventory Correction**
  - Confirmed Git HEAD SHA: `fda5d09dce97ae438a75b968680155d56dd49dc4`.
  - Discrepancy resolved: Test suite consists of 91 tests across 13 test files (`test_gnss_recovery.py` has 13 tests, not 12; total test files is 13, not 15).
  - Authored `reports/CURRENT_BASELINE_AUDIT.md`.
  - Created `reports/REMEDIATION_TRACKER.md`.

- [x] **Phase 1 - Safety Supervisor: Continuous Standstill Verification**
  - Inspected `src/robot_localization/control/fail_safe.py`.
  - Implemented continuous velocity re-verification in `STANDSTILL_HOLD` and `VERIFYING_RECOVERY`.
  - Added velocity freshness and validity checks. Revoked hold when motion is detected, resetting recovery history.
  - Implemented 4 adversarial unit tests in `tests/test_safe_stop.py`. All 11 tests pass.

- [x] **Phase 2 - Particle Filter: Missing or Invalid Scan Handling**
  - Inspected `src/robot_localization/estimation/particle_filter.py`.
  - Ensured prediction updates `estimated_pose` and `covariance` even when no valid LiDAR beams are present.
  - Added regression test `test_particle_filter_predict_updates_estimate_without_scans` in `tests/test_audit_verification.py`.

- [x] **Phase 3 - AMCL: Kidnapped-Robot Recovery and KLD Correctness**
  - Corrected AMCL recovery injection: added `p_random > 0.05` trigger to bypass starvation when $N_{\text{eff}} \ge N/2$.
  - Corrected KLD binning spatial discretization with `math.floor` across Cartesian origin.
  - Computed full 3-DOF sample covariance matrix ($\mathbf{P}_{x\theta}, \mathbf{P}_{y\theta}$) with wrapped heading residuals.
  - Implemented unit tests in `tests/test_audit_verification.py`.

- [x] **Phase 4 - GNSS Recovery Integrity and Metric Semantics**
  - Audited drift reachability envelope and recovery thresholds.
  - Harmonized threshold semantics in `metrics.py` and `realdata.py`.
  - Enforced timestamp boundary checks to prevent extrapolation clamping.
  - Logged evaluated and invalid recovery events explicitly.

- [x] **Phase 5 - Covariance Consistency and Statistical Validity**
  - Re-ran Monte Carlo benchmark across all 6 scenarios in `scripts/run_quick.py`.
  - Verified and documented undercoverage under severe wheel slip ($10.6\%$ to $29.8\%$) as an unvarnished physical reality.
  - Preserved authentic baseline behavior without artificial covariance inflation.

- [x] **Phase 6 - Visual Odometry & Terminology Audit**
  - Audited OpenCV Lucas-Kanade optical flow tracking in `realdata.py`.
  - Disclosed that visual tracking is processed as decoupled frame-to-frame odometry and visual condition monitoring rather than a fused Kalman update.
  - Added caption disclosures for optical flow vector scaling (3.5x).

- [x] **Phase 7 - Comprehensive Figure Readability Audit**
  - Conducted 12-criterion readability audit in `reports/FIGURE_READABILITY_AUDIT.md`.
  - Scored `figure_main`, `figure_realdata`, and `figure_tradeoffs` across contrast, typography, marker collision, and scale.

- [x] **Phase 8 - Redesign `figure_main`**
  - Redesigned `scripts/generate_resilience_figure.py`.
  - Improved rover schematic track layout and spacing.
  - Clarified discrete 1 Hz GNSS fix markers in sensor timeline.

- [x] **Phase 9 - Redesign `figure_realdata`**
  - Redesigned `scripts/generate_figure_realdata.py`.
  - Replaced hardcoded safe-stop intervals with dynamic supervisor state trace (`sup_states_R4_recoverable_temporal`).
  - Removed synthetic fallback re-anchoring marker.
  - Increased KLT flow vector visibility with high-contrast arrows.
  - Decoupled Panel (e) into stacked subplots (Inlier ratio vs Disconnected/Connected RMSE).

- [x] **Phase 10 - Redesign `figure_tradeoffs`**
  - Redesigned `scripts/generate_figure_tradeoffs.py`.
  - Synchronized benchmark latencies directly from `map_benchmark_summary.csv`.
  - Handled asymmetric 95% confidence intervals (`yerr=[low_errs, high_errs]`).
  - Split J0-J4 recovery comparison into distinct position and heading RMSE subpanels.

- [x] **Phase 11 - Export and Format Parity Verification**
  - Implemented automated visual quality and integrity validation script `scripts/verify_figures.py`.
  - Guaranteed format parity (.png, .pdf, .svg) across `assets/`, `figures/`, and `four-wheel-localization/assets/`.
  - Integrated into pytest in `tests/test_audit_verification.py`.

- [x] **Phase 12 - Recheck Benchmarks and Update Numbers**
  - Re-executed `realdata.py`, `scripts/run_map_benchmarks.py`, and `scripts/run_quick.py`.
  - Re-exported all figure assets.

- [x] **Phase 13 - Documentation and Scientific Claim Repair**
  - Authored `reports/SAFETY_AND_RECOVERY_AUDIT.md` and `reports/PARTICLE_FILTER_VALIDATION.md`.
  - Updated `reports/REMEDIATION_TRACKER.md` and `reports/FINAL_VERIFICATION.md`.

- [x] **Phase 14 - Independent Red-Team Verification**
  - Executed 20-point adversarial challenge matrix.

- [x] **Phase 15 - Final Acceptance & Publication Readiness Verdict**
  - Delivered final acceptance matrix and publication readiness verdict.
