# Final Re-Audit Baseline Report

**Execution Timestamp:** 2026-10-18T18:50:00+07:00  
**Repository Working Directory:** `d:\03_Proyek\Kalman_Filter`  
**Git Branch:** `reaudit-figure-redesign`  
**Git HEAD SHA:** `58a6008e7ceeaf74d5c7d88584a37e138b1c466b`  
**Python Environment:** Python 3.13.7 (win32)  
**Core Dependencies:**
- NumPy: 2.5.3
- SciPy: 1.16.2
- OpenCV: 5.0.0
- Pytest: 9.1.1
- Pandas: 2.3.3

---

## 1. Inventory of Test Suite and Baseline Execution

- **Collected Test Count:** 100 tests across 13 test files.
- **Passing Test Count:** 100 passed in 28.75s (0 failures, 0 skipped, 0 warnings).
- **Test Inventory Breakdown:**
  1. `tests/test_audit_verification.py`: 14 tests
  2. `tests/test_core.py`: 7 tests
  3. `tests/test_dependencies_smoke.py`: 2 tests
  4. `tests/test_gnss_recovery.py`: 13 tests
  5. `tests/test_p0_foundation.py`: 9 tests
  6. `tests/test_p1_sensors_fusion.py`: 7 tests
  7. `tests/test_p2_range_sensors.py`: 5 tests
  8. `tests/test_realdata.py`: 5 tests
  9. `tests/test_refinement.py`: 6 tests
  10. `tests/test_ros2_mock.py`: 4 tests
  11. `tests/test_safe_stop.py`: 11 tests
  12. `tests/test_sensor_registry.py`: 6 tests
  13. `tests/test_stereo.py`: 11 tests

---

## 2. Benchmark Artifacts Inventory

- **Map-based Particle Filter & AMCL Benchmark (`results/map_benchmark_summary.csv`):**
  - 10 trials across seeds 5000-5009.
  - Fixed PF ($N=200$): Mean Pos RMSE = $0.0231 \pm 0.0084$ m | Mean Update Time = $11.65$ ms.
  - AMCL (Adaptive KLD): Mean Pos RMSE = $0.0251 \pm 0.0016$ m | Mean Update Time = $7.34$ ms | Mean Particles = $103.3$.
- **Real-Data Hybrid Trajectory Benchmark (`results/realdata_summary.csv`):**
  - 14 configurations spanning R0-R7, visual ablations, and connected VO.
  - Trace file: `results/realdata_traces.npz` (contains timestamps, ground truth, estimator poses, supervisor states).
- **Multi-Scenario Monte Carlo Benchmark (`results/quick_metrics.csv`):**
  - 6 evaluation scenarios (E4, E5, E6, E7, E8, E9, E12).
  - Trace file: `results/saved_benchmark_traces.npz`.

---

## 3. Publication Figures Inventory

Format parity verified across all target directories (`assets/`, `figures/`, `four-wheel-localization/assets/`):
- `figure_main`: .png, .pdf, .svg (6-panel sensor resilience overview)
- `figure_realdata`: .png, .pdf, .svg (6-panel hybrid localization and recovery)
- `figure_tradeoffs`: .png, .pdf, .svg (3-panel Monte Carlo and particle filter trade-offs)

---

## 4. Target Findings for Final Adversarial Remediation

1. **F01 (P0):** Nonfinite GNSS coordinates acceptance in safe-stop verification (`fail_safe.py`).
2. **F02 (P0):** Hardcoded covariance-coverage fallbacks in `scripts/generate_resilience_figure.py`.
3. **F03 (P1):** Stale-velocity verification claim without velocity measurement timestamp checking.
4. **F04 (P1):** Chi-square innovation gating claim absent from supervisor standstill validator.
5. **F05 (P1):** Missing LiDAR observation likelihood returning 0.0, conflating missing scans with kidnapping.
6. **F06 (P1):** Particle filter resampling leaving effective sample size and estimate stale.
7. **F07 (P1):** AMCL recovery tests checking particle injection rather than demonstrated relocalization.
8. **F08 (P1):** Capped/placeholder reprojection errors plotted on invalid/failed stereo observations.
9. **F09 (P1):** Trajectory jump thresholding confusing fast motion with GNSS re-anchoring events.
10. **F10 (P1):** PF/AMCL accuracy discrepancy between current code (0.02-0.03m) and older reports (0.347m).
11. **F11 (P2):** README documentation discrepancies regarding benchmark numbers and visual fusion.
12. **F12 (P2):** Figure verification script checking file existence instead of rendering parity.
13. **F13 (P2):** Crowded annotations, legend ambiguity, and font legibility in figures.
14. **F14 (P2):** Mixed historical and current benchmark results in final verification reports.
