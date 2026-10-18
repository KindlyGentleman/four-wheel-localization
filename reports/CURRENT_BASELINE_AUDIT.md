# Current Baseline Audit and Test Inventory Report

**Date:** 2026-10-17  
**Auditor:** Senior Robotics Research Scientist & Software Verification Engineer  
**Repository Working Directory:** `d:\03_Proyek\Kalman_Filter`  
**Git Branch:** `reaudit-figure-redesign`  
**Inspected Commit SHA:** `fda5d09dce97ae438a75b968680155d56dd49dc4`  
**Prior Reference SHA:** `047430c946311095ba4bf212e625b95eed7230fd`  
**Runtime Environment:** Windows 11 AMD64, Python 3.13.7, CPU-only (NumPy 2.2.6, SciPy 1.15.2, OpenCV 4.12.0, pytest 9.1.1)

---

## 1. Git State and Repository Structure Verification

- `git rev-parse HEAD`: `fda5d09dce97ae438a75b968680155d56dd49dc4`
- `git status`: Working tree clean prior to dedicated branch creation.
- Active branch: `reaudit-figure-redesign` branched directly from `main` (`fda5d09`).
- Python executable: `C:\Python313\python.exe` (Python 3.13.7).

---

## 2. Forensic Audit of Test Inventory Discrepancy

In the prior audit report `reports/FINAL_VERIFICATION.md`, the executive summary claimed:
> "91 passed in 44.10s across 15 test files (100% passing)"
However, the itemized test breakdown in Section 3 listed only 13 test files, and its line for `test_gnss_recovery.py` stated:
> `tests/test_gnss_recovery.py ........................................ 12 passed`
Summing those counts produced: $9+7+2+12+9+7+5+5+6+4+7+6+11 = 90$ tests, contradicting the summary header of 91 tests and 15 test files.

### Independent Verification via pytest `--collect-only -q`:
Actual collected tests: **91 tests across 13 test files**:
1. `tests/test_audit_verification.py` (9 tests)
2. `tests/test_core.py` (7 tests)
3. `tests/test_dependencies_smoke.py` (2 tests)
4. `tests/test_gnss_recovery.py` (**13 tests**, including 3 newly added adversarial tests)
5. `tests/test_p0_foundation.py` (9 tests)
6. `tests/test_p1_sensors_fusion.py` (7 tests)
7. `tests/test_p2_range_sensors.py` (5 tests)
8. `tests/test_realdata.py` (5 tests)
9. `tests/test_refinement.py` (6 tests)
10. `tests/test_ros2_mock.py` (4 tests)
11. `tests/test_safe_stop.py` (7 tests)
12. `tests/test_sensor_registry.py` (6 tests)
13. `tests/test_stereo.py` (11 tests)

**Total:** $9 + 7 + 2 + 13 + 9 + 7 + 5 + 5 + 6 + 4 + 7 + 6 + 11 = 91$ tests.

**Forensic Finding:**
- The actual test suite contains 91 tests across 13 test files.
- The claim of "15 test files" in the prior report was a typographical copy-paste artifact from earlier multi-file planning scaffolds.
- The entry for `test_gnss_recovery.py` underreported its test count by 1 (13 tests actually pass, not 12).
- The inventory is now officially reconciled and verified against live pytest collection.

---

## 3. High-Priority Defect Matrix for Re-Audit

| Issue Key | Severity | Component | File | Suspected Root Cause | Planned Remediation |
|:---|:---:|:---|:---|:---|:---|
| **ISSUE-01** | P1 | Safety Supervisor | `src/robot_localization/control/fail_safe.py` | Standstill is checked at transition into `STANDSTILL_HOLD`, but subsequent states (`STANDSTILL_HOLD`, `VERIFYING_RECOVERY`, `RESUMING`) do not continuously verify physical motion absence or sensor freshness. | Add continuous velocity verification; revoke hold or abort resumption if vehicle slides or moves; add velocity freshness timeout. |
| **ISSUE-02** | P1 | Particle Filter | `src/robot_localization/estimation/particle_filter.py` | `predict()` propagates particles but does not recompute `estimated_pose`. If `compute_scan_weights` receives no valid beams, it returns early, leaving `estimated_pose` stale. | Recompute `estimated_pose` and covariance on prediction when observations are missing or empty without fabricating likelihoods. |
| **ISSUE-03** | P1 | AMCL | `src/robot_localization/estimation/particle_filter.py` | Random particle recovery is injected only inside KLD resampling, which requires $N_{\text{eff}} < N/2$. A kidnapped robot with uniformly low scan likelihood has high $N_{\text{eff}}$, causing injection starvation. | Implement recovery-driven sampling trigger; decouple recovery injection from $N_{\text{eff}}$ collapse when absolute likelihood is critically low. |
| **ISSUE-04** | P2 | AMCL KLD Binning | `src/robot_localization/estimation/particle_filter.py` | Coordinate binning uses integer truncation `int(p[0] / bin_xy_m)`. Truncation rounds toward zero, causing negative $[-0.5, 0.0]$ and positive $[0.0, 0.5]$ coordinates to collapse into the same bin. | Use `math.floor` or `np.floor` for spatial grid discretization; verify circular heading binning. |
| **ISSUE-05** | P2 | Particle Covariance | `src/robot_localization/estimation/particle_filter.py` | Particle filter covariance computes position $2\times 2$ block and scalar heading variance, omitting position-heading cross-covariances $\mathbf{P}_{x\theta}, \mathbf{P}_{y\theta}$. | Compute full $3\times 3$ covariance with angle-wrapped heading residuals $\text{wrap}(\theta^{(i)} - \bar{\theta})$. |
| **ISSUE-06** | P1 | GNSS Recovery | `src/robot_localization/estimation/gnss_recovery.py` | Sub-envelope persistent offsets ($0.3\text{ m} - 2.5\text{ m}$) pass relative consistency buffers. Recovery thresholds disagree between benchmark ($0.35\text{ m}$) and offline scoring ($1.0\text{ m}$). | Document kinematic unobservability of sub-envelope bias; harmonize recovery metric definitions; provide full event accounting. |
| **ISSUE-07** | P1 | Figure Integrity | `scripts/generate_figure_realdata.py` | Safe-stop state timeline uses hardcoded interval timestamps (16.0s, 18.0s, 35.9s, 37.9s); trajectory plot injects synthetic fallback relocalization marker at $t_1 + 1.86$s when no event is logged. | Drive timeline entirely from numerical traces; remove synthetic relocalization fallback markers; annotate unobserved events honestly. |
| **ISSUE-08** | P2 | Figure Readability | `scripts/generate_figure_realdata.py` | Synthetic camera frame KLT flow vectors are tiny and obscured by dark text boxes; dual-axis bar comparisons in panel (e) and legend crowding in panel (f) hinder visual communication. | Redesign figure layout, scale flow vectors visibly with disclosure, move captions outside image, rebalance axes. |
| **ISSUE-09** | P2 | Figure Trade-offs | `scripts/generate_figure_tradeoffs.py` | Discrepancy between README latency claims (22.1 ms / 14.8 ms) and regenerated benchmark values (28.4 ms / 18.5 ms); grouped bar overcrowding. | Re-benchmark latencies under controlled conditions; update figures and README synchronously; clarify CI bounds. |
| **ISSUE-10** | P2 | Figure Main | `scripts/generate_resilience_figure.py` | Kinematics panel (a) labels and coordinate references are crowded; sensor availability lines imply continuous updates instead of discrete fixes. | Clean diagram layout, separate discrete fix indicators from availability timelines. |
