# Pre-Push Audit & Verification Report: `four-wheel-localization`

**Audit Timestamp:** 2026-10-16  
**Auditor:** Antigravity AI Engineering & Verification Assistant  
**Repository Working Directory:** `d:\03_Proyek\Kalman_Filter`  
**Public Export Directory:** `d:\03_Proyek\Kalman_Filter\four-wheel-localization`  

---

## 1. Git Status

- **Current Branch:** `master`
- **Remote Configuration:** None configured (`git remote -v` returned empty). The intended destination is referenced as `https://github.com/KindlyGentleman/four-wheel-localization.git`.
- **Commit History:**
  - `be6886f` (`master`) chore: remove obsolete CITATION.cff
  - `951f913` refactor: audit and prepare reproducible localization benchmarks
  - `184e1b8` feat: complete 4WD robot localization with EKF multi-sensor fusion, tests, figures, and video demonstration
- **Working Tree State:** Clean. All source, test, configuration, result, and asset files have been staged and committed. Uncommitted files are restricted to local development notes and audit records (`reports/`, `AUDIT.md`, `extras/`, `videos/`, and the clean export directory `four-wheel-localization/`), which are protected by `.gitignore`.
- **Bytecode Elimination:** All 36 historical `.pyc` files that were inadvertently tracked in commit `184e1b8` have been untracked and excluded from the git index.

---

## 2. Verification & Reproduction

- **Test Suite Results:**
  - `python -m pytest tests/`
  - **Collected:** 71 items across 12 test files (`test_audit_verification.py`, `test_core.py`, `test_gnss_recovery.py`, `test_p0_foundation.py`, `test_p1_sensors_fusion.py`, `test_p2_range_sensors.py`, `test_realdata.py`, `test_refinement.py`, `test_ros2_mock.py`, `test_safe_stop.py`, `test_sensor_registry.py`, `test_stereo.py`).
  - **Passed:** 71 (100% pass rate).
  - **Failed:** 0.
  - **Skipped:** 0.
  - **Deselected:** 0.
  - **Execution Time:** ~33 seconds.
- **Resolution of Historical 64 vs 71 Test Discrepancy:**
  - Earlier audit logs reported 64 passing tests because `tests/test_core.py` (which contains 7 foundational tests covering angle wrapping, kinematics, EKF cycles, circular manifold UKF, Chi-square gating, AMCL adaptation, and Mahalanobis consistency) was absent from the root test directory.
  - With `test_core.py` synchronized, exactly 71 tests are collected and all 71 pass deterministically.
- **Benchmark Execution & Numerical Consistency:**
  - `python realdata.py`:
    - `R0_dead_reckoning`: RMSE 1.391 m | Peak 2.181 m
    - `R1_standard_ekf`: RMSE 1.023 m | Recovery 2.85 s
    - `R2_bias_aware_ekf`: RMSE 1.062 m | Recovery 1.86 s
    - `R3_strict_gated_ekf`: RMSE 1.362 m | Post-outage acceptance 0.0% (Lockout confirmed)
    - `R4_recoverable_temporal`: RMSE 1.094 m | Recovery 1.86 s | Acceptance 96.3%
    - `R5_recoverable_hysteresis`: RMSE 1.094 m | Recovery 1.86 s
    - `R6_recoverable_cov_adj`: RMSE 1.056 m | Recovery 0.86 s | Acceptance 100%
    - `R7_recoverable_reinit`: RMSE 1.094 m | Recovery 1.86 s (Covariance PSD verified; previous 15.3 m divergence eliminated)
  - **Causal Vision Ablation:**
    - Mode A (Vision Disabled): 1.0939 m RMSE
    - Mode B (Vision Disconnected): 1.0939 m RMSE (Identical to Mode A, zero side-effects)
    - Mode C (Vision Connected - Clean): 0.8370 m RMSE (23.5% reduction during wheel slip)
    - Mode D (Vision Connected - Severe Degradation): 1.0939 m RMSE (Outlier gate safely rejects degraded tracks, falling back to Mode B)
  - `python scripts/run_quick.py`:
    - Nominal traction: Dead reckoning 0.064 m vs Bias-Aware EKF 0.072 m (negative result confirmed)
    - Asymmetric slip (50% left spin): Dead reckoning 11.795 m vs Bias-Aware EKF 0.218 m (54x reduction)
    - Compass disturbance (25 deg offset): Standard EKF 2.292 m vs Robust EKF 0.098 m
    - GNSS multipath (5 m spike): Standard EKF 0.930 m vs Robust EKF 0.092 m
    - 20s GNSS outage: Standard EKF 1.584 m vs Robust EKF 2.094 m (gate lockout)
    - Combined stress: Bias-Aware EKF 0.578 m vs Robust EKF 0.540 m
  - `python scripts/run_map_benchmarks.py`:
    - Fixed Particle Filter ($N=200$): RMSE 0.3421 m, 22.1 ms/step
    - Adaptive AMCL ($N_{\mathrm{min}}=50, N_{\mathrm{max}}=500$): RMSE 0.3421 m, mean 123 particles (38.5% reduction), 14.8 ms/step (33.0% speedup)

---

## 3. Visual Assets Audit

- **Animation GIF (`assets/localization_loop.gif`):**
  - Dimensions: 720 x 540 pixels.
  - Frame count: 120 distinct frames (verified continuous pixel variance, non-static).
  - Frame delay: 80 ms (~12.5 FPS). Total duration: 9.6 seconds.
  - File size: 2.61 MB (well within GitHub web view limits).
  - Visual contents: Dual panel. Left shows rover traversing reference path with continuous odometry frame and discrete re-anchoring offset arrow (REP-105 compliant, no sawtooth lines). Right shows procedural floor texture with active Lucas-Kanade optical flow vectors, telemetry HUD, and experiment timeline.
- **Scientific Figures (`assets/`):**
  - `figure_main.png` / `figure_main.pdf`: 6-panel overview of chassis, slip, bias, outage, stereo, and ablation.
  - `figure_realdata.png` / `figure_realdata.pdf`: 6-panel hybrid benchmark analysis with causal vision ablation and gate lockout failure mode.
  - `figure_tradeoffs.png` / `figure_tradeoffs.pdf`: Multi-scenario comparison and AMCL particle/latency trade-offs (partitioned into separate physical units, eliminating dimensional mixing).
  - `pipeline.svg` / `pipeline.png`: Clean dataflow diagram.
- **Media Exclusions:**
  - Redundant MP4 recordings (`localization_demo.mp4`, `localization_recovery.mp4`, `localization_loop.mp4`) totaling ~24 MB have been excluded from the public export and untracked from Git. The animation GIF and vector figures fully communicate the scientific results without bloating clone size.

---

## 4. Scientific Integrity & Transparency

- **Data Provenance:**
  - The repository does NOT decode raw authentic sensor streams from an M2DGR-plus ROS bag.
  - Ground truth coordinates $(x, y, \theta)$ are authentic 2D trajectory recordings from an outdoor Clearpath Husky rover (`data/anomaly.txt`).
  - Wheel encoders, IMU, GNSS, floor camera, stereo camera, and LiDAR are synthetic observation models conditioned on the reference motion.
  - Injected wheel slip (50% left-side spin) and GNSS blackouts (20-second outage) are controlled synthetic interventions.
  - All documentation clearly identifies the experiment as a **Ground-Truth-Conditioned Hybrid Kinematic Benchmark**.
- **Monte Carlo Statistical Reporting:**
  - Total evaluation count: 480 method-trials.
  - Statistical unit: 6 stress scenarios x 10 independent simulated trajectory seeds x 8 estimator baselines evaluated concurrently per trajectory.
  - Avoids misleading claims of 480 physical rover runs.
- **Negative Results Preserved:**
  - Dead reckoning beats sensor fusion under nominal simulation conditions (0.064 m vs 0.072 m) due to zero-slip, unbiased simulated encoders.
  - Strict innovation gating causes permanent filter lockout (0/27 post-outage fixes accepted).
- **Safe-Stop Controller:**
  - Explicitly documented as evaluated **counterfactually**; the controller correctly outputs $v_{\mathrm{cmd}} = 0$, while the replayed reference trajectory continues along its fixed recording.

---

## 5. Repository Hygiene & Dependencies

- **Public Export Directory Size:** 14.5 MB (down from 46.5 MB, a 69% reduction).
- **Largest Tracked Files:**
  - `assets/localization_loop.gif`: 2.61 MB
  - `results/saved_benchmark_traces.npz`: 2.14 MB
  - `assets/pipeline.png`: 0.65 MB
  - `assets/figure_realdata.png`: 0.56 MB
  - `assets/figure_main.png`: 0.50 MB
- **Excluded Files:**
  - ROS bag recordings (`*.bag`)
  - Video files (`*.mp4`)
  - Python bytecode (`__pycache__/`, `*.pyc`)
  - Test caches (`.pytest_cache/`)
  - Internal audit drafts and working logs (`reports/`, `AUDIT.md`)
- **Dependencies (`requirements.txt` & `pyproject.toml`):**
  - `numpy>=1.24.0`
  - `scipy>=1.10.0`
  - `matplotlib>=3.7.0`
  - `pandas>=2.0.0`
  - `pyyaml>=6.0`
  - `opencv-python>=4.8.0`
  - `pillow>=9.5.0`
  - `pytest>=7.0.0` (dev)
  - `scienceplots>=2.1.0` (dev)
- **Secrets Scan:** 0 tokens, API keys, or private credentials found across the repository.
- **Link & Path Audit:** 0 hardcoded absolute drive paths or `file:///` URLs.
- **Copywriting Hygiene (Anti-Slop):** 0 em dashes (`—`), 0 AI marketing buzzwords, concise direct prose modeled after `KindlyGentleman/fft-diffraction`.

---

## 6. Remaining Blockers

- **P0 Defects:** None.
- **P1 Defects:** None.
- **Operational Requirement (Actionable by User):**
  - The local Git repository currently has no remote URL configured (`git remote -v` is empty).
  - Before pushing, the user must link their remote:
    ```bash
    git remote add origin https://github.com/KindlyGentleman/four-wheel-localization.git
    ```

---

## 7. Audit Verdict

### **READY WITH WARNINGS**

The codebase, mathematical models, experimental results, visual assets, documentation, and tests have met all technical and scientific verification criteria:
1. 71/71 tests pass deterministically.
2. All benchmark figures, tables, and claims match independently reproduced outputs.
3. No secrets or unsupported large binaries are staged.
4. Data provenance and negative results are transparently disclosed.
5. The working tree is cleanly committed.

**Warning:** The Git remote has not been set in the local repository. Once the remote URL is added, the repository is ready for publication.
