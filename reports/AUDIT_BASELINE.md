# Forensic Baseline Inventory and Audit Report

## 1. Executive Identification and Environment

- Target Repository: `four-wheel-localization` (4WD Mobile Robot Multi-Sensor Fusion & Outage Recovery)
- Audit Commit SHA: `73b0622` (branch: `audit-scientific-repair`, checked out from `main`)
- Operating System: Windows 11 (build 10.0.26100), Shell: PowerShell (pwsh)
- Python Runtime: Python 3.13.7 (CPython 64-bit)
- Host Hardware: 13th Gen Intel(R) Core(TM) i7-13620H (10 physical cores, 16 logical threads)
- System Memory: 15.73 GB Total Physical RAM, 1.99 GB Available Physical RAM (strict CPU and memory budgeting active)
- Primary Installed Packages: numpy 2.5.3, scipy 1.16.2, matplotlib 3.10.7, pandas 2.3.3, opencv-python 4.14.0.94, rosbags 0.11.7, open3d 0.20.0, scienceplots 2.1.1, pytest 9.1.1

---

## 2. Repository Architecture and Benchmark Call Graph

The framework provides multi-sensor fusion, fault recovery, and 2D dead-reckoning state estimation for a 4-wheel skid-steer rover.

### Module Call Graph
- `src/robot_localization/models/`: Skid-steer kinematics (`skid_steer.py`), 4WD configuration (`robot_4wd.py`), and planar SE(2) integration (`kinematics.py`).
- `src/robot_localization/sensors/`: Simulated sensor suites: four encoders (`encoder.py`), MPU-9250 IMU (`mpu9250.py`), CMPS12 compass (`cmps12.py`), GNSS receiver (`gnss.py`), 2D landmark LiDAR (`lidar.py`), ultrasonic raycaster (`ultrasonic.py`), stereo camera and ICP (`stereo.py`).
- `src/robot_localization/simulation/`: Multi-rate event scheduler (`scheduler.py`), discrete simulation loop (`robot_simulator.py`), standardized stress scenarios (`scenarios.py`).
- `src/robot_localization/estimation/`:
  - `dead_reckoning.py`: Open-loop 4-wheel kinematic odometry.
  - `ekf.py`: Standard 3D EKF ($[x, y, \theta]$).
  - `bias_aware_ekf.py`: 4D EKF estimating position, heading, and constant gyro bias ($[x, y, \theta, b_g]$).
  - `robust_ekf.py`: 4D EKF with Chi-square innovation gating and wheel-slip covariance scaling.
  - `ukf.py`: Unscented Kalman Filter with circular manifold mean computation.
  - `gnss_recovery.py`: `RecoverableRobustEKF4D` implementing strategies R0 (Strict Gate), R1 (Temporal Consistency), R2 (Health Hysteresis), R3 (Covariance Adjustment), and R4 (Global Reinitialization).
  - `particle_filter.py`: Fixed particle filter and `AMCLAdaptiveParticleFilter2D` with KLD-sampling and dual-rate likelihood tracking.
  - `range_assisted_ekf.py`: Range-only EKF aiding from known landmark anchors.
- `src/robot_localization/control/`: Counterfactual safety supervisor (`fail_safe.py`) monitoring covariance divergence and commanding deceleration.
- `src/robot_localization/evaluation/`: Metric computation (`metrics.py`: RMSE, MAE, NEES, NIS, confidence coverage) and paired Monte Carlo evaluation (`monte_carlo.py`).
- `realdata.py`: Hybrid pose-conditioned benchmark executing 46.2-second SJTU-ViSYS M2DGR-plus rover trajectory (`anomaly.txt`) with synthetic sensor synthesis, KLT optical flow, and recovery configurations R0 to R7.

---

## 3. Forensic Issue Verification Matrix (F01 to F18)

Every suspected finding from the code review has been independently inspected in the current source code:

| Finding ID | Initial Priority | Status | Source Location | Root Cause Mechanism | Scientific / Engineering Impact |
|:---|:---|:---|:---|:---|:---|
| **F01** | P0 | **CONFIRMED** | `realdata.py:34-35`, `requirements.txt`, `pyproject.toml` | `realdata.py` imports `from rosbags.rosbag1 import Reader, Writer` at top level, but `rosbags` is absent from `requirements.txt` and `pyproject.toml`. | Fresh virtual environments fail immediately on `import realdata` or `python reproduce.py --real-data`. |
| **F02** | P0 | **CONFIRMED** | `src/robot_localization/sensors/stereo.py:756-766` | When Open3D is unavailable, `register_point_clouds_icp()` returns hardcoded `success=True`, `fitness=0.80`, `inlier_rmse_m=0.08`, without computing any registration. | Fabricates successful registration and synthetic metrics without performing estimation. |
| **F03** | P0 | **CONFIRMED** | `tests/test_stereo.py:138-162` | `test_point_cloud_icp_registration` asserts `dx_m == -0.08` and `dy_m == -0.03`. When Open3D is missing, the fallback returned `0.0`, failing the test. | Test suite passes only because Open3D happened to be pre-installed in the auditor's local Python environment. |
| **F04** | P1 | **CONFIRMED** | `scripts/generate_figure_realdata.py:297-301` | Inlier ratio in `realdata_summary.csv` is stored as a unit fraction [0, 1] (e.g. 0.9719), but plotted directly on an axis labeled `(%)` with limit [0, 115]. | Renders a bar of height 0.97% instead of 97.2%, making inlier bars visually invisible and misleading. |
| **F05** | P1 | **CONFIRMED** | `scripts/generate_figure_realdata.py:307-310` | For moderate degradation connected vision, the script repeats `float(r_clean_conn["pos_rmse_m"])` because `R4_vision_connected_moderate` was never run or recorded in `realdata_summary.csv`. | Copies clean-vision RMSE (0.837 m) into moderate degradation without running the experiment. |
| **F06** | P1 | **CONFIRMED** | `scripts/generate_figure_realdata.py:298-310` | Script uses inline fallbacks (`else 97.2`, `else 1.094`, `else 0.837`) when CSV records are missing. | Silently outputs plausible-looking figures even when source data is missing or corrupted. |
| **F07** | P1 | **CONFIRMED** | `results/full_monte_carlo_summary.csv:23,31`, `bias_aware_ekf.py:124`, `robust_ekf.py:98` | Process noise covariance during outage is tiny ($q_{pos} = 10^{-4}$ in `BiasAwareEKF4D` and $0.0025$ in `RobustEKF4D`), growing covariance by only 4 to 22 cm over 20s, while vehicle drifts 2.0 m. | Mean position NEES explodes to 41.3 (Combined) and 320.5 (20s Outage); empirical 95% coverage collapses to 4.6% to 9.3%. Filter covariance is grossly overconfident. |
| **F08** | P1 | **CONFIRMED** | `README.md:55`, `realdata.py:340-410` | README claims sensor rates: Encoders (50 Hz), IMU (50 Hz), Camera (10 Hz), GNSS (1 Hz). `data/anomaly.txt` has only 233 points over 46.2s ($dt \approx 0.199$s, ~5 Hz). `realdata.py` loops over GT epochs, so actual effective rates are 5 Hz, 5 Hz, 5 Hz, and 2.5 Hz. | Documentation misleads readers regarding the actual temporal sampling of the hybrid benchmark. |
| **F09** | P1 | **CONFIRMED** | `src/robot_localization/estimation/gnss_recovery.py:73`, `README.md:66` | Code specifies `consistency_window_size: int = 2` as default. `README.md` documents $M = 3$ consecutive fixes required. | Code and documentation contradict each other on the consistency buffer size. |
| **F10** | P1 | **CONFIRMED** | `src/robot_localization/estimation/particle_filter.py:418-436` | `AMCLAdaptiveParticleFilter2D.update_scan_adaptive()` calls `super().update_scan()` which already calls `resample_systematic()`, resetting weights to $1/N$. It then executes `_kld_resample_and_inject()` which samples from the already-flattened weights. | Performs duplicate resampling, depleting particle diversity and distorting KLD-sampling. |
| **F11** | P1 | **CONFIRMED** | `tests/test_gnss_recovery.py:131-152` | Recovery tests only test an isolated 1-epoch spike. Two consecutive correlated multipath outliers bypass `_check_buffer_consistency()` and trigger false recovery. | Insufficient test coverage for correlated false fixes and subtle drift failure modes. |
| **F12** | P1 | **CONFIRMED** | `realdata.py:820-826`, `fail_safe.py` | Safe-stop controller generates $v_{cmd} = 0$, but the robot trajectory is a pre-recorded dataset replay that continues moving regardless of commands. | Safe-stop is open-loop counterfactual; vehicle does not stop in the benchmark. |
| **F13** | P1 | **CONFIRMED** | `realdata.py:554-575` | `generate_synthetic_scene_image()` takes `pose = (x, y, yaw)` but completely ignores `yaw`. The checkerboard and landmarks translate purely in 2D without rotating when the rover turns. | Camera motion synthesis does not represent true camera rotation or calibrated 3D vehicle motion. |
| **F14** | P2 | **CONFIRMED** | `reports/`, `PRE_PUSH_CHECK.md`, `README.md` | Discrepancies exist between report claims, test counts, parameter settings, and gitignore configuration. | Historical audit documents conflict with current repository code. |
| **F15** | P2 | **CONFIRMED** | `data/anomaly.txt`, `data/calibration.txt`, `LICENSE` | Trajectory and calibration files from SJTU-ViSYS M2DGR-plus are stored directly in `data/`, but repository only has a generic MIT license with no explicit attribution to M2DGR authors. | Academic attribution and third-party dataset provenance must be clearly documented. |
| **F16** | P2 | **CONFIRMED** | `scripts/run_quick.py`, `scripts/run_full.py`, `monte_carlo.py` | Statistical aggregation treats correlated time-series steps as independent samples for coverage, while trial-to-trial averaging must use independent random seeds. | Statistical evaluation methodology requires clear documentation of sample independence. |
| **F17** | P2 | **CONFIRMED** | `gnss_recovery.py:338-339, 448-450`, `ros2_mock.py` | Relocalization updates `map_to_odom[:2] += jump` as a simple translation addition, ignoring SE(2) rotation composition. `ros2_mock.py` lacks general transform composition lookup. | Incomplete ROS REP-105 transform composition semantics. |
| **F18** | P2 | **CONFIRMED** | `tests/` | Prior tests verify nominal happy paths and pre-expected numbers, but lack adversarial checks for duplicate resampling, fallback ICP, or yaw rotation in vision. | Test suite gave false confidence of total system correctness. |

---

## 4. Test Baseline

Execution command: `python -m pytest tests/ -v`
- Total tests collected: 71
- Total tests passed: 71
- Total tests failed: 0
- Execution duration: 33.75s
- Status: Pre-existing tests pass in the current environment only because optional dependencies (`rosbags`, `open3d`) happen to be present in the host environment. Clean-environment behavior fails on F01 and F03.

---

## 5. Numerical Baseline

### Hybrid Benchmark (`results/realdata_summary.csv`):
- R0 (Dead Reckoning): 1.391 m Position RMSE, 19.03 deg Heading RMSE, Peak Outage Error: 2.181 m
- R1 (Standard EKF): 1.023 m Position RMSE, 8.18 deg Heading RMSE, Recovery Time: 2.853 s
- R3 (Strict Gated EKF): 1.362 m Position RMSE, 13.76 deg Heading RMSE, Recovery Success: False (Gate Lockout, 0% post-outage fixes accepted)
- R4 (Temporal Consistency): 1.094 m Position RMSE, 11.62 deg Heading RMSE, Recovery Time: 1.857 s (96.3% fix acceptance)
- R6 (Covariance Adjustment): 1.056 m Position RMSE, 10.45 deg Heading RMSE, Recovery Time: 0.859 s (100% fix acceptance)
- R4 Connected Vision (Clean): 0.837 m Position RMSE (reported 23.5% error reduction)
- R4 Connected Vision (Severe): 1.094 m Position RMSE (reported safe fallback)
- R4 Connected Vision (Moderate): Missing from CSV; fabricated in figure as 0.837 m copy of clean vision.

### Monte Carlo Consistency Benchmark (`results/full_monte_carlo_summary.csv`):
- Nominal Traction (Bias-Aware EKF): Pos RMSE: 0.072 m, 95% Coverage: 100.0%, Mean NEES: 0.865
- Nominal Traction (Robust EKF): Pos RMSE: 0.073 m, 95% Coverage: 100.0%, Mean NEES: 0.896
- GNSS Outage 20s (Bias-Aware EKF): Pos RMSE: 1.584 m, 95% Coverage: 4.63%, Mean NEES: 320.54
- GNSS Outage 20s (Robust EKF): Pos RMSE: 2.094 m, 95% Coverage: 4.75%, Mean NEES: 66.33
- Combined Stress (Bias-Aware EKF): Pos RMSE: 0.578 m, 95% Coverage: 9.35%, Mean NEES: 41.30
- Combined Stress (Robust EKF): Pos RMSE: 0.540 m, 95% Coverage: 28.72%, Mean NEES: 15.53

---

## 6. Known Reproducibility Blockers

1. Packaging Blocker: `rosbags` missing from `pyproject.toml` and `requirements.txt`.
2. Hardware/CPU Blocker: `register_point_clouds_icp()` requires Open3D; when absent, returns fake numbers rather than computing actual 2D registration.
3. Visualization Blocker: Plotting script divides percentage by 100 on percentage axis and fabricates moderate degradation result.
4. Estimator Consistency Blocker: Severe underestimation of covariance during blackout and slip.
5. Algorithmic Blocker: AMCL duplicate resampling disrupts KLD particle adaptation.

---

## 7. Staged Remediation Plan

- **Phase 1 (Dependencies and Packaging):** Add `rosbags` to core dependencies, define clear `[project.optional-dependencies]` for open3d, add dependency smoke tests.
- **Phase 2 (Stereo and ICP Correctness):** Replace fake ICP fallback with a genuine, CPU-only 2D SVD/ICP algorithm using NumPy/SciPy; audit camera mounting, projection, and visual odometry.
- **Phase 3 (Figures and Traceability):** Fix percentage scaling in `generate_figure_realdata.py`, run actual moderate-degradation connected-vision experiment, remove hardcoded metric fallbacks.
- **Phase 4 (Covariance and Consistency):** Correct process noise scaling during wheel slip and blackout; calibrate filter uncertainty models; report independent empirical coverage and NEES.
- **Phase 5 (Sensor Fidelity and Provenance):** Clarify 5 Hz hybrid sampling cadence in README and reports; incorporate heading rotation in synthetic scene generation.
- **Phase 6 (GNSS Recovery Robustness):** Harmonize buffer defaults ($M=3$); expand adversarial tests for consecutive multipath outliers and turning blackouts.
- **Phase 7 (Particle Filter and AMCL):** Eliminate duplicate resampling in `AMCLAdaptiveParticleFilter2D`; benchmark fixed vs adaptive KLD filtering fairly.
- **Phase 8 (Safety Supervisor and REP-105):** State open-loop replay limitations explicitly; implement true SE(2) composition for `map -> odom -> base_link`.
- **Phase 9 (Test Architecture and Regression Coverage):** Add regression tests for F01 to F18; implement lightweight GitHub Actions CI configuration.
- **Phase 10 (Documentation and Attribution):** Update README, add M2DGR-plus citation, synchronize all reports.
- **Phase 11 (Full Reproduction):** Run complete clean reproduction and compile final audit deliverables.
