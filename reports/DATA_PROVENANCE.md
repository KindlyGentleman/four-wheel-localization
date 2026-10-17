# Data Provenance and Modality Audit Report

## 1. Executive Summary

This report establishes the physical and digital origin of every data stream, sensor signal, and ground-truth reference used across the `four-wheel-localization` repository.

A central finding of this audit is that **the benchmark does not decode authentic multi-sensor messages from an original M2DGR-plus ROS bag**. Instead, the experimental setup operates as a **hybrid pose-conditioned kinematic simulation**:
1. The reference 2D motion trajectory is derived from authentic ground-truth kinematics (`data/anomaly.txt`, an outdoor rover run from the Shanghai Jiao Tong University M2DGR dataset).
2. Sensor observations (wheel encoders, IMU, GNSS, floor camera, stereo camera, and LiDAR) are **synthetically generated from this reference trajectory** using explicit physics and noise models.
3. The previously provided `create_sample_bag()` function generated a ROS bag synthetically from this trajectory rather than deserializing recorded sensor hardware packets.

Per explicit user authorization, this hybrid simulation approach is fully accepted and scientifically appropriate for controlled algorithmic stress-testing, provided its provenance is transparently disclosed.

---

## 2. Modality-by-Modality Provenance Classification

Each sensor channel is classified into one of six standard provenance tiers:
- **Tier 1:** Recorded real hardware measurement.
- **Tier 2:** Original external ground-truth reference.
- **Tier 3:** Simulation-generated synthetic measurement.
- **Tier 4:** Ground-truth-conditioned synthetic measurement.
- **Tier 5:** Recorded measurement with injected corruption.
- **Tier 6:** Unavailable or unverified.

| Modality / Channel | Provenance Tier | Generator Function / Source | Physical Model & Characteristics | Authenticity Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Ground Truth Pose** $(x, y, \theta)$ | **Tier 2: External GT** | `data/anomaly.txt` via `load_ground_truth()` | Planar 2D pose trajectory from M2DGR ground truth (46.2 s duration, 233 samples @ ~5 Hz, interpolated to 10 Hz / 50 Hz). | Authentic reference trajectory recorded on a real Clearpath Husky rover. |
| **Wheel Encoders** (FL, FR, RL, RR) | **Tier 4: GT-Conditioned** | `run_hybrid_experiment()` in `realdata.py` | Synthesized from differential forward velocity $v$ and yaw rate $\omega$ via inverse kinematics, plus zero-mean Gaussian noise ($\sigma = 0.05$ rad/s). | Synthetic observations conditioned on authentic rover kinematics. |
| **IMU Yaw Rate** ($\omega_z$) | **Tier 4: GT-Conditioned** | `run_hybrid_experiment()` in `realdata.py` | Derived from numerical differentiation of reference heading $\Delta \theta / \Delta t$ with added constant bias ($+0.025$ rad/s) and Gaussian noise ($\sigma = 0.01$ rad/s). | Synthetic observations conditioned on authentic rover kinematics. |
| **GNSS Position Fixes** ($x, y$) | **Tier 4: GT-Conditioned** | `run_hybrid_experiment()` in `realdata.py` | Synthesized at 1 Hz from reference positions $(x, y)$ plus lever-arm translation and Gaussian measurement noise ($\sigma = 0.15$ m). Blackout period (14.0 s to 34.0 s) simulated by suppressing message emission. | Synthetic observations conditioned on authentic rover kinematics with injected blackout. |
| **Monocular Floor Camera** | **Tier 4: GT-Conditioned** | `generate_synthetic_scene_image()` in `realdata.py` | Procedurally generated synthetic floor texture (checkerboard grid + randomized circular gravel particles) rendered according to the rover's exact ground-truth pose. | Synthetic procedural imagery; authentic OpenCV KLT feature tracking applied to synthetic frames. |
| **Stereo Camera Pair** (Left / Right) | **Tier 3: Simulation** | `SyntheticStereoArena` in `sensors/stereo.py` | Pin-hole stereo projection ($f = 400$ px, baseline $b = 0.12$ m) with 3D landmark points rendered into left/right images, followed by OpenCV StereoSGBM disparity computation. | Pure simulation-generated stereo pairs. |
| **LiDAR / Landmark Ranges** | **Tier 3: Simulation** | `KnownLandmarkLiDAR` in `sensors/lidar.py` | Simulated 2D range-bearing observations to pre-mapped landmarks with range noise ($\sigma_r = 0.05$ m) and bearing noise ($\sigma_\theta = 0.02$ rad). | Pure simulation-generated range-bearing observations. |
| **Wheel Slip Anomaly** | **Tier 4: Injected Anomaly** | `run_hybrid_experiment()` in `realdata.py` | Injected 50% wheel-spin slip on left wheels ($v_{\mathrm{left}} \leftarrow 1.5 \cdot v$) between $t = 13.0$ s and $t = 18.0$ s. | Injected synthetic kinematic anomaly during authentic motion segment. |
| **ROS Bag** (`m2dgr_anomaly.bag`) | **Tier 4: Synthesized Bag** | `create_sample_bag()` in `realdata.py` | Synthetically packaged ROS 2 / ROS 1 messages constructed from the GT-conditioned synthetic channels. | Structurally valid bag file, but containing synthetic sensor messages. |

---

## 3. Ground-Truth Leakage Isolation Audit

A critical defect identified in early revisions was the candidate calculation:
$$\Delta v_{\mathrm{slip}} = |v_{\mathrm{wheel}} - v_{\mathrm{true}}|$$

Where $v_{\mathrm{true}}$ was directly extracted from ground-truth velocity arrays. This constituted severe ground-truth leakage into the estimator pipeline.

### Verification of Leakage Elimination:
1. **Observable Slip Detection:** All slip detection logic in `src/robot_localization/` and `realdata.py` has been audited. Slip estimation now relies strictly on:
   - Kinematic disagreement between left and right wheel encoder velocity vs IMU yaw rate:
     $$\Delta \omega_{\mathrm{disagree}} = \left| \frac{v_R - v_L}{B} - \omega_{\mathrm{gyro}} \right|$$
   - Normalized EKF innovation squared (NIS) monitoring:
     $$\epsilon_y = \mathbf{y}_k^T \mathbf{S}_k^{-1} \mathbf{y}_k > \chi^2_{\alpha, p}$$
2. **Ground-Truth Isolation Test:** A regression test was implemented in `tests/test_audit_verification.py`. The ground-truth reference trajectory was subjected to an unannounced affine perturbation ($\Delta x = +5.0$ m, $\Delta y = -3.0$ m).
   - **Result:** Filter estimated state trajectories remained bit-for-bit identical, verifying complete mathematical isolation between ground-truth evaluation arrays and estimator update pipelines. Verdict: **PASS**.

---

## 4. Scientific Justification and Reporting Guidelines

Because all sensor observations are synthetic or pose-conditioned:
- The project documentation must **never** claim "authentic multi-sensor replay of real M2DGR-plus raw data".
- The project is accurately described as a **Hybrid Kinematic Benchmark and Controlled Resilience Simulation**.
- The value of this benchmark lies in controlled, repeatable stress testing where the exact ground-truth error dynamics, sensor failure timings, and recovery mechanics can be deterministically audited.
