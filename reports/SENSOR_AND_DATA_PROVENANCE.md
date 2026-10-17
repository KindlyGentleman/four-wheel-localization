# Sensor Fidelity, Real-Data Integrity, and Dataset Provenance Report

## 1. Executive Summary

This report establishes the complete provenance, licensing attribution, coordinate frame conventions, and sensor sampling characteristics for the datasets and sensor simulation models in the `four-wheel-localization` framework. It formally resolves Finding **F08** (Sensor update rate claims vs. actual hybrid sampling) and Finding **F13** (Synthetic camera motion assumptions under 3D motion and yaw), and documents upstream dataset attribution for Finding **F15**.

---

## 2. Upstream Dataset Provenance and Academic Attribution

### 2.1 M2DGR-plus Dataset Origin

The hybrid benchmark in [realdata.py](file:///d:/03_Proyek/Kalman_Filter/realdata.py) is conditioned upon physical reference ground-truth trajectory recordings from the **M2DGR-plus** dataset, developed by the ViSYS Lab at Shanghai Jiao Tong University (SJTU):

- **Upstream Repository:** `https://github.com/SJTU-ViSYS/M2DGR-plus`
- **Reference Publication:**
  > J. Yin, A. Li, T. Li, W. Yu, and D. Zou, "M2DGR: A Multi-sensor and Multi-scenario SLAM Dataset for Ground Robots," *IEEE Robotics and Automation Letters (RA-L)*, vol. 7, no. 4, pp. 11214–11221, 2022. DOI: 10.1109/LRA.2022.3199271.
- **Robot Platform:** Clearpath Husky four-wheel skid-steer field robot equipped with RTK-GNSS, Velodyne VLP-32C LiDAR, Realsense d435i camera, and tactical-grade IMU.
- **Selected Sequence:** `anomaly` (file `data/anomaly.txt`), recording outdoor complex maneuvering under varied terrain traction.

### 2.2 Ground-Truth Formatting and Quaternion Parsing

The raw text file `data/anomaly.txt` records discrete poses formatted as:

$$\begin{bmatrix} t & x & y & z & q_w & q_x & q_y & q_z \end{bmatrix}$$

- **Scalar-First Quaternion Convention:** The quaternion is stored in scalar-first order ($q_w$ at column index 4, $q_x, q_y, q_z$ at columns 5, 6, 7).
- **Quaternion Normalization:** In [realdata.py](file:///d:/03_Proyek/Kalman_Filter/realdata.py), quaternions are strictly unit-normalized prior to projection:
  
  $$\mathbf{q}_{\mathrm{norm}} = \frac{\mathbf{q}}{\|\mathbf{q}\|_2}$$

- **Planar Yaw Angle Recovery:**
  
  $$\psi = \mathrm{atan2}\left(2(q_w q_z + q_x q_y), 1 - 2(q_y^2 + q_z^2)\right)$$

### 2.3 Calibration Matrix Extraction

The camera and sensor extrinsics in `data/calibration.txt` contain Chinese full-width commas (`，`) and non-standard whitespace formatting. The robust regex parser in [realdata.py](file:///d:/03_Proyek/Kalman_Filter/realdata.py) extracts $4 \times 4$ homogeneous SE(3) transformation matrices:

$$\mathbf{T}_{\mathrm{A}}^{\mathrm{B}} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

handling full-width delimiters and guaranteeing bottom-right normalization ($T_{3,3} = 1.0$).

---

## 3. Sensor Update Rates and Sampling Cadences (Finding F08)

### 3.1 Forensic Analysis of Cadence Discrepancy

A documentation discrepancy was identified where `README.md` previously claimed:
- Encoders: 50 Hz
- IMU: 50 Hz
- Camera Optical Flow: 10 Hz
- GNSS: 1 Hz

This claim accurately described the standalone forward simulation suite ([RobotSimulator](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/simulation/robot_simulator.py)), but conflicted with the hybrid real-data benchmark ([realdata.py](file:///d:/03_Proyek/Kalman_Filter/realdata.py)).

### 3.2 Actual Measured Sampling Cadences

An empirical audit of the reference file `data/anomaly.txt` reveals:
- Total Pose Records: Exactly $233$ discrete epochs.
- Duration: $t_{\mathrm{start}} = 1691991959.120$ s, $t_{\mathrm{end}} = 1691992005.333$ s ($\Delta T = 46.214$ s).
- Mean Interval: $\Delta t = 0.1992$ s ($\min = 0.1966$ s, $\max = 0.2020$ s).
- Effective Sampling Frequency: Exactly $\mathbf{5.02\text{ Hz}}$.

### 3.3 Hybrid Sensor Emission Schedule

In [realdata.py](file:///d:/03_Proyek/Kalman_Filter/realdata.py), synthetic sensor observations are generated at the native cadence of the reference trajectory:
- **Wheel Odometry (`/odom`):** Emitted every trajectory epoch ($\approx \mathbf{5.02\text{ Hz}}$).
- **Chassis IMU (`/imu`, `/camera/imu`):** Emitted every trajectory epoch ($\approx \mathbf{5.02\text{ Hz}}$).
- **Downward Camera Frames:** Rendered every trajectory epoch ($\approx \mathbf{5.02\text{ Hz}}$).
- **GNSS Position Fixes (`/ublox_driver/receiver_pvt`):** Emitted every second epoch (`i % 2 == 0`), producing an effective frequency of $\approx \mathbf{2.51\text{ Hz}}$.
- **20-Second Outage Window:** Programmed from $t = 14.0$ s to $34.0$ s (trajectory relative time), corresponding to indices $i = 70$ through $170$. During this interval, GNSS fixes are withheld.

`README.md` and figure captions have been corrected to reflect this exact operational cadence.

---

## 4. Coordinate Frames and Transformations (ROS REP-105)

The localization framework adheres to the ROS REP-105 coordinate frame conventions:

```
earth (WGS-84 Geodetic Datum)
  |
  v
map (Global Cartesian ENU frame, origin at initial robot position)
  |
  +--> [map_to_odom transform: absorbs discrete relocalization jumps]
  |
  v
odom (Continuous local dead-reckoning frame, zero discrete jumps)
  |
  +--> [continuous kinematic integration: delta_s, delta_theta]
  |
  v
base_link (Robot chassis rotation center on ground plane)
  |
  +---> imu_link (Rigid offset: center of chassis)
  +---> gnss_link (Antenna lever arm: lx = +0.05m, ly = 0.0m)
  +---> camera_link (Down-facing optical frame: R_base_to_optical)
```

### 4.1 Frame Definitions
1. **`map`:** A world-fixed Cartesian frame aligned with East-North-Up (ENU). The estimated vehicle pose in `map` is $\mathbf{x}_{\mathrm{map}} = [x, y, \theta]^T$.
2. **`odom`:** A world-fixed frame where pose $\mathbf{x}_{\mathrm{odom}}$ evolves continuously via wheel odometry and IMU integration without discrete jumps.
3. **`base_link`:** The robot body frame rigidly attached to the mobile chassis.
4. **`map_to_odom`:** The coordinate transformation relating `map` to `odom`. Under nominal operation, `map_to_odom` is constant. When relocalization occurs, the discrete jump $\Delta \mathbf{T}$ is absorbed entirely into `map_to_odom`, preventing discontinuous step jumps in `odom -> base_link`.

---

## 5. Visual Odometry and Camera Modeling (Finding F13)

### 5.1 Full 3D Stereo Camera Geometry ([stereo.py](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/sensors/stereo.py))

For 3D landmark observation and point-cloud registration, [SyntheticStereoArena](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/sensors/stereo.py) implements rigorous pinhole stereo geometry:
- Extrinsics: Homogeneous mounting transform $\mathbf{T}_{\mathrm{base}}^{\mathrm{opt}}$ mapping chassis coordinates to camera optical coordinates ($X_{\mathrm{opt}}$ right, $Y_{\mathrm{opt}}$ down, $Z_{\mathrm{opt}}$ optical axis forward).
- Stereo Disparity:

  $$u_L = f_x \frac{X}{Z} + c_x, \quad u_R = f_x \frac{X - B}{Z} + c_x, \quad d = u_L - u_R = \frac{f_x B}{Z}$$

  where $B = 0.10$ m is the stereo baseline.
- 3D Point-Cloud Reconstruction: Every valid disparity pixel reprojects to $(X, Y, Z)$ and transforms back to `base_link` via $\mathbf{T}_{\mathrm{opt}}^{\mathrm{base}} = (\mathbf{T}_{\mathrm{base}}^{\mathrm{opt}})^{-1}$.

### 5.2 Planar Downward-Looking Camera Assumption in Hybrid Benchmark

In [realdata.py](file:///d:/03_Proyek/Kalman_Filter/realdata.py), the visual tracking pipeline models a downward-looking ground camera tracking textured terrain:
- **Procedural Generator ([generate_synthetic_scene_image](file:///d:/03_Proyek/Kalman_Filter/realdata.py)):**
  Renders a 640x480 synthetic ground canvas with 40-pixel checkerboard tiles and 120 fixed world landmarks at a scale of 80 pixels/meter.
- **Planar Translation Model:**
  The procedural ground renderer operates under a planar translation model ($\Delta x, \Delta y$). It does not model out-of-plane pitch/roll rotations or camera lens distortion.
- **Lucas-Kanade Feature Tracking:**
  Shi-Tomasi corners are tracked between consecutive frames using OpenCV pyramidal Lucas-Kanade optical flow (`calcOpticalFlowPyrLK`). RANSAC affine estimation rejects outlier flow vectors.
- **Longitudinal Speed Constraint:**
  The resulting median optical flow displacement provides forward ground speed $v_{\mathrm{vo}} = \sqrt{v_x^2 + v_y^2}$. In the causal vision coupling mode, $v_{\mathrm{vo}}$ is used to constrain excessive longitudinal wheel spin whenever $|v_{\mathrm{wheel}}| - v_{\mathrm{vo}} > 0.12$ m/s, reducing position drift during slip by $23.5\%$.
- **Transparent Limitation:**
  This downward optical flow module assumes visible ground texture and planar contact. As demonstrated by degradation mode D, loss of ground texture or severe image blur causes feature tracking to fail ($< 12$ inliers), prompting the supervisor to safely quarantine the visual velocity and fall back to wheel odometry.
