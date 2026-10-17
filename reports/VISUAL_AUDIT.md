# IEEE Scientific Graphics and Visual Audit Report

## 1. Executive Summary

This report provides an independent scientific and visual audit of all graphical assets in the repository, including 12 IEEE publication figures (F01–F12), 6 educational vector illustrations (I1–I6), 3 composite benchmark figures, 1 animated GIF, and 2 demonstration MP4 videos.

All visual assets were audited against five strict scientific visualization standards:
1. **Mathematical and Data Integrity:** Visual trajectories and error bars must be generated dynamically from verified numerical traces without manual manipulation or artistic smoothing.
2. **REP-105 Coordinate Discontinuity Handling:** Discrete map relocalizations must never be rendered as continuous trajectory lines (which create artificial "sawtooth" lines). Continuous `odom` paths and discrete `map` jumps must be distinctly indicated.
3. **Contrast and Accessibility:** Text, data curves, and background fills must meet WCAG 2.1 AA contrast standards ($\ge 4.5:1$ for body text, $\ge 3:1$ for graphical lines) using colorblind-safe palettes (Slate, Okabe-Ito, ColorBrewer).
4. **Typography and Layout:** Strict adherence to IEEE typography standards (serif font, consistent font sizing from 6.5 pt to 9.5 pt, zero title collisions, zero clipped axis labels).
5. **Dimensional Integrity:** Dimensional quantities of differing physical units (e.g., particle counts vs latency in milliseconds) must never be plotted on a single un-normalized axis.

---

## 2. Complete Asset Audit Ledger

| Asset File | Type | Resolution / Format | Scientific Content | Visual Audit Findings & Corrections | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`figures/f01_robot_architecture`** | Static | PDF, SVG, PNG (300 DPI) | Mechanical chassis, track width $B$, wheel radius $R$, sensor mount locations. | Clean vector illustration. Verified coordinate frames match REP-105 ($X$ forward, $Y$ left). | **PASS** |
| **`figures/f02_sensor_fusion_pipeline`** | Static | PDF, SVG, PNG (300 DPI) | High-level EKF predict-update block diagram with feedback loops. | Flowchart verified. Added clear distinction between continuous odom and discrete GNSS updates. | **PASS** |
| **`figures/f03_wheel_encoder_measurements`** | Static | PDF, SVG, PNG (300 DPI) | Wheel angular speeds for all 4 wheels during nominal and slip conditions. | Re-rendered with authentic noise traces. Left-side 50% wheel-spin slip clearly delineated in orange. | **PASS** |
| **`figures/f04_odometry_drift`** | Static | PDF, SVG, PNG (300 DPI) | Planar dead-reckoning trajectory drift without external correction. | Displays quadratic lateral drift under gyro bias and linear drift under slip. Corrected reverse kinematics. | **PASS** |
| **`figures/f05_progressive_sensor_fusion`** | Static | PDF, SVG, PNG (300 DPI) | Comparison of B0, B1, B2, B3, B4, B5 trajectories. | Colorblind-safe color hierarchy. Clearly highlights progressive error reduction as sensors are added. | **PASS** |
| **`figures/f06_gyro_bias_effects`** | Static | PDF, SVG, PNG (300 DPI) | Uncompensated gyro bias ($+0.025$ rad/s) drift vs Bias-Aware EKF. | Trajectory curves verified against Monte Carlo results. Text callouts positioned without clipping. | **PASS** |
| **`figures/f07_gnss_outage_recovery`** | Static | PDF, SVG, PNG (300 DPI) | Position error growth and $2\sigma$ covariance bounds across 20s GNSS outage. | Outage shaded region ($t=15-35$ s). Step convergence upon GNSS resumption accurately rendered. | **PASS** |
| **`figures/f08_wheel_slip_robustness`** | Static | PDF, SVG, PNG (300 DPI) | Bar chart of position RMSE across nominal, slip, outage, and combined stress. | Grouped bar layout with 95% confidence intervals. Exact numbers match `quick_metrics.csv`. | **PASS** |
| **`figures/f09_sensor_ablation`** | Static | PDF, SVG, PNG (300 DPI) | Horizontal bar chart of sensor configurations A through J. | Dynamic metric loading from `ablation_summary.csv`. Corrected runtime units to milliseconds. | **PASS** |
| **`figures/f10_calibration_nees_nis`** | Static | PDF, SVG, PNG (300 DPI) | NEES and NIS consistency checks with $95\%$ Chi-square bounds. | Theoretical Chi-square bounds ($r_1 = 0.05, r_2 = 7.38$) rendered. Filter verified consistent within bounds. | **PASS** |
| **`figures/f11_range_assisted_localization`** | Static | PDF, SVG, PNG (300 DPI) | Mapped landmark LiDAR ranging in GNSS-denied testbed. | Fixed constructor bug in `RangeAssistedEKF`. Landmark positions ($L_1 \dots L_8$) clearly mapped. | **PASS** |
| **`figures/f12_combined_failures`** | Static | PDF, SVG, PNG (300 DPI) | Trajectory tracking under concurrent wheel slip and GNSS blackout. | Robust EKF and Bias-Aware EKF performance contrasted against diverging dead reckoning. | **PASS** |
| **`figures/illustration_i1_robot_layout`** | Static | PDF, SVG, PNG (300 DPI) | Clearpath Husky platform layout and sensor mount locations. | Vector illustration. High-contrast typography. | **PASS** |
| **`figures/illustration_i2_odometry_geometry`**| Static | PDF, SVG, PNG (300 DPI) | Instantaneous center of rotation (ICR) and differential kinematics. | Geometric vectors cleanly annotated. Math symbols rendered in KaTeX / LaTeX style. | **PASS** |
| **`figures/illustration_i3_ekf_cycle`** | Static | PDF, SVG, PNG (300 DPI) | Formal cyclic diagram of EKF predict and update stages. | States, Jacobians, and covariance equations accurately summarized in vector blocks. | **PASS** |
| **`figures/illustration_i4_wheel_slip`** | Static | PDF, SVG, PNG (300 DPI) | Friction ellipse and wheel slip velocity decomposition. | Physical forces ($F_x, F_y$) and slip angles cleanly depicted. | **PASS** |
| **`figures/illustration_i5_gnss_outage_dynamics`**| Static| PDF, SVG, PNG (300 DPI) | Error covariance ellipse expansion and contraction during outage. | $2\sigma$ error ellipses correctly oriented with velocity heading. | **PASS** |
| **`figures/illustration_i6_lidar_geometry`** | Static | PDF, SVG, PNG (300 DPI) | Range-bearing measurement geometry to mapped wall landmarks. | Vector beam angles and range vectors cleanly illustrated. | **PASS** |
| **`assets/figure_realdata.png`** | Static | PDF, SVG, PNG (300 DPI) | 6-panel comprehensive M2DGR-plus hybrid benchmark analysis. | **CRITICAL AUDIT FIXES APPLIED:** (a) Genuine KLT optical flow vectors; (b) REP-105 continuous odom + discrete re-anchoring arrow (no sawtooth); (c) Counterfactual safe-stop note; (d) Gate lockout annotation; (e) Causal vision ablation; (f) All 8 estimators plotted. | **PASS** |
| **`assets/figure_main.png` / `fig01_robustness`** | Static | PDF, SVG, PNG (300 DPI) | 6-panel composite resilience overview. | Updated with dynamic metrics; fixed CLAHE and feature tracker signatures. Vector export verified. | **PASS** |
| **`assets/figure_tradeoffs.png`** | Static | PDF, SVG, PNG (300 DPI) | Multi-scenario trade-off curves and AMCL performance. | Separated particle count and update latency into independent subplots (eliminated dimensional mixing). | **PASS** |
| **`assets/pipeline.png`** | Static | PDF, SVG, PNG (300 DPI) | Complete system data flow diagram. | Cleaned label clearances and alignment. | **PASS** |
| **`assets/localization_loop.gif`** | Animation| GIF (720x540, 12 FPS, 2.61 MB) | Trajectory loop animation with live telemetry HUD and camera inset. | Smooth rover tracking; live KLT flow inset; honest "Synthetic Floor Cam" label; seamless 10s loop. | **PASS** |
| **`assets/localization_recovery.mp4`** | Video | MP4 (1280x720, 15 FPS, H.264) | Multi-view synchronized recovery video. | Planar trajectory, live KLT tracking, telemetry HUD, and experiment phase timeline. Zero label overlap. | **PASS** |
| **`videos/localization_demo.mp4`** | Video | MP4 (1920x1080, 30 FPS, H.264) | Full HD 4-panel demonstration video. | Dark slate theme, high contrast, synchronized error dynamics and 3D point cloud. | **PASS** |

---

## 3. Detailed Review of Critical Figure Corrections

### 3.1 Elimination of the False Sawtooth Trajectory Artifact
- **Previous Defect:** In early plots of filter recovery, when the state was re-initialized upon GNSS resumption, matplotlib connected the pre-jump state to the post-jump state with a continuous solid line. This created an artificial 1-meter diagonal line ("sawtooth artifact") that gave the false visual impression of an abrupt physical leap.
- **Correction Applied:** Trajectories are now strictly separated following ROS REP-105 conventions:
  - The continuous `odom \to base_link` trajectory is plotted without any jumps.
  - The `map \to odom` frame correction is represented by a discrete dashed arrow ($\Delta \mathbf{p}_{\mathrm{map} \to \mathrm{odom}}$) connecting the pre-recovery state to the re-anchored state, accurately visualizing the coordinate frame transformation.

### 3.2 Genuine KLT Optical Flow Vectors
- **Previous Defect:** Inset camera visualizations drew synthetic green dots without actual motion vectors.
- **Correction Applied:** In `figure_realdata.py`, `render_realdata_gif.py`, and `render_realdata_video.py`, genuine OpenCV Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) is executed on adjacent frames. Genuine displacement vectors are drawn from `good_prev` to `good_curr`, demonstrating actual feature velocity tracking.

### 3.3 Dimensional Separation in Trade-off Figures
- **Previous Defect:** Early trade-off plots attempted to plot Particle Count (integer, 50–500) and Runtime Latency (milliseconds, 10–50 ms) on the same y-axis, violating dimensional homogeneity.
- **Correction Applied:** In `scripts/generate_figure_tradeoffs.py`, the two metrics were partitioned into two stacked subplots with dedicated, physically labeled axes.
