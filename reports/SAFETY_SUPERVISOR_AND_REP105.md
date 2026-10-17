# Safety Supervisor, Counterfactual Safe-Stop, and ROS REP-105 Verification

## 1. Executive Summary

This report documents the verification and remediation of the fail-safe supervisor and coordinate frame architecture in the `four-wheel-localization` framework:
1. **Finding F12**: Verification that safe-stop commands are counterfactual during fixed dataset replay, ensuring transparent scientific documentation so that open-loop trajectory replay is not confused with closed-loop physical actuation.
2. **Finding F17**: Remediation of ROS REP-105 transform composition from naive translational addition to full analytical $\mathrm{SE}(2)$ composition and inversion, accompanied by a multi-hop $\mathrm{SE}(3)$ TF tree lookup mechanism in `MockTFTree`.

Both issues have been remediated, verified mathematically, and proven empirically through dedicated unit and integration tests.

---

## 2. Audit of Finding F12: Counterfactual Safe-Stop Architecture

### 2.1 Problem Formulation
In offline robot evaluation using pre-recorded datasets (such as M2DGR-plus bags), sensor measurements (wheel odometry, IMU, cameras, GNSS) are fixed historical sequences. If a safety supervisor issues an emergency braking command ($v_{\mathrm{cmd}} = 0$), the physical robot cannot alter its historical motion retroactively.

If an academic publication or software tool claims "autonomous safe-stop execution" without declaring that the trajectory replay is open-loop, it constitutes a misleading claim.

### 2.2 Dual Operating Modes
`SafeStopSupervisor` in `src/robot_localization/control/fail_safe.py` explicitly supports two distinct operational modes:
1. **Closed-Loop Actuation (Simulation & Physical Deployment):**
   The supervisor directly governs commanded linear velocity $v_{\mathrm{cmd}}$ and angular velocity $\omega_{\mathrm{cmd}}$. When GNSS outage exceeds $t_{\mathrm{max}} = 2.0\text{ s}$ or position uncertainty breaches $\sigma_{\mathrm{max}} = 1.2\text{ m}$, the supervisor transitions through:
   $$\text{NOMINAL} \longrightarrow \text{BRAKING} \longrightarrow \text{STANDSTILL\_HOLD} \longrightarrow \text{VERIFYING\_RECOVERY} \longrightarrow \text{RESUMING}$$
   Braking enforces bounded deceleration $a_{\mathrm{brake}} \le 0.6\text{ m/s}^2$ until confirmed standstill ($|v| < 0.015\text{ m/s}$ and $|\omega| < 0.02\text{ rad/s}$).
2. **Open-Loop Counterfactual Annotation (Dataset Replay):**
   When replaying pre-recorded bags, the supervisor operates in observation mode, computing `SafeStopStatus` and `counterfactual_annotation` at every epoch without altering historical measurements.

### 2.3 Documentation and Figure Transparency
The open-loop counterfactual nature of dataset replay is transparently documented across the repository:
- **`README.md` (Line 119):** Explicitly states: "In the hybrid benchmark, the reference trajectory continues unbraked because it replays fixed kinematic recordings."
- **`realdata.py` (Manifest):** `"counterfactual_safe_stop": "Safe-stop supervisor evaluated counterfactually on open-loop trace; physical braking verified in closed-loop simulation"`.
- **`scripts/generate_figure_realdata.py`:** Panel (c) labels the timeline as `"Safe-Stop*"` with explicit caption note: `"*Counterfactual Safe-Stop: reference trajectory continues unbraked"`.
- **`scripts/render_realdata_gif.py` and `scripts/render_realdata_video.py`:** Telemetry overlays display `"Fail-Safe*: ..."` with explicit counterfactual asterisk.

---

## 3. Audit of Finding F17: Analytical SE(2) and REP-105 Coordinate Composition

### 3.1 Defect in Prior Implementation
In ROS REP-105, the standard coordinate frame hierarchy is:
$$\text{map} \longrightarrow \text{odom} \longrightarrow \text{base\_link} \longrightarrow \text{sensors}$$
- `odom -> base_link`: Continuous, smooth local odometry trajectory. Never jumps.
- `map -> odom`: Global drift correction transform. Absorbs discrete relocalization jumps.
- `map -> base_link`: Global estimated robot pose.

The mathematical relation is rigid-body transform composition:
$$T_{\mathrm{map}}^{\mathrm{base}} = T_{\mathrm{map}}^{\mathrm{odom}} \cdot T_{\mathrm{odom}}^{\mathrm{base}}$$

In the prior code (`src/robot_localization/estimation/gnss_recovery.py`), relocalization jumps were recorded via naive translation addition:
```python
# DEFECTIVE
self.map_to_odom[:2] += jump
```
This naive vector addition ignored the orientation of the odom frame. Whenever heading was non-zero or had drifted, $T_{\mathrm{map}}^{\mathrm{odom}} \cdot T_{\mathrm{odom}}^{\mathrm{base}}$ diverged from $T_{\mathrm{map}}^{\mathrm{base}}$.

### 3.2 Remediated SE(2) Rigid-Body Formulation
In planar robotics, rigid-body transformations belong to the Special Euclidean Group $\mathrm{SE}(2)$. Each transform is represented as $\mathbf{T} = [x, y, \theta]^T$.

1. **Transform Composition ($\mathbf{T}_{12} = \mathbf{T}_1 \cdot \mathbf{T}_2$):**
   $$\begin{bmatrix} x_{12} \\ y_{12} \end{bmatrix} = \begin{bmatrix} x_1 \\ y_1 \end{bmatrix} + \begin{bmatrix} \cos\theta_1 & -\sin\theta_1 \\ \sin\theta_1 & \cos\theta_1 \end{bmatrix} \begin{bmatrix} x_2 \\ y_2 \end{bmatrix}$$
   $$\theta_{12} = \operatorname{wrap}(\theta_1 + \theta_2)$$

2. **Analytical Inverse ($\mathbf{T}^{-1}$):**
   $$\begin{bmatrix} x_{\mathrm{inv}} \\ y_{\mathrm{inv}} \end{bmatrix} = -\begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$
   $$\theta_{\mathrm{inv}} = \operatorname{wrap}(-\theta)$$

3. **REP-105 Transform Inversion ($T_{\mathrm{map}}^{\mathrm{odom}} = T_{\mathrm{map}}^{\mathrm{base}} \cdot (T_{\mathrm{odom}}^{\mathrm{base}})^{-1}$):**
   Implemented in `robot_localization.models.kinematics.compute_map_to_odom_se2(map_pose, odom_pose)`.
   When evaluated in `RecoverableRobustEKF4D`:
   ```python
   from robot_localization.models.kinematics import compute_map_to_odom_se2
   self.map_to_odom = compute_map_to_odom_se2(self.x[:3], self.odom_pose)
   ```
   This guarantees that $T_{\mathrm{map}}^{\mathrm{base}} \equiv \operatorname{se2\_compose}(T_{\mathrm{map}}^{\mathrm{odom}}, T_{\mathrm{odom}}^{\mathrm{base}})$ holds identically with zero mathematical truncation error.

### 3.3 Multi-Hop SE(3) Transform Lookup in `MockTFTree`
`MockTFTree` in `src/robot_localization/visualization/ros2_mock.py` was upgraded to implement full multi-hop tree traversal:
- Converts transform representations to $4 \times 4$ homogeneous matrices $\mathbf{T} \in \mathrm{SE}(3)$.
- Identifies Lowest Common Ancestor (LCA) between arbitrary source and target frames.
- Composes chained transforms: $\mathbf{T}_{\mathrm{target}}^{\mathrm{source}} = (\mathbf{T}_{\mathrm{LCA}}^{\mathrm{target}})^{-1} \cdot \mathbf{T}_{\mathrm{LCA}}^{\mathrm{source}}$.
- Converts back to translation and Euler roll-pitch-yaw angles.
- Supports sensor frames such as `camera_left_optical` with REP-105 optical rotation ($-\pi/2, 0, -\pi/2$).

---

## 4. Verification Evidence

All unit and integration tests execute and pass:
- `test_prolonged_outage_triggers_braking_and_standstill_hold`: PASSED
- `test_missing_gnss_indefinitely_maintains_standstill`: PASSED
- `test_single_spurious_fix_never_triggers_resume`: PASSED
- `test_verified_reacquisition_and_conservative_resume`: PASSED
- `test_rep105_coordinate_frame_continuity`: PASSED
- `test_mock_tf_lookup_multi_hop_composition`: PASSED
- `test_se2_kinematics_composition_and_rep105`: PASSED

Findings F12 and F17 are resolved.
