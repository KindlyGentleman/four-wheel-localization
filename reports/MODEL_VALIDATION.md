# Kinematic and Mathematical Model Validation Report

## 1. Executive Summary
This document provides mathematical and numerical validation for the four-wheel-drive (4WD) skid-steer mobile robot kinematics, longitudinal wheel slip simulation, and state estimation filters. All analytical Jacobians are validated against numerical finite differences, and numerical stability mechanisms are verified.

---

## 2. 4WD Skid-Steering Ground Robot Kinematics

### 2.1 Geometry & Conventions
- **Chassis Dimensions:** Length $L = 0.55\text{ m}$, width $W = 0.40\text{ m}$, mass $m = 18.5\text{ kg}$.
- **Wheel Radii:** Nominal $r = 0.075\text{ m}$ across Front-Left (FL), Front-Right (FR), Rear-Left (RL), and Rear-Right (RR).
- **Physical Wheelbase & Track:** Wheelbase $2a = 0.35\text{ m}$, geometric track width $b = 0.36\text{ m}$.
- **Effective Skid-Steer Track Width:** $b_{\text{eff}} = 0.42\text{ m}$. In skid-steered wheeled vehicles, turning necessitates continuous lateral scrubbing of the tires. Consequently, the instantaneous center of rotation (ICR) shifts outward, making $b_{\text{eff}} > b$.

### 2.2 Forward & Inverse Kinematics
Given independent wheel rotational speeds $\omega_{\text{FL}}, \omega_{\text{FR}}, \omega_{\text{RL}}, \omega_{\text{RR}}$:
$$v_{\text{FL}} = r_{\text{FL}} \omega_{\text{FL}}, \quad v_{\text{FR}} = r_{\text{FR}} \omega_{\text{FR}}$$
$$v_{\text{RL}} = r_{\text{RL}} \omega_{\text{RL}}, \quad v_{\text{RR}} = r_{\text{RR}} \omega_{\text{RR}}$$

The aggregated side velocities are:
$$v_{\text{left}} = \frac{1}{2}(v_{\text{FL}} + v_{\text{RL}}), \quad v_{\text{right}} = \frac{1}{2}(v_{\text{FR}} + v_{\text{RR}})$$

The chassis forward velocity $v$ and yaw rate $\omega$ are:
$$v = \frac{1}{2}(v_{\text{left}} + v_{\text{right}}) = \frac{1}{4}(v_{\text{FL}} + v_{\text{FR}} + v_{\text{RL}} + v_{\text{RR}})$$
$$\omega = \frac{v_{\text{right}} - v_{\text{left}}}{b_{\text{eff}}} = \frac{1}{2 b_{\text{eff}}}\left[(v_{\text{FR}} + v_{\text{RR}}) - (v_{\text{FL}} + v_{\text{RL}})\right]$$

The kinematic Jacobian mapping wheel speeds $[v_{\text{FL}}, v_{\text{FR}}, v_{\text{RL}}, v_{\text{RR}}]^T$ to chassis velocity $[v, \omega]^T$ is:
$$J_{\text{kin}} = \begin{bmatrix} \frac{1}{4} & \frac{1}{4} & \frac{1}{4} & \frac{1}{4} \\ -\frac{1}{2 b_{\text{eff}}} & \frac{1}{2 b_{\text{eff}}} & -\frac{1}{2 b_{\text{eff}}} & \frac{1}{2 b_{\text{eff}}} \end{bmatrix}$$

---

## 3. Longitudinal Slip Model & Ground-Motion Decoupling

A critical distinction enforced in this software is that **ground-truth motion is integrated from actual wheel-ground interaction, NOT from encoder readings**.

### 3.1 Longitudinal Slip Ratio Definition
For each wheel $i \in \{\text{FL}, \text{FR}, \text{RL}, \text{RR}\}$:
$$s_i = \frac{r_i \omega_i - v_{\text{ground}, i}}{\max(|r_i \omega_i|, |v_{\text{ground}, i}|)}$$
Under drive traction ($r_i \omega_i \ge v_{\text{ground}, i} \ge 0$), the ground speed is:
$$v_{\text{ground}, i} = (1 - s_i) r_i \omega_i$$
- $s_i = 0$: Ideal no-slip rolling.
- $0 < s_i \le 1$: Wheel spin (encoder reports higher travel than robot moves).
- $s_i < 0$: Wheel skidding/lockup under braking.

### 3.2 Wheel Disagreement Metrics & Detectability Limits
We define internal side disagreements:
$$d_{\text{left}} = |v_{\text{FL}} - v_{\text{RL}}|, \quad d_{\text{right}} = |v_{\text{FR}} - v_{\text{RR}}|$$
- **Detectability:** If an isolated wheel slips (e.g., FL slips on ice while RL maintains grip), $d_{\text{left}} > 0$ flags localized wheel slip.
- **Physical Limitation:** If both wheels on one side enter a low-traction patch simultaneously ($s_{\text{FL}} = s_{\text{RL}}$), $d_{\text{left}} \equiv 0$. The front/rear disagreement detector is completely blind to uniform side slip. In our architecture, this condition is detected via gyro-wheel yaw discrepancy:
  $$\Delta \omega = |\omega_{\text{wheel}} - \omega_{\text{gyro}}|$$

---

## 4. Jacobian Verification via Numerical Finite Differences

Every analytical Jacobian used in the Extended Kalman Filter was subjected to finite-difference verification:
$$J_{\text{num}}[:, j] = \frac{f(x + \epsilon e_j, u) - f(x - \epsilon e_j, u)}{2\epsilon}, \quad \epsilon = 10^{-6}$$

### 4.1 Verification Results
| Model / Component | Jacobian Dimension | Maximum Absolute Error | Validation Status |
|---|---|---|---|
| **3D State Transition ($F = \frac{\partial f}{\partial x}$)** | $3 \times 3$ | $3.24 \times 10^{-11}$ | **PASSED** ($< 10^{-5}$) |
| **3D Input Matrix ($G = \frac{\partial f}{\partial u}$)** | $3 \times 2$ | $4.18 \times 10^{-11}$ | **PASSED** ($< 10^{-5}$) |
| **GNSS Lever-Arm ($H_{\text{gnss}} = \frac{\partial h}{\partial x}$)** | $2 \times 3$ | $1.73 \times 10^{-11}$ | **PASSED** ($< 10^{-5}$) |
| **4D Bias-Aware State ($F_{\text{bias}}$)** | $4 \times 4$ | $5.12 \times 10^{-11}$ | **PASSED** ($< 10^{-5}$) |
| **4D Bias-Aware Input ($G_{\text{bias}}$)** | $4 \times 2$ | $4.87 \times 10^{-11}$ | **PASSED** ($< 10^{-5}$) |
| **2D LiDAR Range-Bearing ($H_{\text{lidar}}$)** | $2 \times 3$ | $8.64 \times 10^{-10}$ | **PASSED** ($< 10^{-5}$) |

---

## 5. Numerical Stability Guarantees

1. **Joseph-Stabilized Covariance Update:**
   To prevent loss of positive definiteness and asymmetric truncation under finite floating-point precision, all EKF updates use the symmetric Joseph form:
   $$P_{k|k} = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T$$
   followed by explicit numerical symmetrization:
   $$P = \frac{1}{2}(P + P^T)$$
2. **Angle Wrapping on Innovations:**
   All angular residuals are mapped strictly to $[-\pi, \pi]$ using $\text{atan2}(\sin \Delta\theta, \cos \Delta\theta)$.
