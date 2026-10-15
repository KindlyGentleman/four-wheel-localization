# Robustness of Four-Wheel Odometry and Extended Kalman Filter Sensor Fusion Under Wheel Slip and GNSS Outages

**Authors:** Robotics & State Estimation Research Team  
**Working Paper:** October 2026  
**Repository:** `4wd-robot-localization-ekf`

---

## Abstract
Dead reckoning for four-wheel-drive (4WD) skid-steered mobile robots suffers from rapid error accumulation due to unmodeled tire scrubbing, longitudinal wheel slip, and wheel radius calibration errors. While multi-sensor fusion using Extended Kalman Filters (EKF) is widely deployed, few open reproductions decouple ground-truth ground kinematics from encoder readings or investigate the failure interactions among asynchronous sensors. In this study, we build a fully reproducible, CPU-based Python simulation of a 4WD planar robot equipped with four independent quantized wheel encoders (50 Hz), a 6-DOF IMU (MPU-9250 style, 50 Hz), a magnetic compass (CMPS12 style, 10 Hz), a GNSS receiver (1 Hz), and an optional range sensor. We reproduce and audit the standard 2D EKF benchmark from PythonRobotics, identifying a significant covariance-versus-standard-deviation noise scaling discrepancy in upstream code. We formulate five progressive fusion baselines and conduct controlled Monte Carlo experiments across nine failure regimes. Results demonstrate that a 5% front-wheel radius mismatch or 35% longitudinal slip causes dead reckoning to diverge by over $13\text{ m}$ to $25\text{ m}$ RMSE ($>60^\circ$ heading error), whereas bias-aware EKF fusion bounds error to $0.17\text{ m}$ RMSE ($0.38^\circ$ heading RMSE). Furthermore, we demonstrate that magnetic compass updates degrade localization by $14\times$ during magnetic interference unless gated, and that online gyroscope bias estimation restores filter consistency from an overconfident NEES of $106.1$ down to $2.19$. All code, data, 18 IEEE-style figures, and a 1080p demonstration video are provided for exact reproduction.

---

## 1. Introduction
Accurate pose estimation is an essential requirement for autonomous ground vehicles (AGVs) operating in unstructured agricultural, industrial, and outdoor environments. Skid-steered four-wheel mobile robots offer mechanical simplicity and high torque by eliminating dedicated steering linkages. However, steering is achieved entirely through differential left/right wheel speeds, forcing tires to scrub and slide laterally against the terrain. Consequently, dead reckoning integrated purely from wheel encoders is notoriously prone to rapid drift.

While sensor fusion combining wheel odometry, Inertial Measurement Units (IMUs), and Global Navigation Satellite Systems (GNSS) is standard practice (e.g. ROS `robot_localization` [1]), published simulation benchmarks frequently exhibit subtle methodological flaws:
1. **Coupled Ground Truth:** Many simulators generate ground truth by integrating commanded unicycle kinematics and then add artificial noise to produce encoder and GNSS readings. Under real-world wheel slip, the chassis ground displacement is strictly smaller or orientationally deviated from wheel-shaft rotation.
2. **Generic Single-Odometry Assumptions:** Many platforms simulate a single aggregated forward speed $v$ and yaw rate $\omega$, omitting independent encoder pulse quantization, tick jitter, and individual wheel disagreements.
3. **Flawed Reference Implementations:** As documented in our audit, popular educational references like PythonRobotics [2] unintentionally multiply covariance matrices directly by standard Gaussian variates, distorting the assumed filter noise ratios by orders of magnitude.

This paper presents a rigorous, reproducible investigation of 4WD skid-steer localization under realistic wheel slip, sensor bias, GNSS blackouts, and magnetic interference.

---

## 2. Four-Wheel Platform & Kinematic Slip Model

### 2.1 Mechanical Configuration
The platform models a 4WD planar chassis of length $L=0.55\text{ m}$, width $W=0.40\text{ m}$, mass $m=18.5\text{ kg}$, nominal wheel radius $r=0.075\text{ m}$, wheelbase $2a=0.35\text{ m}$, geometric track $b=0.36\text{ m}$, and calibrated effective track width $b_{\text{eff}}=0.42\text{ m}$ ($b_{\text{eff}} > b$ to account for tire scrubbing).

Four independent optical encoders provide 2048 pulses per revolution (PPR) directly on each wheel shaft: Front-Left (FL), Front-Right (FR), Rear-Left (RL), and Rear-Right (RR). Discrete cumulative tick counts are integer-quantized with stochastic jitter ($\sigma=0.5\text{ ticks}$) and pulse loss probability ($p=5\times 10^{-4}$).

### 2.2 Decoupled Slip Kinematics
For wheel $i$, longitudinal slip ratio $s_i \in (-\infty, 1]$ is modeled as:
$$v_{\text{ground}, i} = (1 - s_i) r_i \omega_i$$
Chassis ground motion is integrated strictly from ground velocities:
$$v_{\text{ground}} = \frac{1}{4}(v_{\text{ground, FL}} + v_{\text{ground, FR}} + v_{\text{ground, RL}} + v_{\text{ground, RR}})$$
$$\omega_{\text{ground}} = \frac{(v_{\text{ground, FR}} + v_{\text{ground, RR}}) - (v_{\text{ground, FL}} + v_{\text{ground, RL}})}{2 b_{\text{eff}}}$$
whereas wheel encoders measure motor-shaft rotation $\omega_i$. This ensures the true physical trajectory differs substantially from optimistic dead reckoning whenever slip occurs.

---

## 3. Sensor Observation Models

1. **Wheel Encoders (50 Hz):**
   $$v_i = \frac{2\pi r_i}{N \Delta t} \Delta \text{ticks}_i, \quad \text{Var}(v_i) = \left(\frac{2\pi r_i}{N \Delta t}\right)^2 \left(\frac{1}{12} + \sigma_{\text{jitter}}^2\right)$$
   Aggregated chassis inputs $u = [v, \omega]^T = J_{\text{kin}} v_{\text{wheels}}$ propagate full correlated input covariance $Q_u = J_{\text{kin}} \Sigma_{\text{wheels}} J_{\text{kin}}^T$.
2. **MPU-9250 IMU (50 Hz):**
   $$\text{gyro}_z = \omega_{\text{ground}} + b_g + \mathcal{N}(0, \sigma_g^2), \quad \dot{b}_g = w_{bg}$$
   where $\sigma_g = 0.005\text{ rad/s}$ ($0.28^\circ/\text{s}$), $b_g = +2.0^\circ/\text{s}$ initial bias, and $\sigma_{wb} = 10^{-4}\text{ rad/s}/\sqrt{\text{s}}$.
3. **CMPS12 Magnetic Compass (10 Hz):**
   $$\theta_{\text{meas}} = \text{wrap}(\theta_{\text{true}} + \delta_{\text{disturb}} + \mathcal{N}(0, \sigma_c^2))$$
   with $\sigma_c = 1.0^\circ$. Calibration state transitions (0 to 3) reflect magnetic reliability.
4. **GNSS Receiver (1 Hz):**
   $$z_{\text{gnss}} = \begin{bmatrix} p_x \\ p_y \end{bmatrix} + R(\theta) \begin{bmatrix} l_x \\ l_y \end{bmatrix} + \mathcal{N}(0, \Sigma_{\text{gnss}})$$
   incorporating antenna lever-arm offset $[l_x, l_y]^T = [0.05, 0.0]\text{ m}$, random walk drift ($0.005\text{ m/s}$), and contiguous outage windows.

---

## 4. State Estimation Formulations

### 4.1 3-State EKF (B0 - B4)
State: $x = [p_x, p_y, \theta]^T$. State transition uses second-order Runge-Kutta (midpoint) propagation:
$$\theta_{\text{mid}} = \theta_{k-1} + \frac{1}{2}\omega_k \Delta t$$
$$p_{x, k} = p_{x, k-1} + v_k \cos(\theta_{\text{mid}}) \Delta t, \quad p_{y, k} = p_{y, k-1} + v_k \sin(\theta_{\text{mid}}) \Delta t$$
Analytical state Jacobian $F = \frac{\partial f}{\partial x}$ and input Jacobian $G = \frac{\partial f}{\partial u}$ are validated against finite differences to error $< 5 \times 10^{-11}$.

### 4.2 4-State Bias-Aware EKF (B5)
State: $x = [p_x, p_y, \theta, b_g]^T$. Gyroscope bias $b_g$ is co-estimated online:
$$\omega_{\text{eff}} = \omega_{\text{gyro}} - b_g$$
All measurement updates utilize the Joseph-stabilized covariance update:
$$P_{k|k} = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T$$
with mandatory angular innovation wrapping: $\tilde{y}_\theta = \text{atan2}(\sin(z_\theta - \hat{\theta}), \cos(z_\theta - \hat{\theta}))$.

---

## 5. Quantitative Results & Discussion

### 5.1 Nominal Motion vs. Wheel Slip
Under nominal zero-slip conditions (E1), pure wheel odometry (B0) appears adequate ($0.064\text{ m}$ RMSE). However, when 35% asymmetric wheel slip is injected (E4), B0 odometry drifts to **$13.77\text{ m}$ RMSE** ($63.5^\circ$ heading error). Baseline B3 (odometry + gyro + GNSS) limits drift to $0.781\text{ m}$, and B5 (bias-aware EKF) achieves **$0.171\text{ m}$ RMSE** and **$0.38^\circ$ heading RMSE**.

### 5.2 Wheel Radius Calibration Mismatch (E2)
Introducing an unmodeled 5% front-wheel radius asymmetry causes dead reckoning to diverge catastrophically to **$25.74\text{ m}$ RMSE** and $108.8^\circ$ heading error over 60 seconds. In contrast, B4/B5 multi-sensor fusion bounds error to **$0.097\text{ m} / 0.113\text{ m}$ RMSE**, demonstrating extreme robustness to chassis parametric errors.

### 5.3 Gyroscope Bias Estimation (E5)
With an uncompensated gyro bias of $+2.0^\circ/\text{s}$, standard EKF (B3) suffers $1.635\text{ m}$ position RMSE and $22.2^\circ$ heading RMSE with an overconfident NEES of $106.14$. The 4D bias-aware EKF (B5) identifies the bias within $12\text{ s}$, reducing position RMSE to **$0.114\text{ m}$** and heading RMSE to **$0.37^\circ$**, while restoring NEES to a near-ideal **$2.19$** ($98.5\%$ empirical $2\sigma$ coverage).

### 5.4 The Compass Failure Mode (E6)
Introducing local magnetic disturbance ($+35^\circ$ distortion) reveals a critical vulnerability: naive compass fusion (B4) degrades position error by $14\times$ to **$2.568\text{ m}$ RMSE**, whereas B3 (gyro + GNSS only) is immune at **$0.616\text{ m}$**. This verifies that unvalidated compass observations harm state estimation during environmental interference.

### 5.5 GNSS Outages & Blackout Recovery (E8)
During a 20-second complete GNSS blackout with 10% baseline slip, filter covariance expands smoothly from $0.35\text{ m}$ to $2.15\text{ m}$ ($2\sigma$), accurately enclosing the true trajectory error. Upon GNSS fix restoration at $t=35\text{s}$, the Joseph-stabilized update achieves smooth step convergence in $2.1\text{ s}$ without divergence.

---

## 6. Monte Carlo Statistical Summary

Ten paired Monte Carlo trials per scenario yield the following 95% confidence intervals for position RMSE:

| Scenario | B0 (Odometry) [m] | B3 (Primary EKF) [m] | B5 (Bias-Aware EKF) [m] |
|---|---|---|---|
| **Nominal Traction** | $0.064 \pm 0.000$ | $0.663 \pm 0.058$ ($[0.622, 0.705]$) | **$0.072 \pm 0.014$** ($[0.062, 0.083]$) |
| **Asymmetric Slip** | $11.795 \pm 0.000$ | $0.709 \pm 0.065$ ($[0.663, 0.756]$) | **$0.218 \pm 0.033$** ($[0.195, 0.241]$) |
| **20s GNSS Outage** | $1.341 \pm 0.000$ | $2.244 \pm 0.162$ ($[2.129, 2.360]$) | **$1.584 \pm 0.031$** ($[1.562, 1.606]$) |
| **Combined Stress** | $11.925 \pm 0.000$ | $1.587 \pm 0.103$ ($[1.513, 1.661]$) | **$0.578 \pm 0.025$** ($[0.560, 0.596]$) |

---

## 7. Conclusion
This study provides a rigorous, verified reproduction and extension of mobile robot localization. We have shown that four-wheel odometry is physically unreliable under slip and calibration errors, but can be effectively stabilized through consistent multi-sensor fusion. Key takeaways for roboticists include:
1. Online gyro bias co-estimation is essential for heading stability during GNSS denial.
2. Magnetic compass observations require strict innovation gating to avoid catastrophic pull under magnetic distortion.
3. Antenna lever-arm compensation and Joseph-stabilized updates are mandatory for mathematical consistency.

All software, unit tests, figure generation scripts, and high-definition video rendering tools are open-source and deterministically reproducible.

---

## References
- [1] T. Moore and D. Stouch, "A generalized extended Kalman filter implementation for the Robot Operating System," in *Proc. Conf. Intelligent Autonomous Systems (IAS-13)*, 2014, pp. 335–348.
- [2] A. Sakai, D. Ingram, J. Dinius, R. Chawla, A. Raffin, and H. Paarlberg, "PythonRobotics: a Python code collection of robotics algorithms," *arXiv preprint arXiv:1808.10703*, 2018.
- [3] G. Welch and G. Bishop, "An introduction to the Kalman filter," Tech. Rep. TR 95-041, Univ. North Carolina at Chapel Hill, 2006.
- [4] R. C. Smith and P. Cheeseman, "On the representation and estimation of spatial uncertainty," *Int. J. Robot. Res.*, vol. 5, no. 4, pp. 56–68, 1986.
