# Experimental Evaluation Protocol

## 1. Objectives and Benchmark Design
This benchmark evaluates state estimation accuracy, covariance consistency, and failure modes across six progressive sensor configurations (B0 to B5), plus range-assisted (B6) and adaptive covariance (B7) extensions.

---

## 2. Sensor Fusion Baselines

| Baseline Code | Configuration | Inputs & Prediction | Measurement Updates | State Dimension |
|---|---|---|---|---|
| **B0** | 4-Wheel Odometry | $v_{\text{wheel}}, \omega_{\text{wheel}}$ from 4 encoders | None (Dead Reckoning) | 3 ($p_x, p_y, \theta$) |
| **B1** | Odometry + Gyro | $v_{\text{wheel}}, \omega_{\text{gyro}}$ | None (Prediction Only) | 3 ($p_x, p_y, \theta$) |
| **B2** | Odometry + GNSS | $v_{\text{wheel}}, \omega_{\text{wheel}}$ | 1 Hz GNSS Position ($[x, y]$) | 3 ($p_x, p_y, \theta$) |
| **B3** | Primary Multi-Sensor EKF | $v_{\text{wheel}}, \omega_{\text{gyro}}$ | 1 Hz GNSS Position ($[x, y]$) | 3 ($p_x, p_y, \theta$) |
| **B4** | Primary + Compass | $v_{\text{wheel}}, \omega_{\text{gyro}}$ | 1 Hz GNSS + 10 Hz Compass | 3 ($p_x, p_y, \theta$) |
| **B5** | Bias-Aware EKF | $v_{\text{wheel}}, \omega_{\text{gyro}}$ | 1 Hz GNSS + 10 Hz Compass | 4 ($p_x, p_y, \theta, b_g$) |
| **B6** | Range-Assisted EKF | $v_{\text{wheel}}, \omega_{\text{gyro}}$ | 1 Hz GNSS + 5 Hz LiDAR Landmarks | 3 ($p_x, p_y, \theta$) |
| **B7** | Slip-Aware Adaptive EKF | $v_{\text{wheel}}, \omega_{\text{gyro}}$ (Adaptive $Q_u$) | 1 Hz GNSS + 10 Hz Compass | 3 ($p_x, p_y, \theta$) |

---

## 3. Experimental Scenarios (E1 - E12)

1. **E1 (Nominal):** 60-second figure-eight trajectory ($v=0.8\text{ m/s}$, loop radius $4.0\text{ m}$), nominal surface traction (0% slip), full sensor availability.
2. **E2 (Wheel Radius Calibration Error):** 5% asymmetric radius mismatch between front-left and front-right estimator models ($r_{\text{FL}} = 1.05 r_0$, $r_{\text{FR}} = 0.95 r_0$).
3. **E3 (Isolated Wheel Slip):** Isolated front-left wheel slip ($s_{\text{FL}} = 0.40$) between $t=15\text{s}$ and $t=35\text{s}$.
4. **E4 (Asymmetric Differential Slip):** Left-side uniform slip ($s_{\text{FL}} = s_{\text{RL}} = 0.35$) between $t=15\text{s}$ and $t=35\text{s}$.
5. **E5 (Gyroscope Bias & Drift):** Uncalibrated initial gyro bias $b_g = +2.0^\circ/\text{s}$ with random walk drift ($1\times 10^{-4}\text{ rad/s}/\sqrt{\text{s}}$).
6. **E6 (Magnetic Compass Disturbance):** Temporary magnetic distortion inducing $+35^\circ$ heading error and degrading calibration state from $t=20\text{s}$ to $t=35\text{s}$.
7. **E7 (Stochastic GNSS Dropout):** 50% random packet loss on scheduled 1 Hz GNSS updates.
8. **E8 (Contiguous GNSS Outage):** Complete 20-second GNSS blackout between $t=15\text{s}$ and $t=35\text{s}$ with moderate baseline slip (10%).
9. **E9 (GNSS Multipath Disturbance):** Temporary correlated position bias ($+3.5\text{ m}$ East, $+3.5\text{ m}$ North) between $t=20\text{s}$ and $t=35\text{s}$.
10. **E10 (GNSS-Denied LiDAR Localization):** 20-second GNSS outage in a mapped environment with 8 peripheral landmarks.
11. **E11 (Ultrasonic Known-Wall Assistance):** Ultrasonic distance feedback against mapped planar perimeter walls.
12. **E12 (Combined Severe Stress Test):** Simultaneous left-side wheel slip (30%), 20-second GNSS blackout, and uncalibrated gyro bias ($+1.4^\circ/\text{s}$).

---

## 4. Evaluation Metrics & Statistical Testing

- **2D Position RMSE:**
  $$\text{RMSE}_{\text{pos}} = \sqrt{\frac{1}{N} \sum_{k=1}^N \left[(p_{x, k} - \hat{p}_{x, k})^2 + (p_{y, k} - \hat{p}_{y, k})^2\right]}$$
- **Heading RMSE with Angular Wrapping:**
  $$\text{RMSE}_{\theta} = \sqrt{\frac{1}{N} \sum_{k=1}^N \left[\text{wrap}(\theta_k - \hat{\theta}_k)\right]^2}$$
- **Normalized Estimation Error Squared (NEES):**
  $$\epsilon_k = (x_k - \hat{x}_k)^T P_k^{-1} (x_k - \hat{x}_k)$$
  Expected value: $E[\epsilon_k] = d$, where $d=2$ for 2D position and $d=3$ for full pose.
- **Normalized Innovation Squared (NIS):**
  $$\nu_k = y_k^T S_k^{-1} y_k$$
  Tested against $\chi^2$ confidence bounds at 95% and 99% significance levels.
- **Fairness Guarantee:** All baselines evaluated within a trial run on identical sensor streams and disturbance realizations.
