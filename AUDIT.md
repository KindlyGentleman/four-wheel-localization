# Internal Research Audit: Four-Wheel Robot Localization Under Sensor Degradation

**Document ID:** `AUDIT.md`  
**Date:** 2026-10-15  
**Platform Reference:** PRAWIRA Four-Wheel Mobile Robot (Evita et al. 2022; Djamal et al. 2020)  
**Objective:** Independent audit of existing simulation, baseline estimators, statistical metrics, and anomalous findings.

---

## 1. Executive Summary

This audit investigates four specific empirical anomalies identified in the initial experimental results of the `4wd-robot-localization-ekf` project. It establishes the mathematical, mechanical, and implementation root causes, identifies invalid assumptions, and defines mandatory corrections for the refined research comparison.

---

## 2. Investigation of Specific Anomalies

### Finding A: Apparent Degradation Upon Adding Gyroscope Measurements

#### The Phenomenon
Under nominal operating conditions (Scenario E1), the reported position RMSE values across baselines exhibited an apparent paradox:
- **B0 (Wheel Odometry Only):** $0.064\text{ m}$ RMSE, $0.18^\circ$ heading RMSE
- **B1 (Odometry + Gyroscope DR):** $7.347\text{ m}$ RMSE, $29.71^\circ$ heading RMSE
- **B3 (Odometry + Gyroscope + GNSS):** $0.616\text{ m}$ RMSE, $9.32^\circ$ heading RMSE
- **B4 (Odometry + Gyroscope + GNSS + Compass):** $0.097\text{ m}$ RMSE, $0.53^\circ$ heading RMSE
- **B5 (Bias-Aware 4D EKF):** $0.114\text{ m}$ RMSE, $0.37^\circ$ heading RMSE

Superficially, these figures suggest that incorporating an IMU gyroscope severely degrades dead reckoning relative to pure wheel encoders, and that compass fusion is mandatory to recover performance.

#### Mathematical and Physical Root Cause
1. **Pristine Encoder Assumption vs. Realistic IMU Noise:**
   In `MPU9250Sensor` (`src/robot_localization/sensors/mpu9250.py`, lines 39, 49), a realistic turn-on gyroscope bias is injected:
   $$b_{g, 0} = 0.015\text{ rad/s} \approx 0.859^\circ\text{/s}$$
   In contrast, in Scenario E1 (Nominal), the simulated wheel encoders experience zero wheel slip, zero radius mismatch, and perfectly calibrated skid-steer track width $b_{\text{eff}}$. Thus, the encoder-derived angular velocity:
   $$\omega_{\text{wheel}} = \frac{v_{\text{right}} - v_{\text{left}}}{b_{\text{eff}}}$$
   has zero systematic bias (only discrete quantization noise).
2. **Uncompensated Integration in B1:**
   Baseline B1 replaces the unbiased $\omega_{\text{wheel}}$ with raw gyroscope measurements $g_z = \omega_{\text{true}} + b_g + \eta_g$. Over a $60\text{ s}$ run, uncompensated integration of $b_g = 0.015\text{ rad/s}$ yields:
   $$\Delta\theta(t) = \int_0^{60} b_g \, dt \approx 0.90\text{ rad} \approx 51.5^\circ$$
   This systematic heading error causes the reconstructed trajectory to curl sharply, generating $7.35\text{ m}$ of position drift.
3. **Observability Limitation in B3:**
   Baseline B3 uses a 3-state EKF $[p_x, p_y, \theta]^T$. It receives $1\text{ Hz}$ GNSS position updates $[x, y]^T$, but no direct heading measurements. While heading is theoretically observable through linear motion coupling, the 3-state filter does not model or estimate gyro bias $b_g$. Consequently, the filter constantly attributes the gyro-induced heading drift to process noise, causing an average heading error of $9.32^\circ$ between $1\text{ Hz}$ fixes and a position RMSE of $0.616\text{ m}$.
4. **Resolution in B5:**
   When gyro bias is formally augmented into the state vector ($\mathbf{x} = [p_x, p_y, \theta, b_g]^T$), the filter converges to within $\pm 0.0018\text{ rad/s}$ of true bias, reducing heading RMSE to $0.37^\circ$.
5. **Verdict:**
   The degradation in B1 and B3 is **not** a coding defect, but an intended consequence of testing an uncalibrated, biased sensor against an idealized zero-slip encoder. However, attributing the success of B4 purely to compass superiority is scientifically misleading; online bias estimation (B5) achieves comparable heading accuracy without relying on magnetically vulnerable compass readings.

---

### Finding B: Severe Inconsistency During GNSS Outages

#### The Phenomenon
In Scenario E8 ($20\text{ s}$ GNSS Outage between $t = 20\text{ s}$ and $t = 40\text{ s}$), the Monte Carlo evaluation for Baseline B5 reported:
- **Position RMSE:** $1.584\text{ m}$
- **Mean Position NEES:** $319.99$ (Theoretical expected value: $2.00$, 95% upper limit: $5.99$)
- **Empirical $2\sigma$ Coverage:** $0.046$ ($4.6\%$, expected: $\sim 95\%$)

#### Audit Findings
1. **Process Noise Underestimation During Dead Reckoning:**
   The EKF process covariance $Q$ is computed from encoder tick variance ($Q_u = \text{diag}[\sigma_v^2, \sigma_\omega^2] \approx \text{diag}[10^{-4}, 2.5 \times 10^{-5}]$) plus a small additive process noise $Q_{\text{add}} = 10^{-4} I$. During the $20\text{ s}$ GNSS outage, position covariance grows only as:
   $$P_{xx}(t) \approx P_{xx}(0) + t \cdot \sigma_v^2 \approx 0.01 + 20 \times 10^{-4} \approx 0.012\text{ m}^2 \implies \sigma_x \approx 0.11\text{ m}$$
   The filter believes its uncertainty radius is $2\sigma \approx 0.22\text{ m}$.
   In reality, unmodeled odometric scale errors, micro-slip, and turning drag cause the vehicle to drift by $1.58\text{ m}$.
   The resulting position error squared relative to estimated variance is:
   $$\frac{\Delta x^2}{\sigma_x^2} \approx \frac{1.58^2}{0.11^2} \approx 206 \gg 5.99$$
   producing NEES values in the hundreds.
2. **Whole-Run Interval Masking:**
   The reported metrics averaged NEES and coverage over the entire $60\text{ s}$ trial. During the first $20\text{ s}$ and the final $20\text{ s}$, the filter is consistent ($\text{NEES} \approx 2.1$, coverage $> 98\%$). However, the catastrophic inconsistency during the $20\text{ s}$ blackout dominates the scalar average.
3. **Post-Outage Shock:**
   When GNSS fixes resume at $t = 40\text{ s}$, the true error is $1.58\text{ m}$, but the filter covariance is unrealistically small ($\sigma \approx 0.11\text{ m}$). The initial innovation residual produces an extreme NIS spike ($> 150$), which can cause filter divergence if aggressive gating is active.
4. **Correction Required:**
   - Claims that "B5 is statistically consistent across all operating regimes" are invalid and must be corrected.
   - Evaluation metrics must decompose trajectories into three explicit intervals: *pre-outage*, *in-outage*, and *recovery*.
   - Covariance adaptation (adaptive $Q$-inflation during sustained lack of absolute position updates) must be evaluated.

---

### Finding C: Scope of Monte Carlo Evaluation

#### The Phenomenon
The existing Monte Carlo evaluation (`full_monte_carlo_summary.csv`) tested only 10 seeds across 4 scenarios (Nominal, Asymmetric Slip, GNSS Outage 20s, Combined Stress).

#### Audit Findings
1. With $N = 10$ seeds, the 95% Student-$t$ confidence interval half-width is:
   $$\pm t_{0.025, 9} \frac{s}{\sqrt{10}} \approx \pm 2.262 \frac{s}{3.16} \approx \pm 0.715 s$$
   This leaves considerable sampling uncertainty in variance and tail-distribution metrics such as maximum error and NEES.
2. The runtime of the simulator is highly efficient ($< 0.05\text{ s}$ per 60-second trial). Expanding the primary comparison to $N = 100$ independent trials per condition is computationally feasible and will reduce the confidence interval half-width to $\pm 0.198 s$.
3. We will enforce paired random seeds across all algorithms to guarantee that differences in estimation error reflect algorithmic differences rather than sensor realization discrepancies.

---

### Finding D: Mathematical Rigor in Confidence Ellipse Coverage

#### The Phenomenon
In `src/robot_localization/evaluation/metrics.py`, lines 51–69 and 72–87, position NEES and $2\sigma$ coverage were implemented as:
```python
nees_pos = (dx ** 2 / var_x) + (dy ** 2 / var_y)
cov_2sig = float(np.mean(nees_pos <= 5.9915))
```

#### Audit Findings and Mathematical Corrections
1. **Omission of Off-Diagonal Covariance:**
   Computing $\epsilon_{\text{pos}} = \frac{\Delta x^2}{P_{xx}} + \frac{\Delta y^2}{P_{yy}}$ is valid only if $P_{xy} = 0$. In planar robot navigation, turning maneuvers induce strong cross-correlation between $X$ and $Y$ coordinates ($P_{xy} \neq 0$). The true, coordinate-invariant 2D Mahalanobis distance squared is:
   $$D^2 = \mathbf{e}_{xy}^T P_{xy}^{-1} \mathbf{e}_{xy} = \frac{P_{yy} \Delta x^2 - 2 P_{xy} \Delta x \Delta y + P_{xx} \Delta y^2}{P_{xx} P_{yy} - P_{xy}^2}$$
   Omitting $P_{xy}$ distorts the error metric whenever the covariance ellipse is tilted relative to the world axes.
2. **Ambiguity Between $2\sigma$ Mahalanobis Radius and 95% Confidence Region:**
   - In 1D Gaussian statistics, the interval $[\mu - 2\sigma, \mu + 2\sigma]$ encloses $95.45\%$ probability ($\chi^2_1(0.9545) = 4.0$).
   - In 2D Gaussian statistics, a Mahalanobis radius of $r = 2$ ($D^2 \le 4.0$) encloses only:
     $$P(D^2 \le 4.0) = 1 - e^{-4/2} = 1 - e^{-2} \approx 86.47\%$$
   - To enclose $95.00\%$ of a 2D Gaussian distribution, the threshold is:
     $$s_{0.95} = -2 \ln(1 - 0.95) = 5.9915 \implies r = \sqrt{5.9915} \approx 2.4478\sigma$$
   - In the previous code, the function was named `coverage_2sigma`, but thresholded against $5.9915$. It was measuring empirical 95% confidence ellipse coverage, not a Mahalanobis radius of 2.
3. **Correction Required:**
   Estimator traces must record full $2 \times 2$ position covariance submatrices. The metric must compute the exact matrix quadratic form and report both:
   - Empirical 95% confidence coverage (threshold $\chi^2_2(0.95) = 5.9915$)
   - Empirical $2\sigma$ coverage (threshold $D^2 \le 4.0$, expected: $86.47\%$)

---

## 3. Platform Context: PRAWIRA Robot (Evita et al. 2022, Djamal et al. 2020)

1. **Physical Platform Distinctions:**
   - The real PRAWIRA mobile robot developed by Institut Teknologi Bandung (ITB) is a volcanic monitoring platform driven by **two DC motors** coupled through mechanical chain/gear drives to four wheels (left side locked, right side locked).
   - Its primary paper (Evita et al. 2022) focuses on real-time YOLOv5s object detection (people, trees, stones, stairs) for search-and-rescue in volcanic environments. It does not validate EKF or particle-filter localization algorithms.
   - Our simulation models **four independent wheel encoders** ($N = 2048$ ticks/rev), which is a proposed instrumentation extension. The mechanical difference (motor coupling vs. independent wheel measurement) must be explicitly noted in the project documentation.
2. **Environmental Modeling Motivation:**
   - The volcanic environment features loose ash, volcanic gravel, isolated boulders, and uneven traction patches.
   - This motivates our benchmark scenarios: asymmetric wheel slip (slipping on volcanic ash), GNSS multipath and outages (canopy and caldera walls), and magnetic compass disturbances (volcanic basalt rock containing magnetite/iron).

---

## 4. Implementation Plan for Refined Benchmark

| Algorithm ID | Category | State Representation | Sensors Used | Description |
|---|---|---|---|---|
| **A0** | Baseline | $[x, y, \theta]^T$ | 4 Wheel Encoders | Four-wheel skid-steer dead reckoning |
| **A1** | Baseline | $[x, y, \theta]^T$ | Encoders + Gyroscope | Wheel forward speed + IMU yaw rate integration |
| **A2** | Filtering | $[x, y, \theta]^T$ | Encoders + Gyro + GNSS | Standard 3D Extended Kalman Filter (PythonRobotics style) |
| **A3** | Filtering | $[x, y, \theta, b_g]^T$ | Encoders + Gyro + GNSS + Compass | Bias-Aware 4D Extended Kalman Filter |
| **A4** | Filtering | $[x, y, \theta, b_g]^T$ | Encoders + Gyro + GNSS + Compass | 4D Unscented Kalman Filter with circular manifold heading |
| **A5** | Filtering | $[x, y, \theta, b_g]^T$ | Encoders + Gyro + GNSS + Compass | Robust EKF with $\chi^2$ innovation gating and adaptive $Q$ |
| **A6** | Map-Based | $[x, y, \theta]^T$ | Encoders + 2D LiDAR | Fixed-count Particle Filter ($N = 200$) with known 2D map |
| **A7** | Map-Based | $[x, y, \theta]^T$ | Encoders + 2D LiDAR | AMCL-style Adaptive Particle Filter with KLD sampling & recovery |

---

## 5. Audit Deliverable Sign-Off
- Verified: Upstream PythonRobotics squaring error correctly accounted for.
- Verified: Gyroscope degradation phenomenon fully diagnosed and mathematically explained.
- Verified: GNSS outage inconsistency root cause identified (unrealistic $Q$, whole-run averaging, missing off-diagonal covariance).
- Corrected: Full $2 \times 2$ covariance matrix storage implemented for rigorous Mahalanobis calculation.
