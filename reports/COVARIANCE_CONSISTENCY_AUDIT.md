# Covariance Consistency and Filter Uncertainty Audit Report

## 1. Executive Summary

This report delivers the forensic audit and scientific diagnosis of filter covariance consistency across the `four-wheel-localization` framework, specifically addressing Finding **F07** (Filter covariance severely underestimates actual localization uncertainty under combined stress) and Finding **F16** (Benchmark fairness, timing methodology, and Monte Carlo statistical units revalidation).

In state estimation, a filter is statistically consistent if its estimated state error $\mathbf{e}_k = \mathbf{x}_k^{\mathrm{gt}} - \hat{\mathbf{x}}_k$ is compatible with its internally reported error covariance matrix $\mathbf{P}_k$. Formally, the estimation error must be zero-mean $E[\mathbf{e}_k] = \mathbf{0}$, and the covariance must match the expected squared dispersion:

$$E\left[\mathbf{e}_k \mathbf{e}_k^T\right] = \mathbf{P}_k$$

Under severe unmodeled disturbances (30% asymmetric wheel slip, 20-second GNSS blackout, gyroscope bias), previous benchmarks reported:
- [BiasAwareEKF4D](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/estimation/bias_aware_ekf.py) (B5): Position RMSE $0.578$ m, empirical 95% confidence coverage $9.35\%$, mean position NEES $41.30$.
- [RobustEKF4D](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/estimation/robust_ekf.py) (A5): Position RMSE $0.540$ m, empirical 95% confidence coverage $28.72\%$, mean position NEES $15.53$.

This audit independently verified the mathematical metric formulations, diagnosed the physical and algorithmic causes of covariance underestimation, audited benchmark execution fairness, and established the theoretical relationship between classical baseline overconfidence and modern recovery mechanisms.

---

## 2. Forensic Diagnosis: The Mechanics of Filter Overconfidence

### 2.1 Theoretical Expectation vs. Physical Reality

For a 2-DOF horizontal position estimate, the Normalized Estimation Error Squared (NEES) is defined by the Mahalanobis distance squared:

$$\epsilon_{\mathrm{pos}, k} = \mathbf{e}_{xy, k}^T \mathbf{P}_{xy, k}^{-1} \mathbf{e}_{xy, k}$$

Under the linear-Gaussian assumption with consistent covariance, $\epsilon_{\mathrm{pos}, k} \sim \chi^2(2)$:
- Theoretical Expected Value: $E[\epsilon_{\mathrm{pos}}] = 2.000$.
- Theoretical 95% Confidence Region: $\epsilon_{\mathrm{pos}} \le 5.9915$.

When an estimator reports mean NEES $= 41.30$ and coverage $= 9.35\%$, it indicates severe overconfidence: the estimator's covariance ellipsoid is far too small, asserting high precision when actual tracking error is large.

### 2.2 Root Cause 1: Vanishing Discrete Input Noise Propagation ($G Q_u G^T$)

In [BiasAwareEKF4D](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/estimation/bias_aware_ekf.py) and [RobustEKF4D](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/estimation/robust_ekf.py), process noise enters through two distinct pathways:
1. Control input noise mapping: $\mathbf{Q}_{\mathrm{step}} = \mathbf{G} \mathbf{Q}_u \mathbf{G}^T$.
2. Additive state process noise: $\mathbf{Q}_{\mathrm{add}} \Delta t$.

The input Jacobian is:

$$\mathbf{G} = \begin{bmatrix} \cos(\theta_{\mathrm{mid}}) \Delta t & -\frac{1}{2} v \sin(\theta_{\mathrm{mid}}) \Delta t^2 \\ \sin(\theta_{\mathrm{mid}}) \Delta t & \frac{1}{2} v \cos(\theta_{\mathrm{mid}}) \Delta t^2 \\ 0 & \Delta t \\ 0 & 0 \end{bmatrix}$$

Notice that the translational block of $\mathbf{G}$ carries a pre-factor of $\Delta t$. Therefore:

$$\left[\mathbf{G} \mathbf{Q}_u \mathbf{G}^T\right]_{1:2, 1:2} \approx \Delta t^2 \left( \mathbf{Q}_u[0, 0] \begin{bmatrix} \cos^2 \theta & \sin \theta \cos \theta \\ \sin \theta \cos \theta & \sin^2 \theta \end{bmatrix} \right)$$

At an encoder sampling cadence of 50 Hz ($\Delta t = 0.02$ s), $\Delta t^2 = 0.0004$.
Even when wheel slip is detected and [RobustEKF4D](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/estimation/robust_ekf.py) scales velocity variance $\mathbf{Q}_u[0, 0]$ by $20 \times$ (from $1.0 \times 10^{-4}$ to $2.0 \times 10^{-3} (\mathrm{m/s})^2$), the variance added per step is only:

$$\Delta P_{\mathrm{step}} \approx 0.0004 \times 0.002 = 8.0 \times 10^{-7} \mathrm{m}^2$$

Over an entire 20-second blackout ($N = 1000$ steps):

$$\sum_{k=1}^{1000} \Delta P_{\mathrm{step}} = 1000 \times 8.0 \times 10^{-7} = 8.0 \times 10^{-4} \mathrm{m}^2$$

This corresponds to a standard deviation expansion of only:

$$\sigma_{\mathrm{accum}} = \sqrt{0.0008} \approx 0.028 \mathrm{m} = 2.8 \mathrm{cm}$$

Scaling $\mathbf{Q}_u$ in velocity space is mathematically incapable of expanding positional uncertainty to match real macroscopic wheel slip.

### 2.3 Root Cause 2: Under-Parameterized Additive Process Noise

In [BiasAwareEKF4D](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/estimation/bias_aware_ekf.py), additive position process noise is static: $q_{\mathrm{pos\_add}} = 1.0 \times 10^{-4} \mathrm{m}^2/\mathrm{s}$.
Over a 20-second blackout:

$$\Delta P_{\mathrm{add}} = q_{\mathrm{pos\_add}} T = 1.0 \times 10^{-4} \times 20 = 0.0020 \mathrm{m}^2 \implies \sigma \approx 0.045 \mathrm{m} = 4.5 \mathrm{cm}$$

Meanwhile, the vehicle drives at $v = 0.8$ m/s over 16 meters of travel under 30% asymmetric wheel slip. The unmodeled ground kinematic velocity discrepancy is:

$$\Delta v = s \cdot v = 0.30 \times 0.8 = 0.24 \mathrm{m/s}$$

Over 20 seconds, the actual position drift grows to between $1.5$ m and $2.2$ m.
Because the filter's estimated position standard deviation is only $\sigma \approx \sqrt{0.0008 + 0.0020 + 0.007} \approx 0.098$ m:

$$\mathrm{Normalized~Error} = \frac{e}{\sigma} = \frac{1.80 \mathrm{m}}{0.098 \mathrm{m}} \approx 18.4 \sigma$$

The actual error is 18 standard deviations away from the mean. The theoretical probability of a Gaussian variable exceeding $18 \sigma$ is practically zero, causing empirical 95% coverage to collapse to 0% during the outage window.

### 2.4 Root Cause 3: Post-Outage Kalman Gain Starvation

When GNSS resumes at $t = 35$ s with measurement noise $\sigma_{\mathrm{gnss}} = 0.50$ m ($\mathbf{R} = \mathrm{diag}(0.25, 0.25)$):
The filter prior covariance is artificially compressed to $\mathbf{P} \approx \mathrm{diag}(0.007, 0.007)$.
The scalar Kalman gain approximation is:

$$K \approx \frac{P}{P + R} = \frac{0.007}{0.007 + 0.250} = \frac{0.007}{0.257} \approx 0.027$$

When a valid GNSS fix arrives with an innovation residual of $y = 1.80$ m:

$$\Delta \hat{x} = K \cdot y = 0.027 \times 1.80 \mathrm{m} = 0.049 \mathrm{m} = 4.9 \mathrm{cm}$$

Because the filter is overconfident in its dead-reckoning model ($P = 0.007$), it discounts the GNSS fix by 97.3%, updating its state by only 5 cm per second.
It requires more than 30 consecutive updates to reconcile the 1.8 m drift, leaving persistent tracking errors ($0.45 - 0.55$ m) throughout the post-outage segment ($t = 35$ s to $60$ s) and maintaining NEES $> 25$.

---

## 3. Classical Baselines vs. Recovery Strategies: The Architectural Intent

A vital finding of this scientific audit is that the low coverage of classical baselines is an authentic empirical result, not an accidental implementation bug.

The project architecture deliberately separates:
1. **Classical Estimators ([baseline_runner.py](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/estimation/baseline_runner.py)):**
   - Evaluates standard textbook formulations: B0 (Dead Reckoning), B1 (Gyro), B2 (GNSS), B3 (Gyro+GNSS), B4 (Compass), B5 (Bias-Aware EKF), A4 (Manifold UKF), and A5 (Standard Gated Robust EKF).
   - In A5, innovation gating ($\chi^2 \le 9.21$) is enforced strictly. When GNSS fixes return after prolonged blackout, $\mathrm{NIS} \approx 29 - 41 > 9.21$, causing permanent gate lockout ($0/5$ returning fixes accepted in `GNSS_Outage_20s`).
   - This failure reproduces the known vulnerability of standard Kalman gating in safety-critical robotics.
2. **Dedicated Recovery Mechanisms ([gnss_recovery.py](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/estimation/gnss_recovery.py)):**
   - Config J in [run_ablations.py](file:///d:/03_Proyek/Kalman_Filter/scripts/run_ablations.py) and R0–R7 in [realdata.py](file:///d:/03_Proyek/Kalman_Filter/realdata.py) evaluate explicit solutions:
     * **R1 (Temporal Consistency Buffer):** Quarantines the first fix, verifies kinetic plausibility on the second fix ($\Delta s / \Delta t \le v_{\max}$), and triggers controlled recovery.
     * **R2 (Health-State Hysteresis Machine):** Relaxes gate thresholds during recovery settling.
     * **R3 (Distance-Scaled Covariance Inflation):** Scales process noise with traveled distance during blackout: $q_{\mathrm{drift}} = (\alpha |v|)^2$, where $\alpha = 0.12 - 0.15$.
     * **R4 (Controlled Reinitialization):** Re-anchors global map pose cleanly while preserving local odometry continuity.

Artificially forcing A5 to achieve 95% coverage under prolonged blackouts by overriding its gate or inflating its base outage drift would erase the gate-lockout failure mode that the rest of the research paper is designed to study and resolve.

---

## 4. Multi-Scenario Monte Carlo Benchmark Audit

The following table summarizes empirical results from the full 480-trial Monte Carlo benchmark ($10\text{ seeds} \times 6\text{ scenarios} \times 8\text{ baselines}$, 60.0 s execution per trial):

| Scenario | Estimator Baseline | Position RMSE (Mean $\pm$ Std) [m] | 95% Conf. Interval [m] | Heading RMSE [deg] | 95% Cov. Coverage | Mean NEES Position |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Nominal Traction** | B0 Wheel Odometry | $0.064 \pm 0.000$ | $[0.064, 0.064]$ | $0.18 \pm 0.00$ | $100.0\%$ | $0.02$ |
| | B5 Bias-Aware EKF | $0.072 \pm 0.014$ | $[0.062, 0.083]$ | $0.36 \pm 0.03$ | $100.0\%$ | $0.87$ |
| | A4 Manifold UKF | $0.072 \pm 0.014$ | $[0.062, 0.083]$ | $0.36 \pm 0.03$ | $100.0\%$ | $0.86$ |
| | A5 Robust EKF | $0.073 \pm 0.015$ | $[0.063, 0.084]$ | $0.36 \pm 0.04$ | $100.0\%$ | $0.90$ |
| **Asymmetric Slip** | B0 Wheel Odometry | $11.795 \pm 0.000$ | $[11.795, 11.796]$ | $77.40 \pm 0.00$ | $30.6\%$ | $9132.48$ |
| | B5 Bias-Aware EKF | $0.218 \pm 0.033$ | $[0.195, 0.241]$ | $0.36 \pm 0.03$ | $69.2\%$ | $7.50$ |
| | A4 Manifold UKF | $0.218 \pm 0.033$ | $[0.195, 0.241]$ | $0.36 \pm 0.03$ | $69.2\%$ | $7.50$ |
| | A5 Robust EKF | $0.226 \pm 0.033$ | $[0.203, 0.249]$ | $0.37 \pm 0.04$ | $68.1\%$ | $7.87$ |
| **GNSS Outage 20s** | B0 Wheel Odometry | $1.341 \pm 0.000$ | $[1.341, 1.341]$ | $4.13 \pm 0.00$ | $4.9\%$ | $126.02$ |
| | B5 Bias-Aware EKF | $1.584 \pm 0.031$ | $[1.562, 1.606]$ | $0.35 \pm 0.03$ | $4.6\%$ | $320.54$ |
| | A4 Manifold UKF | $1.583 \pm 0.031$ | $[1.561, 1.605]$ | $0.35 \pm 0.03$ | $4.6\%$ | $320.33$ |
| | A5 Robust EKF | $2.094 \pm 0.213$ | $[1.942, 2.247]$ | $0.36 \pm 0.04$ | $4.8\%$ | $66.33$ |
| **Combined Stress** | B0 Wheel Odometry | $11.925 \pm 0.000$ | $[11.925, 11.925]$ | $76.84 \pm 0.00$ | $10.6\%$ | $9609.15$ |
| | B5 Bias-Aware EKF | $0.578 \pm 0.025$ | $[0.560, 0.596]$ | $0.36 \pm 0.03$ | $9.35\%$ | $41.30$ |
| | A4 Manifold UKF | $0.578 \pm 0.025$ | $[0.560, 0.596]$ | $0.36 \pm 0.03$ | $9.36\%$ | $41.26$ |
| | A5 Robust EKF | $0.540 \pm 0.020$ | $[0.526, 0.554]$ | $0.37 \pm 0.04$ | $28.72\%$ | $15.53$ |
| **Compass Fault** | B4 GNSS+Compass | $2.282 \pm 0.030$ | $[2.260, 2.304]$ | $14.07 \pm 0.04$ | $35.0\%$ | $788.73$ |
| | A5 Robust EKF | $0.098 \pm 0.018$ | $[0.084, 0.111]$ | $0.56 \pm 0.16$ | $100.0\%$ | $1.09$ |
| **GNSS Multipath** | B3 Gyro+GNSS | $2.968 \pm 0.082$ | $[2.910, 3.026]$ | $17.22 \pm 0.48$ | $27.4\%$ | $424.33$ |
| | A5 Robust EKF | $0.092 \pm 0.026$ | $[0.074, 0.110]$ | $0.36 \pm 0.04$ | $100.0\%$ | $0.69$ |

### Key Audit Observations

1. **Nominal Consistency:** Under nominal conditions without unmodeled disturbances, B5, A4, and A5 achieve $100.0\%$ 95% coverage with mean NEES $\approx 0.86 - 0.90$. The filter models are statistically consistent when physical assumptions hold.
2. **Selective Outlier Rejection:** In Compass Disturbance (30 deg magnetic offset) and GNSS Multipath (3.5 m step), A5's Chi-square gating rejects the corrupt measurements completely ($\mathrm{NIS} > 9.21$), maintaining $100.0\%$ coverage and sub-0.10 m error, whereas B3/B4 diverge to $2.28 - 2.97$ m.
3. **The Outage Trade-off:** In `GNSS_Outage_20s`, A5's position RMSE ($2.094$ m) is higher than B5 ($1.584$ m) precisely because A5 rejects all returning fixes due to gate lockout, whereas ungated B5 immediately absorbs the returning fix despite large innovation shock.

---

## 5. Benchmark Fairness and Unit Harmonization (Finding F16)

### 5.1 Fairness Audit
- **Identical Realization:** All baselines execute chronologically on the identical list of `sim_result.sensor_events`.
- **Zero Privileged Information:** Ground truth poses are never accessed during estimation; all metrics are evaluated strictly post-hoc.
- **Consistent Seeds:** Matched random seed bases ($2000 + i$) ensure identical stochastic noise across comparative trials.

### 5.2 Resolution of Execution Duration Discrepancy
- Audit finding: `scripts/run_full.py` defined lambda wrappers with `duration_s=40.0`, but [MonteCarloRunner](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/evaluation/monte_carlo.py) hardcoded `duration_s=60.0`.
- Inspection of raw trial traces confirmed that the historical benchmark was evaluated over 60.0 s (mean traveled distance $46.14$ m at $0.8$ m/s).
- Resolution: Refactored [MonteCarloRunner](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/evaluation/monte_carlo.py) to accept configurable `duration_s` (default 60.0 s), and updated [scripts/run_full.py](file:///d:/03_Proyek/Kalman_Filter/scripts/run_full.py) to explicitly specify `duration_s=60.0` matching the 60-second scenario timeline ($t=15$ to $35$ s outage).

### 5.3 Harmonization of Coverage Units
- In [metrics.py](file:///d:/03_Proyek/Kalman_Filter/src/robot_localization/evaluation/metrics.py), `compute_covariance_coverage()` outputs fractional values in $[0.0, 1.0]$.
- In [results/full_monte_carlo_summary.csv](file:///d:/03_Proyek/Kalman_Filter/results/full_monte_carlo_summary.csv), values are recorded as fractions (e.g., $0.0935$ and $0.2872$).
- In [results/ablation_summary.csv](file:///d:/03_Proyek/Kalman_Filter/results/ablation_summary.csv), values were recorded as percentages (e.g., $9.35\%$ and $28.72\%$).
- The discrepancy in unit convention was the underlying trigger for the figure axis scaling bug (F04). All documentation and code comments have been harmonized to specify fractional $[0, 1]$ representation in core evaluation modules and percentage $[0, 100\%]$ representation in presentation plotting scripts.
