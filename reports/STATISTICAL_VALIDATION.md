# Statistical Validation and Consistency Assessment

## 1. Estimation Filter Consistency Criteria
A state estimator is defined as statistically consistent if:
1. The estimation error is zero-mean:
   $$E[x_k - \hat{x}_k] = 0$$
2. The empirical error covariance matches the filter-calculated covariance matrix $P_k$:
   $$E[(x_k - \hat{x}_k)(x_k - \hat{x}_k)^T] = P_k$$

To formally verify consistency across Monte Carlo trials, we compute two normalized metrics:
- **Normalized Estimation Error Squared (NEES):**
  $$\epsilon_k = (x_k - \hat{x}_k)^T P_k^{-1} (x_k - \hat{x}_k)$$
  For state dimension $d$, if the filter is consistent and errors are normally distributed, $\epsilon_k \sim \chi^2(d)$.
  - For 2D position: $E[\epsilon_k] = 2.0$, with 95% confidence interval $[0.05, 5.99]$.
  - For 3D pose: $E[\epsilon_k] = 3.0$, with 95% confidence interval $[0.22, 7.81]$.
- **Normalized Innovation Squared (NIS):**
  $$\nu_k = y_k^T S_k^{-1} y_k$$
  where $y_k$ is the measurement residual and $S_k = H_k P_{k|k-1} H_k^T + R_k$.
  - For 2D GNSS position: $E[\nu_k] = 2.0$, 95% upper limit is $5.99$.
  - For 1D compass heading: $E[\nu_k] = 1.0$, 95% upper limit is $3.84$.

---

## 2. Empirical Consistency Results

| Condition | Baseline | Mean Pos NEES | Theoretical Expectation | $2\sigma$ Ellipse Coverage | Status |
|---|---|---|---|---|---|
| **E1 (Nominal)** | B0 (Odometry) | 0.006 | 2.00 | 100.0% | Conservative |
| **E1 (Nominal)** | B2 (Odom+GNSS) | 1.18 | 2.00 | 100.0% | Consistent |
| **E1 (Nominal)** | B4 (+Compass) | 1.59 | 2.00 | 100.0% | **Consistent** |
| **E1 (Nominal)** | B5 (Bias-Aware) | 2.19 | 2.00 | 98.5% | **Near-Ideal** |
| **E4 (Wheel Slip 35%)** | B0 (Odometry) | 2517.37 | 2.00 | 29.3% | **Severe Divergence** |
| **E4 (Wheel Slip 35%)** | B3 (Primary EKF) | 20.40 | 2.00 | 34.5% | Optimistic |
| **E4 (Wheel Slip 35%)** | B5 (Bias-Aware) | 4.43 | 2.00 | 78.8% | Bounded Error |
| **E5 (Gyro Bias 2°/s)** | B3 (Primary EKF) | 106.14 | 2.00 | 13.9% | Overconfident |
| **E5 (Gyro Bias 2°/s)** | B5 (Bias-Aware) | 2.19 | 2.00 | 98.5% | **Consistent** |

### Findings
1. Under uncompensated gyro bias (E5), standard EKF (B3) exhibits NEES = $106.14 \gg 2.0$, indicating severe overconfidence (the filter trusts corrupt integration and underestimates real error).
2. Baseline B5 successfully estimates gyro bias, restoring NEES to **$2.19$** (close to the theoretical $2.0$), maintaining $98.5\%$ empirical $2\sigma$ coverage.
3. During severe wheel slip (E4), pure odometry experiences catastrophic inconsistency (NEES $> 2500$), while sensor fusion preserves spatial tracking.
