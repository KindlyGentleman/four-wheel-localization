# Particle Filter and AMCL Validation Report

## 1. Executive Summary

This report documents the validation and mathematical remediation of the 2D Particle Filter and Adaptive Monte Carlo Localization (AMCL) module in `src/robot_localization/estimation/particle_filter.py`.

Four technical issues were resolved:
1. Missing or invalid LiDAR scan updates did not update the estimated pose or covariance during prediction.
2. AMCL kidnapping recovery was starved when particle weights were uniformly low ($N_{\text{eff}} \approx N > N/2$).
3. KLD binning spatial discretization truncated negative coordinates toward zero, causing spatial bin collapse across the origin.
4. Covariance estimation was incomplete, computing only position variance and scalar heading variance while omitting pose-heading cross-covariances $\mathbf{P}_{x\theta}, \mathbf{P}_{y\theta}$.

---

## 2. Mathematical Defects and Corrections

### 2.1 State Propagation Synchronization
- **Issue**: In `predict()`, particle state vectors were updated with control inputs and noise, but `self.estimated_pose` and `self.covariance` remained stale until a scan update was invoked. When an empty or corrupted LiDAR scan arrived, early return bypassed estimate calculation.
- **Correction**: `self._compute_estimate()` is now explicitly invoked at the end of `predict()`, as well as on early exits when invalid or zero-beam scans are received. When no beams hit valid obstacles, observation likelihood correctly reports 0.0 rather than fabricating 1.0.

### 2.2 AMCL Kidnapped-Robot Recovery Starvation
- **Issue**: Random particle injection previously occurred only within the KLD resampling loop when $N_{\text{eff}} < N / 2$. However, when the robot is severely displaced (kidnapped), all particles have uniformly low measurement likelihood. Because all particles are equally improbable, the normalized weights remain uniform ($w_i \approx 1/N$), resulting in $N_{\text{eff}} \approx N$. Consequently, resampling was skipped and recovery particles were never injected.
- **Correction**: Resampling and injection is now triggered when $N_{\text{eff}} < N / 2$ OR when $p_{\text{random}} > 0.05$. This guarantees immediate random particle injection across the free workspace when observation likelihood suddenly degrades.

### 2.3 KLD Binning Discretization
- **Issue**: Bin indices were computed using integer casting: `int(x / bin_size)`. In Python, `int(-0.05 / 0.5) == 0` and `int(+0.05 / 0.5) == 0`. Particles spanning $[-0.5, 0.5]$ were incorrectly collapsed into bin 0.
- **Correction**: Replaced with `math.floor(x / bin_size)` and `math.floor(wrap_angle(th) / bin_th)`. This guarantees uniform bin dimensions across the entire coordinate plane and avoids artificial reduction of required particle counts in KLD sampling.

### 2.4 Full 3-DOF Sample Covariance Matrix
- **Issue**: Covariance output was restricted to a $2\times 2$ position block and an isolated scalar variance $\sigma_\theta^2$, omitting cross-terms $\mathbf{P}_{x\theta}$ and $\mathbf{P}_{y\theta}$.
- **Correction**: Implemented complete $3\times 3$ sample covariance:
  $$\mathbf{P} = \sum_{i=1}^N w_i (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$
  with proper circular residual wrapping on heading: $\delta\theta_i = \text{wrap\_angle}(\theta_i - \bar{\theta})$.

---

## 3. Empirical Test Evidence

Unit and regression tests in `tests/test_audit_verification.py` verify all corrections:
- `test_particle_filter_predict_updates_estimate_without_scans`: Verifies pose and covariance update immediately upon motion prediction even with zero LiDAR updates.
- `test_amcl_recovery_injection_under_uniform_low_likelihood`: Verifies particle dispersion and injection across the map when observation likelihood collapses, despite high $N_{\text{eff}}$.
- `test_kld_binning_negative_coordinates`: Verifies distinct binning across the Cartesian origin using `math.floor`.
- `test_particle_filter_full_3dof_covariance`: Verifies $3\times 3$ matrix structure, symmetry, and non-zero $\mathbf{P}_{x\theta}, \mathbf{P}_{y\theta}$ cross-covariances under correlated particle clouds.

All tests pass 100%.
