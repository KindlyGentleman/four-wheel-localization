# Upstream Reference Audit: PythonRobotics 2D EKF Localization

## 1. Reference Metadata
- **Algorithm:** 2D Extended Kalman Filter (EKF) Localization
- **Primary Source:** Sakai, Atsushi, et al. *"PythonRobotics: a Python code collection of robotics algorithms."* arXiv preprint arXiv:1808.10703 (2018).
- **Upstream Repository:** [https://github.com/AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
- **Specific Implementation:** `Localization/extended_kalman_filter/extended_kalman_filter.py`
- **Upstream License:** MIT License

---

## 2. Algorithmic Formulation in Upstream Code

### 2.1 State Vector and Control Input
The upstream implementation defines a 4-state vector:
$$x = \begin{bmatrix} p_x \\ p_y \\ \theta \\ v \end{bmatrix}$$
where $p_x, p_y$ are 2D planar positions, $\theta$ is the yaw heading angle, and $v$ is forward linear speed.

The control input vector is:
$$u = \begin{bmatrix} v_c \\ \omega_c \end{bmatrix}$$
representing commanded forward speed $v_c$ and commanded yaw rate $\omega_c$.

### 2.2 Discrete State Transition Model
The upstream implementation defines:
$$x_k = F x_{k-1} + B u_k$$
with:
$$F = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} \Delta t \cos(\theta_{k-1}) & 0 \\ \Delta t \sin(\theta_{k-1}) & 0 \\ 0 & \Delta t \\ 1 & 0 \end{bmatrix}$$

### 2.3 Upstream Process and Measurement Jacobians
The state Jacobian $J_F = \frac{\partial f}{\partial x}$ is coded as:
$$J_F = \begin{bmatrix} 1 & 0 & -\Delta t \cdot v_c \sin(\theta) & \Delta t \cos(\theta) \\ 0 & 1 & \Delta t \cdot v_c \cos(\theta) & \Delta t \sin(\theta) \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

The measurement model observes 2D Cartesian coordinates:
$$z = H x = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} x, \quad J_H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

---

## 3. Detailed Audit Findings & Technical Inconsistencies

### Finding A-01: Covariance vs. Standard Deviation Matrix Multiplication Bug
In the upstream simulation function `observation(xTrue, xd, u)`:
```python
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2

z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
ud = u + INPUT_NOISE @ np.random.randn(2, 1)
```
- **Issue:** `GPS_NOISE` is computed by squaring the diagonal $[0.5, 0.5]$, yielding the covariance matrix $\Sigma = \text{diag}([0.25, 0.25])$.
- In standard probability theory, generating a zero-mean Gaussian random vector with covariance $\Sigma$ requires multiplying the standard deviation matrix (or Cholesky factor $L = \Sigma^{1/2}$) by standard normal variates $\sim \mathcal{N}(0, I)$.
- In upstream code, multiplying $\Sigma = \text{diag}([0.25, 0.25])$ directly by $\mathcal{N}(0, I)$ results in actual added noise with standard deviation $\sigma = 0.25\text{ m}$, not $0.5\text{ m}$.
- Meanwhile, the measurement covariance matrix passed into the EKF update is:
  ```python
  R = np.diag([1.0, 1.0]) ** 2 # R = diag([1.0, 1.0])
  ```
  Consequently, the filter assumes measurement variance $R = 1.0\text{ m}^2$, while the simulated measurement actually has variance $\sigma^2 = 0.0625\text{ m}^2$. The filter is therefore miscalibrated by a factor of $16\times$ in variance, causing it to under-weight the GPS measurements compared to their true precision.
- Similarly, `INPUT_NOISE` is squared before multiplying by `randn`, scaling input noise by $\sigma^2$ instead of $\sigma$.

### Finding A-02: Kinematic Inconsistency in State Vector and Jacobian
- The state $x_3 = v$ is reset at every time step to $1.0 \cdot u_0$ via matrix $B$. It is not propagated from $x_{k-1}[3]$ (row 4 of $F$ is all zeros).
- However, in the analytical Jacobian $J_F$:
  - Column 4 has terms $\Delta t \cos(\theta)$ and $\Delta t \sin(\theta)$, which would only be valid if $p_{x, k} = p_{x, k-1} + \Delta t \cdot v_{k-1} \cos(\theta_{k-1})$.
  - Diagonal entry $J_F[3, 3] = 1.0$, which implies $v_k = v_{k-1}$, directly contradicting the motion model where $F[3, :] = 0$.
- In our rigorous 4WD formulation, velocity and angular rate are measured kinematic control inputs derived from four independent wheel encoders, and the core pose state is strictly $x = [p_x, p_y, \theta]^T$, or augmented with physical parameters such as gyro bias $x = [p_x, p_y, \theta, b_g]^T$.

### Finding A-03: Lack of Angle Normalization (Wrapping)
- Upstream computes innovation as $y = z - z_{pred}$.
- While $z$ is 2D position in the reference demo, any heading observation or state propagation that rotates through $\pm\pi$ will experience discontinuous jumps if angle wrapping is omitted.
- In our implementation, all angular differences are wrapped strictly to $[-\pi, \pi]$ using $\text{atan2}(\sin\Delta\theta, \cos\Delta\theta)$.

### Finding A-04: Absence of Wheel Kinematics, Skid Steering, and Slip Physics
- The upstream reference models an unconstrained mathematical particle with 2 control inputs. It does not model:
  - Four independent wheels with distinct rotational encoders.
  - Skid-steer kinematics with effective track width $b_{\text{eff}}$.
  - Wheel slip, wheel radius mismatch, or encoder quantization.
  - Realistic sensor error dynamics (colored noise, random walk, magnetic distortion, lever arm).

---

## 4. Empirical Reproduction Verification

We executed `scripts/reproduce_reference.py` under identical seed (`seed=42`, $T=50\text{ s}$, $\Delta t=0.1\text{ s}$):

| Configuration | Dead Reckoning RMSE | EKF Position RMSE | Max EKF Error |
|---|---|---|---|
| **Upstream Verbatim** | 3.8909 m | 0.1702 m | 0.4123 m |
| **Statistically Corrected** | 7.2465 m | 0.3279 m | 0.7791 m |

Under the corrected standard deviation scaling, input noise standard deviation is realistically larger (as originally intended by $1.0\text{ m/s}$ and $30^\circ/\text{s}$ nominal bounds), leading to higher dead-reckoning drift ($7.25\text{ m}$ vs $3.89\text{ m}$). The calibrated EKF successfully fuses GPS measurements and limits the trajectory error to $0.33\text{ m}$ RMSE.

Generated output files:
- Trajectory and error metrics: `results/reproduce_reference_results.npz`
- Reproduction plot: `figures/fig_e0_pythonrobotics_reproduction.png`
