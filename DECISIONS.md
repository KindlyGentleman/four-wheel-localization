# Design & Architectural Decisions Ledger

## Decision D-01: Coordinate Systems and Reference Frames
- **World Frame ($W$):** Right-handed 2D Cartesian coordinate system.
  - $X_W$: Eastward displacement in meters ($+X$ pointing right).
  - $Y_W$: Northward displacement in meters ($+Y$ pointing up).
  - Heading $\theta$: Radians, measured counter-clockwise (CCW) from $+X_W$ (East = $0$, North = $+\pi/2$, West = $\pm\pi$, South = $-\pi/2$).
- **Robot Body Frame ($B$):** Attached to the geometric center of the 4WD chassis.
  - $X_B$: Longitudinal axis pointing forward (direction of travel).
  - $Y_B$: Lateral axis pointing left.
  - Heading aligns with $X_B$.
- **Sensor Frame Conversions:**
  - **CMPS12 compass:** The raw sensor reports degrees clockwise from magnetic north ($0^\circ$ North, $90^\circ$ East). We convert to mathematical radians CCW from $+X_W$:
    $$\theta_{\text{filter}} = \text{wrap}\left(\frac{\pi}{2} - \theta_{\text{compass\_deg}} \cdot \frac{\pi}{180} - \delta_{\text{mag}}\right)$$
  - **GNSS antenna lever-arm:** Positioned at offset $(l_x, l_y)$ in the body frame $B$. The observed position in the world frame is:
    $$p_{\text{ant}} = \begin{bmatrix} p_x \\ p_y \end{bmatrix} + R(\theta) \begin{bmatrix} l_x \\ l_y \end{bmatrix} = \begin{bmatrix} p_x + l_x \cos\theta - l_y \sin\theta \\ p_y + l_x \sin\theta + l_y \cos\theta \end{bmatrix}$$
  - **IMU:** Placed at chassis center $(0,0)$ by default, or with configurable mounting translation.

## Decision D-02: Skid-Steer 4WD Kinematics and Effective Track Width
- For a 4WD skid-steered robot with 4 independently driven wheels:
  - Left wheels: Front-Left (FL), Rear-Left (RL).
  - Right wheels: Front-Right (FR), Rear-Right (RR).
  - In ideal conditions, $v_{\text{left}} = \frac{1}{2}(v_{\text{FL}} + v_{\text{RL}})$, $v_{\text{right}} = \frac{1}{2}(v_{\text{FR}} + v_{\text{RR}})$.
  - Chassis forward speed: $v = \frac{1}{2}(v_{\text{left}} + v_{\text{right}})$.
  - Angular rate under differential steering: $\omega = \frac{v_{\text{right}} - v_{\text{left}}}{b_{\text{eff}}}$.
  - **Physical Reality:** In skid-steer vehicles, turning requires tires to scrub and slide laterally. Therefore, the geometric track width $b$ does not equal the kinematic effective track width $b_{\text{eff}}$ ($b_{\text{eff}} > b$). We model $b_{\text{eff}}$ explicitly as a calibrated parameter, and test sensitivity to calibration mismatch.

## Decision D-03: Longitudinal Wheel Slip Model
- Ground velocity of wheel $i$ under longitudinal slip ratio $s_i \in (-\infty, 1]$:
  $$v_{\text{ground}, i} = (1 - s_i) \cdot r_i \cdot \omega_{\text{wheel}, i}$$
  where $s_i > 0$ represents wheel spin (encoder over-reports travel), $s_i < 0$ represents wheel skid (braking/lockup).
- The chassis motion is integrated from actual wheel-ground interactions, whereas encoders observe motor/wheel-shaft rotation $\omega_{\text{wheel}, i}$.

## Decision D-04: State Representations and Filter Formulations
- **Baseline 3-State EKF (B0–B4):**
  $$x = \begin{bmatrix} p_x & p_y & \theta \end{bmatrix}^T$$
  Control input: $u = \begin{bmatrix} v & \omega \end{bmatrix}^T$ derived from wheel encoders or gyro.
- **Bias-Aware 4-State EKF (B5):**
  $$x = \begin{bmatrix} p_x & p_y & \theta & b_g \end{bmatrix}^T$$
  where $b_g$ is the gyroscope bias, integrated as a Wiener random walk process:
  $$\dot{b}_g = w_{b_g}, \quad w_{b_g} \sim \mathcal{N}(0, \sigma_{b_g}^2)$$
- **Numerical Stability:**
  - Joseph-stabilized covariance update:
    $$P_{k|k} = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T$$
  - Symmetrization enforcement: $P = \frac{1}{2}(P + P^T)$.
  - Innovation angle wrapping for heading: $\tilde{y}_\theta = \text{atan2}(\sin(z_\theta - \hat{\theta}), \cos(z_\theta - \hat{\theta}))$.

## Decision D-05: Statistical Testing and Evaluation Standards
- Monte Carlo trials must be paired across baselines with identical disturbance seeds.
- Trajectory evaluation uses:
  - 2D Position Root Mean Square Error (RMSE)
  - Heading RMSE with wrapping
  - Normalized Estimation Error Squared (NEES) for filter consistency ($k=3$ or $k=4$ DOF)
  - Normalized Innovation Squared (NIS) for measurement updates
  - 95% confidence intervals over independent simulation runs.
