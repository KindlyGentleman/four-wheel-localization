# Sensor Simulation Assumptions and Error Dynamics

## 1. Overview and Scope
This document details the observation models, error parameters, sampling frequencies, and physical assumptions used across all simulated sensor modalities.

These models are **sensor-inspired synthetic models** for reproducible algorithmic benchmarking on a CPU-based workstation. They do not represent live physical hardware connected to hardware-in-the-loop (HIL) test benches.

---

## 2. Sensor Modality Specifications

| Sensor Channel | Measurement Type | Nominal Rate | Noise Model | Systematic Error / Drift | Frame |
|---|---|---|---|---|---|
| **4x Wheel Encoders** | Cumulative signed integer ticks | 50 Hz | Quantization (2048 PPR) + 0.5 tick jitter | Missed pulses ($p=5\times 10^{-4}$), wheel radius mismatch | Wheel hubs |
| **MPU-9250 Gyroscope** | Yaw angular rate $\omega_z$ | 50 Hz | White Gaussian ($\sigma = 0.005\text{ rad/s}$) | Random walk bias ($\sigma_w = 1\times 10^{-4}\text{ rad/s}/\sqrt{\text{s}}$) | Body center |
| **MPU-9250 Accelerometer** | 3-axis specific force | 50 Hz | White Gaussian ($\sigma = 0.05\text{ m/s}^2$) | Centripetal ($v\omega$) + gravity reaction ($+9.81\text{ m/s}^2$) | Body center |
| **CMPS12 Compass** | Absolute heading | 10 Hz | Gaussian ($\sigma = 1.0^\circ = 0.0175\text{ rad}$) | Local magnetic disturbance ($+35^\circ$), calibration states (0-3) | North CW |
| **GNSS Receiver** | 2D horizontal position | 1.0 Hz | Gaussian ($\sigma = 0.50\text{ m}$) | Random walk drift ($0.005\text{ m/s}$), multipath ($+3.5\text{ m}$), outages | World ENU |
| **2D LiDAR** | Range & bearing to landmarks | 5.0 Hz | $\sigma_r = 0.03\text{ m}$, $\sigma_\beta = 0.015\text{ rad}$ | Finite range window ($0.2\text{ m}$ to $12\text{ m}$) | Forward bumper ($+0.20\text{ m}$) |
| **Ultrasonic Array** | Range to planar walls | 10.0 Hz | $\sigma_r = 0.02\text{ m}$ | $15^\circ$ beam spread cutoff, range $0.04\text{ m}$ to $3.5\text{ m}$ | 3 transducers (C, L, R) |

---

## 3. Detailed Modality Formulations

### 3.1 Four Independent Wheel Encoders
- **Resolution:** 2048 pulses per revolution (PPR) directly on wheel shaft (gear ratio 1.0).
- **Physical Integration:**
  $$\theta_i(t) = \int_0^t \omega_i(\tau) d\tau$$
  $$\text{ticks}_i(t) = \text{round}\left(\frac{N}{2\pi} \theta_i(t) + \epsilon_{\text{jitter}}\right)$$
- **Velocity Derivation:**
  $$v_i(k) = \frac{2\pi r_i}{N \Delta t} \Delta \text{ticks}_i(k)$$
- **Quantization Variance:**
  $$\text{Var}(v_i) = \left(\frac{2\pi r_i}{N \Delta t}\right)^2 \left(\frac{1}{12} + \sigma_{\text{jitter}}^2\right)$$

### 3.2 GNSS Antenna Lever-Arm
The GNSS receiver measures the position of its antenna phase center, not the chassis center:
$$p_{\text{ant}} = \begin{bmatrix} p_x \\ p_y \end{bmatrix} + R(\theta) \begin{bmatrix} l_x \\ l_y \end{bmatrix} = \begin{bmatrix} p_x + l_x \cos\theta - l_y \sin\theta \\ p_y + l_x \sin\theta + l_y \cos\theta \end{bmatrix}$$
with measurement Jacobian:
$$H_{\text{gnss}} = \begin{bmatrix} 1 & 0 & -l_x \sin\theta - l_y \cos\theta \\ 0 & 1 & l_x \cos\theta - l_y \sin\theta \end{bmatrix}$$
Omitting the lever arm introduces systematic position errors proportional to angular rate and lever length.

### 3.3 CMPS12 vs. MPU-9250 Profiles & Correlation Warning
- **CMPS12:** In physical hardware, the CMPS12 utilizes an internal BNO080 processor running sensor fusion firmware that blends magnetic and inertial data.
- **Important Statistical Warning:** Because CMPS12 heading is already derived from internal gyro integration, using both CMPS12 heading and raw gyro measurements introduces statistical correlation. In our primary baseline B3, only independent gyro and GNSS are fused to ensure mathematical independence. When CMPS12 is added in B4, we test sensitivity to magnetic distortions and monitor innovation NIS.
- **MPU-9250:** Provides raw inertial and magnetic observations. Gyroscope bias $b_g$ is treated as a dynamic state variable in the 4D bias-aware EKF (B5):
  $$\dot{b}_g = w_{bg}, \quad w_{bg} \sim \mathcal{N}(0, \sigma_{\text{walk}}^2)$$
