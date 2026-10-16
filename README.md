# Playing with Robot Localization

A small, reproducible study on mobile robot state estimation when wheel odometry slips, GNSS cuts out for 20 seconds, and visual tracking degrades.

![Localization Recovery Animation](assets/localization_loop.gif)

*Synchronized planar tracking across a 20-second GNSS blackout and wheel-slip segment. Left: Reference ground-truth path (white), strict innovation gate lockout (red), continuous odometry frame (green dotted), and recoverable EKF map pose with discrete frame shift (emerald solid). Right: Synthetic floor camera with live Lucas-Kanade optical flow, filter telemetry HUD, and experiment phase timeline.*

---

## What I tried

Kalman filter sensor fusion looks simple on paper: predict forward with wheel kinematics, update with satellite fixes, and reject bad measurements with an innovation Mahalanobis gate:

$$d_M^2 = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y} \le \chi^2_{m, 1-\alpha}$$

In textbook examples, this protects the state estimate from multipath jumps and wild outliers. But what happens when traction fails during a prolonged satellite blackout, and valid fixes return after the robot has drifted?

I wanted to test this in a controlled, repeatable environment without running a full robotics simulator stack.

To keep the vehicle kinematics grounded in real motion, I used the 2D ground-truth trajectory from the outdoor M2DGR `Anomaly` dataset run (Clearpath Husky rover, Shanghai Jiao Tong University). I wrote synthetic sensor generators around that reference trajectory in pure Python:
- Four independent wheel encoders with quantization and slip injection (50 Hz)
- Chassis IMU yaw rate with uncompensated rate bias (50 Hz)
- Absolute GNSS position fixes with an injected 20-second blackout (1 Hz)
- Procedural ground texture with OpenCV Lucas-Kanade feature tracking (10 Hz)
- Synthetic stereo camera and 2D landmark LiDAR models

The entire pipeline runs in standalone Python with NumPy, SciPy, Matplotlib, and OpenCV. No ROS installation is required.

---

## When localization fails

The first lesson was that standard textbook filters fail in subtle ways when physical assumptions break.

![Main Resilience Overview](assets/figure_main.png)

*(a) 4WD skid-steer chassis geometry; (b) Asymmetric 50% wheel-spin slip causing severe trajectory divergence; (c) Gyroscope bias (+0.025 rad/s) causing quadratic lateral drift; (d) Error growth and step convergence across a 20-second blackout; (e) Synthetic stereo disparity and point cloud; (f) Sensor configuration ablation across 10 configurations.*

1. **Wheel slip breaks dead reckoning:** When the left wheels spin with 50% slip for 5 seconds, pure odometry diverges by 11.8 m in simulation and 2.18 m peak error on the reference track.
2. **Gyro bias bends straight lines:** An uncompensated bias of just +0.025 rad/s (1.43 deg/s) causes quadratic lateral drift, turning a straight run into a wide spiral (7.36 m RMSE).
3. **Compass declination distorts heading:** A 25-degree magnetic disturbance pulls the standard EKF off-course by 2.29 m unless innovation gating rejects the azimuth update.

---

## GNSS dropout and recovery

The main experiment simulates a 20-second complete GNSS blackout ($t = 14.0$ s to $34.0$ s) while the robot experiences wheel slip ($t = 13.0$ s to $18.0$ s).

![M2DGR Hybrid Benchmark Summary](assets/figure_realdata.png)

*(a) Synthetic camera view with KLT optical flow; (b) Planar trajectory showing continuous odometry frame and discrete map correction arrow (REP-105); (c) Sensor availability timeline; (d) Position error over time highlighting the strict gate lockout failure; (e) Causal vision ablation under clean and degraded conditions; (f) Recovery strategy trade-off: position error vs GNSS fix acceptance.*

During the blackout, the robot navigates purely on dead reckoning and IMU integration. The state error grows to roughly 2.1 m. When satellites reappear at $t = 34.0$ s, filters respond very differently:

- **Strict Innovation Gate (R3 / Gate Lockout):** The filter's estimated covariance grows slower than the actual physical drift. When valid GNSS fixes resume, their innovation exceeds the 99% Chi-square threshold ($\chi^2 > 9.21$). The filter rejects every incoming fix ($0/27$ post-outage fixes accepted). The robot stays permanently lost.
- **Ungated Standard EKF (R1):** The filter immediately accepts the first post-outage fix. However, the large innovation shock causes a transient overshoot before settling after 2.85 seconds.
- **Temporal Consistency Window (R4):** The filter requires $M = 3$ consecutive spatially consistent fixes before triggering a state re-anchoring. It recovers within 1.86 seconds, accepts 96.3% of post-outage fixes, and avoids chasing single multipath outliers.
- **Covariance Inflation (R6):** The filter inflates process covariance proportional to outage duration. This expands the acceptance gate dynamically and absorbs the first fix within 0.86 seconds.
- **Global Re-initialization (R7):** Resets position directly to the verified fix. An early bug reset covariance to an identity matrix ($\mathbf{P} = \mathbf{I}$), which caused the filter to diverge to 15.3 m. Preserving positive-definite block covariance fixed this completely, achieving stable recovery at 1.09 m RMSE.

---

## Does vision help?

A common claim in sensor fusion papers is that adding a camera improves robustness. To verify whether visual tracking actually helped localization or was just being logged as a diagnostic, I ran a causal ablation test:

1. **Mode A (Vision Disabled):** Feature tracker completely turned off. Position RMSE: **1.0939 m**.
2. **Mode B (Vision Disconnected):** Feature tracker running and extracting KLT tracks, but velocity outputs explicitly withheld from the EKF. Position RMSE: **1.0939 m**.
   - Mode A and Mode B produce identical results to 16 decimal places. This confirms there were no hidden side-effects or leaky globals.
3. **Mode C (Vision Connected, Clean Texture):** Visual velocity estimates fused directly into the EKF. Position RMSE drops from 1.0939 m to **0.8370 m** (a 23.5% error reduction). The visual displacement dampens the odometry wheel-spin error during the slip window.
4. **Mode D (Vision Connected, Severe Degradation):** Gaussian noise and blur added to camera frames. Valid feature inliers drop from 286 to 59. The innovation gate identifies the noisy visual velocities and safely rejects them, falling back to the 1.0939 m baseline without diverging.

---

## A few numbers

Summary metrics across 10-seed Monte Carlo evaluations (6 scenarios x 10 seeds x 8 estimators = 480 method-trials) and hybrid benchmark runs:

| Scenario / Configuration | Pure Odometry (B0) | Bias-Aware EKF (B5) | Robust EKF (A5) | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Nominal Traction** | **0.064 m** | 0.072 m | 0.073 m | Pure odometry slightly beats EKF due to zero-drift simulation encoders. |
| **Asymmetric Wheel Slip** | 11.795 m | **0.218 m** | 0.226 m | EKF suppresses 50% wheel slip by 54x. |
| **Compass Disturbance** | 0.064 m | 2.292 m | **0.098 m** | Robust gating rejects 25 deg magnetic offset completely. |
| **GNSS Multipath (5m jump)**| 0.064 m | 0.930 m | **0.092 m** | Huber gating rejects sudden satellite coordinate jumps. |
| **20s GNSS Outage** | 1.341 m | 1.584 m | 2.094 m | Strict gating in A5 causes gate lockout during recovery. |
| **Combined Stress** | 11.925 m | 0.578 m | **0.540 m** | Robust EKF provides best overall balance in multi-fault stress. |
| **Hybrid Outage (R4 Recoverable)** | 1.391 m | 1.094 m | 1.362 m (Lockout)| Temporal recovery restores tracking in 1.86s (96.3% fix acceptance). |
| **Hybrid Outage (R4 + Vision)** | 1.391 m | **0.837 m** | N/A | Fusing visual velocity dampens slip drift by 23.5%. |

---

## What still does not work

Being honest about failure modes is more informative than claiming a universal solution:

1. **Pure odometry beats fusion in ideal conditions:** In simulated nominal conditions with white encoder noise and zero slip, pure wheel odometry achieved 0.064 m error, while the EKF achieved 0.072 m. Fusing noisy IMU and GNSS measurements added small perturbations. Sensor fusion only helped once unmodeled disturbances (slip, bias) appeared.
2. **Gate lockout is a real trap:** If an innovation gate is tuned too strictly ($\chi^2 \le 9.21$), it will permanently reject all post-blackout GNSS fixes after a 20-second drift. Without adaptive covariance or temporal re-anchoring, the filter never recovers.
3. **Floor texture dependency:** The visual velocity module assumes the ground plane has visible features. Over featureless concrete or dark asphalt, inlier count collapses and vision falls back completely to wheel odometry.
4. **Counterfactual safe-stop:** The included safe-stop controller issues emergency stop commands ($v_{\mathrm{cmd}} = 0$) within 2.0 seconds of covariance divergence. In real-world operation this stops the robot. In the hybrid benchmark, the reference trajectory continues unbraked because it replays fixed kinematic recordings.

---

## Reproduce

Clone the repository and install dependencies in an isolated Python environment:

```bash
git clone https://github.com/KindlyGentleman/four-wheel-localization.git
cd four-wheel-localization

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Run tests and experiments:

```bash
# 1. Run all 71 unit and regression tests (~35s)
python -m pytest tests/

# 2. Run the hybrid benchmark and causal vision ablation (~12s)
python realdata.py

# 3. Run the quick multi-scenario benchmark (~20s)
python scripts/run_quick.py

# 4. Run the complete 480-trial Monte Carlo benchmark (~3 min)
python scripts/run_full.py

# 5. Regenerate all IEEE publication figures and illustrations
python scripts/generate_figures.py
python scripts/generate_figure_realdata.py
python scripts/generate_resilience_figure.py
python scripts/generate_figure_tradeoffs.py

# 6. Render the animation GIF
python scripts/render_realdata_gif.py

# Or use the unified reproduction CLI:
python reproduce.py --quick
python reproduce.py --real-data
python reproduce.py --map
python reproduce.py --figure
python reproduce.py --gif
```

---

## References

- M2DGR Dataset: Yin, J. et al., "M2DGR: A Multi-sensor and Multi-scenario SLAM Dataset for Ground Robots", IEEE Robotics and Automation Letters (RA-L), 2022.
- ROS REP-105: "Coordinate Frames for Mobile Platforms" (Foote, 2010).
- Bar-Shalom, Y., Li, X. R., and Kirubarajan, T., *Estimation with Applications to Tracking and Navigation*, John Wiley & Sons, 2001.
- Thrun, S., Burgard, W., and Fox, D., *Probabilistic Robotics*, MIT Press, 2005.
