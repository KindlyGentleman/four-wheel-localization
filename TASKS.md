# Task Tracking Ledger: Four-Wheel Robot Localization with EKF-Based Multi-Sensor Fusion

## Phase P0: Foundation & Reference Reproduction
- [x] P0.1 Audit upstream PythonRobotics EKF reference (`extended_kalman_filter.py`)
- [x] P0.2 Implement exact standalone reference reproduction (`scripts/reproduce_reference.py`)
- [x] P0.3 Document upstream audit findings in `reports/REFERENCE_AUDIT.md`
- [x] P0.4 Implement core 4WD robot kinematic models (`robot_4wd.py`, `skid_steer.py`, `kinematics.py`)
- [x] P0.5 Implement 4 independent wheel encoders with quantization, noise, and tick conversion (`encoder.py`)
- [x] P0.6 Implement 4-wheel dead-reckoning odometry with Jacobian-based covariance propagation (`dead_reckoning.py`)
- [x] P0.7 Implement core 3-state EKF with numerical Jacobian finite-difference validation (`ekf.py`)
- [x] P0.8 Unit test suite for Phase P0 components (`tests/test_p0_foundation.py`)


## Phase P1: Multi-Sensor Simulation & Fusion Core
- [x] P1.1 Implement GNSS sensor model with antenna lever-arm, dropout, and drift (`gnss.py`)
- [x] P1.2 Implement CMPS12 compass and gyro profile with angle wrapping and calibration states (`cmps12.py`)
- [x] P1.3 Implement MPU-9250 6-DOF IMU profile with gyro bias random-walk (`mpu9250.py`)
- [x] P1.4 Implement 4-state bias-aware EKF estimating $[p_x, p_y, \theta, b_g]^T$ (`bias_aware_ekf.py`)
- [x] P1.5 Implement timestamp-driven asynchronous event scheduler (`scheduler.py`, `robot_simulator.py`)
- [x] P1.6 Implement baselines B0 through B5 (`estimation/`, `simulation/scenarios.py`)
- [x] P1.7 Implement benchmark evaluation framework & metrics (RMSE, MAE, NEES, NIS) (`metrics.py`, `monte_carlo.py`)
- [x] P1.8 Run experiments E1-E9 (nominal, asymmetric radius, slip, gyro bias, compass disturbance, GNSS dropouts/outages)
- [x] P1.9 Unit test suite for Phase P1 components (`tests/test_p1_sensors_fusion.py`)

## Phase P2: Optional Range-Assisted Localization & Slip Adaptation
- [x] P2.1 Implement 2D known-landmark LiDAR ray-casting observation model and Jacobians (`lidar.py`)
- [x] P2.2 Implement 3-transducer ultrasonic known-wall observation model (`ultrasonic.py`)
- [x] P2.3 Implement baseline B6 (mapped range assistance during GNSS denial)
- [x] P2.4 Implement baseline B7 (slip-aware adaptive process covariance)
- [x] P2.5 Run experiments E10, E11, E12 (GNSS denial with range sensors, combined failures)
- [x] P2.6 Unit test suite for Phase P2 components (`tests/test_p2_range_sensors.py`)


## Phase P3: Visualization, Demonstration Video & Documentation
- [x] P3.1 Configure IEEE publication styling with scienceplots / standalone fallback (`ieee_style.py`)
- [x] P3.2 Generate IEEE figures F01-F12 programmatically from saved simulation data (`figures.py`, `generate_figures.py`)
- [x] P3.3 Generate programmatic vector illustrations I1-I6 (`illustrations.py`)
- [x] P3.4 Build multi-panel 30-60s animation dashboard and render MP4 video (`animation.py`, `render_video.py`)
- [x] P3.5 Write research reports:
  - [x] `reports/REFERENCE_AUDIT.md`
  - [x] `reports/MODEL_VALIDATION.md`
  - [x] `reports/SENSOR_ASSUMPTIONS.md`
  - [x] `reports/EXPERIMENTAL_PROTOCOL.md`
  - [x] `reports/RESULTS.md`
  - [x] `reports/STATISTICAL_VALIDATION.md`
  - [x] `reports/NOVELTY_AND_LIMITATIONS.md`
  - [x] `reports/REPRODUCIBILITY.md`
  - [x] `reports/MINI_PAPER.md`
- [x] P3.6 Compile final claim ledger and verification audit (`reports/CLAIM_LEDGER.md`)

