# EKF-Based Multi-Sensor Localization for a Four-Wheel Mobile Robot

[![Tests](https://img.shields.io/badge/pytest-21%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Format: IEEE](https://img.shields.io/badge/graphics-IEEE%20Transactions-red.svg)]()

> A reproducible Python simulation investigating four-wheel robot odometry, GNSS/IMU sensor fusion, and Extended Kalman Filter (EKF) localization under realistic sensor noise, wheel slip, and measurement dropout.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Quick Start & Reproduction](#quick-start--reproduction)
- [Sensor Fusion Baselines](#sensor-fusion-baselines)
- [Experimental Scenarios & Results](#experimental-scenarios--results)
- [Scientific Figures & Demonstrations](#scientific-figures--demonstrations)
- [Upstream Audit (PythonRobotics Reproduction)](#upstream-audit-pythonrobotics-reproduction)
- [Documentation & Technical Reports](#documentation--technical-reports)
- [Citation](#citation)
- [License](#license)

---

## Project Overview
This repository provides a modular, reproducible CPU-based Python state estimation framework for planar four-wheel-drive (4WD) skid-steered mobile robots. It investigates localization accuracy, covariance consistency, and failure modes when fusing wheel odometry with inertial, magnetic, satellite, and range sensing.

A central principle of this project is that **ground-truth chassis motion is integrated strictly from ground-wheel contact mechanics**, decoupling true chassis displacement from optimistic encoder readings whenever wheel slip occurs.

```text
              FRONT
          +-----------+
   FL  O--|   LiDAR   |--O  FR  (Front Wheels: Independent Encoders #1 & #2)
          |    IMU    |
          |   GNSS    |
   RL  O--|           |--O  RR  (Rear Wheels:  Independent Encoders #3 & #4)
          +-----------+
               REAR
```

---

## Key Features
- **4 Independent Wheel Encoder Channels (50 Hz):** Simulates integer pulse quantization (2048 PPR), count jitter, and stochastic pulse loss.
- **Skid-Steering Kinematics & Longitudinal Slip:** Explicit model of kinematic scrubbing with calibrated effective track width $b_{\text{eff}} > b$ and dynamic slip ratios $s_i \in (-\infty, 1]$.
- **Inertial & Magnetic Sensing:**
  - 6-DOF IMU (MPU-9250 profile, 50 Hz) with random-walk gyroscope bias and accelerometer specific force.
  - Tilt-compensated magnetic compass (CMPS12 profile, 10 Hz) with calibration state transitions and magnetic interference injection.
- **GNSS Receiver Simulation (1 Hz):** Local East-North-Up (ENU) coordinates, WGS-84 conversion interface, antenna lever-arm compensation, and contiguous blackout intervals.
- **Range Sensing Extensions:** Mapped landmark 2D LiDAR range-bearing updates and 3-transducer ultrasonic planar wall detection.
- **Extended Kalman Filters:**
  - Standard 3-state EKF ($p_x, p_y, \theta$) with analytical Jacobians validated against numerical finite differences ($< 5 \times 10^{-11}$ error).
  - 4-state bias-aware EKF ($p_x, p_y, \theta, b_g$) co-estimating gyroscope bias online.
  - Joseph-stabilized covariance updates and angular innovation wrapping.
- **Publication Deliverables:** 18 IEEE Transactions-compliant scientific figures (PDF, SVG, 300 DPI PNG) and a 1920x1080 30 FPS demonstration video (`videos/localization_demo.mp4`).

---

## Repository Structure
```text
4wd-robot-localization-ekf/
├── README.md
├── pyproject.toml
├── LICENSE
├── CITATION.cff
├── TASKS.md
├── DECISIONS.md
├── configs/
│   ├── robot.yaml               # Chassis dimensions & encoder specs
│   ├── sensors.yaml             # Sampling rates & sensor noise params
│   ├── nominal.yaml             # Nominal trajectory parameters
│   ├── slip.yaml                # Asymmetric slip experiment config
│   ├── gnss_outage.yaml         # GNSS blackout interval config
│   └── benchmark.yaml           # Monte Carlo benchmark settings
├── src/
│   └── robot_localization/
│       ├── models/              # 4WD kinematics, skid steering, numerical integration
│       ├── sensors/             # Encoders, GNSS, CMPS12, MPU-9250, LiDAR, Ultrasonic
│       ├── estimation/          # Dead reckoning, 3D EKF, 4D bias-aware EKF, baseline runner
│       ├── simulation/          # World maps, event scheduler, robot simulator, scenarios
│       ├── evaluation/          # RMSE, MAE, NEES, NIS, covariance coverage, Monte Carlo
│       ├── visualization/       # IEEE styling, figure plotting, vector illustrations, animation
│       └── cli.py               # Command-line entry point
├── tests/                       # 21 comprehensive pytest unit tests
├── scripts/
│   ├── reproduce_reference.py   # Upstream PythonRobotics audit & reproduction
│   ├── run_quick.py             # Deterministic scenarios E1-E9 and E12
│   ├── run_full.py              # Multi-trial Monte Carlo benchmark with 95% CIs
│   ├── generate_figures.py      # Programmatic generation of F01-F12 and I1-I6
│   └── render_video.py          # 1080p demonstration video renderer
├── figures/                     # 18 figures and illustrations (PDF, SVG, PNG)
├── videos/                      # MP4 video, preview PNG, metadata JSON
├── results/                     # Numerical CSV and NPZ benchmark data
└── reports/                     # Detailed research and audit reports
```

---

## Quick Start & Reproduction

### Installation
Clone the repository and install dependencies in editable mode:
```bash
git clone https://github.com/reproduction/4wd-robot-localization-ekf.git
cd 4wd-robot-localization-ekf
pip install -e .[dev]
```

### 1. Run Unit Tests (21 Tests)
Verify analytical Jacobians, kinematics, sensor models, and filter updates:
```bash
python -m pytest tests/ -v
```

### 2. Reproduce Upstream Reference & Audit
```bash
python scripts/reproduce_reference.py
```

### 3. Run Experimental Scenarios (E1 - E9, E12)
```bash
python scripts/run_quick.py
```

### 4. Run Monte Carlo Statistical Trials
```bash
python scripts/run_full.py
```

### 5. Generate Publication Figures & Illustrations
```bash
python scripts/generate_figures.py
```

### 6. Render 1080p Demonstration Video
```bash
python scripts/render_video.py
```

---

## Sensor Fusion Baselines

| Baseline | Inputs & Prediction | Measurement Updates | State Vector |
|---|---|---|---|
| **B0** | 4-Wheel Encoders | None (Dead Reckoning) | $[p_x, p_y, \theta]^T$ |
| **B1** | Wheel Speed + Gyro $Z$ | None (Prediction Only) | $[p_x, p_y, \theta]^T$ |
| **B2** | 4-Wheel Encoders | 1 Hz GNSS Position | $[p_x, p_y, \theta]^T$ |
| **B3** | Wheel Speed + Gyro $Z$ | 1 Hz GNSS Position | $[p_x, p_y, \theta]^T$ |
| **B4** | Wheel Speed + Gyro $Z$ | 1 Hz GNSS + 10 Hz Compass | $[p_x, p_y, \theta]^T$ |
| **B5** | Wheel Speed + Gyro $Z$ | 1 Hz GNSS + 10 Hz Compass | $[p_x, p_y, \theta, b_g]^T$ |

---

## Experimental Scenarios & Results

Summary of 2D Position RMSE (meters) and Heading RMSE (degrees) across benchmark scenarios:

| Scenario | B0 (Odometry) | B1 (Odom+Gyro) | B3 (Primary EKF) | B4 (+Compass) | B5 (Bias-Aware) |
|---|---|---|---|---|---|
| **E1: Nominal** | 0.064 m (0.18°) | 7.347 m (29.7°) | 0.616 m (9.32°) | **0.097 m** (0.53°) | 0.114 m (**0.37°**) |
| **E2: Radius Error (5%)** | 25.74 m (108.8°) | 7.347 m (29.7°) | 0.616 m (9.31°) | **0.097 m** (0.53°) | 0.113 m (**0.37°**) |
| **E3: Isolated FL Slip** | 17.01 m (72.76°) | 5.116 m (29.5°) | 0.815 m (12.39°) | 0.211 m (0.54°) | **0.198 m** (**0.38°**) |
| **E4: Differential Left Slip**| 13.77 m (63.47°) | 4.938 m (29.3°) | 0.781 m (12.92°) | 0.175 m (0.54°) | **0.171 m** (**0.38°**) |
| **E5: Gyro Bias (+2°/s)** | 0.064 m (0.18°) | 15.75 m (68.9°) | 1.635 m (22.21°) | **0.093 m** (1.11°) | 0.114 m (**0.37°**) |
| **E6: Compass Fault (+35°)** | 0.064 m (0.18°) | 7.347 m (29.7°) | **0.616 m** (9.32°) | 2.568 m (16.31°) | 2.585 m (17.14°) |
| **E8: 20s GNSS Outage** | 1.341 m (4.13°) | 7.770 m (29.7°) | 2.466 m (16.85°) | 1.614 m (0.53°) | **1.594 m** (**0.37°**) |
| **E12: Combined Stress** | 11.93 m (76.84°) | 7.426 m (49.2°) | 1.537 m (23.92°) | 0.547 m (0.81°) | **0.530 m** (**0.38°**) |

---

## Scientific Figures & Demonstrations

### Figure Catalog (`figures/`)
- **F01:** Four-wheel mobile robot architecture schematic
- **F02:** Multi-sensor EKF sensor fusion block diagram
- **F03:** Synchronized 4-wheel encoder linear speed traces under slip
- **F04:** Four-wheel odometry drift comparison against ground truth
- **F05:** Progressive sensor fusion trajectories (B0, B1, B3, B4)
- **F06:** Gyroscope bias estimation and error mitigation over time
- **F07:** Error growth and filter recovery across 20-second GNSS outage
- **F08:** Position RMSE vs. longitudinal slip severity
- **F09:** Sensor configuration ablation bar chart (E12 combined stress)
- **F10:** Filter consistency evaluation (NEES & NIS vs. theoretical bounds)
- **F11:** Mapped landmark LiDAR ranging in GNSS-denied zones
- **F12:** Combined failure interaction trajectories and errors

### Research Illustrations (`figures/`)
- **I1:** Robot sensor layout and coordinate frames ($\{W\}, \{B\}, \{L\}$)
- **I2:** Four-wheel skid-steer odometry geometry & effective track width
- **I3:** Discrete Extended Kalman Filter state flow
- **I4:** Longitudinal wheel slip mechanics vs. encoder observation
- **I5:** GNSS blackout uncertainty growth and convergence dynamics
- **I6:** Mapped landmark range-bearing observation geometry

### Demonstration Video (`videos/`)
- **File:** `videos/localization_demo.mp4` (1920x1080, 30 FPS, H.264)
- **Preview:** `videos/localization_demo_preview.png`
- **Metadata:** `videos/video_metadata.json`

---

## Upstream Audit (PythonRobotics Reproduction)
An in-depth mathematical audit of the upstream reference (`extended_kalman_filter.py` by Atsushi Sakai et al.) revealed:
1. **Covariance-versus-Standard-Deviation Multiplication:** Upstream defines `GPS_NOISE = np.diag([0.5, 0.5]) ** 2` and then computes `z = observation_model(xTrue) + GPS_NOISE @ randn(2, 1)`. Because the covariance matrix is squared prior to multiplying by standard normal variates, the actual simulated noise standard deviation is $0.25\text{ m}$ rather than $0.50\text{ m}$. Meanwhile, the filter assumes $R = \text{diag}([1.0, 1.0])^2 = 1.0\text{ m}^2$, creating a $16\times$ over-estimation of measurement variance.
2. **Redundant State Vector:** Upstream includes velocity $v$ in state vector $x=[p_x, p_y, \theta, v]^T$, but simply overwrites $v_k = 1.0 \cdot u_v$ rather than propagating state dynamics.
3. Detailed derivation and empirical comparisons are reported in [`reports/REFERENCE_AUDIT.md`](reports/REFERENCE_AUDIT.md).

---

## Documentation & Technical Reports
- [`reports/REFERENCE_AUDIT.md`](reports/REFERENCE_AUDIT.md): Detailed audit and reproduction of PythonRobotics 2D EKF.
- [`reports/MODEL_VALIDATION.md`](reports/MODEL_VALIDATION.md): Kinematic derivations and finite-difference Jacobian verification.
- [`reports/SENSOR_ASSUMPTIONS.md`](reports/SENSOR_ASSUMPTIONS.md): Sensor noise parameters, rates, and coordinate transforms.
- [`reports/EXPERIMENTAL_PROTOCOL.md`](reports/EXPERIMENTAL_PROTOCOL.md): Benchmark setup and metric definitions.
- [`reports/RESULTS.md`](reports/RESULTS.md): Full empirical tables and comparative analysis.
- [`reports/STATISTICAL_VALIDATION.md`](reports/STATISTICAL_VALIDATION.md): NEES, NIS, and covariance consistency analysis.
- [`reports/NOVELTY_AND_LIMITATIONS.md`](reports/NOVELTY_AND_LIMITATIONS.md): Formal non-claims, novelty bounds, and modeling limits.
- [`reports/REPRODUCIBILITY.md`](reports/REPRODUCIBILITY.md): Step-by-step reproduction instructions.
- [`reports/MINI_PAPER.md`](reports/MINI_PAPER.md): Complete scholarly research manuscript draft.

---

## Citation
```bibtex
@misc{four_wheel_robot_ekf_2026,
  author = {Research Reproduction Team},
  title = {EKF-Based Multi-Sensor Localization for a Four-Wheel Mobile Robot},
  year = {2026},
  url = {https://github.com/reproduction/4wd-robot-localization-ekf}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
