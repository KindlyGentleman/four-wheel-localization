# Exact Reproducibility Guide

## 1. System Environment & Prerequisites
- **Python Version:** Python 3.10+ (tested on Python 3.13.7 64-bit).
- **Core Dependencies:**
  - `numpy>=1.24.0`
  - `scipy>=1.10.0`
  - `matplotlib>=3.7.0`
  - `pandas>=2.0.0`
  - `pyyaml>=6.0`
  - `scienceplots>=2.1.0`
  - `pytest>=7.0.0`
- **Video Rendering:**
  - `ffmpeg` version 7.0+ on system PATH (or `imageio-ffmpeg`).

---

## 2. Step-by-Step Reproduction Instructions

### Step 1: Run Full Unit Test Suite
Execute the 21 unit tests covering kinematics, encoders, EKF Jacobians, sensor profiles, range sensing, and baseline runners:
```bash
python -m pytest tests/ -v
```
*Expected Result:* `21 passed in ~0.4s`.

### Step 2: Reproduce Upstream Reference & Audit
Reproduce the Sakai et al. PythonRobotics 2D EKF localization and verify the noise generation audit:
```bash
python scripts/reproduce_reference.py
```
*Outputs Created:*
- Numerical array: `results/reproduce_reference_results.npz`
- Validation plot: `figures/fig_e0_pythonrobotics_reproduction.png`

### Step 3: Run Deterministic Scenarios (E1 - E9, E12)
Execute all benchmark experiments on identical paired trajectories:
```bash
python scripts/run_quick.py
```
*Outputs Created:*
- Metrics CSV: `results/quick_metrics.csv`
- Compressed trajectory traces: `results/saved_benchmark_traces.npz`

### Step 4: Run Statistical Monte Carlo Trials
Execute 10 paired trials per condition and calculate 95% Student's t confidence intervals:
```bash
python scripts/run_full.py
```
*Outputs Created:*
- Summary table: `results/full_monte_carlo_summary.csv`
- Raw trials table: `results/full_monte_carlo_raw_trials.csv`

### Step 5: Generate IEEE Figures and Illustrations
Export all 12 IEEE scientific figures (F01-F12) and 6 research illustrations (I1-I6) in PDF, SVG, and PNG:
```bash
python scripts/generate_figures.py
```
*Outputs Created:*
- 54 image/vector files in `figures/`
- Manifest: `figures/figure_manifest.csv`

### Step 6: Render High-Definition Demonstration Video
Render the 1920x1080 30-second multi-panel animation:
```bash
python scripts/render_video.py
```
*Outputs Created:*
- Video: `videos/localization_demo.mp4`
- Preview image: `videos/localization_demo_preview.png`
- Metadata: `videos/video_metadata.json`

---

## 3. Seed Determinism & File Integrity
- All experiments use explicit `np.random.RandomState(seed)` seeds (nominal seed = 42, Monte Carlo seed base = 2000).
- Re-running the scripts regenerates identical numerical figures, CSV metrics, and video frames bit-for-bit.
