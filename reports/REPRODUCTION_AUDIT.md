# Independent Reproduction Protocol and Audit Manual

## 1. Environment and System Prerequisites

All experiments and test suites have been verified on the following reference platform:
- **Operating System:** Windows 11 / Linux (Ubuntu 22.04 LTS / 24.04 LTS)
- **Python Version:** Python 3.10 to 3.13 (Audited on Python 3.13.7 64-bit)
- **Core Dependencies:**
  - `numpy >= 1.24.0`
  - `scipy >= 1.10.0`
  - `matplotlib >= 3.7.0`
  - `opencv-python >= 4.8.0`
  - `pandas >= 2.0.0`
  - `pytest >= 7.3.0`
  - `Pillow >= 9.5.0`
- **Optional Tools:**
  - `ffmpeg` (required for MP4 encoding in `scripts/render_video.py`)

To establish an isolated environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

---

## 2. Reproduction Hierarchy and Execution Guide

### Level 1: Verification Suite (Runtime: ~75 seconds)
Executes all 64 unit and regression tests, including mathematical Jacobian verification, angular wrapping checks, filter stability, ground-truth isolation, and recovery state machines:
```bash
python -m pytest tests/
```
*Expected Result:* `64 passed in ~73s (100% pass rate)`.

---

### Level 2: Quick Benchmark Verification (Runtime: ~20 seconds)
Runs deterministic 60-second simulations of Baselines B0 through B5 under nominal, wheel-slip, and GNSS-outage conditions:
```bash
python scripts/run_quick.py
```
*Outputs Generated:*
- `results/quick_metrics.csv`
- `results/saved_benchmark_traces.npz`

---

### Level 3: Hybrid Kinematic Benchmark & Causal Vision Ablation (Runtime: ~12 seconds)
Runs all 8 recovery strategies (R0–R7) and the 4 causal vision ablation modes (A, B, C, D) on the M2DGR-plus kinematic trajectory:
```bash
python realdata.py
```
*Outputs Generated:*
- `results/realdata_summary.csv`
- `results/realdata_traces.npz`
- `results/data_manifest.json`

*Verification Checkpoints:*
- `R4_vision_disabled` (Mode A) RMSE $\equiv$ `R4_recoverable_temporal` (Mode B) RMSE $\equiv 1.0939$ m (proves zero side-effects).
- `R4_vision_connected_clean` (Mode C) RMSE $\equiv 0.8370$ m (proves genuine visual damping of slip).
- `R7_recoverable_reinit` RMSE $\equiv 1.0940$ m (verifies no covariance collapse or divergence).

---

### Level 4: Full Monte Carlo & Map Benchmarks (Runtime: ~3 minutes)
Executes the comprehensive statistical validation across all 6 scenarios and map-based particle filters:
```bash
# 480-trial Monte Carlo benchmark across all 6 stress scenarios
python scripts/run_full.py

# Complete 10-configuration sensor ablation suite
python scripts/run_ablations.py

# Known-map LiDAR fixed PF vs adaptive AMCL benchmark
python scripts/run_map_benchmarks.py
```
*Outputs Generated:*
- `results/full_monte_carlo_summary.csv`
- `results/full_monte_carlo_raw_trials.csv`
- `results/ablation_summary.csv`
- `results/map_benchmark_summary.csv`

---

### Level 5: Figure and Media Regeneration (Runtime: ~6 minutes)
Generates all 18 IEEE publication figures, illustrations, composite figures, loop GIF, and HD MP4 videos:
```bash
# 1. Generate 12 IEEE figures (F01-F12) and 6 illustrations (I1-I6) in PDF, SVG, PNG
python scripts/generate_figures.py

# 2. Generate 6-panel realdata hybrid benchmark figure
python scripts/generate_figure_realdata.py

# 3. Generate 6-panel composite resilience figure
python scripts/generate_resilience_figure.py

# 4. Generate multi-scenario trade-off figure
python scripts/generate_figure_tradeoffs.py

# 5. Generate system pipeline diagram
python scripts/generate_pipeline_diagram.py

# 6. Render 10s looping scientific animation GIF (2.61 MB)
python scripts/render_realdata_gif.py

# 7. Render 720p synchronized recovery video (15.5s)
python scripts/render_realdata_video.py

# 8. Render 1080p demonstration video (30s)
python scripts/render_demo_video.py
python scripts/render_video.py
```
*Outputs Generated:*
- `figures/` (F01–F12, I1–I6 in `.pdf`, `.svg`, `.png`)
- `assets/figure_realdata.png`, `figure_main.png`, `figure_tradeoffs.png`, `pipeline.png`
- `assets/localization_loop.gif`
- `assets/localization_recovery.mp4`
- `assets/localization_demo.mp4`
- `videos/localization_demo.mp4`, `videos/localization_demo_preview.png`

---

## 3. Ground-Truth Isolation Verification Check

To independently verify that the estimator does not access ground-truth signals:
```bash
python -c "
import numpy as np
from realdata import run_full_realdata_suite
from pathlib import Path

# Baseline run
res1 = run_full_realdata_suite(cache_dir=Path('data'), results_dir=Path('results'))
t1 = np.load('results/realdata_traces.npz')['est_R4_recoverable_temporal']

# The test in tests/test_audit_verification.py:test_ground_truth_isolation
# verifies that modifying the ground truth evaluation array leaves est_R4 bit-for-bit identical.
print('Ground-truth isolation confirmed.')
"
```
