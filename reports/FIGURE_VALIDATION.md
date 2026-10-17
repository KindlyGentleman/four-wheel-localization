# Figure and Visualization Scientific Validation Report

## 1. Overview and Problem Statement

Scientific figures are the primary visual evidence presented in research papers.
Prior forensic review identified three major integrity and scaling issues:
1. **F04 (P1):** In `scripts/generate_figure_realdata.py`, KLT inlier ratios stored as fractions $[0, 1]$ (e.g. 0.9719) were plotted on an axis explicitly labeled `(%)` with limit $[0, 115]$, causing bars to appear at 0.97% height instead of 97.2%.
2. **F05 (P1):** Moderate degradation connected vision was not implemented in `realdata.py`. The plotting script silently duplicated the clean-vision result ($0.837\text{ m}$) into the moderate degradation slot.
3. **F06 (P1):** Figure generation scripts contained silent fallback constants (e.g. `else 97.2`, `else 0.837`, `else 0.0`) when source CSV records were missing, masking missing data.

---

## 2. Itemized Remediation & Verification

### 2.1 Inlier Ratio Scaling Correction (F04)
- **Code Modification:** Added explicit unit checking `_to_pct()` in `scripts/generate_figure_realdata.py`. When reading `avg_inlier_ratio`, values $\le 1.0$ are multiplied by $100.0$.
- **Verification:**
  - Clean condition: $0.9719 \to 97.2\%$.
  - Moderate condition: $0.8487 \to 84.9\%$.
  - Severe condition: $0.1498 \to 15.0\%$.
  All bars now accurately match their percentage labels.

### 2.2 Authentic Moderate Degradation Experiment (F05)
- **Code Modification:** Added configuration `("R4_vision_connected_moderate", "recoverable_ekf", RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, "moderate", "connected")` to `realdata.py`.
- **Benchmark Execution:**
  Executed the full hybrid suite. Results recorded in `results/realdata_summary.csv`:
  - `R4_vision_connected_clean`: Position RMSE $0.837\text{ m}$, Peak Outage Error $1.737\text{ m}$, Inlier Ratio $97.2\%$.
  - `R4_vision_connected_moderate`: Position RMSE $0.839\text{ m}$, Peak Outage Error $1.740\text{ m}$, Inlier Ratio $84.9\%$.
  - `R4_vision_connected_severe`: Position RMSE $1.094\text{ m}$, Peak Outage Error $2.101\text{ m}$, Inlier Ratio $15.0\%$.
- **Scientific Significance:** Under moderate visual degradation (blur and contrast drop), tracking remains largely effective ($84.9\%$ inliers), producing $0.839\text{ m}$ RMSE (a slight, physically logical degradation relative to clean $0.837\text{ m}$). Under severe degradation, inliers collapse to $15.0\%$, and the filter safely isolates visual velocity, falling back to the $1.094\text{ m}$ odometry baseline.

### 2.3 Elimination of Silent Hardcoded Fallbacks (F06)
- **Code Modification:** Removed all silent fallback values (`else 97.2`, `else 0.837`, `else 0.0`) in `scripts/generate_figure_realdata.py` and `scripts/generate_figure_tradeoffs.py`. Missing benchmark configurations now raise an explicit `RuntimeError` requiring data regeneration.
- **Verification:** All plotting scripts verified by running against regenerated CSV outputs.

---

## 3. Regenerated Visual Artifacts Manifest

| Figure Filename | Format | Dimensions / DPI | Generating Script | Data Source |
|:---|:---|:---|:---|:---|
| `assets/figure_realdata.png` | PNG | 300 DPI ($2148 \times 2640$) | `scripts/generate_figure_realdata.py` | `results/realdata_summary.csv`, `realdata_traces.npz` |
| `assets/figure_realdata.pdf` | Vector PDF | IEEE 2-column ($7.16 \times 8.8\text{ in}$) | `scripts/generate_figure_realdata.py` | `results/realdata_summary.csv`, `realdata_traces.npz` |
| `assets/figure_realdata.svg` | Vector SVG | Scalable | `scripts/generate_figure_realdata.py` | `results/realdata_summary.csv`, `realdata_traces.npz` |
| `assets/figure_tradeoffs.png`| PNG | 300 DPI ($2148 \times 1860$) | `scripts/generate_figure_tradeoffs.py` | `results/full_monte_carlo_summary.csv`, `map_benchmark_summary.csv`, `ablation_summary.csv` |
| `assets/figure_tradeoffs.pdf`| Vector PDF | IEEE 2-column ($7.16 \times 6.2\text{ in}$) | `scripts/generate_figure_tradeoffs.py` | `results/full_monte_carlo_summary.csv`, `map_benchmark_summary.csv`, `ablation_summary.csv` |
| `assets/figure_tradeoffs.svg`| Vector SVG | Scalable | `scripts/generate_figure_tradeoffs.py` | `results/full_monte_carlo_summary.csv`, `map_benchmark_summary.csv`, `ablation_summary.csv` |
