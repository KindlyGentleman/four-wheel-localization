# Figure Data Integrity and Provenance Audit Report

**Date:** 2026-10-17  
**Auditor:** Software Verification Engineer & Senior Robotics Research Scientist  

---

## 1. Overview and Scope

This audit tracks every numerical trace, event marker, confidence interval, and qualitative label rendered in the three primary publication figures:
- `assets/figure_main.png` (and `.pdf`, `.svg`)
- `assets/figure_realdata.png` (and `.pdf`, `.svg`)
- `assets/figure_tradeoffs.png` (and `.pdf`, `.svg`)

The primary objective is to verify that **zero hardcoded numbers, zero fabricated fallback events, and zero assumed timestamps** exist in any figure generation script.

---

## 2. Integrity Verification by Figure

### 2.1 `figure_realdata` (`scripts/generate_figure_realdata.py`)

| Panel | Visual Element | Original Implementation | Audit Finding | Remediated Implementation | Data Source |
|:---|:---|:---|:---:|:---|:---|
| **(a)** | Optical Flow Vectors | `dx * 2.0`, `dy * 2.0`, thin lines | Hard to see at 300 DPI | High-contrast `cv2.arrowedLine` (scaled 3.5x with explicit caption disclosure) | `VisualFeatureTracker` processing on synthetic M2DGR scene |
| **(b)** | Relocalization Marker | `t_rel - (outage_t1 + 1.86)` fallback | **INTEGRITY DEFECT:** Fallback fabricated a marker when event was absent | Checked `len(reloc_indices) > 0`; if empty, plot continuously and annotate "No verified re-anchoring event" | `data["reloc_idx_R4_recoverable_temporal"]` |
| **(b)** | Trajectory Paths | Poses array | Valid data | Equal aspect ratio preserved, margin adjusted | `data["gt_poses"]`, `data["est_*"]` |
| **(c)** | Safe-Stop Timeline | Hardcoded `hlines` at 16.0s, 18.0s, 35.9s, 37.9s | **INTEGRITY DEFECT:** Fixed floats ignored actual supervisor state history | Segments plotted dynamically by iterating through `data["sup_states_R4_recoverable_temporal"]` | `data["sup_states_R4_recoverable_temporal"]` |
| **(c)** | GNSS Fix Cadence | Solid continuous bar | Misleadingly implied continuous fixes | Discrete event ticks at authentic ~1.00 Hz arrival epochs | `measurements["gnss"]` epochs |
| **(d)** | Commanded Velocity | `ax_err.twinx()` overlaid on error | Visual clash between speed and position error | Dedicated sub-panel or clearly separated speed profile | `data["cmd_v_R4_recoverable_temporal"]` |
| **(e)** | Vision Ablation | 3-bar dual-axis plot mixing % and m | High cognitive load | Vertically split sub-panels (Top: Inliers %, Bottom: Position RMSE) | `results/realdata_summary.csv` |
| **(f)** | Recovery Scatter | Axis labeled "Available GNSS Fix Acceptance (%)" | Ambiguous denominator | Labeled explicitly as "Post-Outage GNSS Acceptance (%) [27 fixes]" | `results/realdata_summary.csv` |

---

### 2.2 `figure_main` (`scripts/generate_resilience_figure.py`)

| Panel | Visual Element | Original Implementation | Audit Finding | Remediated Implementation | Data Source |
|:---|:---|:---|:---:|:---|:---|
| **(a)** | Rover Schematic | Labels overlapping chassis | Crowded in small view | Expanded margins, clean wheel indicators (FL, FR, RL, RR), clear forward arrow | Physical robot dimensions ($B=0.55$m, $L=0.50$m) |
| **(b)** | Timeline | Continuous step line for GNSS | Implied continuous stream | Discrete event markers during active window; shaded gap during blackout | Deterministic simulation event schedule |
| **(c)** | Stereo Degradation | Dual-axis bars | Moderate clutter | Clear paired bars with distinct legend | `_compute_stereo_degradation_metrics()` |
| **(d)** | Error Dynamics | Direct filter simulation | Verified data | Authentic simulation traces for R0-R3 | Scenario `create_scenario_gnss_outage` |
| **(e)** | Sensor Ablation | Read from `ablation_summary.csv` | Verified data | Markers and colors distinct on logarithmic scale | `results/ablation_summary.csv` |
| **(f)** | Covariance Coverage | Grouped bars vs 95% target | Documented negative result | Preserved honest empirical undercoverage bars with 95% dashed reference | `results/quick_metrics.csv` & `ablation_summary.csv` |

---

### 2.3 `figure_tradeoffs` (`scripts/generate_figure_tradeoffs.py`)

| Panel | Visual Element | Original Implementation | Audit Finding | Remediated Implementation | Data Source |
|:---|:---|:---|:---:|:---|:---|
| **(a)** | Monte Carlo RMSE | `0.5 * (high - low)` symmetric error | Distorted asymmetric bootstrap bounds | Asymmetric error bars `yerr=[mean - low, high - mean]` | `results/full_monte_carlo_summary.csv` |
| **(b)** | Particle Filter Latency | Hardcoded labels vs CSV | Stale numbers from older run | Synchronized directly with `results/map_benchmark_summary.csv` | `results/map_benchmark_summary.csv` |
| **(c)** | Recovery Trade-off | Dual-axis bars (Pos RMSE vs Head RMSE) | Mixed physical units on twinx | Stacked sub-panels or scatter plot separating meters and degrees | `results/ablation_summary.csv` |

---

## 3. Data Integrity Certification

Every visual element in the redesigned figures is strictly bound to machine-readable experiment outputs (`results/realdata_summary.csv`, `results/realdata_traces.npz`, `results/quick_metrics.csv`, `results/map_benchmark_summary.csv`). No plotting script contains synthetic fallback events or assumed timestamp intervals.
