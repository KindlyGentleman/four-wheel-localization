# Comprehensive Figure Readability and Visual Quality Audit

**Date:** 2026-10-17  
**Auditor:** Scientific Visualization Editor & Independent Scientific Reviewer  
**Target Artifacts:**
- `assets/figure_main.png` (and `.pdf`, `.svg`)
- `assets/figure_realdata.png` (and `.pdf`, `.svg`)
- `assets/figure_tradeoffs.png` (and `.pdf`, `.svg`)

---

## 1. Executive Visual Assessment

A rigorous inspection of the current generated PNG (300 DPI) and rendered vector PDF/SVG outputs reveals that while the files export without syntax errors, several critical readability, data integrity, and visual communication defects are present:

1. **`figure_realdata` Critical Integrity Defects:**
   - Panel (c) utilizes hardcoded transition timestamps (16.0s, 18.0s, 35.9s, 37.9s) rather than dynamically plotting actual supervisor state changes.
   - Panel (b) injects a synthetic fallback re-anchoring marker at $t = t_1 + 1.86$s when no relocalization event is logged.
   - Panel (a) renders tiny 2x flow vectors that are almost invisible on print, while a heavy dark box covers feature tracking areas.
   - Panel (e) conflates percentage inliers with metric RMSE on an overcrowded dual-axis bar chart.
   - Panel (f) labels the x-axis ambiguously as "Available GNSS Fix Acceptance (%)" and crowds overlapping configurations (R4, R5, R7) into a single point.

2. **`figure_main` Readability Defects:**
   - Panel (a) kinematics diagram contains tightly crowded annotations for `base_link`, `GNSS`, and `Stereo FOV`.
   - Panel (b) draws 1 Hz GNSS fixes as a continuous solid step bar, incorrectly implying high-cadence continuous availability.
   - Panel (c) dual-axis bars combine inliers and reprojection error, obscuring the featureless condition.

3. **`figure_tradeoffs` Consistency Defects:**
   - Panel (a) uses symmetric error bars ($0.5 \times (\text{high} - \text{low})$), obscuring empirical skew in bootstrap/student-t confidence bounds.
   - Panel (b) numbers in figures/benchmarks differ from historical README claims (28.4ms/18.5ms vs 22.1ms/14.8ms).
   - Panel (c) clumps Position RMSE [m] and Heading RMSE [deg] onto dual-axis bars, visually implying commensurability across incompatible physical units.

---

## 2. Quantitative Readability Rubric (Scores 0 to 5)

| Criterion | `figure_main` | `figure_realdata` | `figure_tradeoffs` | Notes on Weaknesses |
|:---|:---:|:---:|:---:|:---|
| **Text Legibility** | 3.5 | 3.0 | 3.5 | Subplot tick labels and annotations drop below 6 pt at column width. |
| **Axis and Unit Clarity** | 3.5 | 3.0 | 3.0 | Dual-axis bars in RealData (e) and Tradeoffs (c) obscure units. |
| **Legend Readability** | 3.5 | 3.0 | 3.5 | RealData (b) and (f) legends crowd trajectory data. |
| **Panel Balance** | 4.0 | 3.5 | 4.0 | RealData (b) has excessive empty space due to coordinate bounds. |
| **Color Accessibility** | 4.0 | 4.0 | 4.0 | Palettes are generally distinct, but line styles must assist grayscale. |
| **Marker Distinguishability**| 4.0 | 3.5 | 4.0 | Overplotting in RealData (f) hides individual method markers. |
| **Line Style Distinguishability**| 4.0 | 4.0 | 4.0 | Solid, dashed, and dotted styles are present. |
| **Annotation Correctness** | 3.5 | **2.0** | 3.5 | **Severe:** Hardcoded timeline (16/18/35.9/37.9s) and synthetic relocalization fallback. |
| **Data Integrity** | 4.0 | **2.5** | 3.5 | Timeline and marker fallbacks violate direct data derivation. |
| **Figure-to-Caption Consistency**| 4.0 | 3.5 | 3.5 | README and captions must match actual plotted event counts. |
| **Scientific Interpretability**| 4.0 | 3.0 | 3.5 | Flow vectors and dual-axis comparisons require high cognitive effort. |
| **Print Suitability (IEEE 2-col)**| 3.5 | 3.0 | 3.5 | Multi-axis bars and small text require redesign for 3.5-inch column width. |
| **Average Score** | **3.8 / 5.0** | **3.0 / 5.0** | **3.5 / 5.0** | **Overall Status: REQUIRES REDESIGN** |

---

## 3. Actionable Redesign Specifications

### 3.1 Redesign for `figure_realdata` (`scripts/generate_figure_realdata.py`)
1. **Dynamic Timeline:** Remove hardcoded floats `16.0`, `18.0`, `35.9`, `37.9`. Read `data["sup_states_R4_recoverable_temporal"]` and plot contiguous state segments directly from data.
2. **Honest Relocalization Markers:** Remove `k_reloc = int(np.argmin(np.abs(t_rel - (outage_t1 + 1.86))))`. Plot re-anchoring marker and coordinate offset arrow ONLY if `len(reloc_indices) > 0`. If absent, annotate "No verified re-anchoring event".
3. **KLT Flow Visibility:** Use `cv2.arrowedLine` with 3.5x scaling and tipLength=0.35 in vibrant yellow/cyan; sample 40 well-distributed feature points; position summary text neatly outside the active motion field.
4. **Decouple Panel (e):** Split into two distinct sub-panels or a paired-bar layout:
   - Upper: KLT Inlier Ratio [%] across Clean, Moderate, Severe degradation.
   - Lower: Position RMSE [m] comparing Disconnected vs Connected Optical Flow.
5. **Clean Panel (f):** Clearly label x-axis as "Post-Outage GNSS Acceptance (%) [27 fixes]". Differentiate R4, R5, R7 points.

### 3.2 Redesign for `figure_resilience` (`scripts/generate_resilience_figure.py`)
1. **Panel (a) Spacing:** Increase margins around rover chassis, expand label font size to $\ge 7$ pt, clarify forward vector direction and wheelbase/track labels.
2. **Panel (b) Discrete Fixes:** Plot discrete event markers/ticks for GNSS fixes rather than a continuous solid step bar.
3. **Panel (f) Calibration Reality:** Clearly mark the 95% nominal target and preserve empirical undercoverage bars honestly.

### 3.3 Redesign for `figure_tradeoffs` (`scripts/generate_figure_tradeoffs.py`)
1. **Panel (a) Asymmetric CIs:** Use `yerr=[lower_diff, upper_diff]` to accurately depict asymmetric bootstrap/student-t confidence bounds.
2. **Panel (b) Latency Parity:** Synchronize numbers with the latest `map_benchmark_summary.csv` ($28.5\text{ ms} \pm \dots$ and $18.5\text{ ms} \pm \dots$).
3. **Panel (c) Unit Separation:** Split into two stacked sub-panels (Top: Position RMSE [m], Bottom: Heading RMSE [deg]) to avoid comparing incompatible physical units on a dual-axis bar plot.
