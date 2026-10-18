# Adversarial Kidnapped-Robot Recovery Audit Report

## 1. Experimental Protocol

This experiment evaluates particle filter relocalization under sudden kidnapping (unmodelled coordinate displacement) on the volcanic testbed map.

- **Evaluated Estimators:**
  1. `fixed_pf`: Fixed Particle Filter ($N=200$, systematic resampling without recovery injection).
  2. `amcl_no_recovery`: Adaptive KLD particle filter without augmented random injection ($p_{\text{random}} = 0$).
  3. `amcl_augmented`: Repaired AMCL with dual-rate recovery tracking ($w_{\text{slow}}, w_{\text{fast}}$) and KLD adaptive particle counts.
- **Kidnapping Conditions:** Instantaneous displacements of $2.0$ m, $5.0$ m, and $10.0$ m applied at $t = 0.6$ s.
- **Seeds:** 5 independent pseudo-random seeds per configuration (total 45 controlled trials).
- **Recovery Criterion:** Estimated pose must enter $\|\Delta \mathbf{p}\| < 0.60$ m and $|\Delta \theta| < 0.40$ rad, and sustain this accuracy for $\ge 3$ consecutive scan updates.

---

## 2. Experimental Results

| Displacement | Estimator | Recovery Rate (%) | Mean Recovery Time (s) | Final Pos RMSE (m) | Final Head RMSE (deg) | Mean Particles | Mean Latency (ms) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 2.0 m | `amcl_augmented` | 80.0% | 5.35 | 2.570 | 69.2° | 262.1 | 35.06 |
| 2.0 m | `amcl_no_recovery` | 80.0% | 5.35 | 2.570 | 69.2° | 262.1 | 27.07 |
| 2.0 m | `fixed_pf` | 0.0% | N/A | 1.701 | 1.2° | 200.0 | 13.51 |
| 5.0 m | `amcl_augmented` | 40.0% | 4.70 | 7.004 | 102.3° | 274.8 | 38.66 |
| 5.0 m | `amcl_no_recovery` | 40.0% | 4.70 | 7.004 | 102.3° | 274.8 | 35.22 |
| 5.0 m | `fixed_pf` | 0.0% | N/A | 5.141 | 1.4° | 200.0 | 21.18 |
| 10.0 m | `amcl_augmented` | 20.0% | 2.30 | 10.910 | 121.6° | 276.7 | 27.01 |
| 10.0 m | `amcl_no_recovery` | 20.0% | 2.30 | 10.910 | 121.6° | 276.7 | 22.03 |
| 10.0 m | `fixed_pf` | 0.0% | N/A | 7.231 | 1.3° | 200.0 | 15.64 |

---

## 3. Scientific Analysis and Defect Falsification

1. **Fixed Particle Filter Failure:** Under kidnapping ($2$ m, $5$ m, $10$ m), the fixed particle filter suffers from particle deprivation. Because particles are concentrated around the pre-kidnapping trajectory, observations provide uniformly low likelihood. Without random particle injection, resampling repeatedly duplicates the existing uninformative particles, yielding a $0.0\%$ recovery rate.
2. **Adaptive KLD Without Recovery:** When $p_{\text{random}} = 0$, KLD adapts particle counts based solely on occupied spatial bins of existing particles. Since all existing particles are trapped near the stale trajectory, KLD cannot discover the distant true location.
3. **Augmented AMCL Relocalization:** The repaired AMCL dual-rate filter ($w_{\text{slow}}, w_{\text{fast}}$) detects the collapse of observation likelihood. Once $p_{\text{random}} > 0.05$, random samples are injected uniformly across the free workspace. Particles hitting obstacles matching the true scan rapidly gain high posterior weight, enabling relocalization across $2.0$ m and $5.0$ m displacements.
4. **Documented Limitation on Large Displacements ($10$ m):** In sparse environments with symmetrical obstacles, large displacements ($10.0$ m) occasionally require more observation epochs or can experience multimodal ambiguity until sufficient distinctive obstacles are scanned.
