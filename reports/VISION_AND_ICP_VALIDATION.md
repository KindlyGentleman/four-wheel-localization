# Stereo Vision, Optical Flow, and ICP Registration Validation Report

## 1. Overview and Problem Statement

The repository incorporates optical flow, visual odometry, stereo depth triangulation, and 3D point cloud registration.
Prior code review flagged three issues:
1. **F02 (P0):** `register_point_clouds_icp()` returned fabricated fitness (0.80), RMSE (0.08m), and success (True) when Open3D was absent.
2. **F03 (P0):** `tests/test_stereo.py` relied on Open3D, failing in clean CPU environments without optional dependencies.
3. **F13 (P1):** Synthetic camera generation in `realdata.py` ignored vehicle yaw, translating floor tiles in 2D without perspective rotation.

---

## 2. ICP Registration Architecture & SVD Implementation

To eliminate fabricated fallback metrics, a genuine CPU-only registration algorithm was implemented directly in `src/robot_localization/sensors/stereo.py`.

### Mathematical Formulation
Given two 3D point sets $\mathcal{P}_{src} = \{\mathbf{p}_i\}_{i=1}^N$ and $\mathcal{P}_{tgt} = \{\mathbf{q}_j\}_{j=1}^M$ in the robot base frame:
1. A KD-tree (`scipy.spatial.cKDTree`) indexes $\mathcal{P}_{tgt}$ in $\mathcal{O}(M \log M)$ time.
2. For iteration $k = 1, \dots, K_{\max}$:
   - Transform source points: $\mathbf{p}_i^{(k)} = \mathbf{R}^{(k)} \mathbf{p}_i + \mathbf{t}^{(k)}$.
   - Query closest target points $\mathbf{q}_{\pi(i)}$ within distance threshold $d_{\max}$.
   - Reject outlier pairings where $\|\mathbf{p}_i^{(k)} - \mathbf{q}_{\pi(i)}\| \ge d_{\max}$.
   - Compute centroids of matched subsets:
     $$\boldsymbol{\mu}_{src} = \frac{1}{|\mathcal{C}|} \sum_{i \in \mathcal{C}} \mathbf{p}_i^{(k)}, \quad \boldsymbol{\mu}_{tgt} = \frac{1}{|\mathcal{C}|} \sum_{i \in \mathcal{C}} \mathbf{q}_{\pi(i)}$$
   - Form centered cross-dispersion matrix:
     $$\mathbf{H} = \sum_{i \in \mathcal{C}} (\mathbf{p}_i^{(k)} - \boldsymbol{\mu}_{src}) (\mathbf{q}_{\pi(i)} - \boldsymbol{\mu}_{tgt})^T$$
   - Compute singular value decomposition: $\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$.
   - Candidate incremental rotation: $\mathbf{R}_{\Delta} = \mathbf{V} \mathbf{U}^T$.
   - If $\det(\mathbf{R}_{\Delta}) < 0$, apply reflection correction: $\mathbf{V}_{:,-1} \leftarrow -\mathbf{V}_{:,-1}$ and recompute $\mathbf{R}_{\Delta}$.
   - Incremental translation: $\mathbf{t}_{\Delta} = \boldsymbol{\mu}_{tgt} - \mathbf{R}_{\Delta} \boldsymbol{\mu}_{src}$.
   - Accumulate rigid transformation:
     $$\mathbf{R}^{(k+1)} = \mathbf{R}_{\Delta} \mathbf{R}^{(k)}, \quad \mathbf{t}^{(k+1)} = \mathbf{R}_{\Delta} \mathbf{t}^{(k)} + \mathbf{t}_{\Delta}$$
3. Post-convergence metric computation:
   - Genuine inliers count: $N_{\text{inlier}} = \sum \mathbb{I}(\|\mathbf{p}_i^{(\text{final})} - \mathbf{q}_{\pi(i)}\| < d_{\max})$.
   - Fitness ratio: $f = N_{\text{inlier}} / N$.
   - Inlier RMSE: $\text{RMSE} = \sqrt{\frac{1}{N_{\text{inlier}}} \sum_{\text{inliers}} d_i^2}$.
   - Success criterion: $f \ge 0.35 \land \text{RMSE} \le 0.25\text{ m} \land N_{\text{inlier}} \ge 15$.

### Verification Matrix
- **Identity test:** Identical point clouds return $\Delta x = 0.0$, $\Delta y = 0.0$, $\Delta \theta = 0.0$, $\text{RMSE} < 10^{-12}\text{ m}$, $f = 1.0$.
- **Known translation test:** Synthetic cluster displaced by $(+0.06\text{ m}, -0.04\text{ m})$ recovers $(-0.06\text{ m}, +0.04\text{ m})$ with $\text{RMSE} < 0.01\text{ m}$.
- **Rotation and translation test:** Displaced by $4.0^\circ$ yaw and $(0.05\text{ m}, 0.02\text{ m})$ recovers $-4.0^\circ$ and target displacement with $\text{RMSE} < 0.02\text{ m}$.
- **Honest rejection test:** Insufficient points ($N=10$) or non-overlapping clouds separated by $10\text{ m}$ honestly return `success=False`, $f = 0.0$, without fabricating fitness.

---

## 3. Stereo Coordinate Geometry and Disparity Triangulation

1. **Reprojection Matrix $\mathbf{Q}$:**
   $$\mathbf{Q} = \begin{bmatrix} 1 & 0 & 0 & -c_x \\ 0 & 1 & 0 & -c_y \\ 0 & 0 & 0 & f_x \\ 0 & 0 & -1/B & 0 \end{bmatrix}$$
   Where $B = 0.10\text{ m}$ is the baseline, $f_x = 400.0\text{ px}$, and $(c_x, c_y) = (320, 240)$.
2. **Depth Triangulation:**
   For valid rectified scanlines ($v_L \approx v_R$) and positive disparity $d = u_L - u_R > 1.0\text{ px}$:
   $$Z = \frac{f_x B}{d}, \quad X = \frac{(u_L - c_x) Z}{f_x}, \quad Y = \frac{(v_L - c_y) Z}{f_y}$$
3. **Frame Transformation:**
   Camera optical frame ($Z$ forward, $X$ right, $Y$ down) transforms to rover `base_link` ($X$ forward, $Y$ left, $Z$ up) via:
   $$\mathbf{R}_{\text{opt}\to\text{base}} = \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}, \quad \mathbf{P}_{\text{base}} = \mathbf{R}_{\text{opt}\to\text{base}} \mathbf{P}_{\text{opt}} + \mathbf{t}_{\text{mount}}$$
   Determinant of $\mathbf{R}_{\text{opt}\to\text{base}}$ is $+1.0$ (proper rotation, zero reflection).
