"""
robot_localization.estimation.particle_filter

Map-Based 2D Particle Filtering and AMCL-Inspired Adaptive Localization:
- OccupancyMap2D: 2D static environment with Euclidean Distance Transform (EDT)
  for O(1) scan likelihood field evaluation.
- ParticleFilter2D (A6): Fixed-count Sequential Monte Carlo pose estimator [x, y, theta]^T
  with circular Riemannian heading statistics, kinematic motion model, and systematic resampling.
- AMCLAdaptiveParticleFilter2D (A7): Adaptive particle filter incorporating KLD-sampling
  for dynamic particle count adjustment and dual-rate (w_slow, w_fast) augmented random
  particle injection for kidnapped-robot recovery.
"""

import math
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
from scipy import ndimage

from robot_localization.models.kinematics import wrap_angle

def wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angles), np.cos(angles))


class OccupancyMap2D:
    """
    2D static obstacle and occupancy map with precomputed Euclidean Distance Transform.
    """

    def __init__(
        self,
        x_min: float = -12.0,
        x_max: float = 12.0,
        y_min: float = -12.0,
        y_max: float = 12.0,
        resolution_m: float = 0.10
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.res = resolution_m

        self.width_cells = int(math.ceil((x_max - x_min) / resolution_m))
        self.height_cells = int(math.ceil((y_max - y_min) / resolution_m))

        # Binary occupancy grid: 1 = obstacle, 0 = free
        self.grid = np.zeros((self.height_cells, self.width_cells), dtype=bool)

        # Distance transform in meters: distance from cell center to nearest obstacle
        self.dist_map = np.zeros((self.height_cells, self.width_cells), dtype=float)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices (col, row)."""
        col = int((x - self.x_min) / self.res)
        row = int((y - self.y_min) / self.res)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (meters)."""
        x = self.x_min + (col + 0.5) * self.res
        y = self.y_min + (row + 0.5) * self.res
        return x, y

    def add_boundary_walls(self, thickness_cells: int = 2):
        """Add impassable boundary walls around perimeter."""
        self.grid[:thickness_cells, :] = True
        self.grid[-thickness_cells:, :] = True
        self.grid[:, :thickness_cells] = True
        self.grid[:, -thickness_cells:] = True

    def add_circular_obstacle(self, cx: float, cy: float, radius_m: float):
        """Add a circular obstacle (e.g. volcanic boulder or tree trunk)."""
        col_c, row_c = self.world_to_grid(cx, cy)
        r_cells = int(math.ceil(radius_m / self.res))

        r_min = max(0, row_c - r_cells)
        r_max = min(self.height_cells, row_c + r_cells + 1)
        c_min = max(0, col_c - r_cells)
        c_max = min(self.width_cells, col_c + r_cells + 1)

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                wx, wy = self.grid_to_world(c, r)
                if math.hypot(wx - cx, wy - cy) <= radius_m:
                    self.grid[r, c] = True

    def add_rectangular_obstacle(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Add rectangular obstacle (e.g. wall segment, stairs, or stone bench)."""
        c1, r1 = self.world_to_grid(x_min, y_min)
        c2, r2 = self.world_to_grid(x_max, y_max)
        col_lo, col_hi = max(0, min(c1, c2)), min(self.width_cells, max(c1, c2) + 1)
        row_lo, row_hi = max(0, min(r1, r2)), min(self.height_cells, max(r1, r2) + 1)
        self.grid[row_lo:row_hi, col_lo:col_hi] = True

    def compute_distance_transform(self):
        """Compute exact Euclidean Distance Transform in meters for fast likelihood lookup."""
        # ndimage.distance_transform_edt computes distance to zero-valued cells
        # So we invert: 0 = obstacle, 1 = free space
        inv_grid = ~self.grid
        edt_cells = ndimage.distance_transform_edt(inv_grid)
        self.dist_map = edt_cells * self.res

    def lookup_distances(self, pts_x: np.ndarray, pts_y: np.ndarray) -> np.ndarray:
        """
        Fast vectorized lookup of distance to nearest obstacle for N points.
        Points outside the map bounds receive a penalty distance.
        """
        cols = np.floor((pts_x - self.x_min) / self.res).astype(int)
        rows = np.floor((pts_y - self.y_min) / self.res).astype(int)

        valid = (cols >= 0) & (cols < self.width_cells) & (rows >= 0) & (rows < self.height_cells)

        dists = np.full(len(pts_x), 10.0) # Penalty for out of bounds
        cols_clamped = np.clip(cols, 0, self.width_cells - 1)
        rows_clamped = np.clip(rows, 0, self.height_cells - 1)

        dists[valid] = self.dist_map[rows_clamped[valid], cols_clamped[valid]]
        return dists

    def simulate_lidar_scan(
        self,
        pose: np.ndarray,
        num_beams: int = 16,
        max_range_m: float = 12.0,
        noise_std_m: float = 0.03,
        rng: Optional[np.random.RandomState] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Raycast 2D LiDAR scan from given robot pose.

        Returns:
            beam_angles_rel: (K,) relative bearing angles in radians
            ranges: (K,) measured ranges in meters
        """
        if rng is None:
            rng = np.random.RandomState()

        px, py, theta = float(pose[0]), float(pose[1]), float(pose[2])
        beam_angles_rel = np.linspace(-math.pi, math.pi, num_beams, endpoint=False)
        ranges = np.zeros(num_beams)

        step_m = self.res * 0.5
        max_steps = int(math.ceil(max_range_m / step_m))

        for k, rel_angle in enumerate(beam_angles_rel):
            ray_theta = theta + rel_angle
            c_ray = math.cos(ray_theta)
            s_ray = math.sin(ray_theta)

            r = max_range_m
            for s in range(1, max_steps):
                curr_dist = s * step_m
                rx = px + curr_dist * c_ray
                ry = py + curr_dist * s_ray

                col, row = self.world_to_grid(rx, ry)
                if not (0 <= col < self.width_cells and 0 <= row < self.height_cells):
                    r = curr_dist
                    break
                if self.grid[row, col]:
                    r = curr_dist
                    break

            # Add noise if not max range
            if r < max_range_m:
                r += rng.normal(0.0, noise_std_m)
                r = float(np.clip(r, 0.1, max_range_m))

            ranges[k] = r

        return beam_angles_rel, ranges

    @classmethod
    def create_volcanic_testbed(cls) -> "OccupancyMap2D":
        """
        Create a 24m x 24m testbed inspired by PRAWIRA's volcanic environment
        (Evita et al. 2022) with boulder clusters, trees, and perimeter boundaries.
        """
        m = cls(x_min=-12.0, x_max=12.0, y_min=-12.0, y_max=12.0, resolution_m=0.10)
        m.add_boundary_walls(thickness_cells=2)

        # Volcanic boulders (Stones)
        m.add_circular_obstacle(cx=4.0, cy=4.0, radius_m=0.8)
        m.add_circular_obstacle(cx=-4.0, cy=5.0, radius_m=1.0)
        m.add_circular_obstacle(cx=-5.0, cy=-4.0, radius_m=0.9)
        m.add_circular_obstacle(cx=5.0, cy=-5.0, radius_m=0.7)

        # Tree trunks (Trees)
        m.add_circular_obstacle(cx=0.0, cy=7.0, radius_m=0.4)
        m.add_circular_obstacle(cx=0.0, cy=-7.0, radius_m=0.4)
        m.add_circular_obstacle(cx=7.0, cy=0.0, radius_m=0.4)
        m.add_circular_obstacle(cx=-7.0, cy=0.0, radius_m=0.4)

        # Rock clusters
        m.add_rectangular_obstacle(x_min=1.5, x_max=2.5, y_min=-1.5, y_max=-0.5)
        m.add_rectangular_obstacle(x_min=-2.5, x_max=-1.5, y_min=1.5, y_max=2.5)

        m.compute_distance_transform()
        return m


class ParticleFilter2D:
    """
    Fixed-count 2D Particle Filter (A6) with circular statistics and systematic resampling.
    """

    def __init__(
        self,
        occupancy_map: OccupancyMap2D,
        num_particles: int = 200,
        initial_pose: Optional[np.ndarray] = None,
        initial_pos_std_m: float = 0.20,
        initial_head_std_rad: float = 0.05,
        laser_sigma_hit_m: float = 0.25,
        laser_z_hit: float = 0.85,
        laser_z_rand: float = 0.15,
        rng: Optional[np.random.RandomState] = None
    ):
        self.map = occupancy_map
        self.N = num_particles
        self.rng = rng if rng is not None else np.random.RandomState()

        self.sigma_hit = laser_sigma_hit_m
        self.z_hit = laser_z_hit
        self.z_rand = laser_z_rand

        # Particles: (N, 3) -> [x, y, theta]
        self.particles = np.zeros((self.N, 3))
        p0 = np.zeros(3) if initial_pose is None else np.array(initial_pose, dtype=float)

        self.particles[:, 0] = p0[0] + self.rng.normal(0.0, initial_pos_std_m, self.N)
        self.particles[:, 1] = p0[1] + self.rng.normal(0.0, initial_pos_std_m, self.N)
        self.particles[:, 2] = wrap_angle_array(p0[2] + self.rng.normal(0.0, initial_head_std_rad, self.N))

        self.weights = np.ones(self.N) / float(self.N)

        # State estimates
        self.estimated_pose = p0.copy()
        self.estimated_cov = np.eye(3) * (initial_pos_std_m ** 2)
        self.n_eff = float(self.N)

    def predict(
        self,
        v_enc: float,
        omega_gyro: float,
        dt: float,
        std_v: float = 0.05,
        std_omega: float = 0.02
    ):
        """Propagate particles through noisy kinematic motion model."""
        noise_v = self.rng.normal(0.0, std_v, self.N)
        noise_w = self.rng.normal(0.0, std_omega, self.N)

        v_eff = v_enc + noise_v
        w_eff = omega_gyro + noise_w

        th = self.particles[:, 2]
        th_mid = th + 0.5 * w_eff * dt

        self.particles[:, 0] += v_eff * np.cos(th_mid) * dt
        self.particles[:, 1] += v_eff * np.sin(th_mid) * dt
        self.particles[:, 2] = wrap_angle_array(th + w_eff * dt)

    def compute_scan_weights(
        self,
        beam_angles_rel: np.ndarray,
        ranges: np.ndarray,
        max_range_m: float = 12.0
    ) -> float:
        """
        Update particle weights using likelihood field observation model and compute estimate.
        Does NOT perform resampling.
        Returns average sample likelihood.
        """
        # Filter valid beams (shorter than max_range)
        valid = (ranges > 0.2) & (ranges < (max_range_m - 0.1))
        if not np.any(valid):
            return 1.0

        v_angles = beam_angles_rel[valid]
        v_ranges = ranges[valid]
        n_beams = len(v_ranges)

        # Compute log-likelihood for each particle
        log_weights = np.zeros(self.N)
        norm_const = 1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_hit)
        rand_prob = self.z_rand / max_range_m

        for i in range(self.N):
            px = self.particles[i, 0]
            py = self.particles[i, 1]
            th = self.particles[i, 2]

            # Ray endpoints in world frame
            ray_th = th + v_angles
            rx = px + v_ranges * np.cos(ray_th)
            ry = py + v_ranges * np.sin(ray_th)

            # Distances to nearest obstacle
            dists = self.map.lookup_distances(rx, ry)

            # Gaussian hit + random uniform mixture
            p_hit = norm_const * np.exp(-0.5 * (dists / self.sigma_hit) ** 2)
            p_beam = self.z_hit * p_hit + rand_prob
            p_beam = np.maximum(p_beam, 1e-12)

            log_weights[i] = np.sum(np.log(p_beam))

        # Absolute observation likelihood metric (geometric mean probability per beam).
        # Bounded and numerical underflow-resistant; drops by orders of magnitude when kidnapped.
        per_beam_likelihood = np.exp(log_weights / max(1, n_beams))
        avg_likelihood = float(np.mean(per_beam_likelihood))

        # Sequential Bayesian importance weight update:
        # log w_k^(i) = log w_{k-1}^(i) + log p(z_k | x_k^(i))
        prior_log_w = np.log(np.maximum(self.weights, 1e-300))
        comb_log_weights = prior_log_w + log_weights

        # Log-sum-exp normalization
        max_log = np.max(comb_log_weights)
        weights_unnorm = np.exp(comb_log_weights - max_log)
        total_w = np.sum(weights_unnorm)

        if total_w > 1e-12:
            self.weights = weights_unnorm / total_w
        else:
            self.weights = np.ones(self.N) / float(self.N)

        # Update estimates and calculate effective sample size
        self._compute_estimate()
        self.n_eff = 1.0 / float(np.sum(self.weights ** 2))

        return avg_likelihood

    def update_scan(
        self,
        beam_angles_rel: np.ndarray,
        ranges: np.ndarray,
        max_range_m: float = 12.0
    ) -> float:
        """
        Update particle weights using likelihood field observation model and
        perform systematic resampling if effective sample size drops below N / 2.
        Returns average sample likelihood.
        """
        avg_likelihood = self.compute_scan_weights(beam_angles_rel, ranges, max_range_m)

        if self.n_eff < (self.N / 2.0):
            self.resample_systematic()

        return avg_likelihood

    def _compute_estimate(self):
        """Compute circular mean and covariance from weighted particles."""
        x_mean = float(np.sum(self.weights * self.particles[:, 0]))
        y_mean = float(np.sum(self.weights * self.particles[:, 1]))

        sin_sum = float(np.sum(self.weights * np.sin(self.particles[:, 2])))
        cos_sum = float(np.sum(self.weights * np.cos(self.particles[:, 2])))
        th_mean = math.atan2(sin_sum, cos_sum)

        self.estimated_pose = np.array([x_mean, y_mean, th_mean])

        # Covariance
        dx = self.particles[:, 0] - x_mean
        dy = self.particles[:, 1] - y_mean
        dth = np.zeros(self.N)
        for i in range(self.N):
            dth[i] = wrap_angle(self.particles[i, 2] - th_mean)

        cov = np.zeros((3, 3))
        cov[0, 0] = np.sum(self.weights * (dx ** 2))
        cov[0, 1] = np.sum(self.weights * dx * dy)
        cov[1, 0] = cov[0, 1]
        cov[1, 1] = np.sum(self.weights * (dy ** 2))
        cov[2, 2] = np.sum(self.weights * (dth ** 2))
        self.estimated_cov = cov

    def resample_systematic(self):
        """Low-variance systematic resampling."""
        positions = (self.rng.random() + np.arange(self.N)) / float(self.N)
        indexes = np.zeros(self.N, dtype=int)
        cumsum = np.cumsum(self.weights)

        i, j = 0, 0
        while i < self.N:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        self.particles = self.particles[indexes].copy()
        self.weights = np.ones(self.N) / float(self.N)


class AMCLAdaptiveParticleFilter2D(ParticleFilter2D):
    """
    AMCL-Inspired Adaptive Particle Filter (A7):
    - KLD-Sampling: adjusts particle count N between N_min and N_max
    - Augmented Recovery: dual-rate (w_slow, w_fast) random particle injection for kidnapped-robot recovery
    """

    def __init__(
        self,
        occupancy_map: OccupancyMap2D,
        min_particles: int = 50,
        max_particles: int = 400,
        initial_pose: Optional[np.ndarray] = None,
        alpha_slow: float = 0.001,
        alpha_fast: float = 0.1,
        rng: Optional[np.random.RandomState] = None
    ):
        super().__init__(
            occupancy_map=occupancy_map,
            num_particles=max_particles,
            initial_pose=initial_pose,
            rng=rng
        )
        self.N_min = min_particles
        self.N_max = max_particles

        # Augmented recovery parameters
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.w_slow = 0.0
        self.w_fast = 0.0
        self.p_random = 0.0

        # Discretization bin sizes for KLD-sampling
        self.bin_xy_m = 0.50
        self.bin_th_rad = math.radians(15.0)

    def update_scan_adaptive(
        self,
        beam_angles_rel: np.ndarray,
        ranges: np.ndarray,
        max_range_m: float = 12.0
    ) -> float:
        """
        Update scan with dual-rate recovery tracking and KLD adaptive resampling.
        Uses compute_scan_weights directly, avoiding premature systematic resampling
        and preserving authentic posterior weights for KLD sampling.
        """
        avg_likelihood = self.compute_scan_weights(beam_angles_rel, ranges, max_range_m)

        # Update short and long term filter likelihood averages
        if self.w_slow == 0.0:
            self.w_slow = avg_likelihood
            self.w_fast = avg_likelihood
        else:
            self.w_slow += self.alpha_slow * (avg_likelihood - self.w_slow)
            self.w_fast += self.alpha_fast * (avg_likelihood - self.w_fast)

        # Recovery injection probability
        if self.w_slow > 1e-12:
            self.p_random = max(0.0, 1.0 - (self.w_fast / self.w_slow))
        else:
            self.p_random = 0.0

        # Apply single KLD-sampling step if effective sample size drops below threshold
        if self.n_eff < (self.N / 2.0):
            self._kld_resample_and_inject()

        return avg_likelihood

    def _kld_resample_and_inject(self):
        """KLD adaptive particle count selection and recovery injection."""
        bins_occupied = set()
        new_particles = []

        # Target particle count initialized to N_min
        N_target = self.N_min
        i = 0

        # Resample with replacement until N_target is reached
        while len(new_particles) < N_target and len(new_particles) < self.N_max:
            # Check random recovery injection
            if self.rng.random() < self.p_random:
                # Uniform random particle in map bounds
                rx = self.rng.uniform(self.map.x_min + 1.0, self.map.x_max - 1.0)
                ry = self.rng.uniform(self.map.y_min + 1.0, self.map.y_max - 1.0)
                rth = self.rng.uniform(-math.pi, math.pi)
                p = np.array([rx, ry, rth])
            else:
                # Roulette sample from current particles
                idx = self.rng.choice(self.N, p=self.weights)
                p = self.particles[idx].copy()
                # Slight diffusion
                p[0] += self.rng.normal(0.0, 0.05)
                p[1] += self.rng.normal(0.0, 0.05)
                p[2] = wrap_angle(p[2] + self.rng.normal(0.0, 0.02))

            new_particles.append(p)

            # Binning for KLD-sampling
            bx = int(p[0] / self.bin_xy_m)
            by = int(p[1] / self.bin_xy_m)
            bth = int(p[2] / self.bin_th_rad)
            bins_occupied.add((bx, by, bth))

            k = len(bins_occupied)
            if k > 1:
                # Wilson-Hilferty transformation for chi-square quantile approximation
                # with epsilon = 0.05, 1 - delta = 0.99 (z = 2.326)
                z = 2.326
                epsilon = 0.05
                term = 1.0 - (2.0 / (9.0 * (k - 1))) + math.sqrt(2.0 / (9.0 * (k - 1))) * z
                N_kld = int(math.ceil(((k - 1) / (2.0 * epsilon)) * (term ** 3)))
                N_target = max(self.N_min, min(self.N_max, N_kld))

        self.N = len(new_particles)
        self.particles = np.array(new_particles)
        self.weights = np.ones(self.N) / float(self.N)
        self.n_eff = float(self.N)
        self._compute_estimate()
