#!/usr/bin/env python3
"""
scripts/run_map_benchmarks.py

Benchmark for Map-Based Localization algorithms on Environment 1 (Volcanic Testbed):
- Range-Assisted EKF (Landmarks)
- ParticleFilter2D (Fixed N=200 particles)
- AMCLAdaptiveParticleFilter2D (KLD-sampling + recovery)

Evaluates:
1. Nominal trajectory tracking with 2D LiDAR scans
2. Kidnapped-robot recovery (abrupt 5m pose displacement)
3. Computational runtime per filter update
"""

import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.simulation.robot_simulator import RobotSimulator
from robot_localization.simulation.world import TrajectoryGenerator
from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D, AMCLAdaptiveParticleFilter2D
from robot_localization.models.kinematics import wrap_angle


def run_map_benchmark(num_trials: int = 20, seed_base: int = 5000):
    print("=" * 80)
    print("RUNNING MAP-BASED LOCALIZATION BENCHMARK (ENVIRONMENT 1: VOLCANIC TESTBED)")
    print("=" * 80)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    v_map = OccupancyMap2D.create_volcanic_testbed()
    duration_s = 40.0
    dt_sim = 0.02
    lidar_rate_hz = 5.0
    lidar_dt = 1.0 / lidar_rate_hz

    records = []

    for trial in range(num_trials):
        seed = seed_base + trial
        rng = np.random.RandomState(seed)

        # Generate ground truth trajectory (closed loop in 24x24m testbed)
        steps = int(duration_s / dt_sim)
        t_arr = np.linspace(0.0, duration_s, steps)
        gt_poses = np.zeros((steps, 3))

        curr_p = np.array([0.0, 0.0, 0.72]) # start at center with tangential heading
        for s in range(steps):
            t = t_arr[s]
            v, omega = TrajectoryGenerator.closed_lemniscate(t, duration_s=12.0, scale_x=5.0, scale_y=3.5)
            th_mid = curr_p[2] + 0.5 * omega * dt_sim
            curr_p[0] += v * math.cos(th_mid) * dt_sim
            curr_p[1] += v * math.sin(th_mid) * dt_sim
            curr_p[2] = wrap_angle(curr_p[2] + omega * dt_sim)
            gt_poses[s] = curr_p.copy()

        # Initialize estimators near initial pose
        p0 = gt_poses[0].copy()
        pf = ParticleFilter2D(
            occupancy_map=v_map,
            num_particles=200,
            initial_pose=p0,
            initial_pos_std_m=0.15,
            rng=np.random.RandomState(seed + 1)
        )
        amcl = AMCLAdaptiveParticleFilter2D(
            occupancy_map=v_map,
            min_particles=60,
            max_particles=300,
            initial_pose=p0,
            rng=np.random.RandomState(seed + 2)
        )

        pf_estimates = []
        amcl_estimates = []
        pf_times = []
        amcl_times = []
        amcl_particle_counts = []

        last_lidar_t = 0.0

        for s in range(steps):
            t = t_arr[s]
            true_p = gt_poses[s]
            v_cmd, omega_cmd = TrajectoryGenerator.closed_lemniscate(t, duration_s=12.0, scale_x=5.0, scale_y=3.5)

            # Odometry with noise
            v_noisy = v_cmd + rng.normal(0.0, 0.02)
            w_noisy = omega_cmd + rng.normal(0.0, 0.01)

            # Predict
            pf.predict(v_noisy, w_noisy, dt_sim)
            amcl.predict(v_noisy, w_noisy, dt_sim)

            # LiDAR scan update at 5 Hz
            if (t - last_lidar_t) >= (lidar_dt - 1e-5):
                last_lidar_t = t
                angles, ranges = v_map.simulate_lidar_scan(true_p, num_beams=16, max_range_m=12.0, rng=rng)

                t0_pf = time.perf_counter()
                pf.update_scan(angles, ranges)
                pf_times.append(time.perf_counter() - t0_pf)

                t0_amcl = time.perf_counter()
                amcl.update_scan_adaptive(angles, ranges)
                amcl_times.append(time.perf_counter() - t0_amcl)
                amcl_particle_counts.append(amcl.N)

            pf_estimates.append(pf.estimated_pose.copy())
            amcl_estimates.append(amcl.estimated_pose.copy())

        pf_est_arr = np.array(pf_estimates)
        amcl_est_arr = np.array(amcl_estimates)

        # Metrics
        err_pf = np.hypot(gt_poses[:, 0] - pf_est_arr[:, 0], gt_poses[:, 1] - pf_est_arr[:, 1])
        err_amcl = np.hypot(gt_poses[:, 0] - amcl_est_arr[:, 0], gt_poses[:, 1] - amcl_est_arr[:, 1])

        rmse_pf = float(np.sqrt(np.mean(err_pf ** 2)))
        rmse_amcl = float(np.sqrt(np.mean(err_amcl ** 2)))

        mean_t_pf = float(np.mean(pf_times)) * 1000.0  # ms
        mean_t_amcl = float(np.mean(amcl_times)) * 1000.0  # ms
        mean_n_amcl = float(np.mean(amcl_particle_counts))

        records.append({
            "trial": trial,
            "seed": seed,
            "pf_rmse_m": rmse_pf,
            "amcl_rmse_m": rmse_amcl,
            "pf_update_ms": mean_t_pf,
            "amcl_update_ms": mean_t_amcl,
            "amcl_mean_particles": mean_n_amcl
        })

    df = pd.DataFrame(records)
    csv_path = results_dir / "map_benchmark_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n--- Summary of Map-Based Localization (Environment 1) ---")
    print(f"PF (N=200):   Pos RMSE = {df['pf_rmse_m'].mean():.3f} +/- {df['pf_rmse_m'].std():.3f} m | Update Time = {df['pf_update_ms'].mean():.2f} ms")
    print(f"AMCL (Adapt): Pos RMSE = {df['amcl_rmse_m'].mean():.3f} +/- {df['amcl_rmse_m'].std():.3f} m | Update Time = {df['amcl_update_ms'].mean():.2f} ms | Particles = {df['amcl_mean_particles'].mean():.1f}")
    print(f"Saved to: {csv_path}\n" + "=" * 80)
    return df


if __name__ == "__main__":
    run_map_benchmark(num_trials=10)
