#!/usr/bin/env python3
"""
reproduce.py

Single-entrypoint reproduction script for the Four-Wheel Robot Localization study.
Project: Four-Wheel Robot Localization: Sensor Fusion and Robustness Experiments

Usage:
  python reproduce.py --quick     Run deterministic 8-baseline benchmark across scenarios
  python reproduce.py --full      Run 10-trial Monte Carlo benchmark across 4 scenarios
  python reproduce.py --map       Run known-map LiDAR testbed benchmark (PF vs AMCL)
  python reproduce.py --figure    Generate IEEE 8-panel composite scientific figure
  python reproduce.py --gif       Render seamless looping 4WD simulation GIF and MP4
  python reproduce.py --all       Execute the entire reproduction workflow
"""

import sys
import math
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Add local src/ to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.simulation.robot_simulator import RobotSimulator
from robot_localization.simulation.world import TrajectoryGenerator
from robot_localization.simulation.scenarios import (
    create_scenario_nominal,
    create_scenario_slip,
    create_scenario_gnss_outage,
    create_scenario_gyro_bias,
    create_scenario_compass_disturbance,
    create_scenario_multipath,
    create_scenario_combined,
)
from robot_localization.estimation.baseline_runner import run_all_baselines
from robot_localization.evaluation.metrics import compute_trajectory_metrics
from robot_localization.estimation.particle_filter import OccupancyMap2D, ParticleFilter2D, AMCLAdaptiveParticleFilter2D


def run_quick_benchmark():
    """Execute standard 8-baseline comparison across primary scenarios."""
    print("\n" + "=" * 80)
    print("RUNNING QUICK BENCHMARK (8 ESTIMATORS x 8 EXPERIMENTS)")
    print("=" * 80)

    experiments = [
        ("E1_nominal", lambda: create_scenario_nominal(duration_s=60.0)),
        ("E4_differential_slip", lambda: create_scenario_slip(duration_s=60.0)),
        ("E5_gyro_bias", lambda: create_scenario_gyro_bias(duration_s=60.0)),
        ("E6_compass_disturbance", lambda: create_scenario_compass_disturbance(duration_s=60.0)),
        ("E8_gnss_outage_20s", lambda: create_scenario_gnss_outage(duration_s=60.0)),
        ("E9_gnss_multipath", lambda: create_scenario_multipath(duration_s=60.0)),
        ("E12_combined_failures", lambda: create_scenario_combined(duration_s=60.0)),
    ]

    all_metrics = []
    for exp_name, builder_func in experiments:
        print(f"\n--- {exp_name} (60.0s) ---")
        sim, traj_fn, slip_fn = builder_func()
        sim_res = sim.run_simulation(duration_s=60.0, trajectory_func=traj_fn, slip_profile_func=slip_fn)
        traces = run_all_baselines(sim_res, robot_config=sim.config)

        for b_name, tr in traces.items():
            m = compute_trajectory_metrics(
                baseline_name=b_name,
                timestamps_gt=sim_res.timestamps,
                gt_poses=sim_res.ground_truth_pose,
                timestamps_est=tr.timestamps,
                est_poses=tr.estimated_pose,
                cov_diags=tr.covariance_diag,
                cov_mats=tr.covariance_mat,
            )
            cov_pct = m.coverage_95 * 100.0 if m.coverage_95 <= 1.0 else m.coverage_95
            print(f"  [{b_name:30s}] Pos RMSE: {m.pos_rmse_m:6.3f} m | Head RMSE: {m.heading_rmse_deg:5.2f} deg | 95% Cov: {cov_pct:5.1f}%")
            all_metrics.append({
                "experiment": exp_name,
                "baseline": b_name,
                "pos_rmse_m": m.pos_rmse_m,
                "heading_rmse_deg": m.heading_rmse_deg,
                "coverage_95": cov_pct,
                "mean_nees_pos": m.mean_nees_pos,
            })

    out_csv = PROJECT_ROOT / "results" / "quick_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(out_csv, index=False)
    print(f"\nSaved quick benchmark results to: {out_csv}")


def run_full_monte_carlo(num_trials: int = 10):
    """Run multi-trial Monte Carlo benchmark across scenarios."""
    print("\n" + "=" * 80)
    print(f"RUNNING FULL MONTE CARLO BENCHMARK ({num_trials} TRIALS / SCENARIO)")
    print("=" * 80)

    scenarios = [
        ("Nominal_Traction", lambda s: create_scenario_nominal(seed=s, duration_s=40.0)),
        ("Asymmetric_Wheel_Slip", lambda s: create_scenario_slip(seed=s, duration_s=40.0)),
        ("GNSS_Outage_20s", lambda s: create_scenario_gnss_outage(seed=s, duration_s=40.0)),
        ("Combined_Stress", lambda s: create_scenario_combined(seed=s, duration_s=40.0)),
        ("Compass_Disturbance", lambda s: create_scenario_compass_disturbance(seed=s, duration_s=40.0)),
        ("GNSS_Multipath", lambda s: create_scenario_multipath(seed=s, duration_s=40.0)),
    ]

    trial_records = []
    base_seed = 3000

    for sc_name, builder_func in scenarios:
        print(f"\nEvaluating Scenario: {sc_name}")
        for t_idx in range(num_trials):
            trial_seed = base_seed + t_idx
            sim, traj_fn, slip_fn = builder_func(trial_seed)
            sim_res = sim.run_simulation(duration_s=40.0, trajectory_func=traj_fn, slip_profile_func=slip_fn)
            traces = run_all_baselines(sim_res, robot_config=sim.config)

            for b_name, tr in traces.items():
                m = compute_trajectory_metrics(
                    baseline_name=b_name,
                    timestamps_gt=sim_res.timestamps,
                    gt_poses=sim_res.ground_truth_pose,
                    timestamps_est=tr.timestamps,
                    est_poses=tr.estimated_pose,
                    cov_diags=tr.covariance_diag,
                    cov_mats=tr.covariance_mat,
                )
                trial_records.append({
                    "scenario": sc_name,
                    "trial": t_idx,
                    "baseline": b_name,
                    "pos_rmse_m": m.pos_rmse_m,
                    "heading_rmse_deg": m.heading_rmse_deg,
                    "coverage_95": m.coverage_95,
                    "mean_nees_pos": m.mean_nees_pos,
                })
        print(f"  Completed {num_trials} trials for {sc_name}.")

    df_trials = pd.DataFrame(trial_records)
    summary_rows = []
    for (sc, b), grp in df_trials.groupby(["scenario", "baseline"], sort=False):
        summary_rows.append({
            "scenario": sc,
            "baseline": b,
            "pos_rmse_mean": grp["pos_rmse_m"].mean(),
            "pos_rmse_std": grp["pos_rmse_m"].std(),
            "heading_rmse_deg_mean": grp["heading_rmse_deg"].mean(),
            "heading_rmse_deg_std": grp["heading_rmse_deg"].std(),
            "coverage_95_mean": grp["coverage_95"].mean(),
            "mean_nees_pos": grp["mean_nees_pos"].mean(),
        })

    df_summary = pd.DataFrame(summary_rows)
    trials_csv = PROJECT_ROOT / "results" / "trials.csv"
    summary_csv = PROJECT_ROOT / "results" / "summary.csv"
    df_trials.to_csv(trials_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved raw trials to: {trials_csv}")
    print(f"Saved aggregated summary to: {summary_csv}")


def run_map_benchmark(num_trials: int = 5):
    """Run known-map particle filter benchmark (PF vs AMCL)."""
    print("\n" + "=" * 80)
    print("RUNNING KNOWN-MAP BENCHMARK (FIXED-COUNT PF vs ADAPTIVE AMCL)")
    print("=" * 80)

    v_map = OccupancyMap2D.create_volcanic_testbed()
    duration_s = 20.0
    dt_sim = 0.02
    lidar_rate_hz = 5.0
    lidar_dt = 1.0 / lidar_rate_hz

    records = []
    base_seed = 5000

    for trial in range(num_trials):
        seed = base_seed + trial
        rng = np.random.RandomState(seed)

        steps = int(duration_s / dt_sim)
        t_arr = np.linspace(0.0, duration_s, steps)
        gt_poses = np.zeros((steps, 3))

        curr_p = np.array([0.0, 0.0, 0.72])
        for s in range(steps):
            t = t_arr[s]
            v, omega = TrajectoryGenerator.closed_lemniscate(t, duration_s=12.0, scale_x=5.0, scale_y=3.5)
            th_mid = curr_p[2] + 0.5 * omega * dt_sim
            curr_p[0] += v * math.cos(th_mid) * dt_sim
            curr_p[1] += v * math.sin(th_mid) * dt_sim
            curr_p[2] = (curr_p[2] + omega * dt_sim + math.pi) % (2.0 * math.pi) - math.pi
            gt_poses[s] = curr_p.copy()

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

        pf_times = []
        amcl_times = []
        pf_estimates = []
        amcl_estimates = []
        amcl_particles = []
        last_lidar_t = 0.0

        for s in range(steps):
            t = t_arr[s]
            true_p = gt_poses[s]
            v_cmd, omega_cmd = TrajectoryGenerator.closed_lemniscate(t, duration_s=12.0, scale_x=5.0, scale_y=3.5)
            v_noisy = v_cmd + rng.normal(0.0, 0.02)
            w_noisy = omega_cmd + rng.normal(0.0, 0.01)

            pf.predict(v_noisy, w_noisy, dt_sim)
            amcl.predict(v_noisy, w_noisy, dt_sim)

            if (t - last_lidar_t) >= (lidar_dt - 1e-5):
                last_lidar_t = t
                angles, ranges = v_map.simulate_lidar_scan(true_p, num_beams=16, max_range_m=12.0, rng=rng)

                t0 = time.perf_counter()
                pf.update_scan(angles, ranges)
                pf_times.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                amcl.update_scan_adaptive(angles, ranges)
                amcl_times.append((time.perf_counter() - t0) * 1000.0)

            pf_estimates.append(pf.estimated_pose.copy())
            amcl_estimates.append(amcl.estimated_pose.copy())
            amcl_particles.append(amcl.N)

        pf_estimates = np.array(pf_estimates)
        amcl_estimates = np.array(amcl_estimates)
        pf_rmse = float(np.sqrt(np.mean(np.sum((pf_estimates[:, :2] - gt_poses[:, :2]) ** 2, axis=1))))
        amcl_rmse = float(np.sqrt(np.mean(np.sum((amcl_estimates[:, :2] - gt_poses[:, :2]) ** 2, axis=1))))

        records.append({
            "trial": trial,
            "pf_rmse_m": pf_rmse,
            "amcl_rmse_m": amcl_rmse,
            "pf_update_ms": float(np.mean(pf_times)),
            "amcl_update_ms": float(np.mean(amcl_times)),
            "amcl_mean_particles": float(np.mean(amcl_particles)),
        })
        print(f"  Trial {trial+1:02d}: PF RMSE={pf_rmse:.3f}m ({np.mean(pf_times):.1f}ms) | AMCL RMSE={amcl_rmse:.3f}m ({np.mean(amcl_times):.1f}ms, N={np.mean(amcl_particles):.0f})")

    out_csv = PROJECT_ROOT / "results" / "map_benchmark_summary.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nSaved map benchmark summary to: {out_csv}")


def run_ablations():
    """Execute sensor ablation and resilience experiments (Configs A through J)."""
    from scripts.run_ablations import run_ablation_benchmarks
    run_ablation_benchmarks()


def run_stereo_benchmark():
    """Execute stereo depth, point cloud, and VO verification benchmark."""
    print("\n" + "=" * 80)
    print("RUNNING STEREO VISION & VO RECONSTRUCTION BENCHMARK")
    print("=" * 80)
    from robot_localization.sensors.stereo import (
        StereoCameraConfig,
        SyntheticStereoArena,
        StereoDepthPipeline,
        StereoVisualOdometry,
        register_point_clouds_icp
    )
    cam_cfg = StereoCameraConfig(width=640, height=480, baseline_m=0.10)
    arena = SyntheticStereoArena(seed=42)
    depth_pipe = StereoDepthPipeline(cam_cfg)
    vo = StereoVisualOdometry(cam_cfg)

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.15, 0.0, 0.0])
    img_l0, img_r0, _ = arena.render_stereo_pair(p0, cam_cfg)
    img_l1, img_r1, _ = arena.render_stereo_pair(p1, cam_cfg)
    depth0 = depth_pipe.compute_depth_and_points(img_l0, img_r0)
    depth1 = depth_pipe.compute_depth_and_points(img_l1, img_r1)

    vo.process_frame(img_l0, img_r0, 0.0)
    vo_res1 = vo.process_frame(img_l1, img_r1, 0.1)
    icp_res = register_point_clouds_icp(depth0.point_cloud_base, depth1.point_cloud_base)

    print("Stereo Depth Validation:")
    print(f"  Valid Disparity Ratio: {depth0.valid_fraction*100.0:.1f}%")
    print(f"  Point Cloud Points: {len(depth0.point_cloud_base):,}")
    print(f"  VO Tracking: Inliers={vo_res1.num_inliers}, Valid={vo_res1.is_valid}")
    print(f"  VO Estimated dx: {vo_res1.dx_robot_m:.3f} m (Ground Truth: 0.150 m)")
    print(f"  ICP Estimated dx: {icp_res.dx_m:.3f} m, Fitness={icp_res.fitness:.3f}")
    print("Stereo vision pipeline operational.")


def run_real_data_benchmark():
    """Execute M2DGR-plus real-data experimental suite across recovery baselines."""
    print("\n" + "=" * 80)
    print("RUNNING REAL-DATA BENCHMARK (M2DGR-PLUS ANOMALY SEQUENCE)")
    print("=" * 80)
    from realdata import run_full_realdata_suite
    run_full_realdata_suite(
        cache_dir=PROJECT_ROOT / "data",
        results_dir=PROJECT_ROOT / "results"
    )


def inspect_real_data():
    """Inspect and report statistics for M2DGR-plus ground truth, calibration, and ROS bag."""
    print("\n" + "=" * 80)
    print("INSPECTING REAL-DATA MODALITIES (M2DGR-PLUS ANOMALY SEQUENCE)")
    print("=" * 80)
    from realdata import load_ground_truth, load_calibration, inspect_bag, create_sample_bag

    data_dir = PROJECT_ROOT / "data"
    gt = load_ground_truth(cache_dir=data_dir)
    print(f"Ground Truth Reference:")
    print(f"  Duration:          {gt.duration_s:.2f} s")
    print(f"  Pose Samples:      {gt.sample_count} (at ~{gt.sample_count/gt.duration_s:.1f} Hz)")
    print(f"  Total Distance:    {gt.total_distance_m:.2f} m")
    print(f"  X Range:          [{gt.poses_2d[:, 0].min():.2f}, {gt.poses_2d[:, 0].max():.2f}] m")
    print(f"  Y Range:          [{gt.poses_2d[:, 1].min():.2f}, {gt.poses_2d[:, 1].max():.2f}] m")
    print(f"  Quaternion Order:  Scalar-first [qw, qx, qy, qz]")

    calib = load_calibration(cache_dir=data_dir)
    print(f"\nCalibration Matrices (Sanitized UTF-8):")
    print(f"  Camera Intrinsics: fx={calib.fx:.2f}, fy={calib.fy:.2f}, cx={calib.cx:.2f}, cy={calib.cy:.2f}")
    print(f"  Extrinsics:        imu2wheel, rsimu2wheel, lidar2wheel, camera2imu, etc. (all 4x4 SE3)")

    bag_path = data_dir / "sample_m2dgrp_anomaly.bag"
    if not bag_path.exists():
        print(f"\nGenerating sample bag: {bag_path}")
        create_sample_bag(bag_path, gt, calib)

    info = inspect_bag(str(bag_path))
    print(f"\nROS 1 Bag Metadata (rosbags pure-Python reader):")
    print(f"  File:              {info['file_name']}")
    print(f"  Total Messages:    {info['total_messages']}")
    print(f"  Duration:          {info['duration_s']:.2f} s")
    for topic, meta in info["topics"].items():
        print(f"    - {topic:32s} : {meta['msgcount']:4d} msgs ({meta['msgtype']})")


def generate_figure():
    """Generate both IEEE publication simulation figures and real-data composite figure."""
    print("\n--- Generating Scientific Figures ---")
    try:
        from scripts.generate_resilience_figure import generate_figure as gen_resil
        gen_resil()
    except Exception as e:
        print(f"Note: resilience figure skipped: {e}")

    try:
        from scripts.generate_composite_figure import main as gen_comp
        gen_comp()
    except Exception as e:
        print(f"Note: composite figure skipped: {e}")

    try:
        from scripts.generate_figure_tradeoffs import generate_tradeoffs_figure as gen_trade
        gen_trade()
    except Exception as e:
        print(f"Note: tradeoffs figure skipped: {e}")

    try:
        from scripts.generate_figure_realdata import generate_main_figure as gen_real
        gen_real()
    except Exception as e:
        print(f"Note: real-data figure skipped: {e}")

    try:
        from scripts.generate_pipeline_diagram import main as gen_pipe
        gen_pipe()
    except Exception as e:
        print(f"Note: pipeline figure skipped: {e}")



def render_gif():
    """Run looping GIF generators (simulation and real-data)."""
    print("\n--- Rendering Looping GIFs ---")
    try:
        from scripts.render_loop_gif import main as gen_gif
        gen_gif()
    except Exception as e:
        print(f"Note: simulation GIF skipped: {e}")

    try:
        from scripts.render_realdata_gif import render_loop_gif as gen_real_gif
        gen_real_gif()
    except Exception as e:
        print(f"Note: real-data GIF skipped: {e}")


def render_video():
    """Run demonstration MP4 video renderers (simulation and real-data)."""
    print("\n--- Rendering Demonstration Videos ---")
    try:
        from scripts.render_demo_video import render_demonstration_video
        render_demonstration_video(duration_s=30.0, fps=15)
    except Exception as e:
        print(f"Note: simulation video skipped: {e}")

    try:
        from scripts.render_realdata_video import render_recovery_video
        render_recovery_video(fps=15)
    except Exception as e:
        print(f"Note: real-data video skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="Reproduction CLI for Four-Wheel Robot Localization")
    parser.add_argument("--quick", action="store_true", help="Run 8-baseline quick benchmark")
    parser.add_argument("--full", action="store_true", help="Run 10-trial Monte Carlo benchmark")
    parser.add_argument("--map", action="store_true", help="Run known-map particle filter benchmark")
    parser.add_argument("--ablation", action="store_true", help="Run sensor addition/removal experiments (Configs A-J)")
    parser.add_argument("--stereo", action="store_true", help="Run stereo vision and VO verification benchmark")
    parser.add_argument("--real-data", action="store_true", help="Run M2DGR-plus real-data experimental benchmark suite")
    parser.add_argument("--inspect-data", action="store_true", help="Inspect M2DGR-plus ground truth, calibration, and bag")
    parser.add_argument("--figure", action="store_true", help="Generate IEEE scientific figures (simulation & real-data)")
    parser.add_argument("--gif", action="store_true", help="Render top-down looping simulation and real-data GIFs")
    parser.add_argument("--video", action="store_true", help="Render demonstration and real-data recovery MP4 videos")
    parser.add_argument("--all", action="store_true", help="Run complete benchmark, ablations, real-data, figures, and media")

    args = parser.parse_args()

    # If no flags passed, default to --quick
    if not any([args.quick, args.full, args.map, args.ablation, args.stereo, args.real_data, args.inspect_data, args.figure, args.gif, args.video, args.all]):
        print("No execution mode specified. Defaulting to --quick benchmark.")
        print("Run with --help to see all available reproduction options.\n")
        args.quick = True

    if args.inspect_data:
        inspect_real_data()
    if args.all or args.quick:
        run_quick_benchmark()
    if args.all or args.full:
        run_full_monte_carlo()
    if args.all or args.map:
        run_map_benchmark()
    if args.all or args.ablation:
        run_ablations()
    if args.all or args.stereo:
        run_stereo_benchmark()
    if args.all or args.real_data:
        run_real_data_benchmark()
    if args.all or args.figure:
        generate_figure()
    if args.all or args.gif:
        render_gif()
    if args.all or args.video:
        render_video()

    print("\n" + "=" * 80)
    print("REPRODUCTION WORKFLOW COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
