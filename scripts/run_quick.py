#!/usr/bin/env python3
"""
scripts/run_quick.py

Quick execution of experimental scenarios E1 through E9.
Generates and saves numerical results and evaluation metrics to `results/quick_benchmark.npz`
and `results/quick_metrics.csv`.
"""

import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def run_experiment(exp_name: str, sim_builder_func, duration_s: float = 60.0):
    print(f"\n--- Running Experiment: {exp_name} ({duration_s}s) ---")
    sim, traj_func, slip_func = sim_builder_func()

    # 1. Forward simulation
    sim_res = sim.run_simulation(
        duration_s=duration_s,
        trajectory_func=traj_func,
        slip_profile_func=slip_func
    )

    # 2. Run all baselines
    traces = run_all_baselines(sim_res, robot_config=sim.config)

    # 3. Compute metrics
    metrics_list = []
    for b_name, trace in traces.items():
        m = compute_trajectory_metrics(
            baseline_name=b_name,
            timestamps_gt=sim_res.timestamps,
            gt_poses=sim_res.ground_truth_pose,
            timestamps_est=trace.timestamps,
            est_poses=trace.estimated_pose,
            cov_diags=trace.covariance_diag,
            nis_gnss=trace.nis_gnss,
            nis_compass=trace.nis_compass,
            cov_mats=trace.covariance_mat
        )
        metrics_list.append({
            "experiment": exp_name,
            "baseline": b_name,
            "pos_rmse_m": m.pos_rmse_m,
            "pos_mae_m": m.pos_mae_m,
            "pos_max_m": m.pos_max_err_m,
            "pos_final_m": m.pos_final_err_m,
            "heading_rmse_deg": m.heading_rmse_deg,
            "drift_percent": m.drift_percent,
            "nees_pos": m.mean_nees_pos,
            "coverage_95": m.coverage_95,
            "coverage_2sigma": m.coverage_2sigma,
        })
        print(f"  [{b_name:30s}] Pos RMSE: {m.pos_rmse_m:6.3f} m | Heading RMSE: {m.heading_rmse_deg:6.2f} deg | 95% Cov: {m.coverage_95*100:5.1f}% | 2-Sig Cov: {m.coverage_2sigma*100:5.1f}%")

    return sim_res, traces, metrics_list


def main():
    print("=" * 80)
    print("4WD ROBOT LOCALIZATION: EKF MULTI-SENSOR BENCHMARK EXECUTION (E1-E9)")
    print("=" * 80)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    saved_traces = {}

    # E1: Nominal odometry & fusion
    sim_res_e1, traces_e1, m_e1 = run_experiment("E1_nominal", lambda: create_scenario_nominal(seed=42))
    all_metrics.extend(m_e1)
    saved_traces["E1"] = {"sim": sim_res_e1, "traces": traces_e1}

    # E2: Asymmetric wheel radius mismatch
    def scenario_e2():
        sim, traj, slip = create_scenario_nominal(seed=42)
        # Create estimator config with 5% radius mismatch on front wheels
        sim.config.wheels["FL"].radius_m *= 1.05
        sim.config.wheels["FR"].radius_m *= 0.95
        return sim, traj, slip

    _, traces_e2, m_e2 = run_experiment("E2_radius_mismatch", scenario_e2)
    all_metrics.extend(m_e2)

    # E3: Isolated front-left wheel slip
    def scenario_e3():
        sim, traj, _ = create_scenario_nominal(seed=42)
        def slip_e3(t):
            if 15.0 <= t <= 35.0:
                return {"FL": 0.40, "FR": 0.0, "RL": 0.0, "RR": 0.0}
            return {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}
        return sim, traj, slip_e3

    sim_res_e3, traces_e3, m_e3 = run_experiment("E3_isolated_slip", scenario_e3)
    all_metrics.extend(m_e3)
    saved_traces["E3"] = {"sim": sim_res_e3, "traces": traces_e3}

    # E4: Left-side differential slip
    sim_res_e4, traces_e4, m_e4 = run_experiment("E4_differential_slip", lambda: create_scenario_slip(seed=42, slip_ratio_left=0.35))
    all_metrics.extend(m_e4)
    saved_traces["E4"] = {"sim": sim_res_e4, "traces": traces_e4}

    # E5: Gyro bias
    sim_res_e5, traces_e5, m_e5 = run_experiment("E5_gyro_bias", lambda: create_scenario_gyro_bias(seed=42, gyro_bias_rad_s=0.035))
    all_metrics.extend(m_e5)
    saved_traces["E5"] = {"sim": sim_res_e5, "traces": traces_e5}

    # E6: Magnetic compass disturbance
    sim_res_e6, traces_e6, m_e6 = run_experiment("E6_compass_disturbance", lambda: create_scenario_compass_disturbance(seed=42, bias_deg=35.0))
    all_metrics.extend(m_e6)
    saved_traces["E6"] = {"sim": sim_res_e6, "traces": traces_e6}

    # E7: GNSS stochastic dropouts (50% dropout)
    def scenario_e7():
        sim, traj, slip = create_scenario_nominal(seed=42)
        sim.gnss.dropout_prob = 0.50
        return sim, traj, slip

    _, traces_e7, m_e7 = run_experiment("E7_gnss_dropout_50pct", scenario_e7)
    all_metrics.extend(m_e7)

    # E8: Contiguous 20s GNSS outage
    sim_res_e8, traces_e8, m_e8 = run_experiment("E8_gnss_outage_20s", lambda: create_scenario_gnss_outage(seed=42, outage_start_s=15.0, outage_end_s=35.0))
    all_metrics.extend(m_e8)
    saved_traces["E8"] = {"sim": sim_res_e8, "traces": traces_e8}

    # E9: GNSS multipath disturbance
    sim_res_e9, traces_e9, m_e9 = run_experiment("E9_gnss_multipath", lambda: create_scenario_multipath(seed=42, bias_m=4.0))
    all_metrics.extend(m_e9)
    saved_traces["E9"] = {"sim": sim_res_e9, "traces": traces_e9}

    # Combined severe stress test
    sim_res_comb, traces_comb, m_comb = run_experiment("E12_combined_failures", lambda: create_scenario_combined(seed=42))
    all_metrics.extend(m_comb)
    saved_traces["Combined"] = {"sim": sim_res_comb, "traces": traces_comb}

    # Save DataFrame
    df = pd.DataFrame(all_metrics)
    csv_path = results_dir / "quick_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics summary table to: {csv_path}")

    # Save key traces for figure generation and video animation
    trace_save_path = results_dir / "saved_benchmark_traces.npz"
    np.savez_compressed(
        trace_save_path,
        # Nominal
        e1_t=sim_res_e1.timestamps,
        e1_gt_pose=sim_res_e1.ground_truth_pose,
        e1_b0_pose=traces_e1["B0_wheel_odometry"].estimated_pose,
        e1_b3_pose=traces_e1["B3_odometry_gyro_gnss"].estimated_pose,
        e1_b4_pose=traces_e1["B4_odometry_gyro_gnss_compass"].estimated_pose,
        # Slip E4
        e4_t=sim_res_e4.timestamps,
        e4_gt_pose=sim_res_e4.ground_truth_pose,
        e4_b0_pose=traces_e4["B0_wheel_odometry"].estimated_pose,
        e4_b1_pose=traces_e4["B1_odometry_gyro"].estimated_pose,
        e4_b3_pose=traces_e4["B3_odometry_gyro_gnss"].estimated_pose,
        e4_d_left=sim_res_e4.wheel_disagreement["d_left"],
        e4_d_right=sim_res_e4.wheel_disagreement["d_right"],
        e4_w_rot=sim_res_e4.wheel_speeds_rotational,
        e4_w_ground=sim_res_e4.wheel_speeds_ground,
        # Outage E8
        e8_t=sim_res_e8.timestamps,
        e8_gt_pose=sim_res_e8.ground_truth_pose,
        e8_b0_pose=traces_e8["B0_wheel_odometry"].estimated_pose,
        e8_b1_pose=traces_e8["B1_odometry_gyro"].estimated_pose,
        e8_b3_pose=traces_e8["B3_odometry_gyro_gnss"].estimated_pose,
        e8_b3_cov=traces_e8["B3_odometry_gyro_gnss"].covariance_diag,
        e8_b5_pose=traces_e8["B5_bias_aware_ekf"].estimated_pose,
        e8_b5_cov=traces_e8["B5_bias_aware_ekf"].covariance_diag,
        e8_a5_pose=traces_e8["A5_robust_ekf"].estimated_pose,
        e8_a5_cov=traces_e8["A5_robust_ekf"].covariance_diag,
        # Combined
        comb_t=sim_res_comb.timestamps,
        comb_gt_pose=sim_res_comb.ground_truth_pose,
        comb_b0_pose=traces_comb["B0_wheel_odometry"].estimated_pose,
        comb_b1_pose=traces_comb["B1_odometry_gyro"].estimated_pose,
        comb_b3_pose=traces_comb["B3_odometry_gyro_gnss"].estimated_pose,
        comb_b4_pose=traces_comb["B4_odometry_gyro_gnss_compass"].estimated_pose,
        comb_b5_pose=traces_comb["B5_bias_aware_ekf"].estimated_pose,
        comb_b5_bg=traces_comb["B5_bias_aware_ekf"].estimated_gyro_bias,
        comb_a4_pose=traces_comb["A4_unscented_kf"].estimated_pose,
        comb_a5_pose=traces_comb["A5_robust_ekf"].estimated_pose,
    )
    print(f"Saved compressed trajectory arrays to: {trace_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
