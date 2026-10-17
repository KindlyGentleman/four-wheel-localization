#!/usr/bin/env python3
"""
scripts/run_full.py

Full multi-trial Monte Carlo statistical benchmark execution.
Runs paired trials with matched random seeds across all baselines under nominal,
wheel-slip, and GNSS outage conditions.
Calculates sample means, sample standard deviations, and 95% Student's t confidence intervals.
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_localization.evaluation.monte_carlo import MonteCarloRunner
from robot_localization.simulation.scenarios import (
    create_scenario_nominal,
    create_scenario_slip,
    create_scenario_gnss_outage,
    create_scenario_combined,
    create_scenario_compass_disturbance,
    create_scenario_multipath,
)


def main():
    print("=" * 80)
    print("RUNNING MULTI-TRIAL MONTE CARLO STATISTICAL BENCHMARK")
    print("=" * 80)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("Nominal_Traction", lambda seed: create_scenario_nominal(seed=seed, duration_s=60.0)),
        ("Asymmetric_Wheel_Slip", lambda seed: create_scenario_slip(seed=seed, duration_s=60.0, slip_ratio_left=0.30)),
        ("GNSS_Outage_20s", lambda seed: create_scenario_gnss_outage(seed=seed, duration_s=60.0, outage_start_s=15.0, outage_end_s=35.0)),
        ("Combined_Stress", lambda seed: create_scenario_combined(seed=seed, duration_s=60.0)),
        ("Compass_Disturbance", lambda seed: create_scenario_compass_disturbance(seed=seed, duration_s=60.0)),
        ("GNSS_Multipath", lambda seed: create_scenario_multipath(seed=seed, duration_s=60.0)),
    ]

    all_summaries = []
    all_raw_dfs = []

    # Run 10 paired trials per condition for statistical confidence intervals
    num_trials = 10
    for sc_name, sc_gen in scenarios:
        print(f"\n>>> Running {num_trials} Monte Carlo trials for scenario: {sc_name}...")
        runner = MonteCarloRunner(scenario_generator_func=sc_gen, num_trials=num_trials, seed_base=2000, duration_s=60.0)
        summary = runner.run(scenario_name=sc_name)

        all_raw_dfs.append(summary.raw_metrics_table)

        print(f"{'Baseline':<32} | {'Pos RMSE Mean [m]':<18} | {'95% CI [m]':<20} | {'Heading RMSE [deg]':<18}")
        print("-" * 95)
        for b_name, b_stat in summary.baseline_stats.items():
            ci_str = f"[{b_stat.pos_rmse_ci95[0]:.3f}, {b_stat.pos_rmse_ci95[1]:.3f}]"
            print(f"{b_name:<32} | {b_stat.pos_rmse_mean:6.3f} +/- {b_stat.pos_rmse_std:5.3f} | {ci_str:<20} | {b_stat.heading_rmse_deg_mean:6.2f} +/- {b_stat.heading_rmse_deg_std:5.2f}")
            all_summaries.append({
                "scenario": sc_name,
                "baseline": b_name,
                "pos_rmse_mean": b_stat.pos_rmse_mean,
                "pos_rmse_std": b_stat.pos_rmse_std,
                "pos_rmse_ci95_low": b_stat.pos_rmse_ci95[0],
                "pos_rmse_ci95_high": b_stat.pos_rmse_ci95[1],
                "heading_rmse_deg_mean": b_stat.heading_rmse_deg_mean,
                "heading_rmse_deg_std": b_stat.heading_rmse_deg_std,
                "final_error_mean": b_stat.final_error_mean,
                "coverage_95_mean": b_stat.coverage_95_mean,
                "coverage_2sigma_mean": b_stat.coverage_2sigma_mean,
                "mean_nees_pos": b_stat.mean_nees_pos,
            })

    # Save outputs
    summary_df = pd.DataFrame(all_summaries)
    summary_path = results_dir / "full_monte_carlo_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    raw_df = pd.concat(all_raw_dfs, ignore_index=True)
    raw_path = results_dir / "full_monte_carlo_raw_trials.csv"
    raw_df.to_csv(raw_path, index=False)

    print("\n" + "=" * 80)
    print(f"Monte Carlo benchmark complete.")
    print(f"Summary table saved to: {summary_path}")
    print(f"Raw trial table saved to: {raw_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
