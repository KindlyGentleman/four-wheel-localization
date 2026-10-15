"""
robot_localization.evaluation.monte_carlo

Monte Carlo statistical benchmark executor.
Runs paired trials with matched random seeds across different sensor fusion baselines,
and calculates statistical metrics: sample mean, sample standard deviation,
and 95% Student's t confidence intervals.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
import pandas as pd
from scipy import stats

from robot_localization.simulation.robot_simulator import RobotSimulator
from robot_localization.estimation.baseline_runner import run_all_baselines
from robot_localization.evaluation.metrics import compute_trajectory_metrics, TrajectoryMetrics


@dataclass
class BaselineStats:
    """Aggregated Monte Carlo statistics for one baseline."""
    baseline: str
    pos_rmse_mean: float
    pos_rmse_std: float
    pos_rmse_ci95: Tuple[float, float]
    heading_rmse_deg_mean: float
    heading_rmse_deg_std: float
    final_error_mean: float
    coverage_2sigma_mean: float
    mean_nees_pos: float


@dataclass
class MonteCarloSummary:
    """Complete summary of a multi-trial experiment."""
    scenario_name: str
    num_trials: int
    baseline_stats: Dict[str, BaselineStats]
    raw_metrics_table: pd.DataFrame


class MonteCarloRunner:
    """
    Executes repeated trials across different noise realizations for a scenario.
    """

    def __init__(
        self,
        scenario_generator_func: Callable[[int], Tuple[RobotSimulator, Callable, Callable]],
        num_trials: int = 20,
        seed_base: int = 1000
    ):
        self.scenario_gen = scenario_generator_func
        self.num_trials = num_trials
        self.seed_base = seed_base

    def run(self, scenario_name: str = "benchmark") -> MonteCarloSummary:
        """Execute paired trials across all baselines."""
        records = []

        for i in range(self.num_trials):
            seed = self.seed_base + i
            sim, traj_func, slip_func = self.scenario_gen(seed)

            # 1. Forward simulation
            sim_result = sim.run_simulation(
                duration_s=60.0,
                trajectory_func=traj_func,
                slip_profile_func=slip_func
            )

            # 2. Run baselines on this identical simulation realization
            baseline_traces = run_all_baselines(sim_result, robot_config=sim.config)

            # 3. Compute metrics for each baseline
            for b_name, trace in baseline_traces.items():
                m = compute_trajectory_metrics(
                    baseline_name=b_name,
                    timestamps_gt=sim_result.timestamps,
                    gt_poses=sim_result.ground_truth_pose,
                    timestamps_est=trace.timestamps,
                    est_poses=trace.estimated_pose,
                    cov_diags=trace.covariance_diag,
                    nis_gnss=trace.nis_gnss,
                    nis_compass=trace.nis_compass
                )
                records.append({
                    "trial": i,
                    "seed": seed,
                    "baseline": b_name,
                    "pos_rmse_m": m.pos_rmse_m,
                    "pos_mae_m": m.pos_mae_m,
                    "pos_max_m": m.pos_max_err_m,
                    "pos_final_m": m.pos_final_err_m,
                    "heading_rmse_deg": m.heading_rmse_deg,
                    "drift_percent": m.drift_percent,
                    "nees_pos": m.mean_nees_pos,
                    "coverage_2sigma": m.coverage_2sigma,
                })

        df = pd.DataFrame(records)

        # Aggregate statistics
        b_stats = {}
        for b_name in df["baseline"].unique():
            sub = df[df["baseline"] == b_name]
            n = len(sub)
            mean_rmse = float(sub["pos_rmse_m"].mean())
            std_rmse = float(sub["pos_rmse_m"].std(ddof=1)) if n > 1 else 0.0

            # 95% Confidence Interval
            if n > 1 and std_rmse > 0:
                t_crit = float(stats.t.ppf(0.975, df=n - 1))
                ci_margin = t_crit * (std_rmse / math.sqrt(n))
                ci = (mean_rmse - ci_margin, mean_rmse + ci_margin)
            else:
                ci = (mean_rmse, mean_rmse)

            b_stats[b_name] = BaselineStats(
                baseline=b_name,
                pos_rmse_mean=mean_rmse,
                pos_rmse_std=std_rmse,
                pos_rmse_ci95=ci,
                heading_rmse_deg_mean=float(sub["heading_rmse_deg"].mean()),
                heading_rmse_deg_std=float(sub["heading_rmse_deg"].std(ddof=1)) if n > 1 else 0.0,
                final_error_mean=float(sub["pos_final_m"].mean()),
                coverage_2sigma_mean=float(sub["coverage_2sigma"].mean()),
                mean_nees_pos=float(sub["nees_pos"].mean())
            )

        return MonteCarloSummary(
            scenario_name=scenario_name,
            num_trials=self.num_trials,
            baseline_stats=b_stats,
            raw_metrics_table=df
        )
