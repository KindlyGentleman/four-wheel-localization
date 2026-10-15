"""
Evaluation and benchmark metrics module: RMSE, MAE, NEES, NIS, covariance coverage, Monte Carlo.
"""

from robot_localization.evaluation.metrics import (
    compute_trajectory_metrics,
    compute_nees_series,
    compute_nis_statistics,
    compute_covariance_coverage,
    TrajectoryMetrics,
)
from robot_localization.evaluation.monte_carlo import (
    MonteCarloRunner,
    MonteCarloSummary,
)

__all__ = [
    "compute_trajectory_metrics",
    "compute_nees_series",
    "compute_nis_statistics",
    "compute_covariance_coverage",
    "TrajectoryMetrics",
    "MonteCarloRunner",
    "MonteCarloSummary",
]
