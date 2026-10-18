#!/usr/bin/env python3
"""
scripts/generate_figure_tradeoffs.py

Generates the 3-panel IEEE trade-off figure (`assets/figure_tradeoffs.png`, `.pdf`, `.svg`)
directly from verified benchmark CSV outputs:
  (a) Multi-scenario 10-trial Monte Carlo Position RMSE (all 6 scenarios, asymmetric 95% CIs)
  (b) Known-Map LiDAR Localization: Particle Count N vs Update Latency [ms] (split subplots)
  (c) GNSS Outage Recovery Strategies (Configs J0-J4): Cleanly split Position vs Heading RMSE subplots
Mirrored to `figures/figure_tradeoffs.*` and `four-wheel-localization/assets/figure_tradeoffs.*`.
"""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FIGURES_DIR = PROJECT_ROOT / "figures"
FWL_ASSETS = PROJECT_ROOT / "four-wheel-localization" / "assets"
for d in [ASSETS_DIR, FIGURES_DIR, FWL_ASSETS]:
    d.mkdir(parents=True, exist_ok=True)


def setup_ieee_style():
    try:
        import scienceplots
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt.style.use("default")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.8,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.3,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.2,
        "axes.unicode_minus": False,
    })


def generate_tradeoffs_figure():
    setup_ieee_style()
    print("Generating IEEE trade-offs figure (assets/figure_tradeoffs.png)...")

    mc_csv = PROJECT_ROOT / "results" / "full_monte_carlo_summary.csv"
    map_csv = PROJECT_ROOT / "results" / "map_benchmark_summary.csv"
    abl_csv = PROJECT_ROOT / "results" / "ablation_summary.csv"

    df_mc = pd.read_csv(mc_csv)
    df_map = pd.read_csv(map_csv)
    df_abl = pd.read_csv(abl_csv)

    fig = plt.figure(figsize=(7.16, 6.4), dpi=300)
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.96, top=0.94, bottom=0.09, hspace=0.40, wspace=0.34)

    ax_mc = fig.add_subplot(gs[0, :])
    
    # Panel (b) subgrid: particles vs latency
    gs_map = gs[1, 0].subgridspec(1, 2, wspace=0.38)
    ax_map_n = fig.add_subplot(gs_map[0, 0])
    ax_map_t = fig.add_subplot(gs_map[0, 1])

    # Panel (c) subgrid: split Position RMSE [m] and Heading RMSE [deg]
    gs_rec = gs[1, 1].subgridspec(2, 1, hspace=0.36)
    ax_rec_pos = fig.add_subplot(gs_rec[0, 0])
    ax_rec_head = fig.add_subplot(gs_rec[1, 0])

    # -------------------------------------------------------------------------
    # Panel (a): 10-Trial Monte Carlo Position RMSE (Asymmetric 95% CIs)
    # -------------------------------------------------------------------------
    scenario_order = [
        ("Nominal_Traction", "Nominal"),
        ("Asymmetric_Wheel_Slip", "Asym. Slip"),
        ("Compass_Disturbance", "Compass Fault"),
        ("GNSS_Multipath", "Multipath"),
        ("GNSS_Outage_20s", "20s Outage"),
        ("Combined_Stress", "Combined"),
    ]
    baselines = [
        ("B0_wheel_odometry", "B0: Wheel DR", "#8B5CF6"),
        ("B5_bias_aware_ekf", "B5: Bias-Aware EKF", "#2563EB"),
        ("A4_unscented_kf", "A4: Manifold UKF", "#0D9488"),
        ("A5_robust_ekf", "A5: Robust EKF", "#DC2626"),
    ]

    x = np.arange(len(scenario_order))
    width = 0.19

    for b_idx, (b_key, b_label, b_col) in enumerate(baselines):
        means = []
        low_errs = []
        high_errs = []
        for sc_key, _ in scenario_order:
            sub = df_mc[(df_mc["scenario"] == sc_key) & (df_mc["baseline"] == b_key)]
            if not sub.empty:
                m = float(sub["pos_rmse_mean"].iloc[0])
                ci_low = float(sub["pos_rmse_ci95_low"].iloc[0])
                ci_high = float(sub["pos_rmse_ci95_high"].iloc[0])
                means.append(m)
                low_errs.append(max(0.0, m - ci_low))
                high_errs.append(max(0.0, ci_high - m))
            else:
                raise RuntimeError(f"Missing Monte Carlo record for scenario '{sc_key}', baseline '{b_key}' in {mc_csv}")
        offset = (b_idx - 1.5) * width
        yerr = np.array([low_errs, high_errs])
        ax_mc.bar(x + offset, means, width, yerr=yerr, capsize=2.0, label=b_label, color=b_col, alpha=0.88, edgecolor="#1E293B", lw=0.5)

    ax_mc.set_yscale("log")
    ax_mc.set_ylim(0.04, 22.0)
    ax_mc.set_xticks(x)
    ax_mc.set_xticklabels([lbl for _, lbl in scenario_order], fontsize=7.2)
    ax_mc.set_ylabel("Position RMSE [m] (log scale)")
    ax_mc.set_title("(a) 10-Trial Monte Carlo Position RMSE Across 6 Scenarios (Mean with Asymmetric 95% CI)", fontweight="bold")
    ax_mc.grid(True, linestyle=":", alpha=0.6, which="both", axis="y")
    ax_mc.legend(loc="upper left", ncol=4, framealpha=0.92, fontsize=6.5)

    # -------------------------------------------------------------------------
    # Panel (b): Known-Map LiDAR Localization (Split Subplots: Particles vs ms)
    # -------------------------------------------------------------------------
    pf_n = 200.0
    amcl_n = float(df_map["amcl_mean_particles"].mean())
    pf_ms = float(df_map["pf_update_ms"].mean())
    amcl_ms = float(df_map["amcl_update_ms"].mean())
    pf_rmse = float(df_map["pf_rmse_m"].mean())
    amcl_rmse = float(df_map["amcl_rmse_m"].mean())

    methods_map = ["Fixed PF", "AMCL"]
    x_m = np.arange(2)

    bars_n = ax_map_n.bar(x_m, [pf_n, amcl_n], width=0.52, color=["#64748B", "#059669"], alpha=0.88, edgecolor="#1E293B", lw=0.6)
    ax_map_n.set_xticks(x_m)
    ax_map_n.set_xticklabels(methods_map, fontsize=6.8)
    ax_map_n.set_ylabel("Mean Particles [count]")
    ax_map_n.set_ylim(0, 245)
    ax_map_n.set_title("(b1) Sample Count", fontweight="bold", fontsize=7.6)
    ax_map_n.grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in bars_n:
        h = bar.get_height()
        ax_map_n.text(bar.get_x() + bar.get_width() / 2, h + 6, f"{h:.0f}", ha="center", va="bottom", fontsize=6.3, fontweight="bold")

    bars_t = ax_map_t.bar(x_m, [pf_ms, amcl_ms], width=0.52, color=["#64748B", "#0284C7"], alpha=0.88, edgecolor="#1E293B", lw=0.6)
    ax_map_t.set_xticks(x_m)
    ax_map_t.set_xticklabels(methods_map, fontsize=6.8)
    ax_map_t.set_ylabel("Scan Update Time [ms]")
    ax_map_t.set_ylim(0, max(pf_ms, amcl_ms) * 1.30)
    ax_map_t.set_title(f"(b2) Latency (RMSE={amcl_rmse:.2f}m)", fontweight="bold", fontsize=7.6)
    ax_map_t.grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in bars_t:
        h = bar.get_height()
        ax_map_t.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}ms", ha="center", va="bottom", fontsize=6.3, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel (c): GNSS Outage Recovery Strategies (Clean Split: Pos vs Heading)
    # -------------------------------------------------------------------------
    j_configs = [
        ("J0_outage_R0_strict_lockout", "J0: Strict\nLockout", "#EF4444"),
        ("J1_outage_R1_temporal_consistency", "J1: Temporal\nConsistency", "#3B82F6"),
        ("J2_outage_R2_health_hysteresis", "J2: Health\nHysteresis", "#0D9488"),
        ("J3_outage_R3_covariance_growth", "J3: Cov\nGrowth", "#D97706"),
        ("J4_outage_R4_global_reinit", "J4: Global\nRe-init", "#8B5CF6"),
    ]
    j_pos_rmse, j_head_rmse, j_labels, j_colors = [], [], [], []
    for j_key, lbl, col in j_configs:
        sub = df_abl[df_abl["config"] == j_key]
        if not sub.empty:
            j_pos_rmse.append(float(sub["pos_rmse_m"].iloc[0]))
            j_head_rmse.append(float(sub["heading_rmse_deg"].iloc[0]))
            j_labels.append(lbl)
            j_colors.append(col)
        else:
            raise RuntimeError(f"Missing ablation record for config '{j_key}' in {abl_csv}")

    x_j = np.arange(len(j_configs))
    w_j = 0.50

    # (c1) Position RMSE [m]
    bars_p = ax_rec_pos.bar(x_j, j_pos_rmse, w_j, color=j_colors, alpha=0.88, edgecolor="#1E293B", lw=0.6)
    ax_rec_pos.set_ylabel("Pos RMSE [m]", fontsize=6.8)
    ax_rec_pos.set_ylim(0, 3.8)
    ax_rec_pos.set_xticks(x_j)
    ax_rec_pos.set_xticklabels([])
    ax_rec_pos.set_title("(c) Simulated Recovery Strategies (J0-J4)", fontweight="bold", fontsize=7.6)
    ax_rec_pos.grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in bars_p:
        h = bar.get_height()
        ax_rec_pos.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}m", ha="center", va="bottom", fontsize=5.8, fontweight="bold")

    # (c2) Heading RMSE [deg]
    bars_h = ax_rec_head.bar(x_j, j_head_rmse, w_j, color=j_colors, alpha=0.88, edgecolor="#1E293B", lw=0.6)
    ax_rec_head.set_ylabel("Head RMSE [deg]", fontsize=6.8)
    ax_rec_head.set_ylim(0, 24.0)
    ax_rec_head.set_xticks(x_j)
    ax_rec_head.set_xticklabels(j_labels, fontsize=6.0)
    ax_rec_head.grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in bars_h:
        h = bar.get_height()
        ax_rec_head.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}°", ha="center", va="bottom", fontsize=5.8, fontweight="bold")

    png_path = ASSETS_DIR / "figure_tradeoffs.png"
    pdf_path = ASSETS_DIR / "figure_tradeoffs.pdf"
    svg_path = ASSETS_DIR / "figure_tradeoffs.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    for target_dir in [FIGURES_DIR, FWL_ASSETS]:
        shutil.copy2(png_path, target_dir / "figure_tradeoffs.png")
        shutil.copy2(pdf_path, target_dir / "figure_tradeoffs.pdf")
        shutil.copy2(svg_path, target_dir / "figure_tradeoffs.svg")

    print(f"Generated and mirrored figure_tradeoffs to {png_path}")


if __name__ == "__main__":
    generate_tradeoffs_figure()
