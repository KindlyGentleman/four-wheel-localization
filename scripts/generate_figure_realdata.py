#!/usr/bin/env python3
"""
scripts/generate_figure_realdata.py

Generates the primary 6-panel IEEE scientific composite figure for the M2DGR-plus
hybrid localization, fault degradation, and global pose recovery benchmark:
- `assets/figure_realdata.pdf`
- `assets/figure_realdata.png` (300 DPI)
- `assets/figure_realdata.svg`

Mirrored to `figures/figure_realdata.*` and `four-wheel-localization/assets/figure_realdata.*`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from realdata import (
    load_ground_truth,
    load_calibration,
    generate_synthetic_scene_image,
    VisualFeatureTracker,
)

ASSETS_DIR = PROJECT_ROOT / "assets"
FIGURES_DIR = PROJECT_ROOT / "figures"
FWL_ASSETS = PROJECT_ROOT / "four-wheel-localization" / "assets"
for d in [ASSETS_DIR, FIGURES_DIR, FWL_ASSETS]:
    d.mkdir(parents=True, exist_ok=True)

# Publication color palette
C_GT = "#0F172A"       # Dark Slate (Ground truth)
C_DR = "#8B5CF6"       # Violet (Dead reckoning)
C_STD = "#2563EB"      # Royal Blue (Standard EKF)
C_STRICT = "#DC2626"   # Crimson (Strict gated EKF / Lockout)
C_RECOV = "#059669"    # Emerald Green (Recoverable EKF)
C_SHADE = "#FEE2E2"    # Soft rose for outage shading
C_SLIP = "#FEF3C7"     # Amber for slip window
C_GRID = "#E2E8F0"     # Slate grid


def setup_ieee_style():
    """Apply SciencePlots IEEE publication styling."""
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
        "axes.titlesize": 8.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.0,
        "figure.titlesize": 9.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.75,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
    })


def generate_main_figure():
    setup_ieee_style()

    gt = load_ground_truth(cache_dir=PROJECT_ROOT / "data")
    calib = load_calibration(cache_dir=PROJECT_ROOT / "data")
    traces_file = PROJECT_ROOT / "results" / "realdata_traces.npz"
    summary_file = PROJECT_ROOT / "results" / "realdata_summary.csv"

    if not traces_file.exists() or not summary_file.exists():
        from realdata import run_full_realdata_suite
        run_full_realdata_suite(cache_dir=PROJECT_ROOT / "data", results_dir=PROJECT_ROOT / "results")

    data = np.load(traces_file)
    df_summary = pd.read_csv(summary_file)

    t_rel = gt.timestamps - gt.timestamps[0]
    outage_t0, outage_t1 = 14.0, 34.0
    slip_t0, slip_t1 = 13.0, 18.0

    # 6-panel layout: 3 rows x 2 cols, width = 7.16 in, height = 9.0 in
    fig = plt.figure(figsize=(7.16, 9.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, left=0.08, right=0.95, top=0.96, bottom=0.06, hspace=0.38, wspace=0.34)

    ax_cam = fig.add_subplot(gs[0, 0])
    ax_traj = fig.add_subplot(gs[0, 1])
    ax_rast = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[1, 1])
    
    # Subgridspec for Panel (e) to separate Inlier Ratio from Position RMSE without dual-axis bars
    gs_vis = gs[2, 0].subgridspec(2, 1, hspace=0.32)
    ax_vis_top = fig.add_subplot(gs_vis[0, 0])
    ax_vis_bot = fig.add_subplot(gs_vis[1, 0])

    ax_trade = fig.add_subplot(gs[2, 1])

    # -------------------------------------------------------------------------
    # Panel (a): Synthetic Floor Camera Frame with High-Contrast Optical Flow
    # -------------------------------------------------------------------------
    tracker = VisualFeatureTracker(calib, seed=10042)
    sample_idx = 60  # t ~ 12.0s during active forward motion
    img_prev = generate_synthetic_scene_image(t_rel[sample_idx - 1], gt.poses_2d[sample_idx - 1])
    img_curr = generate_synthetic_scene_image(t_rel[sample_idx], gt.poses_2d[sample_idx])
    tracker.process_frame(img_prev, degradation_mode="clean")
    v_res = tracker.process_frame(img_curr, degradation_mode="clean")
    disp_img = cv2.cvtColor(v_res["processed_image"], cv2.COLOR_GRAY2RGB)

    good_prev = v_res.get("good_prev", np.empty((0, 2)))
    good_curr = v_res.get("good_curr", np.empty((0, 2)))

    # Select up to 40 well-distributed points for high print clarity
    step_pt = max(1, len(good_prev) // 40)
    selected_indices = list(range(0, len(good_prev), step_pt))[:40]

    for idx in selected_indices:
        p0_pt = good_prev[idx]
        p1_pt = good_curr[idx]
        x0, y0 = int(round(p0_pt[0])), int(round(p0_pt[1]))
        x1, y1 = int(round(p1_pt[0])), int(round(p1_pt[1]))

        # High-contrast flow arrows (3.5x scaled for print visibility)
        dx_vis = int(round((p1_pt[0] - p0_pt[0]) * 3.5))
        dy_vis = int(round((p1_pt[1] - p0_pt[1]) * 3.5))

        cv2.circle(disp_img, (x0, y0), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            disp_img, (x0, y0), (x0 + dx_vis, y0 + dy_vis),
            (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.35
        )

    ax_cam.imshow(disp_img)
    ax_cam.set_title("(a) Synthetic Floor Cam & KLT Flow", fontweight="bold")
    ax_cam.axis("off")
    ax_cam.text(
        0.03, 0.05,
        f"Synthetic Scene | Frame {sample_idx} (t={t_rel[sample_idx]:.1f}s)\n"
        f"KLT Inliers: {v_res['inliers_count']}/{v_res['tracked_count']} ({v_res['inlier_ratio']*100:.1f}%) | Arrows scaled 3.5x",
        transform=ax_cam.transAxes,
        fontsize=6.0,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75, edgecolor="none")
    )

    # -------------------------------------------------------------------------
    # Panel (b): 2D Planar Trajectory (REP-105 Discontinuity Separated)
    # -------------------------------------------------------------------------
    gt_poses = data["gt_poses"]
    dr_poses = data["est_R0_dead_reckoning"]
    strict_poses = data["est_R3_strict_gated_ekf"]
    recov_poses = data["est_R4_recoverable_temporal"]
    odom_poses = data.get("odom_R4_recoverable_temporal", recov_poses)
    reloc_indices = data.get("reloc_idx_R4_recoverable_temporal", np.array([], dtype=int))

    ax_traj.plot(gt_poses[:, 0], gt_poses[:, 1], color=C_GT, label="Ground Truth", lw=1.6, zorder=5)
    ax_traj.plot(dr_poses[:, 0], dr_poses[:, 1], color=C_DR, ls="--", label="Odometry DR", lw=1.1)
    ax_traj.plot(strict_poses[:, 0], strict_poses[:, 1], color=C_STRICT, ls="-.", label="Strict Gate (R3)", lw=1.3)
    ax_traj.plot(odom_poses[:, 0], odom_poses[:, 1], color="#059669", ls=":", lw=1.1, alpha=0.65, label="Cont. Odom (REP-105)")

    has_reloc = (len(reloc_indices) > 0 and reloc_indices[0] < len(recov_poses))
    if has_reloc:
        k_reloc = int(reloc_indices[0])
        ax_traj.plot(recov_poses[:k_reloc, 0], recov_poses[:k_reloc, 1], color=C_RECOV, ls="--", lw=1.2, alpha=0.75)
        ax_traj.plot(recov_poses[k_reloc:, 0], recov_poses[k_reloc:, 1], color=C_RECOV, ls="-", lw=1.6, zorder=6, label="Recovered Map (R4)")

        p_pre = recov_poses[k_reloc - 1]
        p_post = recov_poses[k_reloc]
        ax_traj.annotate(
            "", xy=(p_post[0], p_post[1]), xytext=(p_pre[0], p_pre[1]),
            arrowprops=dict(arrowstyle="->", linestyle="--", color="#059669", lw=1.1, alpha=0.85)
        )
        ax_traj.text(
            0.5 * (p_pre[0] + p_post[0]) + 0.08, 0.5 * (p_pre[1] + p_post[1]) - 0.12,
            r"$\Delta \mathbf{p}_{\mathrm{map}\to\mathrm{odom}}$",
            fontsize=5.8, color="#059669", fontweight="bold"
        )
        ax_traj.scatter(recov_poses[k_reloc, 0], recov_poses[k_reloc, 1], marker="*", color="#059669", s=60, zorder=11, label="Re-anchored")
    else:
        # Honest reporting: no fabricated fallback marker
        ax_traj.plot(recov_poses[:, 0], recov_poses[:, 1], color=C_RECOV, ls="-", label="Recovered Map (R4)", lw=1.5, zorder=6)
        ax_traj.text(
            0.05, 0.90, "No verified re-anchoring event",
            transform=ax_traj.transAxes, fontsize=5.5, color="#DC2626", style="italic"
        )

    idx_out_start = int(np.argmin(np.abs(t_rel - outage_t0)))
    idx_out_end = int(np.argmin(np.abs(t_rel - outage_t1)))

    ax_traj.scatter(gt_poses[idx_out_start, 0], gt_poses[idx_out_start, 1], marker="v", color="#D97706", s=35, zorder=10, label="Outage Start")
    ax_traj.scatter(gt_poses[idx_out_end, 0], gt_poses[idx_out_end, 1], marker="^", color="#2563EB", s=35, zorder=10, label="GNSS Resumed")

    ax_traj.set_xlabel("Local X Position (m)")
    ax_traj.set_ylabel("Local Y Position (m)")
    ax_traj.set_title("(b) 2D Trajectory (REP-105 Separated)", fontweight="bold")
    ax_traj.grid(True, color=C_GRID, linestyle=":")
    ax_traj.set_xlim(-2.6, 1.4)
    ax_traj.set_ylim(-2.6, 1.4)
    ax_traj.set_aspect("equal", "box")
    ax_traj.legend(loc="lower left", ncol=2, framealpha=0.90, fontsize=5.2)

    # -------------------------------------------------------------------------
    # Panel (c): Dynamic Sensor Timeline & Data-Driven Safe-Stop State
    # -------------------------------------------------------------------------
    ax_rast.set_title("(c) Sensor Timeline & Safe-Stop State", fontweight="bold")
    y_labels = ["GNSS Fix (1Hz)", "Wheel Odom", "IMU Gyro", "Safe-Stop*"]
    y_pos = [3, 2, 1, 0]

    # IMU & Odometry timelines
    ax_rast.hlines(1, 0, gt.duration_s, colors="#2563EB", lw=4, label="Active / Nominal")
    ax_rast.hlines(2, 0, slip_t0, colors="#2563EB", lw=4)
    ax_rast.hlines(2, slip_t0, slip_t1, colors="#D97706", lw=4, label="Wheel Slip")
    ax_rast.hlines(2, slip_t1, gt.duration_s, colors="#2563EB", lw=4)

    # GNSS Fix: Discrete event ticks at authentic ~1.00 Hz arrival epochs
    gnss_epochs_active = []
    for i in range(len(t_rel)):
        if i % 5 == 0:
            t_curr = t_rel[i]
            if not (outage_t0 <= t_curr <= outage_t1):
                gnss_epochs_active.append(t_curr)
    
    ax_rast.plot(gnss_epochs_active, [3] * len(gnss_epochs_active), "|", color="#059669", markersize=8, markeredgewidth=1.4, label="GNSS Fix (1Hz)")
    ax_rast.hlines(3, outage_t0, outage_t1, colors="#EF4444", lw=2, linestyle=":")

    # Safe-Stop Timeline: Render dynamically from saved supervisor states
    sup_states = data.get("sup_states_R4_recoverable_temporal", None)
    if sup_states is not None and len(sup_states) == len(t_rel):
        state_colors = {
            "NOMINAL": "#2563EB",
            "BRAKING": "#D97706",
            "STANDSTILL_HOLD": "#DC2626",
            "VERIFYING_RECOVERY": "#7C3AED",
            "RESUMING": "#0284C7"
        }
        for i in range(len(sup_states)):
            t_start = t_rel[i]
            t_end = t_rel[i + 1] if i + 1 < len(t_rel) else gt.duration_s
            c_st = state_colors.get(str(sup_states[i]), "#64748B")
            ax_rast.hlines(0, t_start, t_end, colors=c_st, lw=5)
    else:
        # Fallback if state trace is missing
        ax_rast.hlines(0, 0, gt.duration_s, colors="#64748B", lw=5)

    ax_rast.axvspan(slip_t0, slip_t1, color=C_SLIP, alpha=0.45)
    ax_rast.axvspan(outage_t0, outage_t1, color=C_SHADE, alpha=0.45)

    ax_rast.text(
        0.02, 0.05,
        "*Counterfactual Safe-Stop: reference trajectory continues unbraked",
        transform=ax_rast.transAxes, fontsize=5.3, color="#64748B", style="italic"
    )

    ax_rast.set_yticks(y_pos)
    ax_rast.set_yticklabels(y_labels)
    ax_rast.set_xlabel("Experiment Time (s)")
    ax_rast.set_xlim(0, gt.duration_s)
    ax_rast.set_ylim(-0.65, 3.85)
    ax_rast.grid(True, color=C_GRID, linestyle=":", axis="x")
    ax_rast.legend(loc="upper right", ncol=2, fontsize=5.3, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel (d): Position Error & Safe-Stop Commanded Velocity
    # -------------------------------------------------------------------------
    ax_err.set_title("(d) Position Error Over Time", fontweight="bold")
    ax_err.axvspan(outage_t0, outage_t1, color=C_SHADE, alpha=0.45, label="GNSS Outage")
    ax_err.axvspan(slip_t0, slip_t1, color=C_SLIP, alpha=0.35, label="Wheel Slip")

    err_dr = np.hypot(dr_poses[:, 0] - gt_poses[:, 0], dr_poses[:, 1] - gt_poses[:, 1])
    err_strict = np.hypot(strict_poses[:, 0] - gt_poses[:, 0], strict_poses[:, 1] - gt_poses[:, 1])
    err_std = np.hypot(data["est_R1_standard_ekf"][:, 0] - gt_poses[:, 0], data["est_R1_standard_ekf"][:, 1] - gt_poses[:, 1])
    err_recov = np.hypot(recov_poses[:, 0] - gt_poses[:, 0], recov_poses[:, 1] - gt_poses[:, 1])

    ax_err.plot(t_rel, err_dr, color=C_DR, ls="--", label="Odometry DR", lw=1.1)
    ax_err.plot(t_rel, err_strict, color=C_STRICT, ls="-.", label="Strict Gate (R3)", lw=1.3)
    ax_err.plot(t_rel, err_std, color=C_STD, ls=":", label="Standard EKF (R1)", lw=1.1)
    ax_err.plot(t_rel, err_recov, color=C_RECOV, ls="-", label="Recoverable (R4)", lw=1.5)
    ax_err.axhline(0.35, color="#64748B", ls="--", lw=0.8, alpha=0.8, label="0.35m Threshold")

    idx_38 = int(np.argmin(np.abs(t_rel - 38.0)))
    ax_err.annotate(
        "Gate Lockout\n(55.6% fixes rejected)",
        xy=(t_rel[idx_38], float(err_strict[idx_38])), xytext=(26.5, 2.22),
        arrowprops=dict(arrowstyle="->", color=C_STRICT, lw=0.9),
        fontsize=5.8, color=C_STRICT, fontweight="bold"
    )

    if has_reloc:
        ax_err.annotate(
            f"Re-anchored\n(t={t_rel[k_reloc]:.1f}s, {err_recov[k_reloc]:.2f}m)",
            xy=(t_rel[k_reloc], float(err_recov[k_reloc])), xytext=(35.5, 1.35),
            arrowprops=dict(arrowstyle="->", color=C_RECOV, lw=0.9),
            fontsize=5.8, color=C_RECOV, fontweight="bold"
        )

    ax_err.set_xlabel("Experiment Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_xlim(0, gt.duration_s)
    ax_err.set_ylim(0, 2.75)
    ax_err.grid(True, color=C_GRID, linestyle=":")
    ax_err.legend(loc="upper left", fontsize=5.3, framealpha=0.88)

    # -------------------------------------------------------------------------
    # Panel (e): Decoupled Vision Tracking vs Slip-Dampened EKF
    # -------------------------------------------------------------------------
    def _get_row(cfg_name):
        sub = df_summary[df_summary["configuration"] == cfg_name]
        return sub.iloc[0] if not sub.empty else None

    r_clean_disc = _get_row("R4_recoverable_temporal")
    r_mod_disc = _get_row("R4_moderate_degradation")
    r_sev_disc = _get_row("R4_severe_degradation")
    r_clean_conn = _get_row("R4_vision_connected_clean")
    r_mod_conn = _get_row("R4_vision_connected_moderate")
    r_sev_conn = _get_row("R4_vision_connected_severe")

    modes = ["Clean", "Moderate", "Severe"]
    x_pos = np.arange(len(modes))

    def _to_pct(row):
        val = float(row["avg_inlier_ratio"])
        return val * 100.0 if val <= 1.0 else val

    inlier_ratios = [_to_pct(r_clean_disc), _to_pct(r_mod_disc), _to_pct(r_sev_disc)]
    rmse_disc = [float(r_clean_disc["pos_rmse_m"]), float(r_mod_disc["pos_rmse_m"]), float(r_sev_disc["pos_rmse_m"])]
    rmse_conn = [float(r_clean_conn["pos_rmse_m"]), float(r_mod_conn["pos_rmse_m"]), float(r_sev_conn["pos_rmse_m"])]

    # (e1) Top: KLT Inlier Ratio (%)
    ax_vis_top.bar(x_pos, inlier_ratios, width=0.45, color=["#2563EB", "#0284C7", "#EF4444"], alpha=0.85, edgecolor="#1E293B", lw=0.6)
    ax_vis_top.set_ylabel("Inliers (%)", fontsize=6.8)
    ax_vis_top.set_ylim(0, 115)
    ax_vis_top.set_xticks(x_pos)
    ax_vis_top.set_xticklabels([])
    ax_vis_top.grid(True, color=C_GRID, linestyle=":", axis="y")
    ax_vis_top.set_title("(e) Vision Tracking & Slip Compensation", fontweight="bold")
    for idx_b, val_b in enumerate(inlier_ratios):
        ax_vis_top.text(idx_b, val_b + 3, f"{val_b:.1f}%", ha="center", va="bottom", fontsize=5.8, fontweight="bold")

    # (e2) Bottom: Position RMSE (m) comparing Disconnected vs Connected
    w_pair = 0.35
    ax_vis_bot.bar(x_pos - w_pair / 2, rmse_disc, w_pair, label="Disconnected", color="#94A3B8", alpha=0.88, edgecolor="#1E293B", lw=0.6)
    ax_vis_bot.bar(x_pos + w_pair / 2, rmse_conn, w_pair, label="Connected Flow", color="#10B981", alpha=0.88, edgecolor="#1E293B", lw=0.6)
    ax_vis_bot.set_ylabel("Pos RMSE (m)", fontsize=6.8)
    ax_vis_bot.set_ylim(0, 1.45)
    ax_vis_bot.set_xticks(x_pos)
    ax_vis_bot.set_xticklabels(modes, fontsize=6.5)
    ax_vis_bot.grid(True, color=C_GRID, linestyle=":", axis="y")
    ax_vis_bot.legend(loc="upper left", fontsize=5.2, framealpha=0.90)

    # -------------------------------------------------------------------------
    # Panel (f): Recovery Performance & Tradeoff Comparison
    # -------------------------------------------------------------------------
    ax_trade.set_title("(f) Recovery Tradeoff: Error vs Acceptance", fontweight="bold")

    plot_configs = [
        ("R0_dead_reckoning", "R0: Odometry DR", C_DR, "s"),
        ("R3_strict_gated_ekf", "R3: Strict Gate (Lockout)", C_STRICT, "X"),
        ("R1_standard_ekf", "R1: Standard EKF", C_STD, "o"),
        ("R2_bias_aware_ekf", "R2: Bias-Aware EKF", "#6366F1", "^"),
        ("R6_recoverable_cov_adj", "R6: Cov-Adjusted", "#0284C7", "v"),
        ("R4_recoverable_temporal", "R4/R5/R7: Temporal / Re-init", C_RECOV, "*"),
        ("R4_vision_connected_clean", "R4+VO: Connected Clean", "#15803D", "P"),
    ]

    for cfg_key, label_name, color, marker in plot_configs:
        sub = df_summary[df_summary["configuration"] == cfg_key]
        if not sub.empty:
            rmse = float(sub["pos_rmse_m"].values[0])
            accept = float(sub["gnss_acceptance_pct"].values[0])
            sz = 75 if marker in ["*", "X", "P"] else 45
            ax_trade.scatter(accept, rmse, color=color, marker=marker, s=sz, label=label_name, zorder=6, edgecolors="black", linewidth=0.5)

    ax_trade.axvspan(90, 105, ymin=0.0, ymax=0.55, color="#DCFCE7", alpha=0.45, zorder=1)
    ax_trade.text(48, 0.85, "Target Recovery Zone:\nAcceptance >= 90%, RMSE < 1.15m", fontsize=5.6, color="#15803D", fontweight="bold")

    ax_trade.set_xlabel("Post-Outage GNSS Fix Acceptance (%) [27 fixes]")
    ax_trade.set_ylabel("Position RMSE (m)")
    ax_trade.set_xlim(-6, 108)
    ax_trade.set_ylim(0.75, 1.55)
    ax_trade.grid(True, color=C_GRID, linestyle=":")
    ax_trade.legend(loc="lower left", fontsize=5.1, framealpha=0.90)

    # Save outputs
    pdf_path = ASSETS_DIR / "figure_realdata.pdf"
    png_path = ASSETS_DIR / "figure_realdata.png"
    svg_path = ASSETS_DIR / "figure_realdata.svg"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    import shutil
    for target_dir in [FIGURES_DIR, FWL_ASSETS]:
        shutil.copy2(pdf_path, target_dir / "figure_realdata.pdf")
        shutil.copy2(png_path, target_dir / "figure_realdata.png")
        shutil.copy2(svg_path, target_dir / "figure_realdata.svg")

    plt.close(fig)
    print(f"[Figure] Exported figure_realdata to:\n  {pdf_path}\n  {png_path}\n  {svg_path}")


if __name__ == "__main__":
    generate_main_figure()
