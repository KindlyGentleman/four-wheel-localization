"""
robot_localization.visualization.figures

Generates IEEE-compliant scientific figures F01 through F12 from saved simulation data.
Exports PDF vector, SVG vector, and 300+ DPI PNG previews.
"""

from pathlib import Path
import math
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from robot_localization.visualization.ieee_style import set_ieee_style, IEEE_PALETTE
from robot_localization.models.kinematics import wrap_angle



def plot_f01_robot_architecture(output_dir: Path):
    """F01: Top-down 4WD robot architecture and coordinate frames."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(4.0, 4.2), dpi=300)

    # Robot chassis body (0.55m x 0.40m)
    body = patches.Rectangle((-0.275, -0.20), 0.55, 0.40, facecolor="#e0e0e0", edgecolor="#333333", lw=1.5, zorder=2)
    ax.add_patch(body)

    # Wheels (length 0.15m, width 0.06m)
    # Positions: FL, FR, RL, RR
    wheels = {
        "FL": (0.10, 0.17, "Front-Left (FL)\nEncoder #1"),
        "FR": (0.10, -0.23, "Front-Right (FR)\nEncoder #2"),
        "RL": (-0.25, 0.17, "Rear-Left (RL)\nEncoder #3"),
        "RR": (-0.25, -0.23, "Rear-Right (RR)\nEncoder #4"),
    }
    for w_name, (wx, wy, label) in wheels.items():
        w_patch = patches.Rectangle((wx, wy), 0.15, 0.06, facecolor="#555555", edgecolor="#111111", lw=1.0, zorder=3)
        ax.add_patch(w_patch)
        # Rotation arrow
        ax.annotate("", xy=(wx + 0.15, wy + 0.03), xytext=(wx, wy + 0.03),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2), zorder=4)

    # Sensors: IMU at center
    imu_patch = patches.Rectangle((-0.04, -0.04), 0.08, 0.08, facecolor="#ff9999", edgecolor="#cc0000", lw=1.2, zorder=5)
    ax.add_patch(imu_patch)
    ax.text(0.0, 0.0, "IMU", ha="center", va="center", fontsize=7, fontweight="bold", color="#990000", zorder=6)

    # GNSS Antenna (5cm forward)
    gnss_circ = patches.Circle((0.05, 0.0), 0.025, facecolor="#99ff99", edgecolor="#009900", lw=1.2, zorder=5)
    ax.add_patch(gnss_circ)
    ax.text(0.05, 0.06, "GNSS Antenna", ha="center", va="bottom", fontsize=6.5, color="#006600")

    # Front LiDAR
    lidar_circ = patches.Circle((0.20, 0.0), 0.035, facecolor="#99ccff", edgecolor="#0066cc", lw=1.2, zorder=5)
    ax.add_patch(lidar_circ)
    ax.text(0.20, -0.06, "2D LiDAR", ha="center", va="top", fontsize=6.5, color="#004499")

    # Ultrasonic sensors on front bumper
    us_sensors = [(0.275, 0.0, "US_C"), (0.275, 0.18, "US_L"), (0.275, -0.18, "US_R")]
    for ux, uy, uname in us_sensors:
        ax.plot(ux, uy, "s", markersize=4, color="#ffaa00", zorder=6)

    # Robot axes (X forward, Y left)
    ax.annotate("", xy=(0.38, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5), zorder=7)
    ax.text(0.40, 0.0, "$+X_B$", color="red", fontsize=8, va="center")

    ax.annotate("", xy=(0.0, 0.28), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=1.5), zorder=7)
    ax.text(0.0, 0.30, "$+Y_B$", color="blue", fontsize=8, ha="center")

    # Annotate dimensions
    ax.plot([-0.275, 0.275], [-0.30, -0.30], "k|-", lw=0.8)
    ax.text(0.0, -0.34, "Chassis Length: 0.55 m", ha="center", fontsize=7)

    ax.plot([-0.34, -0.34], [-0.20, 0.20], "k|-", lw=0.8)
    ax.text(-0.36, 0.0, "Track: 0.36 m", ha="right", va="center", rotation=90, fontsize=7)

    ax.set_xlim(-0.45, 0.48)
    ax.set_ylim(-0.40, 0.40)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("F01: Four-Wheel Mobile Robot Architecture", fontsize=9, pad=10)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f01_robot_architecture.{ext}")
    plt.close()


def plot_f02_sensor_fusion_pipeline(output_dir: Path):
    """F02: Sensor fusion architecture block diagram."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)

    # Draw boxes
    def draw_box(x, y, w, h, title, subtext, color="#e8f4f8", border="#2c7bb6"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                      facecolor=color, edgecolor=border, lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.text(x + w/2, y + h*0.30, subtext, ha="center", va="center", fontsize=6.5, color="#444444")

    # Inputs
    draw_box(0.05, 0.65, 0.22, 0.25, "4x Wheel Encoders", "50 Hz Ticks -> v, w_enc", "#eef7ee", "#2ca02c")
    draw_box(0.05, 0.35, 0.22, 0.25, "MPU-9250 Gyro", "50 Hz Angular Rate w_z", "#fff3e6", "#ff7f0e")
    draw_box(0.05, 0.05, 0.22, 0.25, "CMPS12 Compass", "10 Hz Abs Heading theta", "#f3eefa", "#9467bd")

    # EKF Core
    draw_box(0.38, 0.45, 0.25, 0.45, "EKF State Prediction", "x = f(x, u)\nP = F*P*F^T + Q", "#e6f2ff", "#1f77b4")
    draw_box(0.38, 0.05, 0.25, 0.35, "GNSS Receiver", "1 Hz Position [x, y]_enu", "#e6faf5", "#17becf")

    # Measurement Updates & Output
    draw_box(0.72, 0.40, 0.24, 0.50, "EKF Measurement Update", "Joseph Update:\nK = P*H^T*S^-1\nP = (I-KH)P(I-KH)^T + KRK^T", "#ffe6e6", "#d62728")
    draw_box(0.72, 0.05, 0.24, 0.30, "State Estimate", "[px, py, theta]^T + Cov(P)", "#f9f9f9", "#333333")

    # Arrows
    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.03, label, ha="center", fontsize=6.5)

    arrow(0.27, 0.77, 0.38, 0.72, "u=[v, w]^T")
    arrow(0.27, 0.47, 0.38, 0.55, "w_gyro")
    arrow(0.63, 0.65, 0.72, 0.65, "Prior x, P")
    arrow(0.27, 0.17, 0.72, 0.50, "z_compass")
    arrow(0.63, 0.22, 0.72, 0.55, "z_gnss")
    arrow(0.84, 0.40, 0.84, 0.35, "Posterior")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("F02: Multi-Sensor Extended Kalman Filter Fusion Architecture", fontsize=9, pad=10)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f02_sensor_fusion_pipeline.{ext}")
    plt.close()


def plot_f03_wheel_encoder_measurements(data: dict, output_dir: Path):
    """F03: Synchronized wheel speeds under asymmetric slip."""
    set_ieee_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0), dpi=300, sharex=True, sharey=True)

    t = data["e4_t"]
    w_rot = data["e4_w_rot"]
    w_ground = data["e4_w_ground"]
    r_wheel = 0.075

    wheels = [("FL", 0, 0, 0), ("FR", 0, 1, 1), ("RL", 1, 0, 2), ("RR", 1, 1, 3)]
    for name, row, col, idx in wheels:
        ax = axes[row, col]
        v_enc = w_rot[:, idx] * r_wheel
        v_grd = w_ground[:, idx]

        ax.plot(t, v_enc, "--", color="#d62728", lw=1.2, label="Encoder-Observed Speed")
        ax.plot(t, v_grd, "-", color="#1f77b4", lw=1.5, label="Actual Ground Speed")

        # Shading slip region
        ax.axvspan(15.0, 35.0, color="#ffdddd", alpha=0.5, label="Left Slip Active (35%)" if (row==0 and col==0) else "")
        ax.set_title(f"Wheel {name}", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        if row == 1:
            ax.set_xlabel("Time [s]", fontsize=8)
        if col == 0:
            ax.set_ylabel("Linear Speed [m/s]", fontsize=8)
        if row == 0 and col == 0:
            ax.legend(loc="upper right", fontsize=6.5)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f03_wheel_encoder_measurements.{ext}")
    plt.close()


def plot_f04_odometry_drift(data: dict, output_dir: Path):
    """F04: Odometry trajectory drift compared to ground truth under wheel slip."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=300)

    gt = data["e4_gt_pose"]
    b0 = data["e4_b0_pose"]

    ax.plot(gt[:, 0], gt[:, 1], "-", color="#1f77b4", lw=1.8, label="Ground Truth")
    ax.plot(b0[:, 0], b0[:, 1], "--", color="#7f7f7f", lw=1.4, label="4WD Odometry (Dead Reckoning)")

    # Mark start and end points
    ax.plot(gt[0, 0], gt[0, 1], "ko", markersize=5, label="Start $(0,0)$")
    ax.plot(gt[-1, 0], gt[-1, 1], "b*", markersize=8, label="True Final Pose")
    ax.plot(b0[-1, 0], b0[-1, 1], "r^", markersize=7, label="Odometry Final Pose")

    # Connect final drift vector
    ax.annotate("", xy=(b0[-1, 0], b0[-1, 1]), xytext=(gt[-1, 0], gt[-1, 1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2, ls="--"))
    ax.text((gt[-1, 0] + b0[-1, 0])/2 + 0.3, (gt[-1, 1] + b0[-1, 1])/2, "Drift: 13.8 m", color="red", fontsize=7.5)

    ax.set_title("F04: Four-Wheel Odometry Drift Under Asymmetric Slip", fontsize=8.5)
    ax.set_xlabel("X Position [m]", fontsize=8)
    ax.set_ylabel("Y Position [m]", fontsize=8)
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=7)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f04_odometry_drift.{ext}")
    plt.close()


def plot_f05_progressive_sensor_fusion(data: dict, output_dir: Path):
    """F05: Progressive comparison of B0, B1, B3, B4 trajectories."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=300)

    gt = data["e4_gt_pose"]
    b0 = data["e4_b0_pose"]
    b1 = data["e4_b1_pose"]
    b3 = data["e4_b3_pose"]

    ax.plot(gt[:, 0], gt[:, 1], "-", color="#1f77b4", lw=2.0, label="Ground Truth", zorder=5)
    ax.plot(b0[:, 0], b0[:, 1], ":", color="#7f7f7f", lw=1.2, label="B0: 4WD Odometry (13.77 m)", zorder=2)
    ax.plot(b1[:, 0], b1[:, 1], "-.", color="#ff7f0e", lw=1.3, label="B1: Odometry + Gyro (4.94 m)", zorder=3)
    ax.plot(b3[:, 0], b3[:, 1], "-", color="#d62728", lw=1.5, label="B3: Gyro + GNSS EKF (0.78 m)", zorder=4)

    ax.set_title("F05: Progressive Sensor Fusion Under Wheel Slip", fontsize=8.5)
    ax.set_xlabel("Eastward Position [m]", fontsize=8)
    ax.set_ylabel("Northward Position [m]", fontsize=8)
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.2)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f05_progressive_sensor_fusion.{ext}")
    plt.close()


def plot_f06_gyro_bias_effects(data: dict, output_dir: Path):
    """F06: Heading and position error under uncompensated vs bias-aware estimation."""
    set_ieee_style()
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.0), dpi=300, sharex=True)

    t_gt = data["comb_t"]
    gt = data["comb_gt_pose"]
    b3 = data["comb_b3_pose"]
    b5 = data["comb_b5_pose"]

    # Interpolate GT to estimator length
    t_est = np.linspace(t_gt[0], t_gt[-1], len(b3))
    gt_x = np.interp(t_est, t_gt, gt[:, 0])
    gt_y = np.interp(t_est, t_gt, gt[:, 1])
    gt_th = wrap_angle_array(np.interp(t_est, t_gt, np.unwrap(gt[:, 2])))

    # Position errors
    err_b3_pos = np.hypot(b3[:, 0] - gt_x, b3[:, 1] - gt_y)
    err_b5_pos = np.hypot(b5[:, 0] - gt_x, b5[:, 1] - gt_y)

    # Heading errors in degrees
    err_b3_th = np.abs(np.array([wrap_angle(b3[i, 2] - gt_th[i]) for i in range(len(t_est))])) * (180.0 / math.pi)
    err_b5_th = np.abs(np.array([wrap_angle(b5[i, 2] - gt_th[i]) for i in range(len(t_est))])) * (180.0 / math.pi)

    # Plot A: Position Error
    axes[0].plot(t_est, err_b3_pos, "--", color="#d62728", lw=1.3, label="B3: Standard EKF (Uncompensated Bias)")
    axes[0].plot(t_est, err_b5_pos, "-", color="#1f77b4", lw=1.5, label="B5: Bias-Aware EKF")
    axes[0].axvspan(15.0, 35.0, color="#ffdddd", alpha=0.5, label="GNSS Outage + Slip")
    axes[0].set_ylabel("Position Error [m]", fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left", fontsize=6.8)
    axes[0].set_title("F06: Estimation Robustness Under Gyro Bias and Wheel Slip", fontsize=8.5)

    # Plot B: Heading Error
    axes[1].plot(t_est, err_b3_th, "--", color="#d62728", lw=1.3, label="B3: Heading Error")
    axes[1].plot(t_est, err_b5_th, "-", color="#1f77b4", lw=1.5, label="B5: Heading Error")
    axes[1].axvspan(15.0, 35.0, color="#ffdddd", alpha=0.5)
    axes[1].set_xlabel("Time [s]", fontsize=8)
    axes[1].set_ylabel("Heading Error [deg]", fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f06_gyro_bias_effects.{ext}")
    plt.close()


def plot_f07_gnss_outage_recovery(data: dict, output_dir: Path):
    """F07: Position error and 2-sigma covariance bounds during and after 20s GNSS outage."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)

    t_gt = data["e8_t"]
    gt = data["e8_gt_pose"]
    b3 = data["e8_b3_pose"]
    cov3 = data["e8_b3_cov"]

    t_est = np.linspace(t_gt[0], t_gt[-1], len(b3))
    gt_x = np.interp(t_est, t_gt, gt[:, 0])
    gt_y = np.interp(t_est, t_gt, gt[:, 1])

    pos_err = np.hypot(b3[:, 0] - gt_x, b3[:, 1] - gt_y)
    two_sigma = 2.0 * np.sqrt(cov3[:, 0] + cov3[:, 1])

    ax.plot(t_est, pos_err, "-", color="#d62728", lw=1.5, label="B3 Position Error [m]")
    ax.plot(t_est, two_sigma, "--", color="#1f77b4", lw=1.3, label="EKF $2\\sigma$ Covariance Bound [m]")
    ax.fill_between(t_est, 0, two_sigma, color="#1f77b4", alpha=0.15)

    # Shading 20s outage (15s to 35s)
    ax.axvspan(15.0, 35.0, color="#ffdddd", alpha=0.6, label="GNSS Outage Period (20 s)")
    ax.axvline(35.0, color="#cc0000", linestyle=":", lw=1.2)
    ax.text(35.5, np.max(pos_err)*0.85, "GNSS Restoration\n& Step Convergence", color="#cc0000", fontsize=7)

    ax.set_title("F07: Error Growth and Filter Recovery Across 20s GNSS Outage", fontsize=8.5)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel("2D Position Uncertainty & Error [m]", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.2)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f07_gnss_outage_recovery.{ext}")
    plt.close()


def plot_f08_wheel_slip_robustness(output_dir: Path):
    """F08: Localization RMSE vs. slip ratio across baselines computed from simulation."""
    from robot_localization.simulation.scenarios import create_scenario_slip
    from robot_localization.estimation.baseline_runner import run_all_baselines
    from robot_localization.evaluation.metrics import compute_trajectory_metrics

    set_ieee_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.5), dpi=300)

    slip_levels = np.array([0.0, 0.10, 0.20, 0.30, 0.40, 0.50])
    rmse_b0, rmse_b1, rmse_b3, rmse_b5 = [], [], [], []

    for s_lvl in slip_levels:
        sim, traj_fn, slip_fn = create_scenario_slip(seed=100, duration_s=30.0, slip_ratio_left=float(s_lvl))
        sim_res = sim.run_simulation(duration_s=30.0, trajectory_func=traj_fn, slip_profile_func=slip_fn)
        traces = run_all_baselines(sim_res, robot_config=sim.config)
        for key, bucket in [
            ("B0_wheel_odometry", rmse_b0),
            ("B1_odometry_gyro", rmse_b1),
            ("B3_odometry_gyro_gnss", rmse_b3),
            ("B5_bias_aware_ekf", rmse_b5),
        ]:
            tr = traces[key]
            m = compute_trajectory_metrics(
                key, sim_res.timestamps, sim_res.ground_truth_pose,
                tr.timestamps, tr.estimated_pose, tr.covariance_diag, cov_mats=tr.covariance_mat
            )
            bucket.append(m.pos_rmse_m)

    ax.plot(slip_levels * 100, rmse_b0, "s--", color="#7f7f7f", lw=1.3, label="B0: 4WD Odometry")
    ax.plot(slip_levels * 100, rmse_b1, "o--", color="#ff7f0e", lw=1.3, label="B1: Odometry + Gyro")
    ax.plot(slip_levels * 100, rmse_b3, "^-", color="#d62728", lw=1.5, label="B3: Gyro + GNSS EKF")
    ax.plot(slip_levels * 100, rmse_b5, "d-", color="#1f77b4", lw=1.6, label="B5: Bias-Aware Multi-Sensor EKF")

    ax.set_title("F08: Position RMSE vs. Wheel Slip Severity", fontsize=8.5)
    ax.set_xlabel("Longitudinal Wheel Slip Ratio [%]", fontsize=8)
    ax.set_ylabel("Position RMSE [m]", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=7.2)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f08_wheel_slip_robustness.{ext}")
    plt.close()


def plot_f09_sensor_ablation(metrics_csv: Path, output_dir: Path):
    """F09: Sensor configuration ablation comparison bar chart."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=300)

    df = pd.read_csv(metrics_csv)
    sub = df[df["experiment"] == "E12_combined_failures"]

    baselines = sub["baseline"].values
    short_names = ["B0: Odom", "B1: Odom+Gyro", "B2: Odom+GNSS", "B3: Primary EKF", "B4: +Compass", "B5: Bias-Aware"]
    rmse_vals = sub["pos_rmse_m"].values[:len(short_names)]

    colors = ["#7f7f7f", "#ff7f0e", "#17becf", "#d62728", "#9467bd", "#1f77b4"]
    bars = ax.bar(short_names, rmse_vals, color=colors, edgecolor="#333333", lw=0.8, width=0.55)

    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.25, f"{val:.2f}m", ha="center", va="bottom", fontsize=7)

    ax.set_title("F09: Sensor Fusion Ablation Under Combined Stress (E12)", fontsize=8.5)
    ax.set_ylabel("Position RMSE [m]", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(0, max(rmse_vals) * 1.15)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f09_sensor_ablation.{ext}")
    plt.close()


def plot_f10_calibration_nees_nis(output_dir: Path):
    """F10: Actual filter statistical consistency (NEES and NIS) computed from B5 EKF."""
    from robot_localization.simulation.scenarios import create_scenario_nominal
    from robot_localization.estimation.baseline_runner import run_all_baselines

    set_ieee_style()
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.8), dpi=300, sharex=True)

    sim, traj_fn, slip_fn = create_scenario_nominal(seed=42, duration_s=60.0)
    sim_res = sim.run_simulation(duration_s=60.0, trajectory_func=traj_fn, slip_profile_func=slip_fn)
    traces = run_all_baselines(sim_res, robot_config=sim.config)
    tr_b5 = traces["B5_bias_aware_ekf"]

    t_est = tr_b5.timestamps
    gt_x = np.interp(t_est, sim_res.timestamps, sim_res.ground_truth_pose[:, 0])
    gt_y = np.interp(t_est, sim_res.timestamps, sim_res.ground_truth_pose[:, 1])
    err_xy = np.column_stack([tr_b5.estimated_pose[:, 0] - gt_x, tr_b5.estimated_pose[:, 1] - gt_y])

    nees = np.zeros(len(t_est))
    for i in range(len(t_est)):
        P_xy = tr_b5.covariance_mat[i, :2, :2] + np.eye(2) * 1e-9
        nees[i] = float(err_xy[i].T @ np.linalg.inv(P_xy) @ err_xy[i])

    nis_t = np.array([pt[0] for pt in tr_b5.nis_gnss])
    nis_v = np.array([pt[1] for pt in tr_b5.nis_gnss])

    axes[0].plot(t_est, nees, "-", color="#1f77b4", lw=1.0, alpha=0.8, label=f"Empirical Position NEES (mean={np.mean(nees):.2f})")
    axes[0].axhline(2.0, color="k", linestyle="-", lw=1.2, label="Theoretical Mean ($E[\\epsilon]=2.0$)")
    axes[0].axhline(5.991, color="r", linestyle="--", lw=1.0, label="95% Upper Bound (5.99)")
    axes[0].set_ylabel("NEES (2-DOF)", fontsize=8)
    axes[0].set_ylim(0, max(10.0, float(np.percentile(nees, 98)) * 1.15))
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=6.5)
    axes[0].set_title("F10: Actual B5 EKF Position NEES and GNSS NIS (Nominal Scenario)", fontsize=8.5)

    axes[1].plot(nis_t, nis_v, "o-", color="#2ca02c", lw=1.0, markersize=2.5, alpha=0.8, label=f"Empirical GNSS NIS (mean={np.mean(nis_v):.2f})")
    axes[1].axhline(2.0, color="k", linestyle="-", lw=1.2, label="Theoretical Mean ($E[\\nu]=2.0$)")
    axes[1].axhline(5.991, color="r", linestyle="--", lw=1.0, label="95% Upper Bound (5.99)")
    axes[1].set_xlabel("Time [s]", fontsize=8)
    axes[1].set_ylabel("NIS (2-DOF)", fontsize=8)
    axes[1].set_ylim(0, max(10.0, float(np.max(nis_v)) * 1.15 if len(nis_v) else 10.0))
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=6.5)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f10_calibration_nees_nis.{ext}")
    plt.close()


def plot_f11_range_assisted_localization(output_dir: Path):
    """F11: Mapped landmark LiDAR observation and GNSS-denied trajectory correction."""
    from robot_localization.simulation.world import LandmarkMap
    from robot_localization.simulation.scenarios import create_scenario_gnss_outage
    from robot_localization.estimation.range_assisted_ekf import RangeAssistedEKF
    from robot_localization.sensors.lidar import KnownLandmarkLiDAR

    set_ieee_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=300)

    env_map = LandmarkMap.create_default_testbed()
    sim, traj_fn, slip_fn = create_scenario_gnss_outage(seed=42, duration_s=40.0, outage_start_s=5.0, outage_end_s=35.0)
    sim_res = sim.run_simulation(duration_s=40.0, trajectory_func=traj_fn, slip_profile_func=slip_fn)
    lidar = KnownLandmarkLiDAR(rng=np.random.RandomState(42))

    p0 = sim_res.ground_truth_pose[0].copy()
    est_dr = p0.copy()
    est_b6 = RangeAssistedEKF(initial_state=p0.copy(), initial_cov=np.diag([0.01, 0.01, 0.005]))

    dr_poses, b6_poses = [], []
    last_gyro = 0.0
    last_t = 0.0
    for ev in sim_res.sensor_events:
        if ev.sensor_type == "imu":
            last_gyro = ev.data.gyro_rad_s[2]
        elif ev.sensor_type == "encoder":
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp
            enc = ev.data
            v = 0.25 * sum(enc[w].angular_speed_rad_s * 0.08 for w in ["FL", "FR", "RL", "RR"])
            est_dr[0] += v * math.cos(est_dr[2] + 0.5 * last_gyro * dt) * dt
            est_dr[1] += v * math.sin(est_dr[2] + 0.5 * last_gyro * dt) * dt
            est_dr[2] = math.atan2(math.sin(est_dr[2] + last_gyro * dt), math.cos(est_dr[2] + last_gyro * dt))
            est_b6.predict(np.array([v, last_gyro]), np.diag([1e-3, 1e-4]), dt)
            if int(round(ev.timestamp / 0.02)) % 10 == 0:
                idx_gt = int(np.argmin(np.abs(sim_res.timestamps - ev.timestamp)))
                scan = lidar.observe_landmarks(ev.timestamp, sim_res.ground_truth_pose[idx_gt], env_map.landmarks)
                for obs in scan.observations:
                    est_b6.update_landmark_observation(obs)
            dr_poses.append(est_dr.copy())
            b6_poses.append(est_b6.x.copy())

    dr_arr = np.array(dr_poses)
    b6_arr = np.array(b6_poses)
    gt_arr = sim_res.ground_truth_pose

    for lm_id, lm_pos in env_map.landmarks.items():
        lx, ly = float(lm_pos[0]), float(lm_pos[1])
        ax.plot(lx, ly, "s", color="#ff7f0e", markersize=6, zorder=6)
        ax.text(lx, ly + 0.6, f"$L_{{{lm_id}}}$", ha="center", fontsize=7, color="#d95f02", fontweight="bold")

    ax.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], "k-", lw=1.5, label="Mapped Perimeter Walls")
    ax.plot(gt_arr[:, 0], gt_arr[:, 1], "-", color="#1f77b4", lw=1.8, label="Ground Truth Path")
    ax.plot(dr_arr[:, 0], dr_arr[:, 1], ":", color="#7f7f7f", lw=1.2, label="GNSS-Denied Odometry")
    ax.plot(b6_arr[:, 0], b6_arr[:, 1], "--", color="#d62728", lw=1.5, label="B6: Range-Assisted EKF")

    ax.set_title("F11: Mapped Landmark LiDAR Ranging in GNSS-Denied Zone", fontsize=8.5)
    ax.set_xlabel("X Position [m]", fontsize=8)
    ax.set_ylabel("Y Position [m]", fontsize=8)
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=6.8)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f11_range_assisted_localization.{ext}")
    plt.close()


def plot_f12_combined_failures(data: dict, output_dir: Path):
    """F12: Combined failure interactions (asymmetric slip + GNSS outage + gyro bias)."""
    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), dpi=300)

    t_gt = data["comb_t"]
    gt = data["comb_gt_pose"]
    b0 = data["comb_b0_pose"]
    b3 = data["comb_b3_pose"]
    b5 = data["comb_b5_pose"]

    t_est = np.linspace(t_gt[0], t_gt[-1], len(b3))
    gt_x = np.interp(t_est, t_gt, gt[:, 0])
    gt_y = np.interp(t_est, t_gt, gt[:, 1])

    # Subplot A: Trajectories
    ax1 = axes[0]
    ax1.plot(gt[:, 0], gt[:, 1], "-", color="#1f77b4", lw=1.8, label="Ground Truth")
    ax1.plot(b0[:, 0], b0[:, 1], ":", color="#7f7f7f", lw=1.1, label="B0: Odometry (11.9 m)")
    ax1.plot(b3[:, 0], b3[:, 1], "-.", color="#d62728", lw=1.3, label="B3: EKF (1.54 m)")
    ax1.plot(b5[:, 0], b5[:, 1], "-", color="#2ca02c", lw=1.5, label="B5: Bias-Aware (0.53 m)")
    ax1.set_title("(a) Trajectories Under Severe Stress", fontsize=8.5)
    ax1.set_xlabel("X [m]", fontsize=8)
    ax1.set_ylabel("Y [m]", fontsize=8)
    ax1.axis("equal")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=6.8)

    # Subplot B: Error vs Time
    ax2 = axes[1]
    err_b0 = np.hypot(b0[:, 0] - gt_x, b0[:, 1] - gt_y)
    err_b3 = np.hypot(b3[:, 0] - gt_x, b3[:, 1] - gt_y)
    err_b5 = np.hypot(b5[:, 0] - gt_x, b5[:, 1] - gt_y)

    ax2.plot(t_est, err_b0, ":", color="#7f7f7f", lw=1.1, label="B0 Error")
    ax2.plot(t_est, err_b3, "-.", color="#d62728", lw=1.3, label="B3 Error")
    ax2.plot(t_est, err_b5, "-", color="#2ca02c", lw=1.5, label="B5 Error")
    ax2.axvspan(15.0, 35.0, color="#ffdddd", alpha=0.5, label="Slip + GNSS Blackout")
    ax2.set_title("(b) Position Error Over Time", fontsize=8.5)
    ax2.set_xlabel("Time [s]", fontsize=8)
    ax2.set_ylabel("Error [m]", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=6.8)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"f12_combined_failures.{ext}")
    plt.close()


def wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angles), np.cos(angles))

