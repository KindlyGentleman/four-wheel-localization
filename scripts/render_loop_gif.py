#!/usr/bin/env python3
"""
scripts/render_loop_gif.py

Renders a 12-second seamless looping top-down simulation video and GIF:
`assets/localization_loop.mp4` and `assets/localization_loop.gif`.

Visual Features:
  - Dynamic chase camera following the 4WD mobile robot.
  - 4WD robot chassis & 4 wheels drawn to scale (0.50m x 0.45m).
  - Ground truth path, drifting B0 wheel dead-reckoning, and A5 Robust EKF estimate.
  - Dynamic 95% confidence covariance ellipse moving with the filter estimate.
  - GNSS Outage status indicator (flashing red during blackout).
  - Inset global testbed minimap showing the entire periodic lemniscate path and camera FOV.
  - Professional telemetry HUD with real-time tracking errors and speed.
"""

import sys
import math
import shutil
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Ellipse
from matplotlib.animation import FFMpegWriter

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.sensors.gnss import GNSSReceiver
from robot_localization.sensors.cmps12 import CMPS12Sensor
from robot_localization.sensors.mpu9250 import MPU9250Sensor
from robot_localization.simulation.robot_simulator import RobotSimulator
from robot_localization.simulation.world import TrajectoryGenerator
from robot_localization.estimation.baseline_runner import run_all_baselines

ASSETS_DIR = PROJECT_ROOT / "assets"
VIDEOS_DIR = PROJECT_ROOT / "videos"
FIGURES_DIR = PROJECT_ROOT / "figures"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_loop_simulation(duration_s: float = 12.0):
    """Run closed-loop simulation on Gerono lemniscate with GNSS outage."""
    print("Configuring 12s closed-loop simulation...")
    rng = np.random.RandomState(42)
    cfg = Robot4WDConfig()

    # GNSS outage from 4.0s to 8.0s (left curve segment)
    gnss = GNSSReceiver(sample_rate_hz=1.0, outage_intervals=[(4.0, 8.0)], rng=rng)
    cmps = CMPS12Sensor(rng=rng)
    mpu = MPU9250Sensor(rng=rng, gyro_bias_initial_rad_s=0.015)
    sim = RobotSimulator(cfg, gnss, cmps, mpu, rng=rng)

    def traj(t):
        return TrajectoryGenerator.closed_lemniscate(t, duration_s=duration_s, scale_x=4.0, scale_y=2.8)

    def slip(t):
        # Asymmetric slip on left wheels during turn from 3.0s to 7.0s
        if 3.0 <= t <= 7.0:
            return {"FL": 0.20, "FR": 0.0, "RL": 0.20, "RR": 0.0}
        return {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}

    print("Running forward physical simulation (100 Hz)...")
    sim_res = sim.run_simulation(duration_s=duration_s, trajectory_func=traj, slip_profile_func=slip)

    print("Running multi-sensor baseline estimators (50 Hz)...")
    traces = run_all_baselines(sim_res, robot_config=cfg)

    return sim_res, traces


def render_animation(sim_res, traces, duration_s: float = 12.0, fps: int = 15):
    """Render MP4 and convert to seamless GIF."""
    mp4_path = ASSETS_DIR / "localization_loop.mp4"
    gif_path = ASSETS_DIR / "localization_loop.gif"

    total_frames = int(duration_s * fps)
    sim_t = sim_res.timestamps
    gt_pose = sim_res.ground_truth_pose

    b0_trace = traces["B0_wheel_odometry"]
    a5_trace = traces["A5_robust_ekf"]

    # Subsample simulation data to frame timestamps
    frame_times = np.linspace(0, duration_s, total_frames, endpoint=False)

    # Pre-interpolate poses for smooth animation
    gt_x = np.interp(frame_times, sim_t, gt_pose[:, 0])
    gt_y = np.interp(frame_times, sim_t, gt_pose[:, 1])
    gt_th = np.interp(frame_times, sim_t, np.unwrap(gt_pose[:, 2]))

    filt_t = np.arange(len(a5_trace.estimated_pose)) * 0.02
    b0_x = np.interp(frame_times, filt_t, b0_trace.estimated_pose[:, 0])
    b0_y = np.interp(frame_times, filt_t, b0_trace.estimated_pose[:, 1])

    a5_x = np.interp(frame_times, filt_t, a5_trace.estimated_pose[:, 0])
    a5_y = np.interp(frame_times, filt_t, a5_trace.estimated_pose[:, 1])
    a5_th = np.interp(frame_times, filt_t, np.unwrap(a5_trace.estimated_pose[:, 2]))
    a5_cov_xx = np.interp(frame_times, filt_t, a5_trace.covariance_diag[:, 0])
    a5_cov_yy = np.interp(frame_times, filt_t, a5_trace.covariance_diag[:, 1])

    # Obstacles in volcanic testbed
    obstacles = [
        (-2.2, 1.8, 0.45), (2.4, -1.8, 0.50), (-3.2, -1.5, 0.40),
        (3.0, 1.6, 0.45), (0.0, 2.5, 0.35), (0.0, -2.5, 0.35),
        (-4.2, 0.0, 0.55), (4.2, 0.0, 0.55)
    ]

    # Setup figure: 560x560 square format for sharp display and compact GIF size
    fig = plt.figure(figsize=(5.6, 5.6), dpi=100)
    ax_main = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_main.set_aspect("equal")

    # Inset minimap in upper right (clean white card)
    ax_mini = fig.add_axes([0.73, 0.71, 0.24, 0.24])
    ax_mini.set_aspect("equal")
    ax_mini.set_facecolor("#FFFFFF")
    ax_mini.set_xticks([])
    ax_mini.set_yticks([])
    for spine in ax_mini.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)

    # Static elements on minimap
    ax_mini.plot(gt_pose[:, 0], gt_pose[:, 1], color="#94A3B8", lw=1.1, linestyle="--", alpha=0.9)
    # Shaded outage sector on minimap
    outage_mini = Rectangle((-4.8, -1.2), 4.2, 3.6, facecolor="#FEE2E2", alpha=0.7, edgecolor="#FCA5A5", lw=0.6)
    ax_mini.add_patch(outage_mini)
    ax_mini.set_xlim(-6.0, 6.0)
    ax_mini.set_ylim(-4.5, 4.5)

    # Mini robot dot and viewport box
    mini_dot, = ax_mini.plot([], [], "o", color="#F59E0B", markersize=3.8)
    mini_box = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#D97706", lw=0.8, linestyle=":")
    ax_mini.add_patch(mini_box)
    ax_mini.text(0.5, 0.08, "GLOBAL TESTBED", transform=ax_mini.transAxes,
                 fontsize=5.2, color="#64748B", ha="center", fontweight="bold")

    writer = FFMpegWriter(fps=fps, metadata=dict(title="Localization Loop"), bitrate=2000)

    print(f"Rendering {total_frames} animation frames to {mp4_path}...")
    with writer.saving(fig, str(mp4_path), dpi=100):
        for f in range(total_frames):
            ax_main.clear()
            ax_main.set_facecolor("#F8FAFC") # Clean light canvas

            cur_t = frame_times[f]
            rx, ry, rth = gt_x[f], gt_y[f], gt_th[f]
            fx, fy, fth = a5_x[f], a5_y[f], a5_th[f]

            # Chase camera: view centered smoothly on robot position
            view_radius = 2.4
            ax_main.set_xlim(fx - view_radius, fx + view_radius)
            ax_main.set_ylim(fy - view_radius, fy + view_radius)

            # Subtle background grid
            gx_min = math.floor(fx - view_radius)
            gx_max = math.ceil(fx + view_radius)
            gy_min = math.floor(fy - view_radius)
            gy_max = math.ceil(fy + view_radius)
            for gx in range(gx_min, gx_max + 1):
                ax_main.axvline(gx, color="#E2E8F0", lw=0.5, linestyle=":")
            for gy in range(gy_min, gy_max + 1):
                ax_main.axhline(gy, color="#E2E8F0", lw=0.5, linestyle=":")

            # Draw rocks/obstacles in view
            for ox, oy, orad in obstacles:
                if (ox - fx)**2 + (oy - fy)**2 < (view_radius + 1.0)**2:
                    rock = Circle((ox, oy), orad, facecolor="#E2E8F0", edgecolor="#94A3B8", lw=0.8, zorder=2)
                    ax_main.add_patch(rock)

            # Trajectory tails (past 4 seconds)
            tail_steps = int(4.0 * fps)
            tail_start = max(0, f - tail_steps)
            if f > 0:
                # Ground truth tail
                ax_main.plot(gt_x[tail_start:f+1], gt_y[tail_start:f+1],
                             color="#64748B", lw=1.3, linestyle="--", alpha=0.75, zorder=3)
                # B0 Wheel DR tail (drifting)
                ax_main.plot(b0_x[tail_start:f+1], b0_y[tail_start:f+1],
                             color="#8B5CF6", lw=1.2, linestyle=":", alpha=0.85, zorder=4)
                # A5 Robust EKF tail
                ax_main.plot(a5_x[tail_start:f+1], a5_y[tail_start:f+1],
                             color="#0284C7", lw=1.8, linestyle="-", alpha=0.95, zorder=5)

            # 95% Covariance Error Ellipse on Filter Pose
            cov_x = max(a5_cov_xx[f], 1e-4)
            cov_y = max(a5_cov_yy[f], 1e-4)
            ell_w = 2.0 * 2.448 * math.sqrt(cov_x)
            ell_h = 2.0 * 2.448 * math.sqrt(cov_y)
            is_outage = (4.0 <= cur_t <= 8.0)
            ell_color = "#DC2626" if is_outage else "#0284C7"
            cov_ellipse = Ellipse((fx, fy), width=ell_w, height=ell_h, angle=0,
                                  facecolor=ell_color, alpha=0.16, edgecolor=ell_color,
                                  linestyle="--", lw=1.2, zorder=6)
            ax_main.add_patch(cov_ellipse)

            # Robot Chassis Drawing (L = 0.50m, W = 0.45m)
            L, W = 0.50, 0.45
            chassis_corners = np.array([
                [+L/2, +W/2],
                [-L/2, +W/2],
                [-L/2, -W/2],
                [+L/2, -W/2],
            ])
            rot_mat = np.array([
                [math.cos(rth), -math.sin(rth)],
                [math.sin(rth),  math.cos(rth)]
            ])
            transformed_chassis = (rot_mat @ chassis_corners.T).T + np.array([rx, ry])
            poly = patches.Polygon(transformed_chassis, closed=True,
                                   facecolor="#1E293B", edgecolor="#0284C7", lw=1.4, zorder=8)
            ax_main.add_patch(poly)

            # 4 Wheels
            rw, ww = 0.10, 0.04
            wheel_offsets = [
                (+L/2, +W/2), (+L/2, -W/2),
                (-L/2, +W/2), (-L/2, -W/2),
            ]
            for wx_rel, wy_rel in wheel_offsets:
                w_center = rot_mat @ np.array([wx_rel, wy_rel]) + np.array([rx, ry])
                w_corners = np.array([
                    [+rw, +ww/2], [-rw, +ww/2], [-rw, -ww/2], [+rw, -ww/2]
                ])
                transformed_wheel = (rot_mat @ w_corners.T).T + w_center
                w_poly = patches.Polygon(transformed_wheel, closed=True,
                                         facecolor="#0F172A", edgecolor="#64748B", lw=0.7, zorder=9)
                ax_main.add_patch(w_poly)

            # Heading arrow from CG
            arrow_len = 0.32
            ax_main.annotate(
                "",
                xy=(rx + arrow_len * math.cos(rth), ry + arrow_len * math.sin(rth)),
                xytext=(rx, ry),
                arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.8),
                zorder=10,
            )

            # Sensor Hub & Stereo Camera Mount
            ax_main.plot(rx, ry, "o", color="#F59E0B", markersize=3.8, zorder=11)

            # Stereo Camera Mount & FOV Cone
            p_cam_base = np.array([rx + 0.20 * math.cos(rth), ry + 0.20 * math.sin(rth)])
            fov_range = 1.6
            fov_half_angle = math.radians(30)
            p_left = p_cam_base + fov_range * np.array([math.cos(rth + fov_half_angle), math.sin(rth + fov_half_angle)])
            p_right = p_cam_base + fov_range * np.array([math.cos(rth - fov_half_angle), math.sin(rth - fov_half_angle)])
            fov_poly = patches.Polygon([p_cam_base, p_left, p_right], closed=True,
                                       facecolor="#7C3AED", alpha=0.14, edgecolor="#7C3AED", linestyle="--", lw=0.8, zorder=7)
            ax_main.add_patch(fov_poly)

            # Tracked visual landmarks inside FOV
            for ox, oy, _ in obstacles:
                d_cam = math.hypot(ox - p_cam_base[0], oy - p_cam_base[1])
                bearing = math.atan2(oy - p_cam_base[1], ox - p_cam_base[0]) - rth
                bearing = math.atan2(math.sin(bearing), math.cos(bearing))
                if d_cam <= fov_range and abs(bearing) <= fov_half_angle:
                    ax_main.plot(ox, oy, "+", color="#7C3AED", markersize=7, mew=1.6, zorder=12)

            # Telemetry Pill Card (Top Left)
            hud_bg = patches.FancyBboxPatch(
                (0.02, 0.83), 0.68, 0.15, transform=ax_main.transAxes,
                boxstyle="round,pad=0.015,rounding_size=0.04",
                facecolor="#FFFFFF", alpha=0.94, edgecolor="#CBD5E1", lw=0.8, zorder=20
            )
            ax_main.add_patch(hud_bg)

            # Localization Health State
            if cur_t < 4.0:
                health_text = "[HEALTHY] TRACKING"
                health_color = "#059669"
            elif cur_t < 8.0:
                health_text = "[DEGRADED] GNSS OUTAGE (R1 BUFFER)"
                health_color = "#DC2626"
            elif cur_t < 9.5:
                health_text = "[RECOVERING] CONSISTENCY VERIFIED"
                health_color = "#2563EB"
            else:
                health_text = "[RECOVERED] TRACKING NOMINAL"
                health_color = "#0D9488"

            err_dr = math.hypot(b0_x[f] - rx, b0_y[f] - ry)
            err_ekf = math.hypot(fx - rx, fy - ry)

            ax_main.text(0.04, 0.94, f"t = {cur_t:04.1f}s / {duration_s:.1f}s",
                         transform=ax_main.transAxes, fontsize=7.2, fontweight="bold", color="#0F172A", zorder=21)
            ax_main.text(0.28, 0.94, health_text,
                         transform=ax_main.transAxes, fontsize=6.8, fontweight="bold", color=health_color, zorder=21)
            ax_main.text(0.04, 0.88, f"Wheel DR Drift: {err_dr:.2f} m   |   Robust EKF: {err_ekf:.2f} m",
                         transform=ax_main.transAxes, fontsize=6.8, color="#475569", zorder=21)
            ax_main.text(0.04, 0.84, "Sensing: 4 Encoders (50Hz) + IMU (50Hz) + Stereo VO (10Hz) + GNSS (1Hz)",
                         transform=ax_main.transAxes, fontsize=6.0, color="#64748B", zorder=21)

            # Minimap update
            mini_dot.set_data([rx], [ry])
            cam_w = 2.0 * view_radius
            mini_box.set_xy((fx - view_radius, fy - view_radius))
            mini_box.set_width(cam_w)
            mini_box.set_height(cam_w)

            writer.grab_frame()

    plt.close(fig)
    print(f"Saved master video to: {mp4_path} ({mp4_path.stat().st_size / 1024:.1f} KB)")

    # Convert to High-Quality Seamless Looping GIF using FFmpeg palettegen
    print("Converting MP4 to optimized looping GIF with FFmpeg...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vf", "fps=15,scale=460:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
        "-loop", "0", str(gif_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Successfully generated looping GIF:")
    print(f"  -> {gif_path} ({gif_path.stat().st_size / (1024 * 1024):.2f} MB)")

    # Mirror copies to videos/ and figures/
    shutil.copy2(mp4_path, VIDEOS_DIR / "localization_loop.mp4")
    shutil.copy2(gif_path, FIGURES_DIR / "localization_loop.gif")
    print("Mirrored video and GIF copies to videos/ and figures/.")


def main():
    print("=" * 75)
    print("RENDERING SEAMLESS LOOPING 4WD LOCALIZATION SIMULATION GIF")
    print("=" * 75)
    sim_res, traces = run_loop_simulation(duration_s=12.0)
    render_animation(sim_res, traces, duration_s=12.0, fps=15)
    print("=" * 75)


if __name__ == "__main__":
    main()
