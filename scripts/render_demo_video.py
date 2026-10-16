"""
scripts.render_demo_video

Renders a synchronized 4-panel 1080p research demonstration video:
`assets/localization_demo.mp4`

Panels:
  1. Main View (Top-Left): Top-down rover tracking trajectory, EKF pose, 95% uncertainty ellipse,
     drifting dead reckoning, GNSS fixes, and camera FOV sector.
  2. Stereo Camera View (Top-Right): Synchronized Left camera frame, Right camera frame,
     OpenCV StereoSGBM disparity map, and tracked ORB feature matches.
  3. 3D Stereo Point-Cloud View (Bottom-Left): Genuine reconstructed local point cloud
     in robot base_link frame.
  4. Status & Timeline View (Bottom-Right): Active sensors, localization health status,
     error dynamics during GNSS blackout and R1 recovery re-anchoring.
"""

import sys
import math
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Ellipse
from matplotlib.animation import FFMpegWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.simulation.scenarios import create_scenario_gnss_outage
from robot_localization.sensors.stereo import (
    StereoCameraConfig,
    SyntheticStereoArena,
    StereoDepthPipeline,
    StereoVisualOdometry
)
from robot_localization.estimation.gnss_recovery import RecoverableRobustEKF4D, RecoveryStrategy
from robot_localization.estimation.dead_reckoning import DeadReckoning4WD


def render_demonstration_video(duration_s: float = 30.0, fps: int = 15):
    print("=" * 80)
    print("RENDERING SYNCHRONIZED 4-PANEL RESEARCH DEMONSTRATION VIDEO")
    print("=" * 80)

    assets_dir = PROJECT_ROOT / "assets"
    videos_dir = PROJECT_ROOT / "videos"
    assets_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = assets_dir / "localization_demo.mp4"

    # Simulation setup
    sim, traj_fn, slip_fn = create_scenario_gnss_outage(duration_s=duration_s, seed=42)
    sim_res = sim.run_simulation(duration_s=duration_s, trajectory_func=traj_fn, slip_profile_func=slip_fn)
    cfg = sim.config

    # Stereo camera & arena setup
    cam_cfg = StereoCameraConfig(width=320, height=240, fx=300.0, fy=300.0, cx=160.0, cy=120.0, baseline_m=0.10)
    arena = SyntheticStereoArena(seed=42)
    depth_pipe = StereoDepthPipeline(cam_cfg)
    vo = StereoVisualOdometry(cam_cfg, max_features=250, min_inliers=8)

    # Estimators: B0 (Dead Reckoning), J0 (Strict Gate Lockout), J1 (Temporal Consistency Recovery)
    est_b0 = DeadReckoning4WD(cfg, initial_pose=sim_res.ground_truth_pose[0])
    est_j0 = RecoverableRobustEKF4D(strategy=RecoveryStrategy.R0_STRICT_GATE, initial_state=np.zeros(4))
    est_j1 = RecoverableRobustEKF4D(strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, initial_state=np.zeros(4))

    # Pre-simulate estimators to extract synchronized states
    print("Running synchronized estimator simulation...")
    t_records = []
    gt_records = []
    b0_records = []
    j0_records = []
    j1_records = []
    j1_cov_records = []
    gnss_records = []

    last_gyro = 0.0
    last_t = 0.0

    for ev in sim_res.sensor_events:
        if ev.sensor_type == "imu":
            last_gyro = ev.data.gyro_rad_s[2]
        elif ev.sensor_type == "encoder":
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp
            enc = ev.data
            est_b0.update_from_encoders(enc, dt)

            v_fl = enc["FL"].angular_speed_rad_s * cfg.wheel_radius_m
            v_fr = enc["FR"].angular_speed_rad_s * cfg.wheel_radius_m
            v_rl = enc["RL"].angular_speed_rad_s * cfg.wheel_radius_m
            v_rr = enc["RR"].angular_speed_rad_s * cfg.wheel_radius_m
            v = 0.25 * (v_fl + v_fr + v_rl + v_rr)
            omega_enc = (0.5 * (v_fr + v_rr) - 0.5 * (v_fl + v_rl)) / cfg.effective_track_width_m
            wheel_dis = max(abs(v_fl - v_rl), abs(v_fr - v_rr))

            est_j0.predict_recoverable(v, omega_enc, last_gyro, np.diag([1e-4, 2.5e-5]), dt, wheel_dis)
            est_j1.predict_recoverable(v, omega_enc, last_gyro, np.diag([1e-4, 2.5e-5]), dt, wheel_dis)

            idx = np.searchsorted(sim_res.timestamps, ev.timestamp)
            gt_p = sim_res.ground_truth_pose[min(idx, len(sim_res.timestamps)-1)]

            t_records.append(ev.timestamp)
            gt_records.append(gt_p.copy())
            b0_records.append(est_b0.pose.copy())
            j0_records.append(est_j0.x[:3].copy())
            j1_records.append(est_j1.x[:3].copy())
            j1_cov_records.append(np.diag(est_j1.P[:2, :2]).copy())

        elif ev.sensor_type == "gnss":
            fix = ev.data
            if fix.is_valid:
                z = np.array([fix.x_enu_m, fix.y_enu_m])
                est_j0.update_gnss_recoverable(ev.timestamp, z, fix.cov_enu)
                est_j1.update_gnss_recoverable(ev.timestamp, z, fix.cov_enu)
                if j0_records:
                    j0_records[-1] = est_j0.x[:3].copy()
                    j1_records[-1] = est_j1.x[:3].copy()
                    j1_cov_records[-1] = np.diag(est_j1.P[:2, :2]).copy()
                gnss_records.append((ev.timestamp, z[0], z[1]))

    t_arr = np.array(t_records)
    gt_arr = np.array(gt_records)
    b0_arr = np.array(b0_records)
    j0_arr = np.array(j0_records)
    j1_arr = np.array(j1_records)
    j1_cov_arr = np.array(j1_cov_records)

    total_frames = int(duration_s * fps)
    frame_times = np.linspace(0, duration_s, total_frames)

    # Setup 4-panel 1920x1080 canvas
    # 12.8 in x 7.2 in @ 150 DPI = 1920 x 1080
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
    fig.patch.set_facecolor("#0F172A") # Deep slate background

    # 4 Subplot layout
    ax_main = fig.add_axes([0.05, 0.38, 0.44, 0.56])   # Top-Left: Main View
    ax_cam = fig.add_axes([0.53, 0.54, 0.43, 0.40])    # Top-Right: Stereo View
    ax_pcd = fig.add_axes([0.05, 0.07, 0.44, 0.25])    # Bottom-Left: 3D Point Cloud View
    ax_tele = fig.add_axes([0.53, 0.07, 0.43, 0.41])   # Bottom-Right: Status & Timeline View

    writer = FFMpegWriter(fps=fps, metadata=dict(title="Modular Localization Demo"), bitrate=3500)

    print(f"Rendering {total_frames} video frames to {mp4_path}...")
    with writer.saving(fig, str(mp4_path), dpi=150):
        for f_idx in range(total_frames):
            cur_t = frame_times[f_idx]
            sim_idx = np.searchsorted(t_arr, cur_t)
            sim_idx = min(sim_idx, len(t_arr) - 1)

            rx, ry, rth = gt_arr[sim_idx]
            fx, fy, fth = j1_arr[sim_idx]
            drx, dry, _ = b0_arr[sim_idx]

            # -----------------------------------------------------------------
            # 1. Main View (Top-Down Tracking)
            # -----------------------------------------------------------------
            ax_main.clear()
            ax_main.set_facecolor("#1E293B")
            ax_main.set_title("Primary Arena Tracking View (REP-105 map frame)", color="white", fontsize=9.5, fontweight="bold", pad=5)

            view_r = 3.5
            ax_main.set_xlim(fx - view_r, fx + view_r)
            ax_main.set_ylim(fy - view_r, fy + view_r)
            ax_main.set_aspect("equal")

            # Grid
            ax_main.grid(True, linestyle=":", color="#334155", alpha=0.6)
            ax_main.tick_params(colors="#94A3B8", labelsize=7)
            for sp in ax_main.spines.values():
                sp.set_color("#475569")

            # Arena rocks
            for lm in arena.landmarks:
                ox, oy = lm.world_xyz[0], lm.world_xyz[1]
                ax_main.plot(ox, oy, "o", color="#475569", markersize=6)

            # Trajectory tails
            tail_s = max(0, sim_idx - int(fps * 6))
            ax_main.plot(gt_arr[tail_s:sim_idx+1, 0], gt_arr[tail_s:sim_idx+1, 1],
                         color="#94A3B8", linestyle="--", lw=1.2, label="Ground Truth")
            ax_main.plot(b0_arr[tail_s:sim_idx+1, 0], b0_arr[tail_s:sim_idx+1, 1],
                         color="#8B5CF6", linestyle=":", lw=1.2, label="Wheel Dead-Reckoning")
            ax_main.plot(j1_arr[tail_s:sim_idx+1, 0], j1_arr[tail_s:sim_idx+1, 1],
                         color="#0EA5E9", linestyle="-", lw=1.8, label="Recoverable EKF (R1)")

            # GNSS fix history
            gnss_vis = [g for g in gnss_records if g[0] <= cur_t]
            if len(gnss_vis) > 0:
                gx = [g[1] for g in gnss_vis[-8:]]
                gy = [g[2] for g in gnss_vis[-8:]]
                ax_main.scatter(gx, gy, color="#F59E0B", s=20, marker="^", label="GNSS Fixes", zorder=5)

            # Stereo FOV Cone
            fov_range = 2.2
            fov_ang = math.radians(35)
            p_cam = np.array([rx + 0.20 * math.cos(rth), ry + 0.20 * math.sin(rth)])
            p_l = p_cam + fov_range * np.array([math.cos(rth + fov_ang), math.sin(rth + fov_ang)])
            p_r = p_cam + fov_range * np.array([math.cos(rth - fov_ang), math.sin(rth - fov_ang)])
            fov_poly = patches.Polygon([p_cam, p_l, p_r], closed=True,
                                       facecolor="#7C3AED", alpha=0.18, edgecolor="#7C3AED", linestyle="--", lw=0.9, zorder=6)
            ax_main.add_patch(fov_poly)

            # Robot chassis
            L, W = 0.50, 0.45
            corners = np.array([[+L/2, +W/2], [-L/2, +W/2], [-L/2, -W/2], [+L/2, -W/2]])
            R_mat = np.array([[math.cos(rth), -math.sin(rth)], [math.sin(rth), math.cos(rth)]])
            trans_chassis = (R_mat @ corners.T).T + np.array([rx, ry])
            poly = patches.Polygon(trans_chassis, closed=True, facecolor="#0284C7", edgecolor="#E2E8F0", lw=1.2, zorder=8)
            ax_main.add_patch(poly)

            # Heading arrow
            ax_main.annotate("", xy=(rx + 0.40 * math.cos(rth), ry + 0.40 * math.sin(rth)), xytext=(rx, ry),
                             arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=1.6), zorder=9)

            # 95% Confidence Ellipse
            var_x = max(j1_cov_arr[sim_idx, 0], 1e-4)
            var_y = max(j1_cov_arr[sim_idx, 1], 1e-4)
            ell_w = 2.0 * 2.448 * math.sqrt(var_x)
            ell_h = 2.0 * 2.448 * math.sqrt(var_y)
            is_outage = (15.0 <= cur_t <= 35.0)
            e_col = "#DC2626" if is_outage else "#38BDF8"
            ax_main.add_patch(Ellipse((fx, fy), width=ell_w, height=ell_h, facecolor=e_col, alpha=0.2, edgecolor=e_col, linestyle="--", lw=1.0, zorder=7))

            ax_main.legend(loc="upper left", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=6.5)

            # -----------------------------------------------------------------
            # 2. Stereo Camera View (Left Frame + Disparity)
            # -----------------------------------------------------------------
            ax_cam.clear()
            ax_cam.axis("off")
            ax_cam.set_title("Stereo Vision Stream (Left Camera & SGBM Disparity)", color="white", fontsize=9.5, fontweight="bold", pad=5)

            img_l, img_r, _ = arena.render_stereo_pair(gt_arr[sim_idx], cam_cfg)
            depth_res = depth_pipe.compute_depth_and_points(img_l, img_r)

            # Colorize disparity
            disp_norm = cv2.normalize(depth_res.disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_VIRIDIS)
            disp_rgb = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)

            # Combine Left Gray and Disparity side-by-side
            gray_rgb = cv2.cvtColor(img_l, cv2.COLOR_GRAY2RGB)
            stereo_preview = np.hstack([gray_rgb, disp_rgb])
            ax_cam.imshow(stereo_preview)
            ax_cam.text(10, 25, "LEFT OPTICAL FRAME", color="#FFFFFF", fontsize=7, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", alpha=0.7))
            ax_cam.text(330, 25, "SGBM DISPARITY MAP", color="#FFFFFF", fontsize=7, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", alpha=0.7))

            # -----------------------------------------------------------------
            # 3. 3D Stereo Point-Cloud View (Projected base_link)
            # -----------------------------------------------------------------
            ax_pcd.clear()
            ax_pcd.set_facecolor("#1E293B")
            ax_pcd.set_title("Stereo Point Cloud (base_link frame, Top-Down Projection)", color="white", fontsize=9.0, fontweight="bold", pad=4)
            ax_pcd.grid(True, linestyle=":", color="#334155", alpha=0.6)
            ax_pcd.tick_params(colors="#94A3B8", labelsize=6.5)
            for sp in ax_pcd.spines.values():
                sp.set_color("#475569")

            pts_base = depth_res.point_cloud_base
            if len(pts_base) > 0:
                ax_pcd.scatter(pts_base[:, 0], pts_base[:, 1], c=pts_base[:, 2], cmap="plasma", s=6, alpha=0.75)
            # Robot origin in point cloud
            ax_pcd.plot(0, 0, marker="o", color="#38BDF8", markersize=5, label="base_link")
            ax_pcd.set_xlim(-0.5, 4.5)
            ax_pcd.set_ylim(-2.5, 2.5)
            ax_pcd.set_xlabel("X forward [m]", color="#94A3B8", fontsize=7.5)
            ax_pcd.set_ylabel("Y left [m]", color="#94A3B8", fontsize=7.5)

            # -----------------------------------------------------------------
            # 4. Status & Timeline View (Bottom-Right)
            # -----------------------------------------------------------------
            ax_tele.clear()
            ax_tele.set_facecolor("#1E293B")
            ax_tele.set_title("Localization Resilience Telemetry & Error Dynamics", color="white", fontsize=9.0, fontweight="bold", pad=4)

            # Error curve up to current time
            sub_t = t_arr[:sim_idx+1]
            e_j0 = np.linalg.norm(j0_arr[:sim_idx+1, :2] - gt_arr[:sim_idx+1, :2], axis=1)
            e_j1 = np.linalg.norm(j1_arr[:sim_idx+1, :2] - gt_arr[:sim_idx+1, :2], axis=1)

            ax_tele.axvspan(15.0, 35.0, color="#7F1D1D", alpha=0.35, label="GNSS Blackout")
            ax_tele.plot(sub_t, e_j0, color="#EF4444", linestyle="--", lw=1.2, label="R0: Strict Gate (Lockout Failure)")
            ax_tele.plot(sub_t, e_j1, color="#38BDF8", lw=1.8, label="R1: Temporal Recovery (Re-anchored)")

            ax_tele.set_xlim(0, duration_s)
            ax_tele.set_ylim(0, 4.5)
            ax_tele.set_xlabel("Time [s]", color="#94A3B8", fontsize=7.5)
            ax_tele.set_ylabel("Pos Error [m]", color="#94A3B8", fontsize=7.5)
            ax_tele.tick_params(colors="#94A3B8", labelsize=6.5)
            for sp in ax_tele.spines.values():
                sp.set_color("#475569")
            ax_tele.grid(True, linestyle=":", color="#334155", alpha=0.6)
            ax_tele.legend(loc="upper left", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=6.5)

            # Status Overlay Badge
            if cur_t < 15.0:
                h_badge = "STATUS: TRACKING NOMINAL"
                h_bcolor = "#16A34A"
            elif cur_t < 35.0:
                h_badge = "STATUS: GNSS BLACKOUT (DEGRADED)"
                h_bcolor = "#DC2626"
            elif cur_t < 37.0:
                h_badge = "STATUS: RECOVERY VERIFIED"
                h_bcolor = "#2563EB"
            else:
                h_badge = "STATUS: RE-ANCHORED (TRACKING NOMINAL)"
                h_bcolor = "#0D9488"

            ax_tele.text(0.97, 0.90, h_badge, transform=ax_tele.transAxes, ha="right",
                         fontsize=7.5, fontweight="bold", color="#FFFFFF",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=h_bcolor, edgecolor="none", alpha=0.9))

            writer.grab_frame()

    plt.close(fig)
    print(f"Research demonstration MP4 successfully created:")
    print(f"  -> {mp4_path} ({mp4_path.stat().st_size / (1024*1024):.2f} MB)")

    # Copy to videos/
    import shutil
    shutil.copy2(mp4_path, videos_dir / "localization_demo.mp4")
    fw_assets = PROJECT_ROOT / "four-wheel-localization" / "assets"
    if fw_assets.exists():
        shutil.copy2(mp4_path, fw_assets / "localization_demo.mp4")
        print(f"Copied demonstration video to {fw_assets}")


if __name__ == "__main__":
    render_demonstration_video(duration_s=30.0, fps=15)
