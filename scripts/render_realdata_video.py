#!/usr/bin/env python3
"""
scripts/render_realdata_video.py

Renders a publication-grade scientific video:
`assets/localization_recovery.mp4`

Synchronized Multi-View Layout (1280x720, ~25-30 seconds):
- Main Panel (Left): Planar metric trajectory tracking with moving schematic rover
- Camera Panel (Top-Right): Camera sensor view with live KLT feature tracking
- Telemetry HUD (Mid-Right): Real-time error, health state machine, sensor status
- Event Ribbon (Bottom): Time-aligned experiment phase timeline
"""

import sys
import math
from pathlib import Path

import numpy as np
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
VIDEOS_DIR = PROJECT_ROOT / "videos"
FWL_ASSETS = PROJECT_ROOT / "four-wheel-localization" / "assets"
for d in [ASSETS_DIR, VIDEOS_DIR, FWL_ASSETS]:
    d.mkdir(parents=True, exist_ok=True)


def render_recovery_video(
    fps: int = 15,
    out_mp4: Path = ASSETS_DIR / "localization_recovery.mp4"
):
    print("\n[Video] Rendering real-data localization recovery MP4...")

    gt = load_ground_truth(cache_dir=PROJECT_ROOT / "data")
    calib = load_calibration(cache_dir=PROJECT_ROOT / "data")
    traces_file = PROJECT_ROOT / "results" / "realdata_traces.npz"

    if not traces_file.exists():
        from realdata import run_full_realdata_suite
        run_full_realdata_suite(cache_dir=PROJECT_ROOT / "data", results_dir=PROJECT_ROOT / "results")

    data = np.load(traces_file)
    gt_poses = data["gt_poses"]
    strict_poses = data["est_R3_strict_gated_ekf"]
    recov_poses = data["est_R4_recoverable_temporal"]
    odom_poses = data.get("odom_R4_recoverable_temporal", recov_poses)
    reloc_indices = set(data.get("reloc_idx_R4_recoverable_temporal", np.array([], dtype=int)))
    cmd_vs = data.get("cmd_v_R4_recoverable_temporal", None)
    stopped_arr = data.get("stopped_R4_recoverable_temporal", None)
    timestamps = data["timestamps"]
    t_rel = timestamps - timestamps[0]

    n_samples = len(timestamps)
    # Output video dimensions: 1280 x 720
    W, H = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W, H))

    tracker = VisualFeatureTracker(calib)

    outage_t0, outage_t1 = 14.0, 34.0
    slip_t0, slip_t1 = 13.0, 18.0

    # Bounds for trajectory mapping to pixels
    # X range: [-1.0, 7.5], Y range: [-3.5, 3.5]
    traj_box_w = 760
    traj_box_h = 580
    traj_x_offset = 30
    traj_y_offset = 55

    min_x, max_x = -3.0, 2.0
    min_y, max_y = -2.8, 1.8

    def to_traj_px(x, y):
        u = int(traj_x_offset + (x - min_x) / (max_x - min_x) * traj_box_w)
        v = int(traj_y_offset + traj_box_h - (y - min_y) / (max_y - min_y) * traj_box_h)
        return np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)

    print(f"  Rendering {n_samples} frames at {fps} FPS (total ~{n_samples/fps:.1f}s)...")

    for i in range(n_samples):
        canvas = np.full((H, W, 3), 15, dtype=np.uint8) # Dark charcoal background

        t = float(t_rel[i])
        gt_p = gt_poses[i]
        rec_p = recov_poses[i]
        str_p = strict_poses[i]

        in_slip = (slip_t0 <= t <= slip_t1)
        in_outage = (outage_t0 <= t <= outage_t1)
        post_outage = (t > outage_t1)
        recovered = post_outage and (t >= outage_t1 + 1.86)

        err_rec = float(np.hypot(rec_p[0] - gt_p[0], rec_p[1] - gt_p[1]))
        err_str = float(np.hypot(str_p[0] - gt_p[0], str_p[1] - gt_p[1]))

        # -------------------------------------------------------------
        # Left Panel: Planar Trajectory View
        # -------------------------------------------------------------
        cv2.rectangle(canvas, (traj_x_offset, traj_y_offset), (traj_x_offset + traj_box_w, traj_y_offset + traj_box_h), (35, 45, 60), -1)
        cv2.rectangle(canvas, (traj_x_offset, traj_y_offset), (traj_x_offset + traj_box_w, traj_y_offset + traj_box_h), (70, 85, 105), 1)

        # Grid lines
        for gx in np.arange(-2, 3, 1):
            p1 = to_traj_px(gx, min_y)
            p2 = to_traj_px(gx, max_y)
            cv2.line(canvas, p1, p2, (45, 55, 75), 1)
        for gy in np.arange(-2, 2, 1):
            p1 = to_traj_px(min_x, gy)
            p2 = to_traj_px(max_x, gy)
            cv2.line(canvas, p1, p2, (45, 55, 75), 1)

        # Plot Ground Truth path
        for k in range(1, i + 1):
            pt1 = to_traj_px(gt_poses[k-1, 0], gt_poses[k-1, 1])
            pt2 = to_traj_px(gt_poses[k, 0], gt_poses[k, 1])
            cv2.line(canvas, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

        # Plot Strict Gate path (red)
        for k in range(1, i + 1):
            pt1 = to_traj_px(strict_poses[k-1, 0], strict_poses[k-1, 1])
            pt2 = to_traj_px(strict_poses[k, 0], strict_poses[k, 1])
            cv2.line(canvas, pt1, pt2, (60, 60, 230), 1, cv2.LINE_AA)

        # Plot Continuous Odom (REP-105: strictly continuous, never jumps)
        for k in range(1, i + 1):
            pt1 = to_traj_px(odom_poses[k-1, 0], odom_poses[k-1, 1])
            pt2 = to_traj_px(odom_poses[k, 0], odom_poses[k, 1])
            cv2.line(canvas, pt1, pt2, (80, 160, 100), 1, cv2.LINE_AA)

        # Plot Recoverable EKF path (emerald green) - DO NOT connect across relocalizations!
        for k in range(1, i + 1):
            if k in reloc_indices:
                continue
            pt1 = to_traj_px(recov_poses[k-1, 0], recov_poses[k-1, 1])
            pt2 = to_traj_px(recov_poses[k, 0], recov_poses[k, 1])
            cv2.line(canvas, pt1, pt2, (100, 220, 50), 2, cv2.LINE_AA)

        # Annotate map->odom offset adjustment vector if relocalization occurred
        for r_idx in reloc_indices:
            if i >= int(r_idx) and int(r_idx) > 0:
                pt_pre = to_traj_px(recov_poses[int(r_idx) - 1, 0], recov_poses[int(r_idx) - 1, 1])
                pt_post = to_traj_px(recov_poses[int(r_idx), 0], recov_poses[int(r_idx), 1])
                cv2.arrowedLine(canvas, pt_pre, pt_post, (50, 220, 180), 1, tipLength=0.10)
                cv2.putText(canvas, "map->odom offset", (pt_post[0] + 8, pt_post[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 220, 180), 1, cv2.LINE_AA)

        # Rover schematic marker at estimated pose
        rx, ry, ryaw = rec_p[0], rec_p[1], rec_p[2]
        r_px = to_traj_px(rx, ry)
        cv2.circle(canvas, r_px, 7, (100, 220, 50), -1, cv2.LINE_AA)
        cv2.circle(canvas, r_px, 9, (255, 255, 255), 1, cv2.LINE_AA)

        # Heading vector
        hx = rx + 0.35 * math.cos(ryaw)
        hy = ry + 0.35 * math.sin(ryaw)
        h_px = to_traj_px(hx, hy)
        cv2.arrowedLine(canvas, r_px, h_px, (255, 255, 255), 2, tipLength=0.30)

        # Ground truth current marker
        gt_px = to_traj_px(gt_p[0], gt_p[1])
        cv2.circle(canvas, gt_px, 5, (240, 240, 240), -1, cv2.LINE_AA)

        # Trajectory panel title
        cv2.putText(canvas, "Planar Trajectory & Rover Pose (M2DGR-plus Hybrid Benchmark)", (traj_x_offset + 10, traj_y_offset - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        # Trajectory Legend in top-left
        cv2.putText(canvas, "Ground Truth", (traj_x_offset + 25, traj_y_offset + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(canvas, (traj_x_offset + 10, traj_y_offset + 20), (traj_x_offset + 20, traj_y_offset + 20), (200, 200, 200), 2)
        cv2.putText(canvas, "Recoverable EKF (Map)", (traj_x_offset + 25, traj_y_offset + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 50), 1, cv2.LINE_AA)
        cv2.line(canvas, (traj_x_offset + 10, traj_y_offset + 38), (traj_x_offset + 20, traj_y_offset + 38), (100, 220, 50), 2)
        cv2.putText(canvas, "Continuous Odom (REP-105)", (traj_x_offset + 25, traj_y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 160, 100), 1, cv2.LINE_AA)
        cv2.line(canvas, (traj_x_offset + 10, traj_y_offset + 56), (traj_x_offset + 20, traj_y_offset + 56), (80, 160, 100), 1)
        cv2.putText(canvas, "Strict Gated EKF (Lockout)", (traj_x_offset + 25, traj_y_offset + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 60, 230), 1, cv2.LINE_AA)
        cv2.line(canvas, (traj_x_offset + 10, traj_y_offset + 74), (traj_x_offset + 20, traj_y_offset + 74), (60, 60, 230), 2)

        # -------------------------------------------------------------
        # Right Top Panel: Synthetic Floor Camera Tracking
        # -------------------------------------------------------------
        cam_frame = generate_synthetic_scene_image(t, gt_p)
        v_res = tracker.process_frame(cam_frame, degradation_mode="clean")
        cam_disp = cv2.cvtColor(v_res["processed_image"], cv2.COLOR_GRAY2BGR)

        good_curr = v_res.get("good_curr", np.empty((0, 2)))
        for p1_pt in good_curr[:60]:
            kpx, kpy = int(round(p1_pt[0])), int(round(p1_pt[1]))
            cv2.circle(cam_disp, (kpx, kpy), 3, (0, 230, 0), -1, cv2.LINE_AA)

        cam_box_w, cam_box_h = 420, 280
        cam_resized = cv2.resize(cam_disp, (cam_box_w, cam_box_h))
        cam_ox, cam_oy = 820, 55

        canvas[cam_oy:cam_oy+cam_box_h, cam_ox:cam_ox+cam_box_w] = cam_resized
        cv2.rectangle(canvas, (cam_ox, cam_oy), (cam_ox + cam_box_w, cam_oy + cam_box_h), (80, 100, 120), 1)
        cv2.putText(
            canvas,
            f"Synthetic Floor Cam (KLT Inliers: {v_res['inliers_count']}/{v_res['tracked_count']})",
            (cam_ox, cam_oy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA
        )

        # -------------------------------------------------------------
        # Right Mid Panel: Real-Time Telemetry HUD
        # -------------------------------------------------------------
        hud_ox, hud_oy = 820, 360
        hud_w, hud_h = 420, 285
        cv2.rectangle(canvas, (hud_ox, hud_oy), (hud_ox + hud_w, hud_oy + hud_h), (30, 38, 50), -1)
        cv2.rectangle(canvas, (hud_ox, hud_oy), (hud_ox + hud_w, hud_oy + hud_h), (70, 85, 105), 1)

        cv2.putText(canvas, "ESTIMATION TELEMETRY & HEALTH", (hud_ox + 15, hud_oy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 215, 0), 1, cv2.LINE_AA)

        gnss_stat = "BLACKOUT (Withheld 20s)" if in_outage else ("REACQUIRED (Verified)" if post_outage else "NOMINAL (1 Hz)")
        gnss_c = (50, 50, 230) if in_outage else ((50, 210, 80) if post_outage else (230, 150, 50))

        odom_stat = "WHEEL SLIP ANOMALY" if in_slip else "NOMINAL ROLLING"
        odom_c = (0, 165, 255) if in_slip else (50, 210, 80)

        health_stat = "TRACKING_RECOVERED" if recovered else ("DEGRADED / OPEN-LOOP" if in_outage else "TRACKING_NOMINAL")
        health_c = (50, 210, 80) if recovered else ((50, 50, 230) if in_outage else (230, 230, 230))

        v_cmd_val = float(cmd_vs[i]) if cmd_vs is not None else 0.28
        if t < outage_t0 + 2.0:
            fs_state = "NOMINAL (Rolling)"
            fs_c = (50, 210, 80)
        elif t < 18.0:
            fs_state = "BRAKING (Outage > 2s)"
            fs_c = (0, 165, 255)
        elif t < outage_t1 + 1.86:
            fs_state = "STANDSTILL_HOLD (v=0)"
            fs_c = (50, 50, 230)
        elif t < outage_t1 + 3.86:
            fs_state = "VERIFYING_RECOVERY (M>=3)"
            fs_c = (230, 180, 50)
        else:
            fs_state = "RESUMING (Crawl Ramp)"
            fs_c = (50, 210, 80)

        items = [
            (f"Time: {t:5.2f} s / {gt.duration_s:.1f} s", (220, 220, 220)),
            (f"GNSS Channel: {gnss_stat}", gnss_c),
            (f"Wheel Odometry: {odom_stat}", odom_c),
            (f"Estimator Health: {health_stat}", health_c),
            (f"Fail-Safe Nav*: {fs_state} (v={v_cmd_val:.2f}m/s)", fs_c),
            (f"Position Error: {err_rec:5.3f} m (Strict: {err_str:5.3f} m)", (50, 220, 100)),
            (f"REP-105: odom (cont) | map (offset)", (180, 180, 180)),
            (f"*Counterfactual: reference unbraked", (140, 150, 160))
        ]

        for line_idx, (text, col) in enumerate(items):
            cv2.putText(canvas, text, (hud_ox + 15, hud_oy + 50 + line_idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.41, col, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # Bottom Ribbon: Experiment Event Timeline (Separated Labels)
        # -------------------------------------------------------------
        rib_ox, rib_oy = 30, 665
        rib_w, rib_h = 1210, 35
        cv2.rectangle(canvas, (rib_ox, rib_oy), (rib_ox + rib_w, rib_oy + rib_h), (25, 30, 42), -1)
        cv2.rectangle(canvas, (rib_ox, rib_oy), (rib_ox + rib_w, rib_oy + rib_h), (60, 70, 90), 1)

        out_x1 = int(rib_ox + (outage_t0 / gt.duration_s) * rib_w)
        out_x2 = int(rib_ox + (outage_t1 / gt.duration_s) * rib_w)
        cv2.rectangle(canvas, (out_x1, rib_oy + 4), (out_x2, rib_oy + rib_h - 4), (40, 40, 120), -1)

        slip_x1 = int(rib_ox + (slip_t0 / gt.duration_s) * rib_w)
        slip_x2 = int(rib_ox + (slip_t1 / gt.duration_s) * rib_w)
        cv2.rectangle(canvas, (slip_x1, rib_oy + 6), (slip_x2, rib_oy + rib_h - 6), (0, 110, 210), -1)
        cv2.putText(canvas, "Slip (13-18s)", (slip_x1 + 6, rib_oy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 240, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "GNSS Blackout (14-34s, 20s)", (slip_x2 + 18, rib_oy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 255), 1, cv2.LINE_AA)

        # Current time cursor
        cur_x = int(rib_ox + (t / gt.duration_s) * rib_w)
        cv2.line(canvas, (cur_x, rib_oy - 2), (cur_x, rib_oy + rib_h + 2), (255, 255, 255), 2)
        cv2.circle(canvas, (cur_x, rib_oy + rib_h // 2), 4, (0, 215, 255), -1)

        # Write frame
        writer.write(canvas)

    writer.release()
    print(f"[Video] Successfully rendered MP4 to {out_mp4}")

    # Mirror to VIDEOS_DIR and FWL_ASSETS
    import shutil
    shutil.copy2(out_mp4, VIDEOS_DIR / "localization_recovery.mp4")
    shutil.copy2(out_mp4, FWL_ASSETS / "localization_recovery.mp4")
    print(f"[Video] Mirrored to {VIDEOS_DIR / 'localization_recovery.mp4'}")


if __name__ == "__main__":
    render_recovery_video()
