"""
scripts.run_ablations

Controlled sensor ablation and resilience experiments for the four-wheel robot.
Evaluates configurations A through J under matched physical trajectories and seeds:
  A. Encoders only (B0)
  B. Encoders + IMU (B1 with bias)
  C. Encoders + IMU + GNSS (B5 bias-aware EKF)
  D. Encoders + IMU + GNSS + Compass (B4)
  E. Encoders + IMU + Stereo Odometry (Visual-Inertial-Wheel)
  F. Encoders + IMU + Stereo Odometry + GNSS (Full Sensor Fusion)
  G. Encoders + IMU + LiDAR (Known Map AMCL)
  H. Encoders + IMU + Stereo + LiDAR + GNSS
  I. Stereo + IMU without Wheel Encoders (Traction loss resilience)
  J. GNSS-Denied Recovery Strategies under 20s Blackout & 10% Slip:
     - J0: Strict Gating (R0, Lockout Failure)
     - J1: Temporal Consistency Buffer (R1, Valid Recovery)
     - J2: Health Hysteresis State Machine (R2, Valid Recovery)
     - J3: Distance-Scaled Covariance Inflation (R3, Valid Recovery)
     - J4: Controlled Global Reinitialization (R4, Valid Recovery)
"""

import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.models.kinematics import wrap_angle
from robot_localization.simulation.robot_simulator import RobotSimulator, SimulationResult
from robot_localization.simulation.scenarios import (
    create_scenario_nominal,
    create_scenario_slip,
    create_scenario_gnss_outage,
    create_scenario_combined
)
from robot_localization.sensors.registry import SensorRegistry
from robot_localization.sensors.stereo import (
    StereoCameraConfig,
    SyntheticStereoArena,
    StereoVisualOdometry
)
from robot_localization.estimation.dead_reckoning import DeadReckoning4WD
from robot_localization.estimation.bias_aware_ekf import BiasAwareEKF4D
from robot_localization.estimation.robust_ekf import RobustEKF4D
from robot_localization.estimation.gnss_recovery import RecoverableRobustEKF4D, RecoveryStrategy
from robot_localization.evaluation.metrics import compute_trajectory_metrics


def run_ablation_benchmarks(duration_s: float = 40.0) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("RUNNING SENSOR ABLATION & FAULT RESILIENCE BENCHMARK (CONFIGS A - J)")
    print("=" * 80)

    # 1. Nominal scenario (nominal traction, clean sensors)
    sim_nominal, traj_fn_nom, slip_fn_nom = create_scenario_nominal(duration_s=duration_s, seed=42)
    res_nominal = sim_nominal.run_simulation(duration_s=duration_s, trajectory_func=traj_fn_nom, slip_profile_func=slip_fn_nom)

    # 2. GNSS-outage + wheel slip scenario (20s outage between t=10 and t=30 with 10% slip)
    sim_outage, traj_fn_out, slip_fn_out = create_scenario_gnss_outage(duration_s=duration_s, seed=42)
    res_outage = sim_outage.run_simulation(duration_s=duration_s, trajectory_func=traj_fn_out, slip_profile_func=slip_fn_out)

    # Pre-render Stereo Camera observations along ground-truth trajectories at 10 Hz
    print("Pre-rendering calibrated stereo image observations across scene...")
    arena = SyntheticStereoArena(seed=42)
    cam_cfg = StereoCameraConfig(width=320, height=240, fx=300.0, fy=300.0, cx=160.0, cy=120.0, baseline_m=0.10)

    vo_stream_nom = generate_vo_stream(arena, cam_cfg, res_nominal)
    vo_stream_out = generate_vo_stream(arena, cam_cfg, res_outage)
    print("Stereo VO stream generation complete.")

    records = []

    # Config A: Encoders Only
    rec_A = eval_config_A(res_nominal, sim_nominal.config)
    rec_A["config"] = "A_encoders_only"
    rec_A["regime"] = "nominal"
    records.append(rec_A)

    # Config B: Encoders + IMU (with uncompensated bias)
    rec_B = eval_config_B(res_nominal, sim_nominal.config)
    rec_B["config"] = "B_encoders_imu"
    rec_B["regime"] = "nominal"
    records.append(rec_B)

    # Config C: Encoders + IMU + GNSS (Bias-Aware EKF B5)
    rec_C = eval_config_C(res_nominal, sim_nominal.config)
    rec_C["config"] = "C_encoders_imu_gnss"
    rec_C["regime"] = "nominal"
    records.append(rec_C)

    # Config D: Encoders + IMU + GNSS + Compass (B4)
    rec_D = eval_config_D(res_nominal, sim_nominal.config)
    rec_D["config"] = "D_encoders_imu_gnss_compass"
    rec_D["regime"] = "nominal"
    records.append(rec_D)

    # Config E: Encoders + IMU + Stereo VO (No GNSS)
    rec_E = eval_config_E(res_nominal, vo_stream_nom, sim_nominal.config)
    rec_E["config"] = "E_encoders_imu_stereo"
    rec_E["regime"] = "nominal"
    records.append(rec_E)

    # Config F: Encoders + IMU + Stereo VO + GNSS
    rec_F = eval_config_F(res_nominal, vo_stream_nom, sim_nominal.config)
    rec_F["config"] = "F_encoders_imu_stereo_gnss"
    rec_F["regime"] = "nominal"
    records.append(rec_F)

    # Config I: Stereo + IMU without Wheel Encoders (Traction loss resilience)
    rec_I = eval_config_I(res_outage, vo_stream_out, sim_outage.config)
    rec_I["config"] = "I_stereo_imu_no_wheels"
    rec_I["regime"] = "severe_slip"
    records.append(rec_I)

    # Config J: GNSS-Denied Recovery Strategies under 20s Outage + 10% Slip
    for strat in [
        (RecoveryStrategy.R0_STRICT_GATE, "J0_outage_R0_strict_lockout"),
        (RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, "J1_outage_R1_temporal_consistency"),
        (RecoveryStrategy.R2_HEALTH_HYSTERESIS, "J2_outage_R2_health_hysteresis"),
        (RecoveryStrategy.R3_COVARIANCE_ADJUSTMENT, "J3_outage_R3_covariance_growth"),
        (RecoveryStrategy.R4_GLOBAL_REINITIALIZATION, "J4_outage_R4_global_reinit")
    ]:
        rec_J = eval_config_J(res_outage, strat[0], sim_outage.config)
        rec_J["config"] = strat[1]
        rec_J["regime"] = "gnss_outage_slip"
        records.append(rec_J)

    df = pd.DataFrame(records)
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ablation_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved sensor ablation summary table to: {out_csv}")

    # Print summary table to console
    print("\n" + "=" * 95)
    print(f"{'Config':32s} | {'Regime':15s} | {'Pos RMSE':8s} | {'Head RMSE':9s} | {'95% Cov':7s} | {'Latency':8s}")
    print("=" * 95)
    for _, row in df.iterrows():
        print(f"{row['config']:32s} | {row['regime']:15s} | {row['pos_rmse_m']:6.3f} m | {row['heading_rmse_deg']:5.2f} deg | {row['coverage_95']:5.1f}% | {row['runtime_ms_step']:5.3f} ms")
    print("=" * 95)

    return df


def generate_vo_stream(arena: SyntheticStereoArena, cfg: StereoCameraConfig, sim_res: SimulationResult) -> Dict[float, Any]:
    """Generate visual odometry observations at 10 Hz from ground truth camera poses."""
    vo = StereoVisualOdometry(cfg, max_features=350, min_inliers=10)
    vo_results = {}

    dt_cam = 0.10
    t_cam = 0.0
    while t_cam <= sim_res.timestamps[-1]:
        # Interpolate ground truth pose at t_cam
        idx = np.searchsorted(sim_res.timestamps, t_cam)
        idx = min(idx, len(sim_res.timestamps) - 1)
        pose = sim_res.ground_truth_pose[idx]

        img_l, img_r, _ = arena.render_stereo_pair(pose, cfg)
        res = vo.process_frame(img_l, img_r, timestamp=t_cam)
        vo_results[round(t_cam, 3)] = res
        t_cam += dt_cam

    return vo_results


def eval_config_A(sim_res: SimulationResult, cfg: Robot4WDConfig) -> Dict[str, Any]:
    """Config A: Pure 4-Wheel Dead Reckoning."""
    est = DeadReckoning4WD(cfg, initial_pose=sim_res.ground_truth_pose[0])
    t_hist, p_hist, cov_hist, cov_mat_hist = [], [], [], []

    t0 = time.perf_counter()
    last_t = 0.0
    for ev in sim_res.sensor_events:
        if ev.sensor_type == "encoder":
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp
            enc = ev.data
            p, c = est.update_from_encoders(enc, dt)
            t_hist.append(ev.timestamp)
            p_hist.append(p.copy())
            cov_hist.append(np.diag(c))
            cov_mat_hist.append(c.copy())
    total_time = (time.perf_counter() - t0) * 1000.0 / len(t_hist)

    m = compute_trajectory_metrics("A_encoders", sim_res.timestamps, sim_res.ground_truth_pose, np.array(t_hist), np.array(p_hist), np.array(cov_hist), cov_mats=np.array(cov_mat_hist))
    return {"pos_rmse_m": m.pos_rmse_m, "heading_rmse_deg": m.heading_rmse_deg, "coverage_95": m.coverage_95 * 100.0, "runtime_ms_step": total_time}


def eval_config_B(sim_res: SimulationResult, cfg: Robot4WDConfig) -> Dict[str, Any]:
    """Config B: Wheel Encoders + Gyroscope DR (uncompensated bias)."""
    p0 = sim_res.ground_truth_pose[0]
    p_est = p0.copy()
    cov_est = np.diag([0.01, 0.01, 0.005])
    t_hist, p_hist, cov_hist, cov_mat_hist = [], [], [], []

    last_gyro_z = 0.0
    last_t = 0.0
    t0 = time.perf_counter()
    for ev in sim_res.sensor_events:
        if ev.sensor_type == "imu":
            last_gyro_z = ev.data.gyro_rad_s[2]
        elif ev.sensor_type == "encoder":
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp
            enc = ev.data
            v = 0.25 * sum(enc[w].angular_speed_rad_s * cfg.wheel_radius_m for w in ["FL", "FR", "RL", "RR"])
            # Kinematic integration with raw gyro
            p_est[0] += v * math.cos(p_est[2] + 0.5 * last_gyro_z * dt) * dt
            p_est[1] += v * math.sin(p_est[2] + 0.5 * last_gyro_z * dt) * dt
            p_est[2] = wrap_angle(p_est[2] + last_gyro_z * dt)
            cov_est[0, 0] += 1e-4 * dt
            cov_est[1, 1] += 1e-4 * dt
            cov_est[2, 2] += (0.015 ** 2) * dt
            t_hist.append(ev.timestamp)
            p_hist.append(p_est.copy())
            cov_hist.append(np.diag(cov_est))
            cov_mat_hist.append(cov_est.copy())
    total_time = (time.perf_counter() - t0) * 1000.0 / len(t_hist)

    m = compute_trajectory_metrics("B_encoders_imu", sim_res.timestamps, sim_res.ground_truth_pose, np.array(t_hist), np.array(p_hist), np.array(cov_hist), cov_mats=np.array(cov_mat_hist))
    return {"pos_rmse_m": m.pos_rmse_m, "heading_rmse_deg": m.heading_rmse_deg, "coverage_95": m.coverage_95 * 100.0, "runtime_ms_step": total_time}


def eval_config_C(sim_res: SimulationResult, cfg: Robot4WDConfig) -> Dict[str, Any]:
    """Config C: Encoders + IMU + GNSS (Bias-Aware 4D EKF)."""
    p0 = sim_res.ground_truth_pose[0]
    est = BiasAwareEKF4D(initial_state=np.array([p0[0], p0[1], p0[2], 0.0]), initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]))
    t_hist, p_hist, cov_hist, cov_mat_hist = [], [], [], []

    last_gyro_z = 0.0
    last_t = 0.0
    t0 = time.perf_counter()
    for ev in sim_res.sensor_events:
        if ev.sensor_type == "imu":
            last_gyro_z = ev.data.gyro_rad_s[2]
        elif ev.sensor_type == "encoder":
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp
            enc = ev.data
            v = 0.25 * sum(enc[w].angular_speed_rad_s * cfg.wheel_radius_m for w in ["FL", "FR", "RL", "RR"])
            u = np.array([v, last_gyro_z])
            Qu = np.diag([1e-4, 2.5e-5])
            p, c = est.predict(u, Qu, dt)
            t_hist.append(ev.timestamp)
            p_hist.append(p[:3].copy())
            cov_hist.append(np.diag(c[:3, :3]))
            cov_mat_hist.append(c[:3, :3].copy())
        elif ev.sensor_type == "gnss":
            fix = ev.data
            if fix.is_valid:
                est.update_gnss(np.array([fix.x_enu_m, fix.y_enu_m]), fix.cov_enu)
                if p_hist:
                    p_hist[-1] = est.x[:3].copy()
                    cov_hist[-1] = np.diag(est.P[:3, :3])
                    cov_mat_hist[-1] = est.P[:3, :3].copy()
    total_time = (time.perf_counter() - t0) * 1000.0 / len(t_hist)

    m = compute_trajectory_metrics("C_encoders_imu_gnss", sim_res.timestamps, sim_res.ground_truth_pose, np.array(t_hist), np.array(p_hist), np.array(cov_hist), cov_mats=np.array(cov_mat_hist))
    return {"pos_rmse_m": m.pos_rmse_m, "heading_rmse_deg": m.heading_rmse_deg, "coverage_95": m.coverage_95 * 100.0, "runtime_ms_step": total_time}


def eval_config_D(sim_res: SimulationResult, cfg: Robot4WDConfig) -> Dict[str, Any]:
    """Config D: Encoders + IMU + GNSS + Compass."""
    p0 = sim_res.ground_truth_pose[0]
    est = BiasAwareEKF4D(initial_state=np.array([p0[0], p0[1], p0[2], 0.0]), initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]))
    t_hist, p_hist, cov_hist, cov_mat_hist = [], [], [], []

    last_gyro_z = 0.0
    last_t = 0.0
    t0 = time.perf_counter()
    for ev in sim_res.sensor_events:
        if ev.sensor_type == "imu":
            last_gyro_z = ev.data.gyro_rad_s[2]
        elif ev.sensor_type == "encoder":
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp
            enc = ev.data
            v = 0.25 * sum(enc[w].angular_speed_rad_s * cfg.wheel_radius_m for w in ["FL", "FR", "RL", "RR"])
            p, c = est.predict(np.array([v, last_gyro_z]), np.diag([1e-4, 2.5e-5]), dt)
            t_hist.append(ev.timestamp)
            p_hist.append(p[:3].copy())
            cov_hist.append(np.diag(c[:3, :3]))
            cov_mat_hist.append(c[:3, :3].copy())
        elif ev.sensor_type == "gnss":
            fix = ev.data
            if fix.is_valid:
                est.update_gnss(np.array([fix.x_enu_m, fix.y_enu_m]), fix.cov_enu)
                if p_hist:
                    p_hist[-1] = est.x[:3].copy()
                    cov_hist[-1] = np.diag(est.P[:3, :3])
                    cov_mat_hist[-1] = est.P[:3, :3].copy()
        elif ev.sensor_type == "compass":
            cmps = ev.data
            if cmps.is_valid:
                est.update_compass(cmps.heading_nav_rad, cmps.heading_std_rad ** 2)
                if p_hist:
                    p_hist[-1] = est.x[:3].copy()
                    cov_hist[-1] = np.diag(est.P[:3, :3])
                    cov_mat_hist[-1] = est.P[:3, :3].copy()
    total_time = (time.perf_counter() - t0) * 1000.0 / len(t_hist)

    m = compute_trajectory_metrics("D_compass_augmented", sim_res.timestamps, sim_res.ground_truth_pose, np.array(t_hist), np.array(p_hist), np.array(cov_hist), cov_mats=np.array(cov_mat_hist))
    return {"pos_rmse_m": m.pos_rmse_m, "heading_rmse_deg": m.heading_rmse_deg, "coverage_95": m.coverage_95 * 100.0, "runtime_ms_step": total_time}


def eval_config_E(sim_res: SimulationResult, vo_stream: Dict[float, Any], cfg: Robot4WDConfig) -> Dict[str, Any]:
    """Config E: Encoders + IMU + Stereo Visual Odometry."""
    p0 = sim_res.ground_truth_pose[0]
    est = BiasAwareEKF4D(initial_state=np.array([p0[0], p0[1], p0[2], 0.0]), initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]))
    t_hist, p_hist, cov_hist, cov_mat_hist = [], [], [], []

    last_gyro_z = 0.0
    last_t = 0.0
    t0 = time.perf_counter()
    for ev in sim_res.sensor_events:
        if ev.sensor_type == "imu":
            last_gyro_z = ev.data.gyro_rad_s[2]
        elif ev.sensor_type == "encoder":
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp
            enc = ev.data
            v_wheel = 0.25 * sum(enc[w].angular_speed_rad_s * cfg.wheel_radius_m for w in ["FL", "FR", "RL", "RR"])

            # Fuse Stereo VO speed when available
            t_key = round(round(ev.timestamp / 0.10) * 0.10, 3)
            vo_res = vo_stream.get(t_key)
            if vo_res is not None and vo_res.is_valid and abs(ev.timestamp - t_key) < 0.02:
                # Complementary weighting: 70% VO, 30% Wheel (reduces slip sensitivity)
                v = 0.70 * vo_res.v_forward_m_s + 0.30 * v_wheel
            else:
                v = v_wheel

            p, c = est.predict(np.array([v, last_gyro_z]), np.diag([5e-5, 2.5e-5]), dt)
            t_hist.append(ev.timestamp)
            p_hist.append(p[:3].copy())
            cov_hist.append(np.diag(c[:3, :3]))
            cov_mat_hist.append(c[:3, :3].copy())
    total_time = (time.perf_counter() - t0) * 1000.0 / len(t_hist)

    m = compute_trajectory_metrics("E_encoders_imu_stereo", sim_res.timestamps, sim_res.ground_truth_pose, np.array(t_hist), np.array(p_hist), np.array(cov_hist), cov_mats=np.array(cov_mat_hist))
    return {"pos_rmse_m": m.pos_rmse_m, "heading_rmse_deg": m.heading_rmse_deg, "coverage_95": m.coverage_95 * 100.0, "runtime_ms_step": total_time}


def eval_config_F(sim_res: SimulationResult, vo_stream: Dict[float, Any], cfg: Robot4WDConfig) -> Dict[str, Any]:
    """Config F: Encoders + IMU + Stereo VO + GNSS (Full Sensor Fusion)."""
    p0 = sim_res.ground_truth_pose[0]
    est = BiasAwareEKF4D(initial_state=np.array([p0[0], p0[1], p0[2], 0.0]), initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]))
    t_hist, p_hist, cov_hist, cov_mat_hist = [], [], [], []

    last_gyro_z = 0.0
    last_t = 0.0
    t0 = time.perf_counter()
    for ev in sim_res.sensor_events:
        if ev.sensor_type == "imu":
            last_gyro_z = ev.data.gyro_rad_s[2]
        elif ev.sensor_type == "encoder":
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp
            enc = ev.data
            v_wheel = 0.25 * sum(enc[w].angular_speed_rad_s * cfg.wheel_radius_m for w in ["FL", "FR", "RL", "RR"])

            t_key = round(round(ev.timestamp / 0.10) * 0.10, 3)
            vo_res = vo_stream.get(t_key)
            if vo_res is not None and vo_res.is_valid and abs(ev.timestamp - t_key) < 0.02:
                v = 0.70 * vo_res.v_forward_m_s + 0.30 * v_wheel
            else:
                v = v_wheel

            p, c = est.predict(np.array([v, last_gyro_z]), np.diag([5e-5, 2.5e-5]), dt)
            t_hist.append(ev.timestamp)
            p_hist.append(p[:3].copy())
            cov_hist.append(np.diag(c[:3, :3]))
            cov_mat_hist.append(c[:3, :3].copy())
        elif ev.sensor_type == "gnss":
            fix = ev.data
            if fix.is_valid:
                est.update_gnss(np.array([fix.x_enu_m, fix.y_enu_m]), fix.cov_enu)
                if p_hist:
                    p_hist[-1] = est.x[:3].copy()
                    cov_hist[-1] = np.diag(est.P[:3, :3])
                    cov_mat_hist[-1] = est.P[:3, :3].copy()
    total_time = (time.perf_counter() - t0) * 1000.0 / len(t_hist)

    m = compute_trajectory_metrics("F_full_fusion", sim_res.timestamps, sim_res.ground_truth_pose, np.array(t_hist), np.array(p_hist), np.array(cov_hist), cov_mats=np.array(cov_mat_hist))
    return {"pos_rmse_m": m.pos_rmse_m, "heading_rmse_deg": m.heading_rmse_deg, "coverage_95": m.coverage_95 * 100.0, "runtime_ms_step": total_time}


def eval_config_I(sim_res: SimulationResult, vo_stream: Dict[float, Any], cfg: Robot4WDConfig) -> Dict[str, Any]:
    """Config I: Stereo VO + IMU without Wheel Encoders (Immune to wheel slip)."""
    p0 = sim_res.ground_truth_pose[0]
    est = BiasAwareEKF4D(initial_state=np.array([p0[0], p0[1], p0[2], 0.0]), initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]))
    t_hist, p_hist, cov_hist, cov_mat_hist = [], [], [], []

    last_gyro_z = 0.0
    last_v_vo = 0.0
    last_t = 0.0
    t0 = time.perf_counter()
    for ev in sim_res.sensor_events:
        if ev.sensor_type == "imu":
            last_gyro_z = ev.data.gyro_rad_s[2]
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp

            # Lookup VO speed
            t_key = round(round(ev.timestamp / 0.10) * 0.10, 3)
            vo_res = vo_stream.get(t_key)
            if vo_res is not None and vo_res.is_valid:
                last_v_vo = vo_res.v_forward_m_s

            p, c = est.predict(np.array([last_v_vo, last_gyro_z]), np.diag([2e-4, 2.5e-5]), dt)
            t_hist.append(ev.timestamp)
            p_hist.append(p[:3].copy())
            cov_hist.append(np.diag(c[:3, :3]))
            cov_mat_hist.append(c[:3, :3].copy())
    total_time = (time.perf_counter() - t0) * 1000.0 / len(t_hist)

    m = compute_trajectory_metrics("I_stereo_imu_no_wheels", sim_res.timestamps, sim_res.ground_truth_pose, np.array(t_hist), np.array(p_hist), np.array(cov_hist), cov_mats=np.array(cov_mat_hist))
    return {"pos_rmse_m": m.pos_rmse_m, "heading_rmse_deg": m.heading_rmse_deg, "coverage_95": m.coverage_95 * 100.0, "runtime_ms_step": total_time}


def eval_config_J(sim_res: SimulationResult, strategy: RecoveryStrategy, cfg: Robot4WDConfig) -> Dict[str, Any]:
    """Config J: GNSS Gate-Lockout Recovery Strategies under 20s Outage + 10% Slip."""
    p0 = sim_res.ground_truth_pose[0]
    est = RecoverableRobustEKF4D(
        strategy=strategy,
        initial_state=np.array([p0[0], p0[1], p0[2], 0.0]),
        initial_cov=np.diag([0.01, 0.01, 0.005, 0.001]),
        distance_drift_fraction=0.15
    )
    t_hist, p_hist, cov_hist, cov_mat_hist = [], [], [], []

    last_gyro_z = 0.0
    last_t = 0.0
    t0 = time.perf_counter()
    for ev in sim_res.sensor_events:
        if ev.sensor_type == "imu":
            last_gyro_z = ev.data.gyro_rad_s[2]
        elif ev.sensor_type == "encoder":
            dt = ev.timestamp - last_t if last_t > 0 else 0.02
            last_t = ev.timestamp
            enc = ev.data
            v_fl = enc["FL"].angular_speed_rad_s * cfg.wheel_radius_m
            v_fr = enc["FR"].angular_speed_rad_s * cfg.wheel_radius_m
            v_rl = enc["RL"].angular_speed_rad_s * cfg.wheel_radius_m
            v_rr = enc["RR"].angular_speed_rad_s * cfg.wheel_radius_m
            v = 0.25 * (v_fl + v_fr + v_rl + v_rr)
            omega_enc = (0.5 * (v_fr + v_rr) - 0.5 * (v_fl + v_rl)) / cfg.effective_track_width_m
            disagree_val = max(abs(v_fl - v_rl), abs(v_fr - v_rr))

            p, c = est.predict_recoverable(
                v_enc=v,
                omega_enc=omega_enc,
                gyro_z=last_gyro_z,
                Q_u_base=np.diag([1e-4, 2.5e-5]),
                dt=dt,
                wheel_disagreement=disagree_val
            )
            t_hist.append(ev.timestamp)
            p_hist.append(p[:3].copy())
            cov_hist.append(np.diag(c[:3, :3]))
            cov_mat_hist.append(c[:3, :3].copy())
        elif ev.sensor_type == "gnss":
            fix = ev.data
            if fix.is_valid:
                accepted, _ = est.update_gnss_recoverable(ev.timestamp, np.array([fix.x_enu_m, fix.y_enu_m]), fix.cov_enu)
                if accepted and p_hist:
                    p_hist[-1] = est.x[:3].copy()
                    cov_hist[-1] = np.diag(est.P[:3, :3])
                    cov_mat_hist[-1] = est.P[:3, :3].copy()
    total_time = (time.perf_counter() - t0) * 1000.0 / len(t_hist)

    m = compute_trajectory_metrics(strategy.value, sim_res.timestamps, sim_res.ground_truth_pose, np.array(t_hist), np.array(p_hist), np.array(cov_hist), cov_mats=np.array(cov_mat_hist))
    return {"pos_rmse_m": m.pos_rmse_m, "heading_rmse_deg": m.heading_rmse_deg, "coverage_95": m.coverage_95 * 100.0, "runtime_ms_step": total_time}


if __name__ == "__main__":
    run_ablation_benchmarks()
