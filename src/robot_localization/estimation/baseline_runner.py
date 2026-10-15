"""
robot_localization.estimation.baseline_runner

Unified baseline execution engine.
Applies baselines B0 through B5 on identical simulated sensor data:
- B0: Four-wheel odometry only (DeadReckoning4WD)
- B1: Wheel speed + Gyro angular rate (EKF prediction only)
- B2: Wheel odometry + GNSS updates
- B3: Wheel speed + Gyro + GNSS updates (Primary baseline)
- B4: Wheel speed + Gyro + GNSS + Compass updates
- B5: Bias-Aware 4D EKF (Wheel speed + Gyro + GNSS + Compass + Gyro bias estimation)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.models.kinematics import wrap_angle
from robot_localization.estimation.dead_reckoning import DeadReckoning4WD
from robot_localization.estimation.ekf import ExtendedKalmanFilter3D
from robot_localization.estimation.bias_aware_ekf import BiasAwareEKF4D
from robot_localization.simulation.robot_simulator import SimulationResult
from robot_localization.sensors.encoder import EncoderSample
from robot_localization.sensors.gnss import GNSSFix
from robot_localization.sensors.cmps12 import CMPS12Sample
from robot_localization.sensors.mpu9250 import MPU9250Sample


@dataclass
class BaselineTrace:
    """Estimated state trajectory and covariance trace for a specific baseline."""
    name: str
    timestamps: np.ndarray             # (M,)
    estimated_pose: np.ndarray         # (M, 3): [x, y, theta]
    covariance_diag: np.ndarray        # (M, 3): [var_x, var_y, var_theta]
    estimated_gyro_bias: Optional[np.ndarray] = None # (M,) for B5
    nis_gnss: List[Tuple[float, float]] = field(default_factory=list) # [(t, nis), ...]
    nis_compass: List[Tuple[float, float]] = field(default_factory=list)


def run_all_baselines(
    sim_result: SimulationResult,
    robot_config: Optional[Robot4WDConfig] = None
) -> Dict[str, BaselineTrace]:
    """
    Execute all baselines B0 - B5 on the provided simulation result.

    Returns:
        dict: baseline_name -> BaselineTrace
    """
    cfg = robot_config if robot_config is not None else Robot4WDConfig()

    # Initial states
    p0 = sim_result.ground_truth_pose[0]
    initial_pose = p0.copy()
    initial_cov3 = np.diag([0.01, 0.01, 0.005])
    initial_cov4 = np.diag([0.01, 0.01, 0.005, 0.001])

    # Instantiate estimators
    est_b0 = DeadReckoning4WD(cfg, initial_pose=initial_pose, initial_cov=initial_cov3)
    est_b1 = ExtendedKalmanFilter3D(initial_state=initial_pose, initial_cov=initial_cov3)
    est_b2 = ExtendedKalmanFilter3D(initial_state=initial_pose, initial_cov=initial_cov3)
    est_b3 = ExtendedKalmanFilter3D(initial_state=initial_pose, initial_cov=initial_cov3)
    est_b4 = ExtendedKalmanFilter3D(initial_state=initial_pose, initial_cov=initial_cov3)
    est_b5 = BiasAwareEKF4D(
        initial_state=np.array([initial_pose[0], initial_pose[1], initial_pose[2], 0.0]),
        initial_cov=initial_cov4
    )

    # Buffers
    traces = {
        "B0_wheel_odometry": {"t": [], "pose": [], "cov": []},
        "B1_odometry_gyro": {"t": [], "pose": [], "cov": []},
        "B2_odometry_gnss": {"t": [], "pose": [], "cov": [], "nis_gnss": []},
        "B3_odometry_gyro_gnss": {"t": [], "pose": [], "cov": [], "nis_gnss": []},
        "B4_odometry_gyro_gnss_compass": {"t": [], "pose": [], "cov": [], "nis_gnss": [], "nis_compass": []},
        "B5_bias_aware_ekf": {"t": [], "pose": [], "cov": [], "bg": [], "nis_gnss": [], "nis_compass": []},
    }

    # Tracking last values
    last_gyro_z = 0.0
    last_gyro_var = 0.005 ** 2
    last_enc_time = 0.0

    # Process events strictly chronologically
    for event in sim_result.sensor_events:
        t = event.timestamp

        if event.sensor_type == "imu":
            # Update latest gyro rate
            sample_imu: MPU9250Sample = event.data
            last_gyro_z = sample_imu.gyro_rad_s[2]

        elif event.sensor_type == "encoder":
            # Step odometry prediction on all filters
            dt = t - last_enc_time if last_enc_time > 0 else 0.02
            last_enc_time = t
            samples_dict: Dict[str, EncoderSample] = event.data

            # B0: 4WD dead reckoning
            p_b0, cov_b0 = est_b0.update_from_encoders(samples_dict, dt)
            traces["B0_wheel_odometry"]["t"].append(t)
            traces["B0_wheel_odometry"]["pose"].append(p_b0.copy())
            traces["B0_wheel_odometry"]["cov"].append(np.diag(cov_b0))

            # Forward speed from encoders
            v_enc = est_b0.last_v
            var_v = est_b0.last_input_cov[0, 0]
            omega_enc = est_b0.last_omega
            var_omega_enc = est_b0.last_input_cov[1, 1]

            # B1: v_enc + gyro_z (Prediction only)
            u_b1 = np.array([v_enc, last_gyro_z])
            Qu_b1 = np.diag([var_v, last_gyro_var])
            p_b1, cov_b1 = est_b1.predict(u_b1, Qu_b1, dt)
            traces["B1_odometry_gyro"]["t"].append(t)
            traces["B1_odometry_gyro"]["pose"].append(p_b1.copy())
            traces["B1_odometry_gyro"]["cov"].append(np.diag(cov_b1))

            # B2: v_enc + omega_enc (Prediction)
            u_b2 = np.array([v_enc, omega_enc])
            Qu_b2 = np.diag([var_v, var_omega_enc])
            p_b2, cov_b2 = est_b2.predict(u_b2, Qu_b2, dt)
            traces["B2_odometry_gnss"]["t"].append(t)
            traces["B2_odometry_gnss"]["pose"].append(p_b2.copy())
            traces["B2_odometry_gnss"]["cov"].append(np.diag(cov_b2))

            # B3: v_enc + gyro_z (Prediction)
            p_b3, cov_b3 = est_b3.predict(u_b1, Qu_b1, dt)
            traces["B3_odometry_gyro_gnss"]["t"].append(t)
            traces["B3_odometry_gyro_gnss"]["pose"].append(p_b3.copy())
            traces["B3_odometry_gyro_gnss"]["cov"].append(np.diag(cov_b3))

            # B4: v_enc + gyro_z (Prediction)
            p_b4, cov_b4 = est_b4.predict(u_b1, Qu_b1, dt)
            traces["B4_odometry_gyro_gnss_compass"]["t"].append(t)
            traces["B4_odometry_gyro_gnss_compass"]["pose"].append(p_b4.copy())
            traces["B4_odometry_gyro_gnss_compass"]["cov"].append(np.diag(cov_b4))

            # B5: v_enc + gyro_z (Prediction with online bias)
            p_b5, cov_b5 = est_b5.predict(u_b1, Qu_b1, dt)
            traces["B5_bias_aware_ekf"]["t"].append(t)
            traces["B5_bias_aware_ekf"]["pose"].append(p_b5[:3].copy())
            traces["B5_bias_aware_ekf"]["cov"].append(np.diag(cov_b5[:3, :3]))
            traces["B5_bias_aware_ekf"]["bg"].append(p_b5[3])

        elif event.sensor_type == "gnss":
            fix: GNSSFix = event.data
            if fix.is_valid:
                z_pos = np.array([fix.x_enu_m, fix.y_enu_m])
                R_pos = fix.cov_enu

                # B2 Update
                if est_b2.update_gnss(z_pos, R_pos):
                    traces["B2_odometry_gnss"]["nis_gnss"].append((t, est_b2.last_nis_gnss))

                # B3 Update
                if est_b3.update_gnss(z_pos, R_pos):
                    traces["B3_odometry_gyro_gnss"]["nis_gnss"].append((t, est_b3.last_nis_gnss))

                # B4 Update
                if est_b4.update_gnss(z_pos, R_pos):
                    traces["B4_odometry_gyro_gnss_compass"]["nis_gnss"].append((t, est_b4.last_nis_gnss))

                # B5 Update
                if est_b5.update_gnss(z_pos, R_pos):
                    traces["B5_bias_aware_ekf"]["nis_gnss"].append((t, est_b5.last_nis_gnss))

        elif event.sensor_type == "compass":
            cmps_sample: CMPS12Sample = event.data
            if cmps_sample.is_valid:
                z_head = cmps_sample.heading_nav_rad
                R_head = cmps_sample.heading_std_rad ** 2

                # B4 compass update
                if est_b4.update_compass(z_head, R_head):
                    traces["B4_odometry_gyro_gnss_compass"]["nis_compass"].append((t, est_b4.last_nis_compass))

                # B5 compass update
                if est_b5.update_compass(z_head, R_head):
                    traces["B5_bias_aware_ekf"]["nis_compass"].append((t, est_b5.last_nis_compass))

    # Convert to BaselineTrace objects
    results = {}
    for name, data in traces.items():
        results[name] = BaselineTrace(
            name=name,
            timestamps=np.array(data["t"]),
            estimated_pose=np.array(data["pose"]),
            covariance_diag=np.array(data["cov"]),
            estimated_gyro_bias=np.array(data["bg"]) if "bg" in data else None,
            nis_gnss=data.get("nis_gnss", []),
            nis_compass=data.get("nis_compass", [])
        )
    return results
