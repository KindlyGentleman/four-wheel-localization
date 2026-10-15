"""
robot_localization.simulation.robot_simulator

Full 4WD robot multi-sensor simulation engine.
Integrates ground truth robot dynamics at high rate (100 Hz), applies longitudinal wheel slip,
and schedules asynchronous discrete sensor events (Encoders 50Hz, IMU 50Hz, Compass 10Hz, GNSS 1Hz).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.models.skid_steer import SkidSteerKinematics, ChassisVelocity, WheelSpeeds
from robot_localization.models.kinematics import integrate_pose_rk2, wrap_angle
from robot_localization.sensors.encoder import FourWheelEncoders, EncoderSample
from robot_localization.sensors.gnss import GNSSReceiver, GNSSFix
from robot_localization.sensors.cmps12 import CMPS12Sensor, CMPS12Sample
from robot_localization.sensors.mpu9250 import MPU9250Sensor, MPU9250Sample
from robot_localization.simulation.scheduler import EventScheduler, SensorEvent


@dataclass
class SimulationResult:
    """Precomputed ground truth and sensor observation traces for reproducible evaluation."""
    timestamps: np.ndarray             # High-rate ground truth time vector
    ground_truth_pose: np.ndarray      # (N, 3): [x, y, theta]
    ground_truth_chassis_vel: np.ndarray # (N, 2): [v_ground, omega_ground]
    wheel_speeds_rotational: np.ndarray # (N, 4): [FL, FR, RL, RR] in rad/s
    wheel_speeds_ground: np.ndarray     # (N, 4): [FL, FR, RL, RR] in m/s
    applied_slip_ratios: np.ndarray    # (N, 4): [FL, FR, RL, RR]
    wheel_disagreement: Dict[str, np.ndarray] # 'd_left', 'd_right', 'speed_variance'

    # Chronologically ordered sensor event stream
    sensor_events: List[SensorEvent] = field(default_factory=list)

    # Fast indexed lookup dictionaries by timestamp or sequential arrays
    encoder_samples: List[Dict[str, EncoderSample]] = field(default_factory=list)
    gnss_samples: List[GNSSFix] = field(default_factory=list)
    cmps12_samples: List[CMPS12Sample] = field(default_factory=list)
    mpu9250_samples: List[MPU9250Sample] = field(default_factory=list)


class RobotSimulator:
    """
    Continuous ground truth integration and multi-rate sensor sampling coordinator.
    """

    def __init__(
        self,
        robot_config: Robot4WDConfig,
        gnss_receiver: Optional[GNSSReceiver] = None,
        cmps12_sensor: Optional[CMPS12Sensor] = None,
        mpu9250_sensor: Optional[MPU9250Sensor] = None,
        dt_sim_s: float = 0.01, # 100 Hz ground-truth physics
        rate_encoder_hz: float = 50.0,
        rate_imu_hz: float = 50.0,
        rate_compass_hz: float = 10.0,
        rate_gnss_hz: float = 1.0,
        rng: Optional[np.random.RandomState] = None
    ):
        self.config = robot_config
        self.kinematics = SkidSteerKinematics(robot_config)
        self.dt_sim = dt_sim_s
        self.rng = rng if rng is not None else np.random.RandomState(42)

        # Sensors
        self.encoders = FourWheelEncoders(robot_config, rng=self.rng)
        self.gnss = gnss_receiver if gnss_receiver is not None else GNSSReceiver(rng=self.rng)
        self.cmps12 = cmps12_sensor if cmps12_sensor is not None else CMPS12Sensor(rng=self.rng)
        self.mpu9250 = mpu9250_sensor if mpu9250_sensor is not None else MPU9250Sensor(rng=self.rng)

        # Sampling rates & periods
        self.period_encoder = 1.0 / rate_encoder_hz
        self.period_imu = 1.0 / rate_imu_hz
        self.period_compass = 1.0 / rate_compass_hz
        self.period_gnss = 1.0 / rate_gnss_hz

    def run_simulation(
        self,
        duration_s: float,
        trajectory_func: Callable[[float], Tuple[float, float]],
        slip_profile_func: Optional[Callable[[float], Dict[str, float]]] = None,
        initial_pose: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """
        Execute forward simulation over specified duration.

        Args:
            duration_s: Total simulation time
            trajectory_func: Callable t -> (v_cmd, omega_cmd)
            slip_profile_func: Callable t -> {'FL': s, 'FR': s, 'RL': s, 'RR': s}
            initial_pose: [x0, y0, theta0]
        """
        scheduler = EventScheduler()
        t = 0.0
        pose = np.zeros(3) if initial_pose is None else np.array(initial_pose, dtype=float).copy()

        # Schedule initial sensor events
        scheduler.schedule(0.0, "encoder")
        scheduler.schedule(0.0, "imu")
        scheduler.schedule(0.0, "compass")
        scheduler.schedule(0.0, "gnss")

        # Ground truth history buffers
        t_hist = []
        pose_hist = []
        vel_hist = []
        wheel_rot_hist = []
        wheel_ground_hist = []
        slip_hist = []
        d_left_hist = []
        d_right_hist = []
        var_hist = []

        all_events: List[SensorEvent] = []
        enc_samples: List[Dict[str, EncoderSample]] = []
        gnss_samples: List[GNSSFix] = []
        cmps_samples: List[CMPS12Sample] = []
        mpu_samples: List[MPU9250Sample] = []

        last_encoder_time = 0.0
        last_imu_time = 0.0
        last_compass_time = 0.0
        last_gnss_time = 0.0

        prev_v_cmd = 0.0

        while t <= duration_s:
            # 1. Commanded chassis velocities
            v_cmd, omega_cmd = trajectory_func(t)
            cmd = ChassisVelocity(v=v_cmd, omega=omega_cmd)

            # 2. Inverse kinematics: ideal wheel rotational speeds
            w_rot = self.kinematics.chassis_to_wheel_speeds(cmd)

            # 3. Dynamic wheel slip ratio
            if slip_profile_func is not None:
                slips = slip_profile_func(t)
            else:
                slips = {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}

            # 4. Actual wheel-ground linear speeds and resulting chassis ground velocity
            vg_fl = (1.0 - slips["FL"]) * self.config.wheels["FL"].radius_m * w_rot.fl
            vg_fr = (1.0 - slips["FR"]) * self.config.wheels["FR"].radius_m * w_rot.fr
            vg_rl = (1.0 - slips["RL"]) * self.config.wheels["RL"].radius_m * w_rot.rl
            vg_rr = (1.0 - slips["RR"]) * self.config.wheels["RR"].radius_m * w_rot.rr

            chassis_ground = self.kinematics.compute_ground_chassis_velocity(w_rot, slips)

            # Disagreement metrics
            speeds_linear_ideal = {
                "FL": self.config.wheels["FL"].radius_m * w_rot.fl,
                "FR": self.config.wheels["FR"].radius_m * w_rot.fr,
                "RL": self.config.wheels["RL"].radius_m * w_rot.rl,
                "RR": self.config.wheels["RR"].radius_m * w_rot.rr,
            }
            disag = self.kinematics.compute_wheel_disagreement(speeds_linear_ideal)

            # 5. Integrate ground-truth chassis pose using actual ground velocity
            pose = integrate_pose_rk2(pose, chassis_ground.v, chassis_ground.omega, self.dt_sim)

            # Record high-rate ground truth
            t_hist.append(t)
            pose_hist.append(pose.copy())
            vel_hist.append(chassis_ground.to_array())
            wheel_rot_hist.append(w_rot.to_array())
            wheel_ground_hist.append([vg_fl, vg_fr, vg_rl, vg_rr])
            slip_hist.append([slips["FL"], slips["FR"], slips["RL"], slips["RR"]])
            d_left_hist.append(disag["d_left"])
            d_right_hist.append(disag["d_right"])
            var_hist.append(disag["speed_variance"])

            # 6. Service sensor events up to current simulation time t
            while scheduler.has_next() and scheduler.peek_next_time() <= (t + 1e-9):
                ev = scheduler.pop_next()
                ev_time = ev.timestamp

                if ev.sensor_type == "encoder":
                    dt_enc = max(ev_time - last_encoder_time, 1e-6) if last_encoder_time > 0 else self.period_encoder
                    last_encoder_time = ev_time
                    rot_dict = {"FL": w_rot.fl, "FR": w_rot.fr, "RL": w_rot.rl, "RR": w_rot.rr}
                    sample_dict = self.encoders.update(ev_time, rot_dict, dt=dt_enc)
                    ev.data = sample_dict
                    enc_samples.append(sample_dict)
                    all_events.append(ev)
                    # Schedule next encoder
                    scheduler.schedule(ev_time + self.period_encoder, "encoder")

                elif ev.sensor_type == "imu":
                    dt_imu = max(ev_time - last_imu_time, 1e-6) if last_imu_time > 0 else self.period_imu
                    last_imu_time = ev_time
                    accel_forward = (v_cmd - prev_v_cmd) / self.dt_sim
                    imu_sample = self.mpu9250.update(
                        timestamp=ev_time,
                        true_theta_rad=pose[2],
                        true_v_m_s=chassis_ground.v,
                        true_yaw_rate_rad_s=chassis_ground.omega,
                        true_linear_accel_m_s2=accel_forward,
                        dt_since_last=dt_imu
                    )
                    ev.data = imu_sample
                    mpu_samples.append(imu_sample)
                    all_events.append(ev)
                    scheduler.schedule(ev_time + self.period_imu, "imu")

                elif ev.sensor_type == "compass":
                    last_compass_time = ev_time
                    cmps_sample = self.cmps12.update(
                        timestamp=ev_time,
                        true_theta_nav_rad=pose[2],
                        true_yaw_rate_rad_s=chassis_ground.omega
                    )
                    ev.data = cmps_sample
                    cmps_samples.append(cmps_sample)
                    all_events.append(ev)
                    scheduler.schedule(ev_time + self.period_compass, "compass")

                elif ev.sensor_type == "gnss":
                    dt_gnss = max(ev_time - last_gnss_time, 1e-6) if last_gnss_time > 0 else self.period_gnss
                    last_gnss_time = ev_time
                    fix = self.gnss.update(ev_time, pose, dt_since_last=dt_gnss)
                    ev.data = fix
                    gnss_samples.append(fix)
                    all_events.append(ev)
                    scheduler.schedule(ev_time + self.period_gnss, "gnss")

            prev_v_cmd = v_cmd
            t += self.dt_sim

        return SimulationResult(
            timestamps=np.array(t_hist),
            ground_truth_pose=np.array(pose_hist),
            ground_truth_chassis_vel=np.array(vel_hist),
            wheel_speeds_rotational=np.array(wheel_rot_hist),
            wheel_speeds_ground=np.array(wheel_ground_hist),
            applied_slip_ratios=np.array(slip_hist),
            wheel_disagreement={
                "d_left": np.array(d_left_hist),
                "d_right": np.array(d_right_hist),
                "speed_variance": np.array(var_hist)
            },
            sensor_events=all_events,
            encoder_samples=enc_samples,
            gnss_samples=gnss_samples,
            cmps12_samples=cmps_samples,
            mpu9250_samples=mpu_samples,
        )
