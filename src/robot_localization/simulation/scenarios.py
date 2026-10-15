"""
robot_localization.simulation.scenarios

Preconfigured benchmark scenario generators for reproducible evaluations:
- Nominal motion
- Asymmetric wheel slip (left side, right side, isolated wheel)
- GNSS outages and stochastic dropouts
- Gyroscope bias and drift
- Magnetic compass disturbances
- GNSS multipath bias events
- Combined failure scenarios
"""

import math
from typing import Dict, Tuple, Optional, Callable
import numpy as np

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.sensors.gnss import GNSSReceiver
from robot_localization.sensors.cmps12 import CMPS12Sensor
from robot_localization.sensors.mpu9250 import MPU9250Sensor
from robot_localization.simulation.robot_simulator import RobotSimulator
from robot_localization.simulation.world import TrajectoryGenerator


def create_scenario_nominal(
    seed: int = 42,
    duration_s: float = 60.0,
    speed_m_s: float = 0.8
) -> Tuple[RobotSimulator, Callable, Callable]:
    """Nominal scenario: No wheel slip, full sensor availability."""
    rng = np.random.RandomState(seed)
    cfg = Robot4WDConfig()
    gnss = GNSSReceiver(sample_rate_hz=1.0, dropout_prob=0.0, rng=rng)
    cmps = CMPS12Sensor(rng=rng)
    mpu = MPU9250Sensor(rng=rng)

    sim = RobotSimulator(
        robot_config=cfg,
        gnss_receiver=gnss,
        cmps12_sensor=cmps,
        mpu9250_sensor=mpu,
        rng=rng
    )

    def traj(t):
        return TrajectoryGenerator.figure_eight(t, speed=speed_m_s, loop_radius=4.0)

    def slip(t):
        return {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}

    return sim, traj, slip


def create_scenario_slip(
    seed: int = 42,
    duration_s: float = 60.0,
    slip_ratio_left: float = 0.35,
    t_start: float = 15.0,
    t_end: float = 35.0
) -> Tuple[RobotSimulator, Callable, Callable]:
    """Asymmetric wheel slip on left side between t_start and t_end."""
    rng = np.random.RandomState(seed)
    cfg = Robot4WDConfig()
    gnss = GNSSReceiver(sample_rate_hz=1.0, dropout_prob=0.0, rng=rng)
    cmps = CMPS12Sensor(rng=rng)
    mpu = MPU9250Sensor(rng=rng)

    sim = RobotSimulator(cfg, gnss, cmps, mpu, rng=rng)

    def traj(t):
        return TrajectoryGenerator.figure_eight(t, speed=0.8, loop_radius=4.0)

    def slip(t):
        if t_start <= t <= t_end:
            return {"FL": slip_ratio_left, "FR": 0.0, "RL": slip_ratio_left, "RR": 0.0}
        return {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}

    return sim, traj, slip


def create_scenario_gnss_outage(
    seed: int = 42,
    duration_s: float = 60.0,
    outage_start_s: float = 15.0,
    outage_end_s: float = 35.0,
    moderate_slip: float = 0.10
) -> Tuple[RobotSimulator, Callable, Callable]:
    """Contiguous 20-second GNSS outage with moderate baseline slip."""
    rng = np.random.RandomState(seed)
    cfg = Robot4WDConfig()
    gnss = GNSSReceiver(
        sample_rate_hz=1.0,
        outage_intervals=[(outage_start_s, outage_end_s)],
        rng=rng
    )
    cmps = CMPS12Sensor(rng=rng)
    mpu = MPU9250Sensor(rng=rng)

    sim = RobotSimulator(cfg, gnss, cmps, mpu, rng=rng)

    def traj(t):
        return TrajectoryGenerator.figure_eight(t, speed=0.8, loop_radius=4.0)

    def slip(t):
        return {"FL": moderate_slip, "FR": moderate_slip, "RL": moderate_slip, "RR": moderate_slip}

    return sim, traj, slip


def create_scenario_gyro_bias(
    seed: int = 42,
    duration_s: float = 60.0,
    gyro_bias_rad_s: float = 0.035 # ~2.0 deg/s
) -> Tuple[RobotSimulator, Callable, Callable]:
    """Significant constant + random walk gyro bias."""
    rng = np.random.RandomState(seed)
    cfg = Robot4WDConfig()
    gnss = GNSSReceiver(sample_rate_hz=1.0, rng=rng)
    cmps = CMPS12Sensor(gyro_bias_rad_s=gyro_bias_rad_s, rng=rng)
    mpu = MPU9250Sensor(
        gyro_bias_initial_rad_s=gyro_bias_rad_s,
        gyro_bias_walk_std_rad_s_sqrt_s=0.0005,
        rng=rng
    )

    sim = RobotSimulator(cfg, gnss, cmps, mpu, rng=rng)

    def traj(t):
        return TrajectoryGenerator.figure_eight(t, speed=0.8, loop_radius=4.0)

    def slip(t):
        return {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}

    return sim, traj, slip


def create_scenario_compass_disturbance(
    seed: int = 42,
    duration_s: float = 60.0,
    disturb_start_s: float = 20.0,
    disturb_end_s: float = 35.0,
    bias_deg: float = 30.0
) -> Tuple[RobotSimulator, Callable, Callable]:
    """Temporary magnetic interference distorting compass heading."""
    rng = np.random.RandomState(seed)
    cfg = Robot4WDConfig()
    gnss = GNSSReceiver(sample_rate_hz=1.0, rng=rng)
    cmps = CMPS12Sensor(rng=rng)
    mpu = MPU9250Sensor(rng=rng)

    # Schedule disturbance triggers
    def pre_step_callback(t: float):
        if disturb_start_s <= t <= disturb_end_s:
            cmps.trigger_magnetic_disturbance(bias_rad=math.radians(bias_deg), active=True)
        else:
            cmps.trigger_magnetic_disturbance(active=False)

    sim = RobotSimulator(cfg, gnss, cmps, mpu, rng=rng)

    def traj(t):
        pre_step_callback(t)
        return TrajectoryGenerator.figure_eight(t, speed=0.8, loop_radius=4.0)

    def slip(t):
        return {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}

    return sim, traj, slip


def create_scenario_multipath(
    seed: int = 42,
    duration_s: float = 60.0,
    t_start: float = 20.0,
    t_end: float = 35.0,
    bias_m: float = 3.5
) -> Tuple[RobotSimulator, Callable, Callable]:
    """Temporary GNSS multipath position jump."""
    rng = np.random.RandomState(seed)
    cfg = Robot4WDConfig()
    gnss = GNSSReceiver(sample_rate_hz=1.0, rng=rng)
    cmps = CMPS12Sensor(rng=rng)
    mpu = MPU9250Sensor(rng=rng)

    def pre_step(t: float):
        if t_start <= t <= t_end:
            gnss.trigger_multipath_event(bias_east=bias_m, bias_north=bias_m, active=True)
        else:
            gnss.trigger_multipath_event(active=False)

    sim = RobotSimulator(cfg, gnss, cmps, mpu, rng=rng)

    def traj(t):
        pre_step(t)
        return TrajectoryGenerator.figure_eight(t, speed=0.8, loop_radius=4.0)

    def slip(t):
        return {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}

    return sim, traj, slip


def create_scenario_combined(
    seed: int = 42,
    duration_s: float = 60.0
) -> Tuple[RobotSimulator, Callable, Callable]:
    """Combined severe stress test: Left-side slip + 20s GNSS outage + gyro bias."""
    rng = np.random.RandomState(seed)
    cfg = Robot4WDConfig()
    gnss = GNSSReceiver(
        sample_rate_hz=1.0,
        outage_intervals=[(15.0, 35.0)],
        rng=rng
    )
    cmps = CMPS12Sensor(rng=rng)
    mpu = MPU9250Sensor(gyro_bias_initial_rad_s=0.025, rng=rng)

    sim = RobotSimulator(cfg, gnss, cmps, mpu, rng=rng)

    def traj(t):
        return TrajectoryGenerator.figure_eight(t, speed=0.8, loop_radius=4.0)

    def slip(t):
        if 15.0 <= t <= 35.0:
            return {"FL": 0.30, "FR": 0.0, "RL": 0.30, "RR": 0.0}
        return {"FL": 0.05, "FR": 0.05, "RL": 0.05, "RR": 0.05}

    return sim, traj, slip
