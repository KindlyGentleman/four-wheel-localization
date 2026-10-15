"""
Simulation engine: World maps, event scheduling, robot simulator, and scenario builders.
"""

from robot_localization.simulation.world import SimulationWorld, LandmarkMap
from robot_localization.simulation.scheduler import EventScheduler, SensorEvent
from robot_localization.simulation.robot_simulator import RobotSimulator, SimulationResult
from robot_localization.simulation.scenarios import (
    create_scenario_nominal,
    create_scenario_slip,
    create_scenario_gnss_outage,
    create_scenario_gyro_bias,
    create_scenario_compass_disturbance,
    create_scenario_multipath,
    create_scenario_combined,
)

__all__ = [
    "SimulationWorld",
    "LandmarkMap",
    "EventScheduler",
    "SensorEvent",
    "RobotSimulator",
    "SimulationResult",
    "create_scenario_nominal",
    "create_scenario_slip",
    "create_scenario_gnss_outage",
    "create_scenario_gyro_bias",
    "create_scenario_compass_disturbance",
    "create_scenario_multipath",
    "create_scenario_combined",
]
