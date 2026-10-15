"""
robot_localization.simulation.world

Environment representation: 2D landmark map, wall segments, and reference trajectories.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class LandmarkMap:
    """Known 2D point landmarks and linear wall segments."""
    landmarks: Dict[int, np.ndarray] = field(default_factory=dict)  # id -> np.array([x, y])
    walls: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list) # [([x1, y1], [x2, y2]), ...]

    @classmethod
    def create_default_testbed(cls) -> "LandmarkMap":
        """
        Create a standard 20m x 20m research testbed with 8 peripheral landmarks
        and perimeter boundary walls.
        """
        lms = {
            1: np.array([8.0, 8.0]),
            2: np.array([-8.0, 8.0]),
            3: np.array([-8.0, -8.0]),
            4: np.array([8.0, -8.0]),
            5: np.array([0.0, 9.0]),
            6: np.array([0.0, -9.0]),
            7: np.array([9.0, 0.0]),
            8: np.array([-9.0, 0.0]),
        }
        # Perimeter boundary walls
        w1 = (np.array([-10.0, -10.0]), np.array([10.0, -10.0]))
        w2 = (np.array([10.0, -10.0]), np.array([10.0, 10.0]))
        w3 = (np.array([10.0, 10.0]), np.array([-10.0, 10.0]))
        w4 = (np.array([-10.0, 10.0]), np.array([-10.0, -10.0]))
        return cls(landmarks=lms, walls=[w1, w2, w3, w4])


class TrajectoryGenerator:
    """Generates continuous reference commanded velocities v(t) and omega(t)."""

    @staticmethod
    def figure_eight(t: float, speed: float = 0.8, loop_radius: float = 4.0) -> Tuple[float, float]:
        """
        Figure-eight trajectory:
        Omega alternates smoothly to form clockwise and counter-clockwise loops.
        """
        # Period of one loop
        loop_omega = speed / loop_radius
        period = 4.0 * math.pi / loop_omega
        # Modulate angular rate with period
        phase = (2.0 * math.pi * t) / (period / 2.0)
        omega = loop_omega * math.cos(phase)
        return speed, omega

    @staticmethod
    def straight_and_turns(t: float, speed: float = 0.8) -> Tuple[float, float]:
        """Trajectory consisting of straight segments, 90 deg turns, and point turns."""
        cycle = t % 20.0
        if cycle < 5.0:
            # Straight
            return speed, 0.0
        elif cycle < 8.0:
            # 90 deg turn
            return speed, 0.5236 # ~30 deg/s
        elif cycle < 13.0:
            # Straight
            return speed, 0.0
        elif cycle < 15.0:
            # Point turn (stop and pivot)
            return 0.0, 0.7854 # 45 deg/s
        else:
            # Return straight
            return speed, -0.35


class SimulationWorld:
    """Container holding the environment map and trajectory generator."""

    def __init__(self, landmark_map: Optional[LandmarkMap] = None):
        self.map = landmark_map if landmark_map is not None else LandmarkMap.create_default_testbed()
        self.traj_gen = TrajectoryGenerator()
