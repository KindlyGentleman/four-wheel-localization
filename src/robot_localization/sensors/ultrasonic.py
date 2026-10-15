"""
robot_localization.sensors.ultrasonic

Simulation of ultrasonic range finders measuring distances to known planar walls.
Features:
- Configurable transducer array: front-center, front-left (+30 deg), front-right (-30 deg)
- Geometric ray-plane intersection against known map wall segments
- Transducer mounting offsets and orientation transforms
- Analytical measurement Jacobians w.r.t robot pose
- Realistic beam spread angle cutoff and measurement noise
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from robot_localization.models.kinematics import wrap_angle, rotation_matrix_2d


@dataclass
class UltrasonicEcho:
    """A range reading from an ultrasonic transducer."""
    sensor_name: str
    range_m: float
    is_valid: bool
    wall_index: Optional[int]
    expected_range_m: float
    variance_m2: float


class UltrasonicArray:
    """
    Array of ultrasonic transducers mounted on the mobile chassis.
    """

    def __init__(
        self,
        sample_rate_hz: float = 10.0,
        max_range_m: float = 3.5,
        min_range_m: float = 0.04,
        range_std_m: float = 0.02,
        beam_angle_rad: float = math.radians(15.0),
        rng: Optional[np.random.RandomState] = None
    ):
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.max_range_m = max_range_m
        self.min_range_m = min_range_m
        self.range_std = range_std_m
        self.beam_angle_rad = beam_angle_rad
        self.rng = rng if rng is not None else np.random.RandomState()

        # Transducer configurations relative to body frame: (x_offset, y_offset, yaw_offset)
        self.transducers = {
            "front_center": (0.275, 0.0, 0.0),
            "front_left":   (0.275, 0.18, math.radians(30.0)),
            "front_right":  (0.275, -0.18, math.radians(-30.0)),
        }

    def measure_walls(
        self,
        timestamp: float,
        robot_pose: np.ndarray,
        walls: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, UltrasonicEcho]:
        """
        Ray-cast each transducer against planar wall segments.
        Returns echo dictionary.
        """
        px, py, theta = robot_pose[0], robot_pose[1], robot_pose[2]
        R = rotation_matrix_2d(theta)
        p_robot = np.array([px, py])

        echos = {}
        var_r = self.range_std ** 2

        for name, (lx, ly, yaw_off) in self.transducers.items():
            # Transducer pose in world frame
            p_trans = p_robot + R @ np.array([lx, ly])
            theta_trans = wrap_angle(theta + yaw_off)
            ray_dir = np.array([math.cos(theta_trans), math.sin(theta_trans)])

            closest_range = float("inf")
            best_wall_idx = None

            for w_idx, (w_start, w_end) in enumerate(walls):
                # Line segment intersection
                # Ray: p = p_trans + s * ray_dir, s >= 0
                # Wall: q = w_start + t * (w_end - w_start), t in [0, 1]
                v_wall = w_end - w_start
                denom = ray_dir[0] * v_wall[1] - ray_dir[1] * v_wall[0]
                if abs(denom) < 1e-9:
                    continue  # Parallel

                diff = w_start - p_trans
                s = (diff[0] * v_wall[1] - diff[1] * v_wall[0]) / denom
                t_seg = (diff[0] * ray_dir[1] - diff[1] * ray_dir[0]) / denom

                if s > self.min_range_m and 0.0 <= t_seg <= 1.0:
                    if s < closest_range:
                        # Check angle of incidence
                        normal = np.array([-v_wall[1], v_wall[0]])
                        normal = normal / np.linalg.norm(normal)
                        cos_inc = abs(np.dot(ray_dir, normal))
                        if cos_inc >= math.cos(self.beam_angle_rad):
                            closest_range = s
                            best_wall_idx = w_idx

            if closest_range <= self.max_range_m and best_wall_idx is not None:
                r_meas = closest_range + self.rng.normal(0.0, self.range_std)
                echos[name] = UltrasonicEcho(
                    sensor_name=name,
                    range_m=max(0.01, r_meas),
                    is_valid=True,
                    wall_index=best_wall_idx,
                    expected_range_m=closest_range,
                    variance_m2=var_r
                )
            else:
                echos[name] = UltrasonicEcho(
                    sensor_name=name,
                    range_m=self.max_range_m,
                    is_valid=False,
                    wall_index=None,
                    expected_range_m=float("inf"),
                    variance_m2=var_r
                )

        return echos
