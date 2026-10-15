"""
robot_localization.sensors.lidar

2D Range-Bearing LiDAR simulation against known environment landmarks.
Features:
- Mounting offset transformation relative to robot center
- Range and bearing observations to identifiable environmental landmarks
- Analytical measurement Jacobians for range and bearing
- Measurement noise, maximum range cutoff, and line-of-sight visibility
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from robot_localization.models.kinematics import wrap_angle, rotation_matrix_2d


@dataclass
class LandmarkObservation:
    """A range-bearing observation to a specific landmark."""
    landmark_id: int
    range_m: float
    bearing_rad: float           # Relative bearing in sensor frame [-pi, pi]
    landmark_pos_world: np.ndarray # [Lx, Ly]
    cov_rb: np.ndarray           # 2x2 covariance [var_r, 0; 0, var_bearing]


@dataclass
class LiDARScan:
    """A collection of landmark observations at a single timestamp."""
    timestamp: float
    sensor_pos_world: np.ndarray # [sx, sy]
    observations: List[LandmarkObservation]


class KnownLandmarkLiDAR:
    """
    Simulated 2D range-bearing sensor observing mapped landmark features.
    """

    def __init__(
        self,
        sample_rate_hz: float = 5.0,
        max_range_m: float = 12.0,
        min_range_m: float = 0.20,
        range_std_m: float = 0.03,
        bearing_std_rad: float = 0.015,
        mounting_offset_m: Tuple[float, float] = (0.20, 0.0),
        rng: Optional[np.random.RandomState] = None
    ):
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.max_range_m = max_range_m
        self.min_range_m = min_range_m
        self.range_std = range_std_m
        self.bearing_std = bearing_std_rad
        self.mounting_offset = np.array(mounting_offset_m, dtype=float)

        self.rng = rng if rng is not None else np.random.RandomState()

    def get_sensor_pose_world(self, robot_pose: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute (x_sensor, y_sensor) and theta_sensor in world frame."""
        px, py, theta = robot_pose[0], robot_pose[1], robot_pose[2]
        R = rotation_matrix_2d(theta)
        p_sensor = np.array([px, py]) + R @ self.mounting_offset
        return p_sensor, theta

    def observe_landmarks(
        self,
        timestamp: float,
        robot_pose: np.ndarray,
        landmarks: Dict[int, np.ndarray]
    ) -> LiDARScan:
        """
        Generate range-bearing observations for visible landmarks within sensor range.
        """
        p_sensor, theta = self.get_sensor_pose_world(robot_pose)
        obs_list = []

        cov_rb = np.diag([self.range_std ** 2, self.bearing_std ** 2])

        for lm_id, lm_pos in landmarks.items():
            dx = lm_pos[0] - p_sensor[0]
            dy = lm_pos[1] - p_sensor[1]
            r_true = math.hypot(dx, dy)

            # Check range limits
            if self.min_range_m <= r_true <= self.max_range_m:
                # Bearing in sensor frame: angle to landmark minus robot heading
                bearing_true = wrap_angle(math.atan2(dy, dx) - theta)

                # Add noise
                r_meas = max(0.01, r_true + self.rng.normal(0.0, self.range_std))
                b_meas = wrap_angle(bearing_true + self.rng.normal(0.0, self.bearing_std))

                obs_list.append(LandmarkObservation(
                    landmark_id=lm_id,
                    range_m=r_meas,
                    bearing_rad=b_meas,
                    landmark_pos_world=lm_pos.copy(),
                    cov_rb=cov_rb.copy()
                ))

        return LiDARScan(
            timestamp=timestamp,
            sensor_pos_world=p_sensor,
            observations=obs_list
        )

    @classmethod
    def compute_observation_jacobian(
        cls,
        robot_state: np.ndarray,
        landmark_pos: np.ndarray,
        mounting_offset: Tuple[float, float] = (0.20, 0.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute predicted observation [r, beta]^T and Jacobian H (2x3) w.r.t robot state [px, py, theta].
        """
        px, py, theta = robot_state[0], robot_state[1], robot_state[2]
        lx, ly = mounting_offset[0], mounting_offset[1]

        c = math.cos(theta)
        s = math.sin(theta)

        # Sensor position
        sx = px + lx * c - ly * s
        sy = py + lx * s + ly * c

        # Derivatives of sensor position w.r.t robot state
        # dsx/dpx = 1, dsx/dpy = 0, dsx/dtheta = -lx*s - ly*c
        # dsy/dpx = 0, dsy/dpy = 1, dsy/dtheta =  lx*c - ly*s
        dsx_dth = -lx * s - ly * c
        dsy_dth =  lx * c - ly * s

        dx = landmark_pos[0] - sx
        dy = landmark_pos[1] - sy
        q = dx * dx + dy * dy
        r = math.sqrt(q)

        # Predicted observation
        beta = wrap_angle(math.atan2(dy, dx) - theta)
        z_pred = np.array([r, beta])

        # Jacobian H (2x3)
        # dr/dsx = -dx/r, dr/dsy = -dy/r
        # dbeta/dsx = dy/q, dbeta/dsy = -dx/q, dbeta/dtheta = -1
        dr_dpx = -dx / r
        dr_dpy = -dy / r
        dr_dth = (-dx / r) * dsx_dth + (-dy / r) * dsy_dth

        dbeta_dpx = dy / q
        dbeta_dpy = -dx / q
        dbeta_dth = (dy / q) * dsx_dth + (-dx / q) * dsy_dth - 1.0

        H = np.array([
            [dr_dpx, dr_dpy, dr_dth],
            [dbeta_dpx, dbeta_dpy, dbeta_dth]
        ])

        return z_pred, H
