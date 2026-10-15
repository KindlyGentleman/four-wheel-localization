"""
robot_localization.models.kinematics

Fundamental kinematic routines: pose representation, coordinate transformations,
angle normalization, and numerical integration methods.
"""

import math
import numpy as np


def wrap_angle(angle: float) -> float:
    """
    Wrap angle in radians to interval [-pi, pi].

    Args:
        angle: Angle in radians.

    Returns:
        Angle wrapped to [-pi, pi].
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    """
    Wrap an array of angles in radians to [-pi, pi].
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Return 2x2 rotation matrix R(theta) mapping body coordinates to world coordinates.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]])


def body_to_world(point_body: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    """
    Transform 2D point from robot body frame to world frame.

    Args:
        point_body: [x_b, y_b] or array of shape (2,)
        robot_pose: [x, y, theta] or array of shape (3,)

    Returns:
        point_world: [x_w, y_w] of shape (2,)
    """
    theta = float(robot_pose[2])
    R = rotation_matrix_2d(theta)
    pos = np.array([robot_pose[0], robot_pose[1]])
    return pos + R @ point_body


def world_to_body(point_world: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    """
    Transform 2D point from world frame to robot body frame.

    Args:
        point_world: [x_w, y_w]
        robot_pose: [x, y, theta]

    Returns:
        point_body: [x_b, y_b]
    """
    theta = float(robot_pose[2])
    R = rotation_matrix_2d(theta)
    pos = np.array([robot_pose[0], robot_pose[1]])
    return R.T @ (point_world - pos)


def integrate_pose_euler(pose: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """
    Forward Euler integration of planar kinematic unicycle model.

    Args:
        pose: [x, y, theta] in meters and radians.
        v: Forward linear speed in m/s.
        omega: Angular velocity in rad/s.
        dt: Time increment in seconds.

    Returns:
        New pose [x_next, y_next, theta_next].
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    x_next = x + v * math.cos(theta) * dt
    y_next = y + v * math.sin(theta) * dt
    theta_next = wrap_angle(theta + omega * dt)
    return np.array([x_next, y_next, theta_next])


def integrate_pose_rk2(pose: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """
    Second-order Runge-Kutta (midpoint) integration for planar mobile robot.
    Significantly reduces truncation error on curved paths compared to Euler.

    Args:
        pose: [x, y, theta]
        v: Forward linear speed in m/s.
        omega: Angular velocity in rad/s.
        dt: Time increment in seconds.

    Returns:
        New pose [x_next, y_next, theta_next].
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    # Midpoint heading
    theta_mid = theta + 0.5 * omega * dt
    x_next = x + v * math.cos(theta_mid) * dt
    y_next = y + v * math.sin(theta_mid) * dt
    theta_next = wrap_angle(theta + omega * dt)
    return np.array([x_next, y_next, theta_next])
