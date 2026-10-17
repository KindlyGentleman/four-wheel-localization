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


def se2_compose(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Compose two 2D rigid-body transformations in SE(2): T_12 = T_1 * T_2.
    Each transform is represented as [x, y, theta] with theta in radians.

    Args:
        t1: Parent transform [x1, y1, theta1]
        t2: Child transform [x2, y2, theta2]

    Returns:
        Composed transform [x_12, y_12, theta_12] in SE(2).
    """
    x1, y1, th1 = float(t1[0]), float(t1[1]), float(t1[2])
    x2, y2, th2 = float(t2[0]), float(t2[1]), float(t2[2])
    c1, s1 = math.cos(th1), math.sin(th1)
    x12 = x1 + c1 * x2 - s1 * y2
    y12 = y1 + s1 * x2 + c1 * y2
    th12 = wrap_angle(th1 + th2)
    return np.array([x12, y12, th12])


def se2_inverse(t: np.ndarray) -> np.ndarray:
    """
    Compute analytical inverse of an SE(2) rigid-body transform: T^-1.
    Satisfies se2_compose(t, se2_inverse(t)) == [0, 0, 0] within floating tolerance.

    Args:
        t: Transform [x, y, theta]

    Returns:
        Inverse transform [x_inv, y_inv, theta_inv].
    """
    x, y, th = float(t[0]), float(t[1]), float(t[2])
    c, s = math.cos(th), math.sin(th)
    x_inv = -c * x - s * y
    y_inv = s * x - c * y
    th_inv = wrap_angle(-th)
    return np.array([x_inv, y_inv, th_inv])


def compute_map_to_odom_se2(map_pose: np.ndarray, odom_pose: np.ndarray) -> np.ndarray:
    """
    Compute SE(2) transform from map frame to odom frame following ROS REP-105:
      T_map_base = T_map_odom * T_odom_base
      => T_map_odom = T_map_base * (T_odom_base)^-1

    Args:
        map_pose: Global robot pose in map frame [x_m, y_m, theta_m]
        odom_pose: Smooth local robot pose in odom frame [x_o, y_o, theta_o]

    Returns:
        map_to_odom transform [x_mo, y_mo, theta_mo] in SE(2).
    """
    odom_inv = se2_inverse(odom_pose)
    return se2_compose(map_pose, odom_inv)

