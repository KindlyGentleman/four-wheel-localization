"""
tests.test_ros2_mock

Unit tests for ROS 2 mock event bus and REP-105 TF tree.
"""

import pytest
import math

from robot_localization.visualization.ros2_mock import ROS2MockBus, MockTFTree


def test_tf_tree_rep105_hierarchy():
    tree = MockTFTree()
    # Check static sensor frames
    tf_camera = tree.get_transform("base_link", "camera_link")
    assert tf_camera is not None
    assert tf_camera.translation == (0.20, 0.0, 0.20)

    tf_optical = tree.get_transform("camera_link", "camera_left_optical")
    assert tf_optical is not None
    # REP-105 optical rotation
    assert math.isclose(tf_optical.rotation_rpy[0], -math.pi/2, abs_tol=1e-4)

    tf_gnss = tree.get_transform("base_link", "gnss_antenna")
    assert tf_gnss is not None
    assert tf_gnss.translation[0] == 0.05


def test_ros2_mock_bus_publish_subscribe():
    bus = ROS2MockBus()
    received = []

    def on_odom(msg):
        received.append(msg)

    bus.subscribe("/odometry/local", on_odom)

    # Publish messages
    bus.publish("/odometry/local", {"pose": [1.0, 2.0, 0.1]}, timestamp=1.0)
    bus.publish("/odometry/local", {"pose": [1.1, 2.0, 0.1]}, timestamp=1.02)

    assert len(received) == 2
    summary = bus.get_topic_summary()
    odom_stat = next(s for s in summary if s["topic"] == "/odometry/local")
    assert odom_stat["count"] == 2
    assert odom_stat["rate_hz"] > 0


def test_mock_tf_lookup_multi_hop_composition():
    """Verify that MockTFTree composes multi-hop transformations along the REP-105 hierarchy."""
    import numpy as np
    tree = MockTFTree()

    # Dynamic transform: map -> odom
    # map -> odom: translation = (10.0, 5.0, 0.0), yaw = 0.5 rad (~28.6 deg)
    tree.update_dynamic_transform("map", "odom", (10.0, 5.0, 0.0), (0.0, 0.0, 0.5), stamp=1.0)

    # Dynamic transform: odom -> base_link
    # odom -> base_link: translation = (2.0, 0.0, 0.0), yaw = 0.1 rad
    tree.update_dynamic_transform("odom", "base_link", (2.0, 0.0, 0.0), (0.0, 0.0, 0.1), stamp=1.0)

    # 1. Lookup single hop: odom -> base_link
    tf_single = tree.lookup_transform("odom", "base_link")
    assert tf_single is not None
    assert math.isclose(tf_single.translation[0], 2.0, abs_tol=1e-5)
    assert math.isclose(tf_single.rotation_rpy[2], 0.1, abs_tol=1e-5)

    # 2. Lookup multi-hop: map -> base_link
    tf_composed = tree.lookup_transform("map", "base_link")
    assert tf_composed is not None
    # Expected translation: [10, 5] + R(0.5) @ [2, 0] = [10 + 2*cos(0.5), 5 + 2*sin(0.5)]
    expected_x = 10.0 + 2.0 * math.cos(0.5)
    expected_y = 5.0 + 2.0 * math.sin(0.5)
    expected_yaw = 0.5 + 0.1
    assert math.isclose(tf_composed.translation[0], expected_x, abs_tol=1e-5)
    assert math.isclose(tf_composed.translation[1], expected_y, abs_tol=1e-5)
    assert math.isclose(tf_composed.rotation_rpy[2], expected_yaw, abs_tol=1e-5)

    # 3. Lookup inverse: base_link -> map
    tf_inv = tree.lookup_transform("base_link", "map")
    assert tf_inv is not None
    # Composing tf_composed and tf_inv must yield identity
    from robot_localization.visualization.ros2_mock import _tf_to_mat4
    T1 = _tf_to_mat4(tf_composed)
    T2 = _tf_to_mat4(tf_inv)
    np.testing.assert_allclose(T1 @ T2, np.eye(4), atol=1e-6)

    # 4. Multi-hop sensor lookup: map -> lidar_link (map -> odom -> base_link -> lidar_link)
    tf_lidar = tree.lookup_transform("map", "lidar_link")
    assert tf_lidar is not None
    assert tf_lidar.translation[2] > 0.20 # height of lidar


def test_se2_kinematics_composition_and_rep105():
    """Verify SE(2) composition, inversion, and map_to_odom calculation functions in kinematics."""
    import numpy as np
    from robot_localization.models.kinematics import (
        se2_compose,
        se2_inverse,
        compute_map_to_odom_se2,
        wrap_angle
    )

    # Case 1: Simple translation and heading
    t1 = np.array([5.0, 3.0, 0.5])
    t2 = np.array([2.0, -1.0, -0.2])
    t12 = se2_compose(t1, t2)
    # Inverse composition
    t2_inv = se2_inverse(t2)
    t1_recovered = se2_compose(t12, t2_inv)
    np.testing.assert_allclose(t1_recovered, t1, atol=1e-12)

    # Case 2: REP-105 Identity: T_map_base = se2_compose(T_map_odom, T_odom_base)
    odom_pose = np.array([12.5, -4.2, 1.1])
    map_pose = np.array([14.8, -3.1, 1.3])

    map_to_odom = compute_map_to_odom_se2(map_pose, odom_pose)
    map_pose_reconstructed = se2_compose(map_to_odom, odom_pose)
    np.testing.assert_allclose(map_pose_reconstructed, map_pose, atol=1e-12)

