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
