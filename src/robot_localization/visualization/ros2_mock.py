"""
robot_localization.visualization.ros2_mock

In-process ROS 2 architectural simulation mockup and topic event-bus.
Adheres to ROS REP-105 coordinate frame conventions:
  map -> odom -> base_link -> [imu_link, gnss_antenna, lidar_link, camera_link -> camera_left_optical]
Provides topic-style publication, subscription, and TF tree transform lookups.
Works out-of-the-box in standalone Python environments without requiring ROS 2 installation.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ROSHeader:
    seq: int
    stamp: float                     # Timestamp in seconds
    frame_id: str


@dataclass
class TransformStamped:
    header: ROSHeader
    child_frame_id: str
    translation: Tuple[float, float, float]
    rotation_rpy: Tuple[float, float, float] # Roll, pitch, yaw (rad)


class MockTFTree:
    """
    Maintains coordinate frame relationships adhering strictly to ROS REP-105.
    Tree structure:
      map -> odom -> base_link -> [imu_link, gnss_antenna, lidar_link, camera_link -> camera_left_optical]
    """

    def __init__(self):
        # Static transforms relative to parent
        self.static_transforms: Dict[str, TransformStamped] = {}
        # Dynamic transforms updated by estimators
        self.dynamic_transforms: Dict[str, TransformStamped] = {}

        self._setup_default_transforms()

    def _setup_default_transforms(self):
        t0 = 0.0
        # base_link -> imu_link
        self.set_static_transform("base_link", "imu_link", (0.0, 0.0, 0.10), (0.0, 0.0, 0.0), t0)
        # base_link -> gnss_antenna (lever arm)
        self.set_static_transform("base_link", "gnss_antenna", (0.05, 0.0, 0.25), (0.0, 0.0, 0.0), t0)
        # base_link -> lidar_link
        self.set_static_transform("base_link", "lidar_link", (0.15, 0.0, 0.22), (0.0, 0.0, 0.0), t0)
        # base_link -> camera_link
        self.set_static_transform("base_link", "camera_link", (0.20, 0.0, 0.20), (0.0, 0.0, 0.0), t0)
        # camera_link -> camera_left_optical (REP-105 optical rotation)
        # roll = -90 deg (-pi/2), pitch = 0, yaw = -90 deg (-pi/2)
        self.set_static_transform("camera_link", "camera_left_optical", (0.0, 0.0, 0.0), (-math.pi/2, 0.0, -math.pi/2), t0)

    def set_static_transform(self, parent: str, child: str, trans: Tuple[float, float, float], rpy: Tuple[float, float, float], stamp: float):
        tf = TransformStamped(
            header=ROSHeader(seq=0, stamp=stamp, frame_id=parent),
            child_frame_id=child,
            translation=trans,
            rotation_rpy=rpy
        )
        self.static_transforms[f"{parent}->{child}"] = tf

    def update_dynamic_transform(self, parent: str, child: str, trans: Tuple[float, float, float], rpy: Tuple[float, float, float], stamp: float):
        tf = TransformStamped(
            header=ROSHeader(seq=0, stamp=stamp, frame_id=parent),
            child_frame_id=child,
            translation=trans,
            rotation_rpy=rpy
        )
        self.dynamic_transforms[f"{parent}->{child}"] = tf

    def get_transform(self, parent: str, child: str) -> Optional[TransformStamped]:
        key = f"{parent}->{child}"
        if key in self.dynamic_transforms:
            return self.dynamic_transforms[key]
        if key in self.static_transforms:
            return self.static_transforms[key]
        return None

    def get_frame_hierarchy(self) -> Dict[str, Any]:
        """Return human-readable representation of TF hierarchy."""
        return {
            "root": "map",
            "children": {
                "odom": {
                    "parent": "map",
                    "type": "dynamic",
                    "description": "Continuous local odometry drift correction"
                },
                "base_link": {
                    "parent": "odom",
                    "type": "dynamic",
                    "description": "Robot kinematic center of rotation"
                },
                "sensors": {
                    "imu_link": {"parent": "base_link", "type": "static", "offset": [0.0, 0.0, 0.10]},
                    "gnss_antenna": {"parent": "base_link", "type": "static", "offset": [0.05, 0.0, 0.25]},
                    "lidar_link": {"parent": "base_link", "type": "static", "offset": [0.15, 0.0, 0.22]},
                    "camera_link": {"parent": "base_link", "type": "static", "offset": [0.20, 0.0, 0.20]},
                    "camera_left_optical": {"parent": "camera_link", "type": "static", "rep105": "Z-forward X-right Y-down"}
                }
            }
        }


@dataclass
class TopicStats:
    topic_name: str
    message_count: int = 0
    last_publish_time: float = 0.0
    hz: float = 0.0
    subscribers_count: int = 0


class ROS2MockBus:
    """
    Lightweight, thread-safe in-process publish/subscribe message bus.
    Supports standard ROS 2 topic names and topic rate statistics.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self._stats: Dict[str, TopicStats] = {}
        self.tf_tree = MockTFTree()

        # Pre-register standard ROS 2 robotics topic names
        self.standard_topics = [
            "/clock",
            "/robot/wheel/encoders",
            "/imu/data",
            "/compass/heading",
            "/gps/fix",
            "/stereo/left/image",
            "/stereo/right/image",
            "/stereo/disparity",
            "/stereo/points",
            "/scan",
            "/odom/wheel",
            "/odom/visual",
            "/odometry/local",
            "/odometry/global",
            "/localization/pose",
            "/localization/health",
            "/tf",
            "/tf_static"
        ]
        for t in self.standard_topics:
            self._stats[t] = TopicStats(topic_name=t)

    def publish(self, topic: str, msg: Any, timestamp: Optional[float] = None):
        """Publish a message onto the mock bus."""
        t_now = timestamp if timestamp is not None else time.time()
        if topic not in self._stats:
            self._stats[topic] = TopicStats(topic_name=topic)

        stat = self._stats[topic]
        stat.message_count += 1
        if stat.last_publish_time > 0:
            dt = t_now - stat.last_publish_time
            if dt > 1e-5:
                stat.hz = 0.9 * stat.hz + 0.1 * (1.0 / dt) if stat.hz > 0 else (1.0 / dt)
        stat.last_publish_time = t_now

        # Route to subscribers
        if topic in self._subscribers:
            for cb in self._subscribers[topic]:
                try:
                    cb(msg)
                except Exception as e:
                    pass

    def subscribe(self, topic: str, callback: Callable[[Any], None]):
        """Register a callback for a specific topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
        if topic not in self._stats:
            self._stats[topic] = TopicStats(topic_name=topic)
        self._stats[topic].subscribers_count = len(self._subscribers[topic])

    def get_topic_summary(self) -> List[Dict[str, Any]]:
        """Return active topics, publication rates, and message counts."""
        summary = []
        for t_name, stat in sorted(self._stats.items()):
            if stat.message_count > 0:
                summary.append({
                    "topic": t_name,
                    "count": stat.message_count,
                    "rate_hz": round(stat.hz, 1),
                    "last_timestamp_s": round(stat.last_publish_time, 3),
                    "subscribers": stat.subscribers_count
                })
        return summary
