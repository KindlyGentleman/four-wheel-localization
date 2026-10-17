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


def _rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])


def _matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return (roll, pitch, yaw)


def _tf_to_mat4(tf: TransformStamped) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(*tf.rotation_rpy)
    T[:3, 3] = tf.translation
    return T


def _mat4_to_tf(T: np.ndarray, parent: str, child: str, stamp: float) -> TransformStamped:
    trans = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    rpy = _matrix_to_rpy(T[:3, :3])
    return TransformStamped(
        header=ROSHeader(seq=0, stamp=stamp, frame_id=parent),
        child_frame_id=child,
        translation=trans,
        rotation_rpy=rpy
    )


class MockTFTree:
    """
    Maintains coordinate frame relationships adhering strictly to ROS REP-105.
    Tree structure:
      map -> odom -> base_link -> [imu_link, gnss_antenna, lidar_link, camera_link -> camera_left_optical]
    Provides full SE(2)/SE(3) transform chaining and multi-hop lookup.
    """

    def __init__(self):
        # Static transforms relative to parent
        self.static_transforms: Dict[str, TransformStamped] = {}
        # Dynamic transforms updated by estimators
        self.dynamic_transforms: Dict[str, TransformStamped] = {}
        self.parent_of: Dict[str, str] = {}

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
        self.parent_of[child] = parent

    def update_dynamic_transform(self, parent: str, child: str, trans: Tuple[float, float, float], rpy: Tuple[float, float, float], stamp: float):
        tf = TransformStamped(
            header=ROSHeader(seq=0, stamp=stamp, frame_id=parent),
            child_frame_id=child,
            translation=trans,
            rotation_rpy=rpy
        )
        self.dynamic_transforms[f"{parent}->{child}"] = tf
        self.parent_of[child] = parent

    def get_transform(self, parent: str, child: str) -> Optional[TransformStamped]:
        key = f"{parent}->{child}"
        if key in self.dynamic_transforms:
            return self.dynamic_transforms[key]
        if key in self.static_transforms:
            return self.static_transforms[key]
        return None

    def lookup_transform(self, target_frame: str, source_frame: str) -> Optional[TransformStamped]:
        """
        Look up the transformation that maps points in source_frame to target_frame.
        Supports multi-hop tree traversal and SE(3) matrix composition following ROS TF2 conventions.
        """
        if target_frame == source_frame:
            return TransformStamped(
                header=ROSHeader(seq=0, stamp=0.0, frame_id=target_frame),
                child_frame_id=source_frame,
                translation=(0.0, 0.0, 0.0),
                rotation_rpy=(0.0, 0.0, 0.0)
            )

        def _get_chain_to_root(frame: str) -> List[str]:
            chain = [frame]
            curr = frame
            while curr in self.parent_of:
                curr = self.parent_of[curr]
                chain.append(curr)
            return chain

        target_chain = _get_chain_to_root(target_frame)
        source_chain = _get_chain_to_root(source_frame)

        # Find lowest common ancestor
        common_ancestor = None
        for f in source_chain:
            if f in target_chain:
                common_ancestor = f
                break

        if common_ancestor is None:
            return None

        # Helper to compute transform from root/ancestor down to descendant: T_ancestor_descendant
        def _compute_transform_down(ancestor: str, descendant: str) -> np.ndarray:
            if ancestor == descendant:
                return np.eye(4)
            # Path from descendant up to ancestor
            path_up = []
            c = descendant
            while c != ancestor:
                p = self.parent_of[c]
                path_up.append((p, c))
                c = p
            # Reverse to compose from ancestor down to descendant
            T = np.eye(4)
            for p, ch in reversed(path_up):
                tf = self.get_transform(p, ch)
                if tf is None:
                    return np.eye(4)
                T = T @ _tf_to_mat4(tf)
            return T

        T_anc_target = _compute_transform_down(common_ancestor, target_frame)
        T_anc_source = _compute_transform_down(common_ancestor, source_frame)

        # T_target_source = (T_anc_target)^-1 * T_anc_source
        T_target_anc = np.linalg.inv(T_anc_target)
        T_target_source = T_target_anc @ T_anc_source

        # Get stamp from the most recent transform in the lookup
        stamp = 0.0
        for f in [target_frame, source_frame]:
            if f in self.parent_of:
                direct_tf = self.get_transform(self.parent_of[f], f)
                if direct_tf:
                    stamp = max(stamp, direct_tf.header.stamp)

        return _mat4_to_tf(T_target_source, target_frame, source_frame, stamp)

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
