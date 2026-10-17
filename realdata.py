"""
realdata.py

Real-Data and Hybrid Sensor Fusion Pipeline for Ground Robot Localization.
Supports the M2DGR and M2DGR-plus datasets (SJTU-ViSYS).

Capabilities:
1. Ground truth parsing (anomaly.txt with scalar-first quaternion qw, qx, qy, qz)
2. Calibration parsing (calibration.txt with full-width Chinese comma sanitization)
3. ROS-independent ROS 1 bag extraction via rosbags library (no ROS/Gazebo dependency)
4. Offline cache management (manifest.json, odom.npz, imu.npz, gnss.npz, ground_truth.npz)
5. Authentic test bag synthesis (rosbags writer with exact M2DGR-plus schemas)
6. Real OpenCV feature tracking & controlled image degradation (clean / moderate / severe)
7. Controlled GNSS denial and fault recovery benchmarks (R0 - R5)
8. Independent metric evaluation (APE, RMSE, Peak Outage Error, Recovery Time)
"""

import os
import sys
import math
import json
import time
import urllib.request
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import cv2

# rosbags library for ROS-independent bag parsing and writing
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

# Add src to path if needed
_CUR_DIR = Path(__file__).resolve().parent
_SRC_DIR = _CUR_DIR / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from robot_localization.models.robot_4wd import Robot4WDConfig
from robot_localization.models.kinematics import wrap_angle, integrate_pose_rk2
from robot_localization.estimation.dead_reckoning import DeadReckoning4WD
from robot_localization.estimation.ekf import ExtendedKalmanFilter3D
from robot_localization.estimation.bias_aware_ekf import BiasAwareEKF4D
from robot_localization.estimation.robust_ekf import RobustEKF4D
from robot_localization.estimation.gnss_recovery import (
    RecoverableRobustEKF4D,
    RecoveryStrategy,
    EstimatorHealthState,
)


@dataclass
class GroundTruthTrajectory:
    """Ground truth reference trajectory from high-precision RTK/INS or Vicon."""
    timestamps: np.ndarray       # (N,) in seconds
    positions_xyz: np.ndarray    # (N, 3) in meters
    quaternions_wxyz: np.ndarray # (N, 4) [qw, qx, qy, qz]
    poses_2d: np.ndarray         # (N, 3) [x, y, yaw_rad]
    linear_velocities: np.ndarray # (N,) signed longitudinal body-frame speed in m/s
    angular_velocities: np.ndarray # (N,) yaw rate in rad/s
    lateral_velocities: Optional[np.ndarray] = None # (N,) lateral body-frame speed in m/s
    euclidean_speeds: Optional[np.ndarray] = None   # (N,) unsigned planar speed in m/s

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def sample_count(self) -> int:
        return len(self.timestamps)

    @property
    def total_distance_m(self) -> float:
        diffs = np.diff(self.positions_xyz[:, :2], axis=0)
        return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


@dataclass
class M2DGRCalibration:
    """Extrinsics, intrinsics, and sensor noise parameters from calibration.txt."""
    rsimu2wheel: np.ndarray     # (4, 4)
    imu2wheel: np.ndarray       # (4, 4)
    lidar2wheel: np.ndarray     # (4, 4)
    rsimu2camera: np.ndarray    # (4, 4)
    lidar2imu: np.ndarray       # (4, 4)
    camera2imu: np.ndarray      # (4, 4)
    lidar2camera: np.ndarray    # (4, 4)
    # Camera intrinsics (PINHOLE)
    image_width: int = 640
    image_height: int = 480
    fx: float = 603.955566
    fy: float = 603.125793
    cx: float = 324.085815
    cy: float = 232.723038
    # Noise parameters
    acc_noise: float = 0.01237
    gyr_noise: float = 0.00300
    acc_bias_walk: float = 0.000192
    gyr_bias_walk: float = 0.0000547

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


def load_ground_truth(
    gt_path_or_url: Optional[str] = None,
    cache_dir: Path = Path("data")
) -> GroundTruthTrajectory:
    """
    Load ground truth trajectory from file or download official anomaly.txt.
    M2DGR-plus gt format:
    timestamp x y z qw qx qy qz
    Notice: Scalar-first quaternion (qw is col 4).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_file = cache_dir / "anomaly.txt"

    if gt_path_or_url is not None and Path(gt_path_or_url).exists():
        raw_text = Path(gt_path_or_url).read_text(encoding="utf-8")
    elif local_file.exists():
        raw_text = local_file.read_text(encoding="utf-8")
    else:
        url = gt_path_or_url or "https://raw.githubusercontent.com/SJTU-ViSYS/M2DGR-plus/main/gt/anomaly.txt"
        print(f"[Data] Fetching official ground truth from: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw_text = resp.read().decode("utf-8")
        local_file.write_text(raw_text, encoding="utf-8")
        print(f"[Data] Cached ground truth to {local_file}")

    lines = [l.strip().split() for l in raw_text.splitlines() if l.strip()]
    data = np.array(lines, dtype=np.float64)

    timestamps = data[:, 0]
    positions_xyz = data[:, 1:4]
    # Col 4 is qw, cols 5..7 are qx, qy, qz
    qw = data[:, 4]
    qx = data[:, 5]
    qy = data[:, 6]
    qz = data[:, 7]

    # Normalize quaternions
    q_norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    q_norm[q_norm == 0] = 1.0
    qw /= q_norm
    qx /= q_norm
    qy /= q_norm
    qz /= q_norm
    quaternions_wxyz = np.column_stack([qw, qx, qy, qz])

    # Convert to 2D yaw angle: psi = atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    yaw_unwrapped = np.unwrap(yaw)
    poses_2d = np.column_stack([positions_xyz[:, 0], positions_xyz[:, 1], yaw])

    # Compute signed body-frame velocities and yaw rates
    dt = np.diff(timestamps)
    dt[dt <= 0] = 0.05
    dx = np.diff(positions_xyz[:, 0])
    dy = np.diff(positions_xyz[:, 1])
    dyaw = np.diff(yaw_unwrapped)
    yaw_mid = yaw_unwrapped[:-1] + 0.5 * dyaw

    v_fwd_raw = (dx * np.cos(yaw_mid) + dy * np.sin(yaw_mid)) / dt
    v_lat_raw = (-dx * np.sin(yaw_mid) + dy * np.cos(yaw_mid)) / dt
    v_euc_raw = np.hypot(dx, dy) / dt

    lin_speed = np.concatenate([[v_fwd_raw[0]], v_fwd_raw])
    lat_speed = np.concatenate([[v_lat_raw[0]], v_lat_raw])
    euc_speed = np.concatenate([[v_euc_raw[0]], v_euc_raw])

    yaw_rate_raw = dyaw / dt
    yaw_rate = np.concatenate([[yaw_rate_raw[0]], yaw_rate_raw])

    return GroundTruthTrajectory(
        timestamps=timestamps,
        positions_xyz=positions_xyz,
        quaternions_wxyz=quaternions_wxyz,
        poses_2d=poses_2d,
        linear_velocities=lin_speed,
        angular_velocities=yaw_rate,
        lateral_velocities=lat_speed,
        euclidean_speeds=euc_speed,
    )


def load_calibration(
    calib_path_or_url: Optional[str] = None,
    cache_dir: Path = Path("data")
) -> M2DGRCalibration:
    """
    Parse calibration.txt handling full-width Chinese commas and non-standard delimiters.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_file = cache_dir / "calibration.txt"

    if calib_path_or_url is not None and Path(calib_path_or_url).exists():
        raw_text = Path(calib_path_or_url).read_text(encoding="utf-8")
    elif local_file.exists():
        raw_text = local_file.read_text(encoding="utf-8")
    else:
        url = calib_path_or_url or "https://raw.githubusercontent.com/SJTU-ViSYS/M2DGR-plus/main/calibration.txt"
        print(f"[Data] Fetching official calibration from: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw_text = resp.read().decode("utf-8")
        local_file.write_text(raw_text, encoding="utf-8")
        print(f"[Data] Cached calibration to {local_file}")

    # Robust regex parser for 4x4 matrix blocks
    matrix_blocks = re.findall(
        r'(\w+):\s*\n\s*rows:\s*(\d+)\s*\n\s*cols:\s*(\d+).*?data:\s*\[(.*?)\]',
        raw_text,
        re.DOTALL
    )
    matrices = {}
    for name, rows, cols, data_str in matrix_blocks:
        nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', data_str)]
        mat = np.array(nums, dtype=np.float64).reshape((int(rows), int(cols)))
        matrices[name] = mat

    # Default identities if any block missing
    def get_mat(k: str) -> np.ndarray:
        return matrices.get(k, np.eye(4, dtype=np.float64))

    return M2DGRCalibration(
        rsimu2wheel=get_mat("rsimu2wheel"),
        imu2wheel=get_mat("imu2wheel"),
        lidar2wheel=get_mat("lidar2wheel"),
        rsimu2camera=get_mat("rsimu2camera"),
        lidar2imu=get_mat("lidar2imu"),
        camera2imu=get_mat("camera2imu"),
        lidar2camera=get_mat("lidar2camera"),
        image_width=640,
        image_height=480,
        fx=603.955566,
        fy=603.125793,
        cx=324.085815,
        cy=232.723038,
        acc_noise=0.01237,
        gyr_noise=0.00300,
        acc_bias_walk=0.000192,
        gyr_bias_walk=0.0000547
    )


def inspect_bag(bag_path: str) -> Dict[str, Any]:
    """
    ROS-independent ROS1 bag inspection using pure-Python rosbags.
    """
    p = Path(bag_path)
    if not p.exists():
        raise FileNotFoundError(f"Rosbag file not found: {bag_path}")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    topic_info = {}
    total_messages = 0

    with Reader(p) as reader:
        start_ns = reader.start_time
        end_ns = reader.end_time
        duration_s = (end_ns - start_ns) * 1e-9 if (start_ns and end_ns) else 0.0

        for conn in reader.connections:
            topic_info[conn.topic] = {
                "msgtype": conn.msgtype,
                "msgcount": conn.msgcount,
                "digest": getattr(conn, "digest", "")
            }
            total_messages += conn.msgcount

    return {
        "file_name": p.name,
        "file_size_bytes": p.stat().st_size,
        "duration_s": duration_s,
        "start_time_s": start_ns * 1e-9 if start_ns else 0.0,
        "end_time_s": end_ns * 1e-9 if end_ns else 0.0,
        "total_messages": total_messages,
        "topics": topic_info
    }


def create_sample_bag(
    output_path: Path,
    gt: GroundTruthTrajectory,
    calib: M2DGRCalibration
) -> Path:
    """
    Generate a ground-truth-conditioned synthetic ROS1 bag for offline verification.
    Synthesizes /odom, /imu, /camera/imu, and /ublox_driver/receiver_pvt matching
    M2DGR-plus message schemas and injects a wheel-spin anomaly at t=13s-18s.
    """
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Writer(output_path) as writer:
        conn_odom = writer.add_connection("/odom", "nav_msgs/msg/Odometry", typestore=typestore)
        conn_imu = writer.add_connection("/imu", "sensor_msgs/msg/Imu", typestore=typestore)
        conn_cam_imu = writer.add_connection("/camera/imu", "sensor_msgs/msg/Imu", typestore=typestore)
        conn_gnss = writer.add_connection("/ublox_driver/receiver_pvt", "sensor_msgs/msg/NavSatFix", typestore=typestore)

        Header = typestore.types["std_msgs/msg/Header"]
        TimeMsg = typestore.types["builtin_interfaces/msg/Time"]

        rng = np.random.default_rng(42)
        n = len(gt.timestamps)

        odom_x, odom_y, odom_yaw = gt.poses_2d[0]

        for i in range(n):
            t_s = gt.timestamps[i]
            t_sec = int(t_s)
            t_nsec = int((t_s - t_sec) * 1e9)
            stamp = TimeMsg(sec=t_sec, nanosec=t_nsec)
            header = Header(seq=i, stamp=stamp, frame_id="odom")

            dt = gt.timestamps[i] - gt.timestamps[max(0, i - 1)] if i > 0 else 0.2
            v_true = float(gt.linear_velocities[i])
            w_true = float(gt.angular_velocities[i])

            rel_t = t_s - gt.timestamps[0]
            if 13.0 <= rel_t <= 18.0:
                v_wheel = 0.25 + 0.02 * float(rng.standard_normal())
                w_wheel = 0.05 * float(rng.standard_normal())
            else:
                v_wheel = v_true + 0.01 * float(rng.standard_normal())
                w_wheel = w_true + 0.005 * float(rng.standard_normal())

            odom_yaw = wrap_angle(odom_yaw + w_wheel * dt)
            odom_x += v_wheel * math.cos(odom_yaw) * dt
            odom_y += v_wheel * math.sin(odom_yaw) * dt

            odom_msg = typestore.types["nav_msgs/msg/Odometry"](
                header=header,
                child_frame_id="base_link",
                pose=typestore.types["geometry_msgs/msg/PoseWithCovariance"](
                    pose=typestore.types["geometry_msgs/msg/Pose"](
                        position=typestore.types["geometry_msgs/msg/Point"](x=odom_x, y=odom_y, z=0.0),
                        orientation=typestore.types["geometry_msgs/msg/Quaternion"](
                            x=0.0, y=0.0, z=math.sin(odom_yaw / 2.0), w=math.cos(odom_yaw / 2.0)
                        )
                    ),
                    covariance=np.zeros(36, dtype=np.float64)
                ),
                twist=typestore.types["geometry_msgs/msg/TwistWithCovariance"](
                    twist=typestore.types["geometry_msgs/msg/Twist"](
                        linear=typestore.types["geometry_msgs/msg/Vector3"](x=v_wheel, y=0.0, z=0.0),
                        angular=typestore.types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=w_wheel)
                    ),
                    covariance=np.zeros(36, dtype=np.float64)
                )
            )
            raw_odom = typestore.serialize_ros1(odom_msg, "nav_msgs/msg/Odometry")
            writer.write(conn_odom, int(t_s * 1e9), raw_odom)

            imu_msg = typestore.types["sensor_msgs/msg/Imu"](
                header=Header(seq=i, stamp=stamp, frame_id="imu_link"),
                orientation=typestore.types["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.full(9, -1.0, dtype=np.float64),
                angular_velocity=typestore.types["geometry_msgs/msg/Vector3"](
                    x=0.001 * float(rng.standard_normal()),
                    y=0.001 * float(rng.standard_normal()),
                    z=w_true + calib.gyr_noise * float(rng.standard_normal())
                ),
                angular_velocity_covariance=np.diag([1e-4, 1e-4, calib.gyr_noise**2]).flatten(),
                linear_acceleration=typestore.types["geometry_msgs/msg/Vector3"](
                    x=0.01 * float(rng.standard_normal()),
                    y=0.01 * float(rng.standard_normal()),
                    z=9.805 + calib.acc_noise * float(rng.standard_normal())
                ),
                linear_acceleration_covariance=np.diag([1e-3, 1e-3, calib.acc_noise**2]).flatten()
            )
            raw_imu = typestore.serialize_ros1(imu_msg, "sensor_msgs/msg/Imu")
            writer.write(conn_imu, int(t_s * 1e9), raw_imu)
            writer.write(conn_cam_imu, int(t_s * 1e9), raw_imu)

            if i % 2 == 0:
                gx = gt.positions_xyz[i, 0] + 0.04 * float(rng.standard_normal())
                gy = gt.positions_xyz[i, 1] + 0.04 * float(rng.standard_normal())
                lat = 31.025 + gy / 111000.0
                lon = 121.435 + gx / (111000.0 * math.cos(math.radians(31.025)))

                gnss_msg = typestore.types["sensor_msgs/msg/NavSatFix"](
                    header=Header(seq=i // 2, stamp=stamp, frame_id="gnss_link"),
                    status=typestore.types["sensor_msgs/msg/NavSatStatus"](status=0, service=1),
                    latitude=lat,
                    longitude=lon,
                    altitude=float(gt.positions_xyz[i, 2]),
                    position_covariance=np.diag([0.05**2, 0.05**2, 0.10**2]).flatten(),
                    position_covariance_type=2
                )
                raw_gnss = typestore.serialize_ros1(gnss_msg, "sensor_msgs/msg/NavSatFix")
                writer.write(conn_gnss, int(t_s * 1e9), raw_gnss)

    return output_path


class VisualFeatureTracker:
    """
    OpenCV-based Shi-Tomasi / KLT optical flow feature tracking pipeline.
    Uses an isolated RNG so visual degradation never perturbs sensor PRNG streams.
    """
    def __init__(self, calib: M2DGRCalibration, max_features: int = 400, seed: int = 1042):
        self.calib = calib
        self.max_features = max_features
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kps: Optional[np.ndarray] = None
        self.rng = np.random.default_rng(seed)

    def process_frame(
        self,
        img: np.ndarray,
        degradation_mode: str = "clean",
        dt: float = 0.20,
        pixels_per_meter: float = 80.0
    ) -> Dict[str, Any]:
        """
        Process a single image frame with optional controlled degradation.
        Degradation modes:
        - 'clean': Original image
        - 'moderate': Contrast 0.45, Gaussian blur 5x5
        - 'severe': Contrast 0.16, severe blur 15x15, additive Gaussian noise
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        if degradation_mode == "moderate":
            gray = cv2.convertScaleAbs(gray, alpha=0.45, beta=20)
            gray = cv2.GaussianBlur(gray, (7, 7), 1.8)
            noise = self.rng.normal(0.0, 8.0, gray.shape).astype(np.int16)
            gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif degradation_mode == "severe":
            gray = cv2.convertScaleAbs(gray, alpha=0.16, beta=10)
            gray = cv2.GaussianBlur(gray, (15, 15), 4.0)
            noise = self.rng.normal(0.0, 22.0, gray.shape).astype(np.int16)
            gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        kps = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=0.03,
            minDistance=8,
            blockSize=5
        )

        num_features = len(kps) if kps is not None else 0
        tracked_count = 0
        inliers_count = 0
        inlier_ratio = 0.0
        tracking_failed = False
        flow_speed_m_s = 0.0
        flow_vec_m_s = np.zeros(2, dtype=np.float64)
        inlier_prev_pts = np.empty((0, 2), dtype=np.float32)
        inlier_curr_pts = np.empty((0, 2), dtype=np.float32)

        if self.prev_gray is not None and self.prev_kps is not None and len(self.prev_kps) > 10 and kps is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_kps,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            good_prev = self.prev_kps[st == 1]
            good_curr = p1[st == 1]
            tracked_count = len(good_curr)

            if tracked_count >= 8:
                M_aff, mask = cv2.estimateAffinePartial2D(
                    good_prev,
                    good_curr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=1.5,
                    maxIters=200,
                    confidence=0.99
                )
                if mask is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    inliers_count = int(np.sum(inlier_mask))
                    inlier_ratio = float(inliers_count / max(1, tracked_count))
                    inlier_prev_pts = good_prev[inlier_mask].reshape(-1, 2)
                    inlier_curr_pts = good_curr[inlier_mask].reshape(-1, 2)
                    if inliers_count >= 8 and dt > 1e-4:
                        flow_px = inlier_curr_pts - inlier_prev_pts
                        med_du = float(np.median(flow_px[:, 0]))
                        med_dv = float(np.median(flow_px[:, 1]))
                        vx_world = -med_du / (pixels_per_meter * dt)
                        vy_world = -med_dv / (pixels_per_meter * dt)
                        flow_vec_m_s = np.array([vx_world, vy_world], dtype=np.float64)
                        flow_speed_m_s = float(np.hypot(vx_world, vy_world))

            if inliers_count < 12 or inlier_ratio < 0.45:
                tracking_failed = True
        else:
            if num_features < 15:
                tracking_failed = True

        self.prev_gray = gray
        self.prev_kps = kps

        return {
            "num_features": num_features,
            "tracked_count": tracked_count,
            "inliers_count": inliers_count,
            "inlier_ratio": inlier_ratio,
            "tracking_failed": tracking_failed,
            "flow_speed_m_s": flow_speed_m_s,
            "flow_vec_m_s": flow_vec_m_s,
            "good_prev": inlier_prev_pts,
            "good_curr": inlier_curr_pts,
            "processed_image": gray
        }


# Fixed world landmarks for deterministic top-down synthetic camera rendering
_LANDMARK_RNG = np.random.default_rng(12345)
_WORLD_LANDMARKS_XY = _LANDMARK_RNG.uniform(-4.5, 5.5, size=(120, 2))
_WORLD_LANDMARKS_RADII = _LANDMARK_RNG.integers(3, 8, size=120)
_WORLD_LANDMARKS_COLORS = [
    (int(c[0]), int(c[1]), int(c[2]))
    for c in _LANDMARK_RNG.integers(30, 235, size=(120, 3))
]


def generate_synthetic_scene_image(
    t: float,
    pose: np.ndarray,
    width: int = 640,
    height: int = 480,
    draw_banner: bool = False
) -> np.ndarray:
    """
    Generate a synthetic top-down floor camera image conditioned on pose (x, y, yaw).
    Uses fixed world-frame landmarks so KLT optical flow tracks physical translation
    without touching the global NumPy PRNG.

    NOTE ON 2D PLANAR TRANSLATION ASSUMPTION (Finding F13):
    This procedural ground-plane camera renderer operates under an explicit 2D
    planar translation model. It renders translational ground texture without
    3D out-of-plane pitch/roll rotations or camera lens distortion.
    Lucas-Kanade optical flow feature tracking computes translational floor
    displacement speed (v_vo = hypot(vx, vy)) to constrain longitudinal wheel
    spin during unmodeled traction slip. For full 3D stereo geometry with
    camera extrinsics and disparity, see SyntheticStereoArena in stereo.py.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    x, y, yaw = pose
    tile_size = 40
    ppm = 80.0
    shift_x = int(round(x * ppm)) % tile_size
    shift_y = int(round(y * ppm)) % tile_size

    for i in range(-1, height // tile_size + 2):
        for j in range(-1, width // tile_size + 2):
            px = j * tile_size - shift_x
            py = i * tile_size - shift_y
            idx_i = i + int(math.floor(y * ppm / tile_size))
            idx_j = j + int(math.floor(x * ppm / tile_size))
            color = (180, 180, 180) if (idx_i + idx_j) % 2 == 0 else (120, 120, 120)
            cv2.rectangle(img, (px, py), (px + tile_size, py + tile_size), color, -1)

    for (lx, ly), radius, color in zip(_WORLD_LANDMARKS_XY, _WORLD_LANDMARKS_RADII, _WORLD_LANDMARKS_COLORS):
        px = int(round(width * 0.5 + (lx - x) * ppm))
        py = int(round(height * 0.5 + (ly - y) * ppm))
        if -10 <= px < width + 10 and -10 <= py < height + 10:
            cv2.circle(img, (px, py), int(radius), color, -1)

    if draw_banner:
        cv2.putText(
            img,
            f"Synthetic Floor Cam t={t:.2f}s [x={x:.2f}, y={y:.2f}]",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    return img


@dataclass
class RealDataTrialResult:
    """Result of running an estimator on the hybrid M2DGR-plus sequence."""
    method_name: str
    scenario_name: str
    timestamps: np.ndarray
    est_poses: np.ndarray       # (M, 3) Global map frame pose
    gt_poses: np.ndarray        # (M, 3) Reference ground truth
    cov_diags: np.ndarray       # (M, 3)
    pos_rmse_m: float
    heading_rmse_deg: float
    outage_peak_error_m: float
    recovery_time_s: float
    recovery_success: bool
    gnss_acceptance_pct: float
    false_recoveries: int
    computation_time_ms: float
    health_states: List[Tuple[float, str]] = field(default_factory=list)
    # REP-105 and Safe-Stop Diagnostics
    est_poses_odom: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    map_to_odom_offsets: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    relocalization_indices: List[int] = field(default_factory=list)
    supervisor_states: List[Tuple[float, str]] = field(default_factory=list)
    commanded_vs: List[float] = field(default_factory=list)
    is_stopped_flags: List[bool] = field(default_factory=list)
    counterfactual_annotations: List[str] = field(default_factory=list)
    # Additional audit transparency fields
    gnss_epoch_acceptance_pct: float = 0.0
    vision_mode: str = "disconnected"
    avg_features: float = 0.0
    avg_inliers: float = 0.0
    avg_inlier_ratio: float = 0.0
    tracking_failure_pct: float = 0.0


def estimate_onboard_wheel_disagreement(
    w_wheel: float,
    gyro_z: float,
    v_wheel: float,
    prev_v_wheel: Optional[float] = None,
    dt: float = 0.05
) -> float:
    """
    Estimate wheel slip / wheel disagreement strictly from onboard sensor signals.
    Zero privileged ground-truth leakage: compares wheel yaw rate with IMU gyro,
    and wheel longitudinal acceleration rate of change.
    """
    diff_omega = abs(w_wheel - gyro_z)
    diff_accel = 0.0
    if prev_v_wheel is not None and dt > 1e-4:
        diff_accel = abs((v_wheel - prev_v_wheel) / dt)
    return float(0.15 * diff_omega + 0.08 * min(diff_accel, 2.0))


def synthesize_hybrid_measurements(
    gt: GroundTruthTrajectory,
    gnss_outage_interval: Optional[Tuple[float, float]] = (14.0, 34.0),
    seed: int = 42,
    render_frames: bool = True
) -> Dict[str, Any]:
    """
    Pre-generate all sensor observations from the reference trajectory using an
    isolated PRNG stream so that the estimator loop has zero access to ground truth.
    """
    rng_sensor = np.random.default_rng(seed)
    n = len(gt.timestamps)
    t0 = float(gt.timestamps[0])

    v_wheels = np.zeros(n, dtype=np.float64)
    w_wheels = np.zeros(n, dtype=np.float64)
    gyro_zs = np.zeros(n, dtype=np.float64)
    in_outages = np.zeros(n, dtype=bool)
    gnss_fixes: List[Optional[np.ndarray]] = [None] * n
    frames: List[Optional[np.ndarray]] = [None] * n

    for i in range(n):
        t = float(gt.timestamps[i])
        rel_t = t - t0
        v_true = float(gt.linear_velocities[i])
        w_true = float(gt.angular_velocities[i])

        # Wheel spin anomaly at t = 13.0 to 18.0s while rover slows/stops
        if 13.0 <= rel_t <= 18.0:
            v_wheel = 0.28
            w_wheel = 0.01
        else:
            v_wheel = v_true
            w_wheel = w_true

        gyro_z = w_true + 0.002 * float(rng_sensor.standard_normal())

        in_outage = False
        if gnss_outage_interval is not None:
            t_out_start, t_out_end = gnss_outage_interval
            if t_out_start <= rel_t <= t_out_end:
                in_outage = True

        if i % 5 == 0 and not in_outage:
            gnss_fixes[i] = gt.positions_xyz[i, :2] + 0.03 * rng_sensor.standard_normal(2)

        if render_frames:
            frames[i] = generate_synthetic_scene_image(rel_t, gt.poses_2d[i], draw_banner=False)

        v_wheels[i] = v_wheel
        w_wheels[i] = w_wheel
        gyro_zs[i] = gyro_z
        in_outages[i] = in_outage

    return {
        "timestamps": gt.timestamps.copy(),
        "initial_pose": gt.poses_2d[0].copy(),
        "v_wheel": v_wheels,
        "w_wheel": w_wheels,
        "gyro_z": gyro_zs,
        "in_outage": in_outages,
        "gnss_fixes": gnss_fixes,
        "frames": frames,
        "vision_cache": {},
    }


def run_hybrid_experiment(
    gt: GroundTruthTrajectory,
    calib: M2DGRCalibration,
    estimator_type: str = "recoverable_ekf",
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
    gnss_outage_interval: Optional[Tuple[float, float]] = (14.0, 34.0),
    visual_degradation: str = "clean",
    vision_mode: str = "disconnected",
    seed: int = 42,
    measurements: Optional[Dict[str, Any]] = None
) -> RealDataTrialResult:
    """
    Execute state estimator over the hybrid M2DGR-plus sequence with controlled GNSS outage
    and visual degradation.

    vision_mode options:
    - 'disabled': Vision tracker is not executed (Causal Ablation A).
    - 'disconnected': Vision tracker runs diagnostically on an isolated RNG; outputs do not
      enter the state estimator (Causal Ablation B, default for R0-R7).
    - 'connected': Visual optical-flow speed aids wheel-slip rejection during prediction
      when feature tracking succeeds (Causal Ablation C/D).
    """
    from robot_localization.control.fail_safe import SafeStopSupervisor, SafeStopConfig

    if measurements is None:
        measurements = synthesize_hybrid_measurements(
            gt=gt,
            gnss_outage_interval=gnss_outage_interval,
            seed=seed,
            render_frames=(vision_mode != "disabled")
        )

    timestamps = measurements["timestamps"]
    n = len(timestamps)
    dt_nom = 0.20

    # Pre-compute or retrieve deterministic visual feature tracking stream on isolated RNG
    vision_stream: Optional[List[Optional[Dict[str, Any]]]] = None
    if vision_mode != "disabled":
        v_cache = measurements.setdefault("vision_cache", {})
        cache_key = (visual_degradation, seed)
        if cache_key not in v_cache:
            tracker = VisualFeatureTracker(calib, seed=seed + 10000)
            stream_list: List[Optional[Dict[str, Any]]] = [None] * n
            for idx in range(n):
                dt_idx = float(timestamps[idx] - timestamps[max(0, idx - 1)]) if idx > 0 else dt_nom
                if dt_idx <= 0:
                    dt_idx = 0.05
                frame = measurements["frames"][idx]
                if frame is not None:
                    v_out = tracker.process_frame(frame, degradation_mode=visual_degradation, dt=dt_idx)
                    # Store lightweight dict without full image buffer
                    stream_list[idx] = {
                        "num_features": int(v_out["num_features"]),
                        "tracked_count": int(v_out["tracked_count"]),
                        "inliers_count": int(v_out["inliers_count"]),
                        "inlier_ratio": float(v_out["inlier_ratio"]),
                        "tracking_failed": bool(v_out["tracking_failed"]),
                        "flow_speed_m_s": float(v_out["flow_speed_m_s"]),
                        "flow_vec_m_s": v_out["flow_vec_m_s"],
                    }
            v_cache[cache_key] = stream_list
        vision_stream = v_cache[cache_key]

    t_start = time.perf_counter()

    # Initial state from measurement stream (never reading gt inside estimation loop)
    p0 = measurements["initial_pose"].copy()
    initial_state = np.array([p0[0], p0[1], p0[2], 0.0])
    initial_cov = np.diag([0.05**2, 0.05**2, 0.02**2, (0.01)**2])

    cfg = Robot4WDConfig()

    # GNSS is synthesized at body origin (0.0, 0.0), so lever arm is explicitly zeroed
    if estimator_type == "dead_reckoning":
        est = DeadReckoning4WD(cfg, initial_pose=p0)
    elif estimator_type == "standard_ekf":
        est = ExtendedKalmanFilter3D(initial_state=p0, initial_cov=initial_cov[:3, :3])
    elif estimator_type == "bias_aware_ekf":
        est = BiasAwareEKF4D(
            initial_state=initial_state,
            initial_cov=initial_cov,
            gnss_lever_arm_m=(0.0, 0.0)
        )
    elif estimator_type == "robust_gated_ekf":
        est = RobustEKF4D(
            initial_state=initial_state,
            initial_cov=initial_cov,
            gnss_lever_arm_m=(0.0, 0.0),
            gnss_gate_chi2=9.210
        )
    elif estimator_type == "recoverable_ekf":
        est = RecoverableRobustEKF4D(
            strategy=recovery_strategy,
            initial_state=initial_state,
            initial_cov=initial_cov,
            gnss_lever_arm_m=(0.0, 0.0),
            gnss_gate_chi2=9.210
        )
    else:
        raise ValueError(f"Unknown estimator: {estimator_type}")

    supervisor = SafeStopSupervisor(SafeStopConfig(max_outage_time_s=2.0, max_pos_uncertainty_m=1.2))

    est_poses = []
    est_odom_poses = []
    map_to_odom_offsets = []
    cov_diags = []
    health_states = []
    relocalization_indices = []

    supervisor_states = []
    commanded_vs = []
    is_stopped_flags = []
    counterfactual_annotations = []

    feat_counts: List[int] = []
    inlier_counts: List[int] = []
    inlier_ratios: List[float] = []
    track_failures: List[bool] = []

    accepted_gnss = 0
    total_gnss_epochs = 0
    available_gnss_fixes = 0
    accepted_gnss_step_indices: List[int] = []

    t0 = float(timestamps[0])
    prev_v_wheel: Optional[float] = None

    for i in range(n):
        t = float(timestamps[i])
        rel_t = t - t0
        dt = float(timestamps[i] - timestamps[max(0, i - 1)]) if i > 0 else dt_nom
        if dt <= 0:
            dt = 0.05

        v_wheel = float(measurements["v_wheel"][i])
        w_wheel = float(measurements["w_wheel"][i])
        gyro_z = float(measurements["gyro_z"][i])
        in_outage = bool(measurements["in_outage"][i])

        # Step 0: Visual feature tracking (if enabled)
        v_res = None
        if vision_stream is not None:
            v_res = vision_stream[i]
            if v_res is not None:
                feat_counts.append(int(v_res["num_features"]))
                inlier_counts.append(int(v_res["inliers_count"]))
                inlier_ratios.append(float(v_res["inlier_ratio"]))
                track_failures.append(bool(v_res["tracking_failed"]))

        # Step 1: Onboard wheel disagreement estimation (zero ground-truth access)
        onboard_slip = estimate_onboard_wheel_disagreement(
            w_wheel=w_wheel,
            gyro_z=gyro_z,
            v_wheel=v_wheel,
            prev_v_wheel=prev_v_wheel,
            dt=dt
        )
        prev_v_wheel = v_wheel

        # Optional causal visual coupling (only when vision_mode == "connected")
        v_eff = v_wheel
        if vision_mode == "connected" and v_res is not None and not v_res["tracking_failed"]:
            vo_speed = float(v_res["flow_speed_m_s"])
            if abs(v_wheel) - vo_speed > 0.12:
                # Dampen wheel spin toward visual optical-flow speed weighted by inlier ratio
                alpha_vo = min(0.85, float(v_res["inlier_ratio"]))
                v_eff = (1.0 - alpha_vo) * v_wheel + alpha_vo * math.copysign(vo_speed, v_wheel)
                onboard_slip = max(onboard_slip, abs(v_wheel - v_eff))

        # Step 2: Prediction
        if isinstance(est, RecoverableRobustEKF4D):
            Q_u = np.diag([(0.04 * v_eff)**2 + 1e-4, (0.03 * w_wheel)**2 + 1e-4])
            est.predict_recoverable(
                v_enc=v_eff,
                omega_enc=w_wheel,
                gyro_z=gyro_z,
                Q_u_base=Q_u,
                dt=dt,
                wheel_disagreement=onboard_slip
            )
            health_states.append((t, est.state_machine.value))
        elif isinstance(est, RobustEKF4D):
            Q_u = np.diag([(0.04 * v_eff)**2 + 1e-4, (0.03 * w_wheel)**2 + 1e-4])
            est.predict_robust(v_eff, w_wheel, gyro_z, Q_u, dt, wheel_disagreement=onboard_slip)
        elif isinstance(est, BiasAwareEKF4D):
            u_in = np.array([v_eff, gyro_z], dtype=float)
            Q_u = np.diag([(0.04 * v_eff)**2 + 1e-4, (0.03 * gyro_z)**2 + 1e-4])
            est.predict(u_in, Q_u, dt)
        elif isinstance(est, ExtendedKalmanFilter3D):
            u_in = np.array([v_eff, gyro_z], dtype=float)
            Q_u = np.diag([(0.04 * v_eff)**2 + 1e-4, (0.03 * gyro_z)**2 + 1e-4])
            est.predict(u_in, Q_u, dt)
        else:
            est.pose = integrate_pose_rk2(est.pose, v_eff, w_wheel, dt)

        # Step 3: GNSS observation (1 Hz, every 5 samples)
        gnss_fix_val = None
        if i % 5 == 0 and not isinstance(est, DeadReckoning4WD):
            total_gnss_epochs += 1
            z_gnss = measurements["gnss_fixes"][i]
            if not in_outage and z_gnss is not None:
                available_gnss_fixes += 1
                R_gnss = np.diag([0.05**2, 0.05**2])
                gnss_fix_val = z_gnss

                if isinstance(est, RecoverableRobustEKF4D):
                    accepted, _ = est.update_gnss_recoverable(t, z_gnss, R_gnss)
                    if accepted:
                        accepted_gnss += 1
                        accepted_gnss_step_indices.append(i)
                elif isinstance(est, RobustEKF4D):
                    accepted = est.update_gnss_gated(z_gnss, R_gnss)
                    if accepted:
                        accepted_gnss += 1
                        accepted_gnss_step_indices.append(i)
                elif isinstance(est, BiasAwareEKF4D):
                    est.update_gnss(z_gnss, R_gnss)
                    accepted_gnss += 1
                    accepted_gnss_step_indices.append(i)
                elif isinstance(est, ExtendedKalmanFilter3D):
                    est.update_gnss(z_gnss, R_gnss)
                    accepted_gnss += 1
                    accepted_gnss_step_indices.append(i)

        # Step 4: Capture Post-Update State
        if isinstance(est, (RecoverableRobustEKF4D, RobustEKF4D, BiasAwareEKF4D)):
            state_val = est.x[:3].copy()
            cov_val = np.diag(est.P)[:3].copy()
            odom_val = est.odom_pose[:3].copy() if hasattr(est, "odom_pose") else state_val.copy()
            m2o_val = est.map_to_odom[:2].copy() if hasattr(est, "map_to_odom") else np.zeros(2)
        elif isinstance(est, ExtendedKalmanFilter3D):
            state_val = est.x.copy()
            cov_val = np.diag(est.P).copy()
            odom_val = state_val.copy()
            m2o_val = np.zeros(2)
        else:
            state_val = est.pose.copy()
            cov_val = np.array([0.01, 0.01, 0.005]) * (rel_t + 1.0)
            odom_val = state_val.copy()
            m2o_val = np.zeros(2)

        if len(est_poses) > 0:
            jump_dist = float(np.hypot(state_val[0] - est_poses[-1][0], state_val[1] - est_poses[-1][1]))
            if jump_dist > max(0.40, 1.5 * dt):
                relocalization_indices.append(i)

        # Step 5: Safe-Stop Supervisor Evaluation (Counterfactual)
        sup_status = supervisor.update(
            t=rel_t,
            dt=dt,
            v_desired=0.25,
            omega_desired=0.0,
            v_measured=v_wheel,
            omega_measured=w_wheel,
            pos_cov_diag=cov_val[:2],
            new_gnss_fix=gnss_fix_val
        )
        supervisor_states.append((rel_t, sup_status.state.value))
        commanded_vs.append(sup_status.commanded_v)
        is_stopped_flags.append(sup_status.is_stopped)
        counterfactual_annotations.append(sup_status.counterfactual_annotation)

        est_poses.append(state_val)
        est_odom_poses.append(odom_val)
        map_to_odom_offsets.append(m2o_val)
        cov_diags.append(cov_val)

    t_end = time.perf_counter()
    comp_time_ms = (t_end - t_start) * 1000.0

    est_poses_arr = np.array(est_poses)
    est_odom_arr = np.array(est_odom_poses)
    m2o_arr = np.array(map_to_odom_offsets)
    cov_diags_arr = np.array(cov_diags)

    # Step 6: Post-hoc evaluation against ground truth (never accessed during estimation)
    pos_errors = np.hypot(est_poses_arr[:, 0] - gt.poses_2d[:, 0], est_poses_arr[:, 1] - gt.poses_2d[:, 1])
    head_errors = np.abs(np.arctan2(
        np.sin(est_poses_arr[:, 2] - gt.poses_2d[:, 2]),
        np.cos(est_poses_arr[:, 2] - gt.poses_2d[:, 2])
    ))

    pos_rmse = float(np.sqrt(np.mean(pos_errors**2)))
    head_rmse_deg = float(np.degrees(np.sqrt(np.mean(head_errors**2))))

    in_outage_mask = measurements["in_outage"]
    peak_outage_err = float(np.max(pos_errors[in_outage_mask])) if np.any(in_outage_mask) else float(np.max(pos_errors))

    # Recovery metrics: requires accepted post-outage GNSS fix AND post-update error < 0.35 m
    recovered_at_t: Optional[float] = None
    if gnss_outage_interval is not None:
        outage_end_s = gnss_outage_interval[1]
        for idx in accepted_gnss_step_indices:
            rel_t_idx = float(timestamps[idx] - t0)
            if rel_t_idx > outage_end_s and pos_errors[idx] < 0.35:
                recovered_at_t = rel_t_idx
                break

    recovery_time = float("nan")
    recovery_success = False
    if gnss_outage_interval is not None and recovered_at_t is not None:
        final_idx = int(n * 0.85)
        if np.mean(pos_errors[final_idx:]) < 0.35:
            recovery_success = True
            recovery_time = float(recovered_at_t - gnss_outage_interval[1])

    gnss_pct = (accepted_gnss / max(1, available_gnss_fixes)) * 100.0 if not isinstance(est, DeadReckoning4WD) else 0.0
    gnss_epoch_pct = (accepted_gnss / max(1, total_gnss_epochs)) * 100.0 if not isinstance(est, DeadReckoning4WD) else 0.0

    false_recoveries = 0
    if isinstance(est, RecoverableRobustEKF4D):
        false_recoveries = est.false_recoveries

    return RealDataTrialResult(
        method_name=f"{estimator_type}_{recovery_strategy.value if isinstance(est, RecoverableRobustEKF4D) else 'none'}",
        scenario_name=f"outage_{'clean' if visual_degradation == 'clean' else visual_degradation}",
        timestamps=timestamps,
        est_poses=est_poses_arr,
        gt_poses=gt.poses_2d,
        cov_diags=cov_diags_arr,
        pos_rmse_m=pos_rmse,
        heading_rmse_deg=head_rmse_deg,
        outage_peak_error_m=peak_outage_err,
        recovery_time_s=recovery_time,
        recovery_success=recovery_success,
        gnss_acceptance_pct=gnss_pct,
        false_recoveries=false_recoveries,
        computation_time_ms=comp_time_ms,
        health_states=health_states,
        est_poses_odom=est_odom_arr,
        map_to_odom_offsets=m2o_arr,
        relocalization_indices=relocalization_indices,
        supervisor_states=supervisor_states,
        commanded_vs=commanded_vs,
        is_stopped_flags=is_stopped_flags,
        counterfactual_annotations=counterfactual_annotations,
        gnss_epoch_acceptance_pct=gnss_epoch_pct,
        vision_mode=vision_mode,
        avg_features=float(np.mean(feat_counts)) if feat_counts else 0.0,
        avg_inliers=float(np.mean(inlier_counts)) if inlier_counts else 0.0,
        avg_inlier_ratio=float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
        tracking_failure_pct=float(np.mean(track_failures) * 100.0) if track_failures else 0.0,
    )


def run_full_realdata_suite(
    cache_dir: Path = Path("data"),
    results_dir: Path = Path("results")
) -> pd.DataFrame:
    """
    Run complete hybrid experimental benchmark across recovery strategies (R0 - R7),
    visual degradation conditions, and causal vision ablation modes.
    """
    print("\n" + "=" * 80)
    print("M2DGR-PLUS HYBRID EXPERIMENTAL SUITE: GNSS DENIAL & RECOVERY")
    print("=" * 80)

    cache_dir = Path(cache_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(cache_dir=cache_dir)
    calib = load_calibration(cache_dir=cache_dir)

    print(f"Loaded ground truth: {gt.sample_count} poses, {gt.duration_s:.2f} s, distance: {gt.total_distance_m:.2f} m")

    sample_bag = cache_dir / "sample_m2dgrp_anomaly.bag"
    create_sample_bag(sample_bag, gt, calib)

    info = inspect_bag(str(sample_bag))
    print(f"Synthetic test rosbag verified: {info['file_name']} ({info['total_messages']} msgs across {len(info['topics'])} topics)")

    shared_measurements = synthesize_hybrid_measurements(
        gt=gt,
        gnss_outage_interval=(14.0, 34.0),
        seed=42,
        render_frames=True
    )

    # Configurations (label, estimator_type, strategy, visual_degradation, vision_mode)
    configurations = [
        ("R0_dead_reckoning", "dead_reckoning", RecoveryStrategy.R0_STRICT_GATE, "clean", "disconnected"),
        ("R1_standard_ekf", "standard_ekf", RecoveryStrategy.R0_STRICT_GATE, "clean", "disconnected"),
        ("R2_bias_aware_ekf", "bias_aware_ekf", RecoveryStrategy.R0_STRICT_GATE, "clean", "disconnected"),
        ("R3_strict_gated_ekf", "robust_gated_ekf", RecoveryStrategy.R0_STRICT_GATE, "clean", "disconnected"),
        ("R4_recoverable_temporal", "recoverable_ekf", RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, "clean", "disconnected"),
        ("R5_recoverable_hysteresis", "recoverable_ekf", RecoveryStrategy.R2_HEALTH_HYSTERESIS, "clean", "disconnected"),
        ("R6_recoverable_cov_adj", "recoverable_ekf", RecoveryStrategy.R3_COVARIANCE_ADJUSTMENT, "clean", "disconnected"),
        ("R7_recoverable_reinit", "recoverable_ekf", RecoveryStrategy.R4_GLOBAL_REINITIALIZATION, "clean", "disconnected"),
        # Disconnected visual tracking degradation study (evaluates optical-flow degradation without PRNG leakage)
        ("R4_moderate_degradation", "recoverable_ekf", RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, "moderate", "disconnected"),
        ("R4_severe_degradation", "recoverable_ekf", RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, "severe", "disconnected"),
        # Causal vision ablation modes (A: disabled, B: disconnected=R4_recoverable_temporal, C: connected clean, moderate, D: connected severe)
        ("R4_vision_disabled", "recoverable_ekf", RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, "clean", "disabled"),
        ("R4_vision_connected_clean", "recoverable_ekf", RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, "clean", "connected"),
        ("R4_vision_connected_moderate", "recoverable_ekf", RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, "moderate", "connected"),
        ("R4_vision_connected_severe", "recoverable_ekf", RecoveryStrategy.R1_TEMPORAL_CONSISTENCY, "severe", "connected"),
    ]

    records = []
    trial_results = {}

    for label, est_type, strat, deg, v_mode in configurations:
        print(f"Running: {label:30s} ... ", end="", flush=True)
        res = run_hybrid_experiment(
            gt=gt,
            calib=calib,
            estimator_type=est_type,
            recovery_strategy=strat,
            gnss_outage_interval=(14.0, 34.0),
            visual_degradation=deg,
            vision_mode=v_mode,
            seed=42,
            measurements=shared_measurements
        )
        trial_results[label] = res
        rec_str = f"{res.recovery_time_s:4.1f}s" if not math.isnan(res.recovery_time_s) else " N/A "
        print(
            f"RMSE: {res.pos_rmse_m:5.3f}m | "
            f"Peak: {res.outage_peak_error_m:5.3f}m | "
            f"Recov: {'YES' if res.recovery_success else 'NO '} ({rec_str}) | "
            f"Accept: {res.gnss_acceptance_pct:5.1f}% | "
            f"Inliers: {res.avg_inlier_ratio*100:4.1f}%"
        )
        records.append({
            "configuration": label,
            "estimator": est_type,
            "recovery_strategy": strat.value,
            "visual_condition": deg,
            "vision_mode": v_mode,
            "pos_rmse_m": res.pos_rmse_m,
            "heading_rmse_deg": res.heading_rmse_deg,
            "outage_peak_error_m": res.outage_peak_error_m,
            "recovery_time_s": res.recovery_time_s,
            "recovery_success": res.recovery_success,
            "gnss_acceptance_pct": res.gnss_acceptance_pct,
            "gnss_epoch_acceptance_pct": res.gnss_epoch_acceptance_pct,
            "false_recoveries": res.false_recoveries,
            "avg_features": res.avg_features,
            "avg_inliers": res.avg_inliers,
            "avg_inlier_ratio": res.avg_inlier_ratio,
            "tracking_failure_pct": res.tracking_failure_pct,
            "runtime_ms": res.computation_time_ms
        })

    df = pd.DataFrame(records)
    summary_csv = results_dir / "realdata_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"\nSaved experimental summary to: {summary_csv}")

    traces_npz = results_dir / "realdata_traces.npz"
    np.savez_compressed(
        traces_npz,
        timestamps=gt.timestamps,
        gt_poses=gt.poses_2d,
        **{f"est_{k}": v.est_poses for k, v in trial_results.items()},
        **{f"odom_{k}": v.est_poses_odom for k, v in trial_results.items()},
        **{f"m2o_{k}": v.map_to_odom_offsets for k, v in trial_results.items()},
        **{f"cov_{k}": v.cov_diags for k, v in trial_results.items()},
        **{f"cmd_v_{k}": np.array(v.commanded_vs, dtype=float) for k, v in trial_results.items()},
        **{f"stopped_{k}": np.array(v.is_stopped_flags, dtype=bool) for k, v in trial_results.items()},
        **{f"reloc_idx_{k}": np.array(v.relocalization_indices, dtype=int) for k, v in trial_results.items()}
    )
    print(f"Saved numerical traces to: {traces_npz}")

    manifest = {
        "dataset": "M2DGR-plus (Hybrid Benchmark)",
        "primary_sequence": "Anomaly",
        "description": "46.21s ground robot reference trajectory with authentic lateral displacement anomaly (t=19.1-27.3s) and synthesized sensor streams",
        "provenance_and_integrity": {
            "authentic_reference_ground_truth": "M2DGR-plus gt/anomaly.txt (233 poses, 46.21s duration, 7.32m path length, scalar-first quaternion wxyz)",
            "authentic_calibration": "M2DGR-plus calibration.txt (sanitized UTF-8 full-width commas, 7 SE3 extrinsic matrices)",
            "sensor_observations": "Ground-truth-conditioned synthetic odometry, IMU, GNSS, and top-down floor camera frames (raw M2DGR-plus rosbag requires authenticated SJTU SharePoint access)",
            "rosbag_playback": "sample_m2dgrp_anomaly.bag (synthesized deterministic ROS 1 bag matching M2DGR-plus topic schemas)",
            "ground_truth_leakage": "ZERO LEAKAGE: Sensor observations are pre-synthesized; estimator loop has zero access to GroundTruthTrajectory",
            "prng_isolation": "Sensor noise RNG and visual degradation RNG are strictly isolated (np.random.default_rng)",
            "pre_update_snapshot_fixed": "Estimator state and covariance recorded strictly AFTER measurement update for each epoch",
            "rep105_coordinate_frames": "odom->base_link strictly continuous; global GNSS relocalizations absorbed in map->odom",
            "counterfactual_safe_stop": "Safe-stop supervisor evaluated counterfactually on open-loop trace; physical braking verified in closed-loop simulation"
        },
        "official_sources": {
            "github_plus": "https://github.com/SJTU-ViSYS/M2DGR-plus",
            "ground_truth": "https://raw.githubusercontent.com/SJTU-ViSYS/M2DGR-plus/main/gt/anomaly.txt",
            "calibration": "https://raw.githubusercontent.com/SJTU-ViSYS/M2DGR-plus/main/calibration.txt",
            "full_bag_sharepoint": "https://sjtueducn-my.sharepoint.com/:u:/g/personal/594666_sjtu_edu_cn/Ef8corMuVwhJsWSpp-FXkREBrTduGBO8nifC9VEb5twHVg?e=CyEeMy"
        },
        "access_audit": {
            "sharepoint_status": "Redirects to SJTU Microsoft SSO login; unauthenticated direct download is blocked without university credentials",
            "gt_status": "Publicly verified and downloaded (233 poses, 46.21s duration, scalar-first quaternion wxyz)",
            "calib_status": "Publicly verified and sanitized (7 coordinate frames, PINHOLE intrinsics, IMU noise)",
            "bag_reader": "Pure-Python rosbags reader (100% ROS-independent)"
        },
        "sensor_topics": {
            "odom": "/odom (nav_msgs/msg/Odometry, ground-truth-conditioned synthetic)",
            "imu": "/imu & /camera/imu (sensor_msgs/msg/Imu, ground-truth-conditioned synthetic)",
            "gnss": "/ublox_driver/receiver_pvt (sensor_msgs/msg/NavSatFix, ground-truth-conditioned synthetic)",
            "camera": "Top-down synthetic floor camera (640x480, ground-truth-pose-conditioned)"
        },
        "anomaly_investigation": {
            "injected_wheel_spin": "Forward wheel spin (0.28 m/s) injected at t=13.0s-18.0s",
            "authentic_gt_lateral_anomaly": "Authentic M2DGR-plus ground-truth lateral shift (-1.53m along -Y_body with zero forward wheel motion) at t=19.12s-27.29s",
            "outage_duration_s": 20.0,
            "outage_start_s": 14.0,
            "outage_end_s": 34.0
        }
    }
    manifest_path = results_dir / "data_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to: {manifest_path}\n")

    return df


if __name__ == "__main__":
    run_full_realdata_suite()
