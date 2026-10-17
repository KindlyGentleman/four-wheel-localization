"""
robot_localization.sensors.stereo

Calibrated passive stereo camera system, depth pipeline, 3D point cloud generation,
stereo visual odometry (VO), and ICP-based geometric registration.
Supports:
- Mode A: Deterministic synthetic geometric stereo simulator with rocks, walls, and textures
- Mode B: Real calibrated stereo image sequence ingestion
- OpenCV StereoSGBM disparity computation and depth triangulation
- 3D point cloud generation with range filtering and voxel downsampling
- Feature-based stereo visual odometry (ORB + stereo triangulation + PnP RANSAC)
- Extreme-environment visual degradation simulation and CLAHE enhancement
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from robot_localization.models.kinematics import wrap_angle


@dataclass
class StereoCameraConfig:
    """Intrinsics, extrinsics, and disparity parameters for calibrated stereo pair."""
    width: int = 640
    height: int = 480
    fx: float = 400.0                   # Focal length X in pixels
    fy: float = 400.0                   # Focal length Y in pixels
    cx: float = 320.0                   # Principal point X
    cy: float = 240.0                   # Principal point Y
    baseline_m: float = 0.10            # Stereo baseline in meters (10 cm)
    frame_id: str = "camera_link"
    optical_frame_id: str = "camera_left_optical"
    mounting_offset_base_xyz: Tuple[float, float, float] = (0.20, 0.0, 0.20)
    mounting_rpy_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0) # Pitch, roll, yaw
    # Disparity parameters
    min_disparity: int = 0
    num_disparities: int = 64          # Must be multiple of 16
    block_size: int = 5
    min_depth_m: float = 0.25
    max_depth_m: float = 12.0

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)

    @property
    def q_matrix(self) -> np.ndarray:
        """Standard 4x4 reprojection disparity-to-depth matrix."""
        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0, 0] = 1.0
        Q[0, 3] = -self.cx
        Q[1, 1] = 1.0
        Q[1, 3] = -self.cy
        Q[2, 3] = self.fx
        Q[3, 2] = -1.0 / self.baseline_m
        Q[3, 3] = 0.0
        return Q


# Coordinate transforms between REP-105 standard frames
# base_link: X forward, Y left, Z up
# camera_left_optical: Z forward (optical axis), X right, Y down
R_BASE_TO_OPTICAL = np.array([
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0, -1.0],
    [ 1.0,  0.0,  0.0]
], dtype=np.float64)

R_OPTICAL_TO_BASE = R_BASE_TO_OPTICAL.T


def transform_optical_to_base(pts_optical: np.ndarray, mounting_xyz: Tuple[float, float, float] = (0.20, 0.0, 0.20)) -> np.ndarray:
    """Transform (N, 3) 3D points from camera optical frame to robot base_link frame."""
    if len(pts_optical) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    # P_base = R_optical_to_base * P_opt + offset
    pts_base = (R_OPTICAL_TO_BASE @ pts_optical.T).T + np.array(mounting_xyz)
    return pts_base


def transform_base_to_optical(pts_base: np.ndarray, mounting_xyz: Tuple[float, float, float] = (0.20, 0.0, 0.20)) -> np.ndarray:
    """Transform (N, 3) 3D points from robot base_link frame to camera optical frame."""
    if len(pts_base) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts_rel = pts_base - np.array(mounting_xyz)
    pts_opt = (R_BASE_TO_OPTICAL @ pts_rel.T).T
    return pts_opt


# -----------------------------------------------------------------------------
# Mode A: Synthetic Geometric Scene and Stereo Renderer
# -----------------------------------------------------------------------------

@dataclass
class VisualLandmark3D:
    id: int
    world_xyz: np.ndarray           # [x, y, z] in world frame (m)
    size_m: float = 0.40
    color: Tuple[int, int, int] = (180, 180, 180)
    texture_type: str = "cross"     # "cross", "checker", "circle"


class SyntheticStereoArena:
    """
    Deterministic 3D environment for synthetic stereo simulation.
    Features static textured rocks, vertical pillars/walls, and ground texture.
    """

    def __init__(self, arena_size_m: float = 20.0, seed: int = 42):
        self.arena_size = arena_size_m
        self.rng = np.random.RandomState(seed)
        self.landmarks: List[VisualLandmark3D] = []
        self._build_deterministic_scene()

    def _build_deterministic_scene(self):
        # Create boundary pillars / rocks in a 20m arena
        landmark_coords = [
            (2.0,  1.5, 0.4, 0.35, "checker"),
            (3.5, -2.0, 0.5, 0.45, "cross"),
            (5.0,  0.5, 0.6, 0.50, "circle"),
            (4.0,  3.0, 0.4, 0.40, "checker"),
            (6.5, -1.5, 0.5, 0.40, "cross"),
            (7.0,  2.5, 0.7, 0.55, "circle"),
            (1.5, -2.5, 0.4, 0.30, "checker"),
            (8.5,  0.0, 0.6, 0.50, "cross"),
            (9.0, -3.0, 0.5, 0.45, "circle"),
            (10.5, 1.5, 0.8, 0.60, "checker"),
        ]
        for idx, (x, y, z, s, t_type) in enumerate(landmark_coords):
            self.landmarks.append(VisualLandmark3D(
                id=idx,
                world_xyz=np.array([x, y, z], dtype=np.float64),
                size_m=s,
                texture_type=t_type
            ))

    def render_stereo_pair(
        self,
        robot_pose_xy_theta: np.ndarray,
        config: StereoCameraConfig,
        degradation: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Render calibrated left and right stereo images (grayscale uint8)
        at the specified robot ground-truth 2D pose [x, y, theta].
        """
        rx, ry, r_theta = robot_pose_xy_theta[0], robot_pose_xy_theta[1], robot_pose_xy_theta[2]
        w, h = config.width, config.height

        # Base background with textured ground/dust noise
        img_left = np.full((h, w), 120, dtype=np.uint8)
        img_right = np.full((h, w), 120, dtype=np.uint8)

        # Add deterministic procedural background surface pattern
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        bg_pattern = (np.sin(grid_x * 0.05 + rx * 2.0) * np.cos(grid_y * 0.05 + ry * 2.0) * 15.0).astype(np.int16)
        img_left = np.clip(img_left.astype(np.int16) + bg_pattern, 0, 255).astype(np.uint8)
        img_right = np.clip(img_right.astype(np.int16) + bg_pattern, 0, 255).astype(np.uint8)

        # Robot world to base_link transform
        c_th = math.cos(r_theta)
        s_th = math.sin(r_theta)
        R_world_to_base = np.array([
            [ c_th,  s_th, 0.0],
            [-s_th,  c_th, 0.0],
            [  0.0,   0.0, 1.0]
        ])

        visible_landmarks = []

        # Sort landmarks by distance (back to front rendering)
        dists = [np.linalg.norm(lm.world_xyz[:2] - np.array([rx, ry])) for lm in self.landmarks]
        sorted_indices = np.argsort(dists)[::-1]

        for s_idx in sorted_indices:
            lm = self.landmarks[s_idx]
            # World to base
            p_rel_world = lm.world_xyz - np.array([rx, ry, 0.0])
            p_base = R_world_to_base @ p_rel_world

            # Base to left camera optical frame
            p_opt = transform_base_to_optical(p_base.reshape(1, 3), config.mounting_offset_base_xyz)[0]
            Z_left = p_opt[2]
            X_left = p_opt[0]
            Y_left = p_opt[1]

            # Check if landmark is in front of camera
            if Z_left < config.min_depth_m or Z_left > config.max_depth_m:
                continue

            # Left image projection
            u_left = config.fx * (X_left / Z_left) + config.cx
            v_left = config.fy * (Y_left / Z_left) + config.cy

            # Right camera: baseline offset B along optical X axis
            # In standard rectified stereo, right camera is shifted by +B in optical X
            # (or X_right = X_left - B)
            X_right = X_left - config.baseline_m
            Z_right = Z_left
            Y_right = Y_left

            u_right = config.fx * (X_right / Z_right) + config.cx
            v_right = config.fy * (Y_right / Z_right) + config.cy # Identical scanline

            disparity_px = u_left - u_right

            # Landmark radius in pixels
            radius_px = int(max(config.fx * (lm.size_m / Z_left), 3.0))

            # Check image boundaries
            if (-radius_px <= u_left < w + radius_px) and (-radius_px <= v_left < h + radius_px):
                self._draw_landmark_patch(img_left, int(u_left), int(v_left), radius_px, lm.texture_type)
                self._draw_landmark_patch(img_right, int(u_right), int(v_right), radius_px, lm.texture_type)
                visible_landmarks.append({
                    "id": lm.id,
                    "u_left": float(u_left),
                    "v_left": float(v_left),
                    "u_right": float(u_right),
                    "v_right": float(v_right),
                    "disparity": float(disparity_px),
                    "depth_m": float(Z_left),
                    "p_optical": p_opt.copy()
                })

        # Apply synthetic degradation if requested
        if degradation is not None:
            img_left, img_right = apply_visual_degradation(img_left, img_right, degradation)

        return img_left, img_right, visible_landmarks

    def _draw_landmark_patch(self, img: np.ndarray, u: int, v: int, radius: int, t_type: str):
        """Draw distinct geometric textured pattern on image canvas."""
        h, w = img.shape
        cv2.circle(img, (u, v), radius, 220, -1)
        cv2.circle(img, (u, v), radius, 30, 2)
        if t_type == "cross":
            cv2.line(img, (u - radius, v), (u + radius, v), 40, 2)
            cv2.line(img, (u, v - radius), (u, v + radius), 40, 2)
        elif t_type == "checker":
            cv2.rectangle(img, (u - radius//2, v - radius//2), (u, v), 40, -1)
            cv2.rectangle(img, (u, v), (u + radius//2, v + radius//2), 40, -1)
        elif t_type == "circle":
            cv2.circle(img, (u, v), max(radius // 2, 2), 50, -1)


# -----------------------------------------------------------------------------
# Extreme Visual Degradation & Preprocessing
# -----------------------------------------------------------------------------

def apply_visual_degradation(img_l: np.ndarray, img_r: np.ndarray, degradation_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate extreme outdoor/volcanic visual degradations.
    """
    out_l = img_l.copy()
    out_r = img_r.copy()

    if degradation_type == "low_illumination":
        # Dark volcanic night / shadowed terrain
        out_l = (out_l * 0.22).astype(np.uint8)
        out_r = (out_r * 0.22).astype(np.uint8)

    elif degradation_type == "saturated_exposure":
        # Direct sunlight glare / overexposure
        out_l = np.clip(out_l.astype(np.float32) * 1.8 + 60, 0, 255).astype(np.uint8)
        out_r = np.clip(out_r.astype(np.float32) * 1.8 + 60, 0, 255).astype(np.uint8)

    elif degradation_type == "brightness_mismatch":
        # Left camera shadowed, right camera glare
        out_l = (out_l * 0.40).astype(np.uint8)
        out_r = np.clip(out_r.astype(np.float32) * 1.40, 0, 255).astype(np.uint8)

    elif degradation_type == "motion_blur":
        # Rapid skid turn motion blur (horizontal 15px kernel)
        kernel = np.zeros((1, 15), dtype=np.float32)
        kernel[0, :] = 1.0 / 15.0
        out_l = cv2.filter2D(out_l, -1, kernel)
        out_r = cv2.filter2D(out_r, -1, kernel)

    elif degradation_type == "dust_haze":
        # Volcanic ash / dust contrast attenuation
        # I = I_orig * (1 - A) + A * atmospheric_light
        fog = 180
        alpha = 0.65
        out_l = cv2.addWeighted(out_l, 1.0 - alpha, np.full_like(out_l, fog), alpha, 0)
        out_r = cv2.addWeighted(out_r, 1.0 - alpha, np.full_like(out_r, fog), alpha, 0)

    elif degradation_type == "reduced_texture":
        # Flat featureless ash bed: low contrast smoothing
        out_l = cv2.GaussianBlur(out_l, (21, 21), 0)
        out_l = (out_l * 0.3 + 100).astype(np.uint8)
        out_r = cv2.GaussianBlur(out_r, (21, 21), 0)
        out_r = (out_r * 0.3 + 100).astype(np.uint8)

    return out_l, out_r


def enhance_stereo_images(img_l: np.ndarray, img_r: np.ndarray, method: str = "clahe") -> Tuple[np.ndarray, np.ndarray]:
    """
    CPU-efficient image enhancement (Contrast Limited Adaptive Histogram Equalization).
    """
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enh_l = clahe.apply(img_l)
        enh_r = clahe.apply(img_r)
        return enh_l, enh_r
    elif method == "normalize":
        enh_l = cv2.normalize(img_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        enh_r = cv2.normalize(img_r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return enh_l, enh_r
    return img_l, img_r


# -----------------------------------------------------------------------------
# Disparity, Metric Depth, and 3D Point Cloud Pipeline
# -----------------------------------------------------------------------------

@dataclass
class StereoDepthResult:
    disparity_map: np.ndarray        # (H, W) float32 in pixels
    depth_map_m: np.ndarray          # (H, W) float32 in meters
    valid_mask: np.ndarray           # (H, W) bool
    valid_fraction: float            # Fraction of valid pixels
    median_depth_m: float
    point_cloud_optical: np.ndarray  # (N, 3) XYZ in camera optical frame
    point_cloud_base: np.ndarray     # (N, 3) XYZ in robot base_link frame


class StereoDepthPipeline:
    """
    OpenCV StereoSGBM disparity and metric 3D point cloud generation.
    """

    def __init__(self, config: Optional[StereoCameraConfig] = None):
        self.config = config if config is not None else StereoCameraConfig()
        # Initialize OpenCV StereoSGBM
        # Parameters configured for low-cost rover cameras
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=self.config.min_disparity,
            numDisparities=self.config.num_disparities,
            blockSize=self.config.block_size,
            P1=8 * self.config.block_size ** 2,
            P2=32 * self.config.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute_depth_and_points(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        voxel_downsample_m: float = 0.05,
        max_points: int = 1500
    ) -> StereoDepthResult:
        """
        Compute disparity, metric depth, and downsampled 3D point cloud.
        """
        # Ensure grayscale
        if len(img_left.shape) == 3:
            gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_l = img_left
            gray_r = img_right

        # StereoSGBM returns 16-bit signed fixed point with 4 fractional bits (multiply by 16)
        disp_raw = self.sgbm.compute(gray_l, gray_r)
        disparity_px = disp_raw.astype(np.float32) / 16.0

        # Valid disparity mask
        valid_mask = (disparity_px > 0.5) & (disparity_px < self.config.num_disparities)

        # Metric depth Z = (fx * B) / disparity
        depth_m = np.zeros_like(disparity_px, dtype=np.float32)
        depth_m[valid_mask] = (self.config.fx * self.config.baseline_m) / disparity_px[valid_mask]

        # Enforce valid range
        range_mask = valid_mask & (depth_m >= self.config.min_depth_m) & (depth_m <= self.config.max_depth_m)
        depth_m[~range_mask] = 0.0

        valid_fraction = float(np.count_nonzero(range_mask)) / float(disparity_px.size)
        median_depth = float(np.median(depth_m[range_mask])) if np.any(range_mask) else 0.0

        # 3D Point Cloud Projection
        v_idx, u_idx = np.nonzero(range_mask)
        z_valid = depth_m[v_idx, u_idx]

        if len(z_valid) > 0:
            x_valid = (u_idx - self.config.cx) * z_valid / self.config.fx
            y_valid = (v_idx - self.config.cy) * z_valid / self.config.fy
            pts_opt = np.stack([x_valid, y_valid, z_valid], axis=-1)

            # Subsample for CPU efficiency
            if len(pts_opt) > max_points:
                step = len(pts_opt) // max_points
                pts_opt = pts_opt[::step][:max_points]

            pts_base = transform_optical_to_base(pts_opt, self.config.mounting_offset_base_xyz)
        else:
            pts_opt = np.zeros((0, 3), dtype=np.float64)
            pts_base = np.zeros((0, 3), dtype=np.float64)

        return StereoDepthResult(
            disparity_map=disparity_px,
            depth_map_m=depth_m,
            valid_mask=range_mask,
            valid_fraction=valid_fraction,
            median_depth_m=median_depth,
            point_cloud_optical=pts_opt,
            point_cloud_base=pts_base
        )


# -----------------------------------------------------------------------------
# Stereo Visual Odometry (Camera-Based Motion Estimation)
# -----------------------------------------------------------------------------

@dataclass
class VisualOdometryResult:
    is_valid: bool
    dx_robot_m: float                  # Forward displacement in robot base_link
    dy_robot_m: float                  # Lateral displacement in robot base_link
    dtheta_robot_rad: float            # Yaw rotation in robot base_link
    v_forward_m_s: float
    omega_yaw_rad_s: float
    covariance: np.ndarray             # 3x3 [x, y, theta] uncertainty
    num_detected_features: int
    num_stereo_matches: int
    num_temporal_matches: int
    num_inliers: int
    inlier_ratio: float
    reprojection_rmse_px: float
    dt: float


class StereoVisualOdometry:
    """
    Lightweight, CPU-friendly feature-based stereo visual odometry.
    Pipeline:
    1. Detect ORB features on Left image.
    2. Triangulate 3D coordinates via epipolar disparity matching on Right image.
    3. Match 2D features temporally between Left(t-1) and Left(t).
    4. Estimate 3D-to-2D relative pose using PnP with RANSAC.
    5. Transform relative camera motion to robot base_link frame.
    """

    def __init__(
        self,
        config: Optional[StereoCameraConfig] = None,
        max_features: int = 500,
        min_inliers: int = 10,
        reprojection_thresh_px: float = 3.0
    ):
        self.config = config if config is not None else StereoCameraConfig()
        self.max_features = max_features
        self.min_inliers = min_inliers
        self.reprojection_thresh_px = reprojection_thresh_px

        # Feature detector & descriptor matcher
        self.orb = cv2.ORB_create(
            nfeatures=self.max_features,
            scaleFactor=1.2,
            nlevels=4,
            edgeThreshold=15,
            fastThreshold=20
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Previous frame state
        self.prev_left_img: Optional[np.ndarray] = None
        self.prev_kpts_left: Optional[List[cv2.KeyPoint]] = None
        self.prev_desc_left: Optional[np.ndarray] = None
        self.prev_pts_3d_opt: Optional[np.ndarray] = None
        self.prev_time: Optional[float] = None

    def process_frame(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        timestamp: float
    ) -> VisualOdometryResult:
        """
        Process stereo image pair and estimate relative robot motion.
        """
        # Ensure grayscale
        if len(img_left.shape) == 3:
            gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_l = img_left
            gray_r = img_right

        # 1. Detect ORB features on Left and Right
        kp_l, desc_l = self.orb.detectAndCompute(gray_l, None)
        kp_r, desc_r = self.orb.detectAndCompute(gray_r, None)

        if desc_l is None or desc_r is None or len(kp_l) < self.min_inliers or len(kp_r) < self.min_inliers:
            self._save_state(gray_l, kp_l, desc_l, None, timestamp)
            return self._create_invalid_result(0.0, len(kp_l) if kp_l else 0)

        # 2. Stereo matching between Left and Right at current frame
        stereo_matches = self.matcher.match(desc_l, desc_r)

        # Epipolar validation: in rectified stereo, keypoints must have near-identical y-coordinates (v_l ~= v_r)
        # and positive disparity (u_l > u_r)
        valid_stereo_pts_3d = []
        valid_kpt_indices = []

        for m in stereo_matches:
            pt_l = kp_l[m.queryIdx].pt
            pt_r = kp_r[m.trainIdx].pt
            v_diff = abs(pt_l[1] - pt_r[1])
            disp = pt_l[0] - pt_r[0]

            if v_diff < 2.0 and disp > 1.0: # Valid epipolar geometry
                z = (self.config.fx * self.config.baseline_m) / disp
                if self.config.min_depth_m <= z <= self.config.max_depth_m:
                    x = (pt_l[0] - self.config.cx) * z / self.config.fx
                    y = (pt_l[1] - self.config.cy) * z / self.config.fy
                    valid_stereo_pts_3d.append([x, y, z])
                    valid_kpt_indices.append(m.queryIdx)

        if len(valid_stereo_pts_3d) < self.min_inliers:
            self._save_state(gray_l, kp_l, desc_l, None, timestamp)
            return self._create_invalid_result(0.0, len(kp_l), len(stereo_matches))

        curr_pts_3d_opt = np.array(valid_stereo_pts_3d, dtype=np.float64)

        # 3. If first frame, save state and return
        if self.prev_left_img is None or self.prev_pts_3d_opt is None or self.prev_time is None:
            self._save_state(gray_l, kp_l, desc_l, (valid_kpt_indices, curr_pts_3d_opt), timestamp)
            return self._create_invalid_result(0.0, len(kp_l), len(stereo_matches))

        dt = timestamp - self.prev_time
        if dt <= 1e-4:
            return self._create_invalid_result(0.0, len(kp_l), len(stereo_matches))

        # 4. Temporal feature matching between Left(t-1) and Left(t)
        prev_kpt_idxs, prev_pts_3d = self.prev_pts_3d_opt
        prev_desc_tracked = self.prev_desc_left[prev_kpt_idxs]

        temporal_matches = self.matcher.match(prev_desc_tracked, desc_l)

        if len(temporal_matches) < self.min_inliers:
            self._save_state(gray_l, kp_l, desc_l, (valid_kpt_indices, curr_pts_3d_opt), timestamp)
            return self._create_invalid_result(dt, len(kp_l), len(stereo_matches), len(temporal_matches))

        # Prepare 3D-to-2D correspondences for PnP
        pts_3d_prev_matched = []
        pts_2d_curr_matched = []

        for m in temporal_matches:
            pts_3d_prev_matched.append(prev_pts_3d[m.queryIdx])
            pts_2d_curr_matched.append(kp_l[m.trainIdx].pt)

        pts_3d_prev_matched = np.array(pts_3d_prev_matched, dtype=np.float64)
        pts_2d_curr_matched = np.array(pts_2d_curr_matched, dtype=np.float64)

        # 5. PnP with RANSAC: estimates camera pose change (R_vec, t_vec)
        K = self.config.camera_matrix
        dist_coeffs = np.zeros(4, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d_prev_matched,
            imagePoints=pts_2d_curr_matched,
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            reprojectionError=self.reprojection_thresh_px,
            iterationsCount=150,
            flags=cv2.SOLVEPNP_EPNP
        )

        num_inliers = len(inliers) if inliers is not None else 0
        inlier_ratio = float(num_inliers) / float(len(temporal_matches)) if len(temporal_matches) > 0 else 0.0

        if not success or num_inliers < self.min_inliers:
            self._save_state(gray_l, kp_l, desc_l, (valid_kpt_indices, curr_pts_3d_opt), timestamp)
            return self._create_invalid_result(dt, len(kp_l), len(stereo_matches), len(temporal_matches), num_inliers)

        # Reprojection RMSE of inliers
        R_cam, _ = cv2.Rodrigues(rvec)
        proj_pts, _ = cv2.projectPoints(pts_3d_prev_matched[inliers[:, 0]], rvec, tvec, K, dist_coeffs)
        residuals = np.linalg.norm(proj_pts.reshape(-1, 2) - pts_2d_curr_matched[inliers[:, 0]], axis=1)
        rmse_px = float(np.sqrt(np.mean(residuals ** 2)))

        # 6. Transform relative camera motion to robot base_link frame
        # Camera motion: P_curr = R_cam * P_prev + tvec
        # In optical frame: Z forward, X right, Y down
        # Robot base_link: X forward, Y left, Z up
        # Convert translation:
        t_cam = tvec.flatten()
        # Invert relative transform to get motion of camera in previous camera frame
        R_delta_cam = R_cam.T
        t_delta_cam = -R_delta_cam @ t_cam

        # Transform to base_link:
        t_delta_base = R_OPTICAL_TO_BASE @ t_delta_cam
        R_delta_base = R_OPTICAL_TO_BASE @ R_delta_cam @ R_BASE_TO_OPTICAL

        # Extract planar displacements [dx, dy, dtheta]
        dx = float(t_delta_base[0])
        dy = float(t_delta_base[1])
        # Yaw angle from rotation matrix
        dtheta = float(math.atan2(R_delta_base[1, 0], R_delta_base[0, 0]))

        v_forward = dx / dt
        omega_yaw = dtheta / dt

        # Empirical covariance based on reprojection error and inlier count
        scale = max(rmse_px / 1.0, 1.0) / max(num_inliers / 20.0, 0.5)
        cov = np.diag([
            (0.04 * scale) ** 2,
            (0.04 * scale) ** 2,
            (0.02 * scale) ** 2
        ])

        self._save_state(gray_l, kp_l, desc_l, (valid_kpt_indices, curr_pts_3d_opt), timestamp)

        return VisualOdometryResult(
            is_valid=True,
            dx_robot_m=dx,
            dy_robot_m=dy,
            dtheta_robot_rad=dtheta,
            v_forward_m_s=v_forward,
            omega_yaw_rad_s=omega_yaw,
            covariance=cov,
            num_detected_features=len(kp_l),
            num_stereo_matches=len(stereo_matches),
            num_temporal_matches=len(temporal_matches),
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            reprojection_rmse_px=rmse_px,
            dt=dt
        )

    def _save_state(self, gray_l, kp_l, desc_l, pts_data, timestamp):
        self.prev_left_img = gray_l
        self.prev_kpts_left = kp_l
        self.prev_desc_left = desc_l
        self.prev_pts_3d_opt = pts_data
        self.prev_time = timestamp

    def _create_invalid_result(
        self,
        dt: float,
        detected: int = 0,
        stereo_m: int = 0,
        temporal_m: int = 0,
        inliers: int = 0
    ) -> VisualOdometryResult:
        return VisualOdometryResult(
            is_valid=False,
            dx_robot_m=0.0,
            dy_robot_m=0.0,
            dtheta_robot_rad=0.0,
            v_forward_m_s=0.0,
            omega_yaw_rad_s=0.0,
            covariance=np.diag([100.0, 100.0, 10.0]),
            num_detected_features=detected,
            num_stereo_matches=stereo_m,
            num_temporal_matches=temporal_m,
            num_inliers=inliers,
            inlier_ratio=0.0,
            reprojection_rmse_px=999.0,
            dt=dt
        )


# -----------------------------------------------------------------------------
# Stereo Point Cloud ICP Registration
# -----------------------------------------------------------------------------

@dataclass
class ICPRegistrationResult:
    success: bool
    transformation: np.ndarray      # 4x4 SE(3) matrix
    fitness: float                  # Overlap ratio (0 to 1)
    inlier_rmse_m: float
    correspondence_count: int
    dx_m: float
    dy_m: float
    dtheta_rad: float


def _register_point_clouds_cpu_svd(
    source_pts_base: np.ndarray,
    target_pts_base: np.ndarray,
    init_T: np.ndarray,
    max_correspondence_dist_m: float = 0.50,
    max_iterations: int = 50,
    tolerance: float = 1e-5
) -> Tuple[bool, np.ndarray, float, float, int]:
    """
    Genuine CPU-only point-to-point ICP registration using KD-tree matching
    and SVD rigid transformation (Kabsch / Arun algorithm).
    """
    from scipy.spatial import cKDTree

    src = np.asarray(source_pts_base, dtype=np.float64)
    tgt = np.asarray(target_pts_base, dtype=np.float64)
    tree = cKDTree(tgt)

    T = init_T.copy().astype(np.float64)
    R = T[:3, :3]
    t = T[:3, 3]

    prev_error = float("inf")

    for _ in range(max_iterations):
        curr_src = (R @ src.T).T + t
        dists, indices = tree.query(curr_src)
        valid = dists < max_correspondence_dist_m
        num_valid = int(np.sum(valid))

        if num_valid < 10:
            break

        pts_src_matched = curr_src[valid]
        pts_tgt_matched = tgt[indices[valid]]

        centroid_src = np.mean(pts_src_matched, axis=0)
        centroid_tgt = np.mean(pts_tgt_matched, axis=0)

        src_centered = pts_src_matched - centroid_src
        tgt_centered = pts_tgt_matched - centroid_tgt

        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R_delta = Vt.T @ U.T

        # Reflection correction
        if np.linalg.det(R_delta) < 0:
            Vt[-1, :] *= -1
            R_delta = Vt.T @ U.T

        t_delta = centroid_tgt - R_delta @ centroid_src

        R = R_delta @ R
        t = R_delta @ t + t_delta

        mean_err = float(np.mean(dists[valid]))
        if abs(prev_error - mean_err) < tolerance:
            break
        prev_error = mean_err

    # Compute genuine fitness and inlier RMSE from actual points
    final_src = (R @ src.T).T + t
    dists, _ = tree.query(final_src)
    inliers = dists < max_correspondence_dist_m
    inlier_count = int(np.sum(inliers))
    fitness = float(inlier_count) / float(len(src))
    rmse = float(np.sqrt(np.mean(dists[inliers] ** 2))) if inlier_count > 0 else 999.0

    T_final = np.eye(4, dtype=np.float64)
    T_final[:3, :3] = R
    T_final[:3, 3] = t

    success = bool((fitness > 0.35) and (rmse < 0.25) and (inlier_count >= 15))
    return success, T_final, fitness, rmse, inlier_count


def register_point_clouds_icp(
    source_pts_base: np.ndarray,
    target_pts_base: np.ndarray,
    initial_transform: Optional[np.ndarray] = None,
    max_correspondence_dist_m: float = 0.50,
    force_cpu: bool = False
) -> ICPRegistrationResult:
    """
    Register two consecutive stereo point clouds in the robot base frame using ICP.
    Uses Open3D when available and not forced to CPU; otherwise executes genuine CPU SVD ICP.
    """
    if len(source_pts_base) < 20 or len(target_pts_base) < 20:
        return ICPRegistrationResult(
            success=False,
            transformation=np.eye(4),
            fitness=0.0,
            inlier_rmse_m=999.0,
            correspondence_count=0,
            dx_m=0.0, dy_m=0.0, dtheta_rad=0.0
        )

    init_T = initial_transform if initial_transform is not None else np.eye(4)

    if HAS_OPEN3D and not force_cpu:
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(source_pts_base)
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(target_pts_base)

        reg = o3d.pipelines.registration.registration_icp(
            pcd_src,
            pcd_tgt,
            max_correspondence_dist_m,
            init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        T = reg.transformation
        fitness = float(reg.fitness)
        rmse = float(reg.inlier_rmse)
        success = fitness > 0.35 and rmse < 0.25
        dx = float(T[0, 3])
        dy = float(T[1, 3])
        dtheta = float(math.atan2(T[1, 0], T[0, 0]))

        return ICPRegistrationResult(
            success=success,
            transformation=T,
            fitness=fitness,
            inlier_rmse_m=rmse,
            correspondence_count=int(fitness * len(source_pts_base)),
            dx_m=dx,
            dy_m=dy,
            dtheta_rad=dtheta
        )
    else:
        # Genuine CPU-only 2D/3D SVD nearest-neighbor registration
        success, T, fitness, rmse, inlier_count = _register_point_clouds_cpu_svd(
            source_pts_base=source_pts_base,
            target_pts_base=target_pts_base,
            init_T=init_T,
            max_correspondence_dist_m=max_correspondence_dist_m
        )
        dx = float(T[0, 3])
        dy = float(T[1, 3])
        dtheta = float(math.atan2(T[1, 0], T[0, 0]))

        return ICPRegistrationResult(
            success=success,
            transformation=T,
            fitness=fitness,
            inlier_rmse_m=rmse,
            correspondence_count=inlier_count,
            dx_m=dx,
            dy_m=dy,
            dtheta_rad=dtheta
        )
