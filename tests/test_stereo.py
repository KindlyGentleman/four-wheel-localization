"""
tests.test_stereo

Unit tests for calibrated stereo camera, synthetic scene generator,
StereoSGBM disparity & metric depth, 3D point clouds, visual odometry,
and ICP point-cloud registration.
"""

import math
import pytest
import numpy as np
import cv2

from robot_localization.sensors.stereo import (
    StereoCameraConfig,
    SyntheticStereoArena,
    StereoDepthPipeline,
    StereoVisualOdometry,
    transform_optical_to_base,
    transform_base_to_optical,
    apply_visual_degradation,
    enhance_stereo_images,
    register_point_clouds_icp,
    R_BASE_TO_OPTICAL,
    R_OPTICAL_TO_BASE
)


def test_coordinate_transforms_roundtrip_and_rep105():
    """Verify REP-105 coordinate frame transforms between camera optical and base_link."""
    offset = (0.20, 0.05, 0.15)

    # Point at [1.0, 0.0, 0.0] in base_link (1m directly in front of robot)
    pt_base = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    pt_opt = transform_base_to_optical(pt_base, mounting_xyz=offset)

    # In optical frame: Z is optical forward (distance along camera viewing axis)
    # Relative to camera mounted at x=0.20, distance forward is 0.80m
    assert math.isclose(pt_opt[0, 2], 0.80, abs_tol=1e-5)
    # Point centered in Y in base_link (y=0.0) relative to mount y=0.05 -> X_opt = -(0.0 - 0.05) = +0.05
    assert math.isclose(pt_opt[0, 0], 0.05, abs_tol=1e-5)

    # Roundtrip check
    pt_base_rec = transform_optical_to_base(pt_opt, mounting_xyz=offset)
    np.testing.assert_allclose(pt_base, pt_base_rec, atol=1e-6)


def test_synthetic_stereo_render_and_disparity():
    """Verify that synthetic scene renders stereo pair and StereoSGBM extracts valid disparity and depth."""
    arena = SyntheticStereoArena(seed=123)
    cfg = StereoCameraConfig(width=320, height=240, fx=300.0, fy=300.0, cx=160.0, cy=120.0, baseline_m=0.10)

    # Robot at origin facing forward along X axis
    pose = np.array([0.0, 0.0, 0.0])
    img_l, img_r, visible = arena.render_stereo_pair(pose, cfg)

    assert img_l.shape == (240, 320)
    assert img_r.shape == (240, 320)
    assert len(visible) > 0, "Expected visible landmarks in scene"

    # Compute depth pipeline
    pipeline = StereoDepthPipeline(cfg)
    res = pipeline.compute_depth_and_points(img_l, img_r)

    assert res.valid_fraction > 0.01, f"Expected valid disparity fraction, got {res.valid_fraction}"
    assert res.median_depth_m > 0.5, f"Expected positive median depth, got {res.median_depth_m}"
    assert len(res.point_cloud_base) > 0, "Expected non-empty point cloud"
    assert res.point_cloud_base.shape[1] == 3


def test_stereo_visual_odometry_forward_motion():
    """Verify that stereo VO recovers known forward motion between consecutive frames."""
    arena = SyntheticStereoArena(seed=42)
    cfg = StereoCameraConfig(width=640, height=480, fx=400.0, fy=400.0, cx=320.0, cy=240.0, baseline_m=0.10)
    vo = StereoVisualOdometry(cfg, max_features=500, min_inliers=8)

    # Frame 0: at origin
    pose_0 = np.array([0.0, 0.0, 0.0])
    img_l0, img_r0, _ = arena.render_stereo_pair(pose_0, cfg)
    res_0 = vo.process_frame(img_l0, img_r0, timestamp=0.0)
    # First frame initializes keypoints, returns invalid motion
    assert not res_0.is_valid

    # Frame 1: moved forward by 0.10m in 0.1s (v = 1.0 m/s)
    pose_1 = np.array([0.10, 0.0, 0.0])
    img_l1, img_r1, _ = arena.render_stereo_pair(pose_1, cfg)
    res_1 = vo.process_frame(img_l1, img_r1, timestamp=0.10)

    assert res_1.is_valid, f"VO failed to track frame 1: inliers={res_1.num_inliers}"
    assert res_1.num_inliers >= 8
    # Check estimated forward displacement dx matches true 0.10m within tolerance
    assert math.isclose(res_1.dx_robot_m, 0.10, abs_tol=0.035), f"Expected dx~0.10m, got {res_1.dx_robot_m:.3f}m"
    # Check yaw rotation is near zero
    assert abs(res_1.dtheta_robot_rad) < 0.05
    # Check velocity
    assert math.isclose(res_1.v_forward_m_s, 1.0, abs_tol=0.35)


def test_vo_tracking_failure_on_blank_images():
    """Verify that VO reports tracking failure cleanly without crashing when texture is absent."""
    cfg = StereoCameraConfig(width=320, height=240)
    vo = StereoVisualOdometry(cfg)

    blank_l = np.full((240, 320), 128, dtype=np.uint8)
    blank_r = np.full((240, 320), 128, dtype=np.uint8)

    res = vo.process_frame(blank_l, blank_r, timestamp=1.0)
    assert not res.is_valid
    assert res.num_inliers == 0
    assert res.dx_robot_m == 0.0


def test_extreme_visual_degradations_and_clahe():
    """Verify synthetic degradation generators and CLAHE contrast enhancement."""
    arena = SyntheticStereoArena(seed=99)
    cfg = StereoCameraConfig(width=320, height=240)
    pose = np.array([0.0, 0.0, 0.0])

    img_l, img_r, _ = arena.render_stereo_pair(pose, cfg)

    # 1. Low illumination
    dark_l, dark_r = apply_visual_degradation(img_l, img_r, "low_illumination")
    assert np.mean(dark_l) < np.mean(img_l) * 0.40

    # Enhance with CLAHE
    enh_l, enh_r = enhance_stereo_images(dark_l, dark_r, "clahe")
    assert np.std(enh_l) > np.std(dark_l), "CLAHE should increase contrast/std"

    # 2. Motion blur
    blur_l, blur_r = apply_visual_degradation(img_l, img_r, "motion_blur")
    assert blur_l.shape == img_l.shape

    # 3. Dust haze
    haze_l, haze_r = apply_visual_degradation(img_l, img_r, "dust_haze")
    assert np.mean(haze_l) > np.mean(img_l)


def test_point_cloud_icp_registration():
    """Verify ICP registration between two synthetic point clouds under small displacement."""
    rng = np.random.RandomState(42)
    # Generate non-degenerate 3D cluster of surface points (e.g. boulders / surface rocks)
    target_pts = rng.uniform(-2.0, 2.0, (400, 3))

    # Displace source by dx = 0.08m, dy = 0.03m
    dx_true = 0.08
    dy_true = 0.03
    source_pts = target_pts.copy()
    source_pts[:, 0] += dx_true
    source_pts[:, 1] += dy_true

    res = register_point_clouds_icp(
        source_pts_base=source_pts,
        target_pts_base=target_pts,
        max_correspondence_dist_m=0.30
    )

    assert res.success
    assert res.fitness > 0.80
    # To map source back to target, dx is -0.08, dy is -0.03
    assert math.isclose(res.dx_m, -dx_true, abs_tol=0.01)
    assert math.isclose(res.dy_m, -dy_true, abs_tol=0.01)
