"""
tests/test_realdata.py

Unit and regression tests for real-data parsing, rosbags interaction,
calibration handling, visual feature tracking degradation, and
the GNSS gate-lockout vs recovery dynamics on M2DGR-plus.
"""

from pathlib import Path
import numpy as np
import pytest

from realdata import (
    load_ground_truth,
    load_calibration,
    inspect_bag,
    create_sample_bag,
    generate_synthetic_scene_image,
    VisualFeatureTracker,
    run_hybrid_experiment,
    RecoveryStrategy,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def test_ground_truth_loading_and_conventions():
    """Verify ground truth parsing, duration, sample count, and scalar-first quaternion."""
    gt = load_ground_truth(cache_dir=DATA_DIR)

    assert gt.sample_count == 233, f"Expected 233 poses, got {gt.sample_count}"
    assert 45.0 <= gt.duration_s <= 48.0, f"Expected ~46.2s, got {gt.duration_s}"
    assert gt.total_distance_m > 7.0, f"Expected distance > 7m, got {gt.total_distance_m}"
    assert gt.poses_2d.shape == (233, 3)
    assert gt.quaternions_wxyz.shape == (233, 4)

    # First pose quaternion: anomaly.txt starts with 1.000000 0.000000 0.000000 0.000000 (scalar first qw=1.0)
    qw_0 = gt.quaternions_wxyz[0, 0]
    assert np.isclose(qw_0, 1.0, atol=1e-3), f"Expected scalar-first qw near 1.0, got {qw_0}"


def test_calibration_parsing_and_matrices():
    """Verify calibration regex parsing handles full-width Chinese commas and yields valid SE(3) matrices."""
    calib = load_calibration(cache_dir=DATA_DIR)

    # Check 4x4 matrix dimensions
    for name in ["rsimu2wheel", "imu2wheel", "lidar2wheel", "rsimu2camera", "lidar2imu", "camera2imu", "lidar2camera"]:
        mat = getattr(calib, name)
        assert mat.shape == (4, 4), f"{name} shape should be (4, 4), got {mat.shape}"
        assert np.isclose(mat[3, 3], 1.0), f"{name} bottom-right element should be 1.0"

    # Check camera intrinsics
    assert 600.0 <= calib.fx <= 610.0
    assert 600.0 <= calib.fy <= 610.0
    assert 320.0 <= calib.cx <= 330.0
    assert 230.0 <= calib.cy <= 235.0


def test_rosbag_inspect_and_read():
    """Verify rosbags pure-Python reader reads valid messages without ROS 1 or ROS 2 installed."""
    bag_path = DATA_DIR / "sample_m2dgrp_anomaly.bag"
    if not bag_path.exists():
        gt = load_ground_truth(cache_dir=DATA_DIR)
        calib = load_calibration(cache_dir=DATA_DIR)
        create_sample_bag(bag_path, gt, calib)

    info = inspect_bag(str(bag_path))
    assert info["total_messages"] > 0
    assert info["duration_s"] > 40.0
    assert "/odom" in info["topics"]
    assert "/imu" in info["topics"]
    assert "/camera/imu" in info["topics"]
    assert "/ublox_driver/receiver_pvt" in info["topics"]
    assert info["topics"]["/odom"]["msgtype"] == "nav_msgs/msg/Odometry"


def test_visual_feature_tracker_degradation():
    """Verify visual feature tracking degradation matches expected quality drop."""
    calib = load_calibration(cache_dir=DATA_DIR)
    tracker = VisualFeatureTracker(calib)

    # Generate test scene images
    img0 = generate_synthetic_scene_image(0.0, np.array([0.0, 0.0, 0.0]))
    img1 = generate_synthetic_scene_image(0.2, np.array([0.1, 0.0, 0.0]))

    # Test clean tracking
    res0 = tracker.process_frame(img0, degradation_mode="clean")
    res1 = tracker.process_frame(img1, degradation_mode="clean")
    assert res1["tracked_count"] > 20
    assert res1["inlier_ratio"] > 0.80

    # Test severe degradation tracking
    tracker_deg = VisualFeatureTracker(calib)
    tracker_deg.process_frame(img0, degradation_mode="severe")
    res_deg = tracker_deg.process_frame(img1, degradation_mode="severe")
    assert res_deg["inlier_ratio"] < res1["inlier_ratio"]


def test_gnss_gate_lockout_and_recovery():
    """Verify that strict gating reproduces gate lockout while temporal consistency recovers."""
    gt = load_ground_truth(cache_dir=DATA_DIR)
    calib = load_calibration(cache_dir=DATA_DIR)

    # 1. Strict Gate (Lockout Reproduction)
    res_strict = run_hybrid_experiment(
        gt=gt,
        calib=calib,
        estimator_type="robust_gated_ekf",
        recovery_strategy=RecoveryStrategy.R0_STRICT_GATE,
        gnss_outage_interval=(14.0, 34.0),
        visual_degradation="clean"
    )
    # Under 20s outage and wheel slip, strict gating gets permanently locked out after outage
    assert not res_strict.recovery_success
    assert np.isnan(res_strict.recovery_time_s)
    assert res_strict.pos_rmse_m > 1.30
    assert res_strict.gnss_acceptance_pct < 60.0

    # 2. Recoverable EKF with Temporal Consistency
    res_recov = run_hybrid_experiment(
        gt=gt,
        calib=calib,
        estimator_type="recoverable_ekf",
        recovery_strategy=RecoveryStrategy.R1_TEMPORAL_CONSISTENCY,
        gnss_outage_interval=(14.0, 34.0),
        visual_degradation="clean"
    )
    # Re-anchors GNSS within ~1.86 seconds and achieves lower RMSE and <0.35m post-recovery error
    assert res_recov.recovery_success
    assert res_recov.recovery_time_s > 0.0
    assert res_recov.recovery_time_s < 3.0
    assert res_recov.pos_rmse_m < 1.15
    assert res_recov.pos_rmse_m < res_strict.pos_rmse_m
    assert res_recov.gnss_acceptance_pct > 90.0
    assert res_recov.false_recoveries == 0
