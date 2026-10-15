#!/usr/bin/env python3
"""
scripts/generate_figures.py

Batch generation of all 12 IEEE publication figures (F01-F12) and 6 research illustrations (I1-I6).
Outputs vector formats (PDF, SVG) and high-resolution 300+ DPI PNG images to `figures/`.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_localization.visualization.figures import (
    plot_f01_robot_architecture,
    plot_f02_sensor_fusion_pipeline,
    plot_f03_wheel_encoder_measurements,
    plot_f04_odometry_drift,
    plot_f05_progressive_sensor_fusion,
    plot_f06_gyro_bias_effects,
    plot_f07_gnss_outage_recovery,
    plot_f08_wheel_slip_robustness,
    plot_f09_sensor_ablation,
    plot_f10_calibration_nees_nis,
    plot_f11_range_assisted_localization,
    plot_f12_combined_failures,
)
from robot_localization.visualization.illustrations import (
    draw_illustration_i1,
    draw_illustration_i2,
    draw_illustration_i3,
    draw_illustration_i4,
    draw_illustration_i5,
    draw_illustration_i6,
)


def main():
    print("=" * 75)
    print("GENERATING IEEE SCIENTIFIC FIGURES & VECTOR ILLUSTRATIONS")
    print("=" * 75)

    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results"

    # Load saved benchmark traces
    data_file = results_dir / "saved_benchmark_traces.npz"
    metrics_file = results_dir / "quick_metrics.csv"

    if not data_file.exists() or not metrics_file.exists():
        print(f"Error: Required results not found at {data_file}. Run scripts/run_quick.py first.")
        sys.exit(1)

    data = np.load(data_file)

    print("Generating Figure F01: Robot Architecture...")
    plot_f01_robot_architecture(fig_dir)

    print("Generating Figure F02: Sensor Fusion Pipeline...")
    plot_f02_sensor_fusion_pipeline(fig_dir)

    print("Generating Figure F03: Four Wheel Encoder Traces...")
    plot_f03_wheel_encoder_measurements(data, fig_dir)

    print("Generating Figure F04: Four-Wheel Odometry Drift...")
    plot_f04_odometry_drift(data, fig_dir)

    print("Generating Figure F05: Progressive Sensor Fusion...")
    plot_f05_progressive_sensor_fusion(data, fig_dir)

    print("Generating Figure F06: Gyro Bias and Slip Effects...")
    plot_f06_gyro_bias_effects(data, fig_dir)

    print("Generating Figure F07: GNSS Outage Recovery...")
    plot_f07_gnss_outage_recovery(data, fig_dir)

    print("Generating Figure F08: Wheel Slip Robustness...")
    plot_f08_wheel_slip_robustness(fig_dir)

    print("Generating Figure F09: Sensor Configuration Ablation...")
    plot_f09_sensor_ablation(metrics_file, fig_dir)

    print("Generating Figure F10: Calibration NEES and NIS...")
    plot_f10_calibration_nees_nis(fig_dir)

    print("Generating Figure F11: Range-Assisted Localization...")
    plot_f11_range_assisted_localization(fig_dir)

    print("Generating Figure F12: Combined Failure Scenarios...")
    plot_f12_combined_failures(data, fig_dir)

    print("\n--- Generating Educational Vector Illustrations I1 through I6 ---")
    print("Generating Illustration I1: Platform Layout...")
    draw_illustration_i1(fig_dir)

    print("Generating Illustration I2: Odometry Geometry...")
    draw_illustration_i2(fig_dir)

    print("Generating Illustration I3: EKF Predict-Update Cycle...")
    draw_illustration_i3(fig_dir)

    print("Generating Illustration I4: Wheel Slip Mechanics...")
    draw_illustration_i4(fig_dir)

    print("Generating Illustration I5: GNSS Outage Dynamics...")
    draw_illustration_i5(fig_dir)

    print("Generating Illustration I6: Mapped LiDAR Geometry...")
    draw_illustration_i6(fig_dir)

    # Manifest
    manifest = [
        {"id": "F01", "name": "f01_robot_architecture", "description": "Four-wheel mobile robot architecture schematic"},
        {"id": "F02", "name": "f02_sensor_fusion_pipeline", "description": "Multi-sensor EKF fusion block diagram"},
        {"id": "F03", "name": "f03_wheel_encoder_measurements", "description": "Synchronized 4-wheel encoder traces under slip"},
        {"id": "F04", "name": "f04_odometry_drift", "description": "Odometry trajectory drift comparison"},
        {"id": "F05", "name": "f05_progressive_sensor_fusion", "description": "Progressive sensor fusion trajectories (B0, B1, B3, B4)"},
        {"id": "F06", "name": "f06_gyro_bias_effects", "description": "Gyro bias estimation and error mitigation"},
        {"id": "F07", "name": "f07_gnss_outage_recovery", "description": "Error growth and filter recovery across 20s GNSS outage"},
        {"id": "F08", "name": "f08_wheel_slip_robustness", "description": "RMSE vs longitudinal slip ratio across baselines"},
        {"id": "F09", "name": "f09_sensor_ablation", "description": "Ablation bar chart across all 6 baselines"},
        {"id": "F10", "name": "f10_calibration_nees_nis", "description": "Filter consistency evaluation (NEES & NIS)"},
        {"id": "F11", "name": "f11_range_assisted_localization", "description": "Mapped LiDAR range updates in GNSS-denied zones"},
        {"id": "F12", "name": "f12_combined_failures", "description": "Combined stress scenario performance (slip + outage + bias)"},
        {"id": "I01", "name": "illustration_i1_robot_layout", "description": "Robot sensor layout and body coordinate frames"},
        {"id": "I02", "name": "illustration_i2_odometry_geometry", "description": "Skid-steer kinematic geometry and effective track width"},
        {"id": "I03", "name": "illustration_i3_ekf_cycle", "description": "Discrete Extended Kalman Filter state flow"},
        {"id": "I04", "name": "illustration_i4_wheel_slip", "description": "Longitudinal wheel slip vs encoder measurement"},
        {"id": "I05", "name": "illustration_i5_gnss_outage_dynamics", "description": "Covariance growth during GNSS blackout and recovery"},
        {"id": "I06", "name": "illustration_i6_lidar_geometry", "description": "Mapped landmark range-bearing observation geometry"},
    ]
    pd.DataFrame(manifest).to_csv(fig_dir / "figure_manifest.csv", index=False)
    print(f"\nSaved complete figure manifest to: {fig_dir / 'figure_manifest.csv'}")
    print(f"All 18 figures and illustrations exported successfully in PDF, SVG, and PNG.")
    print("=" * 75)


if __name__ == "__main__":
    main()
