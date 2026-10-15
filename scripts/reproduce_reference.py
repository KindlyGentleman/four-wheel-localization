#!/usr/bin/env python3
"""
scripts/reproduce_reference.py

Reproduction and audit of the original PythonRobotics 2D Extended Kalman Filter (EKF)
localization benchmark by Atsushi Sakai et al.

References:
    - Sakai et al., "PythonRobotics: a Python code collection of robotics algorithms",
      arXiv:1808.10703, 2018.
    - Upstream source:
      https://github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/extended_kalman_filter/extended_kalman_filter.py
"""

import math
import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def run_upstream_ekf(sim_time: float = 50.0, dt: float = 0.1, seed: int = 42, fix_noise_math: bool = False):
    """
    Run the 2D EKF localization as implemented in PythonRobotics.

    Args:
        sim_time: Total simulation duration in seconds.
        dt: Time step in seconds.
        seed: Random seed for reproducibility.
        fix_noise_math: If False, reproduces upstream bug where covariance matrix is
                        multiplied directly by randn. If True, uses standard deviations.

    Returns:
        dict containing simulation history arrays and computed RMSE metrics.
    """
    rng = np.random.RandomState(seed)

    # Filter covariances
    Q = np.diag([
        0.1,  # variance of location on x-axis
        0.1,  # variance of location on y-axis
        np.deg2rad(1.0),  # variance of yaw angle
        1.0  # variance of velocity
    ]) ** 2

    R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

    # Upstream simulation noise parameters
    # Note: In upstream, INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
    # which is the covariance matrix.
    input_noise_sigma = np.diag([1.0, np.deg2rad(30.0)])
    gps_noise_sigma = np.diag([0.5, 0.5])

    if not fix_noise_math:
        # Upstream quirk: multiplies squared matrix (covariance) directly by randn
        input_noise_mat = input_noise_sigma ** 2
        gps_noise_mat = gps_noise_sigma ** 2
    else:
        # Correct statistical practice: standard deviation matrix
        input_noise_mat = input_noise_sigma
        gps_noise_mat = gps_noise_sigma
        # In the corrected mode, also ensure R matches the actual GPS noise covariance
        R = gps_noise_mat @ gps_noise_mat.T

    # Initial states: [x, y, yaw, v]^T
    x_true = np.zeros((4, 1))
    x_est = np.zeros((4, 1))
    x_dr = np.zeros((4, 1))
    P_est = np.eye(4)

    def calc_input():
        v = 1.0  # [m/s]
        yawrate = 0.1  # [rad/s]
        return np.array([[v], [yawrate]])

    def motion_model(x, u):
        F = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
        B = np.array([
            [dt * math.cos(x[2, 0]), 0.0],
            [dt * math.sin(x[2, 0]), 0.0],
            [0.0, dt],
            [1.0, 0.0]
        ])
        return F @ x + B @ u

    def observation_model(x):
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        return H @ x

    def jacob_f(x, u):
        yaw = x[2, 0]
        v = u[0, 0]
        return np.array([
            [1.0, 0.0, -dt * v * math.sin(yaw), dt * math.cos(yaw)],
            [0.0, 1.0, dt * v * math.cos(yaw), dt * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def jacob_h():
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

    def ekf_estimation(xEst, PEst, z, u):
        # Predict
        xPred = motion_model(xEst, u)
        jF = jacob_f(xEst, u)
        PPred = jF @ PEst @ jF.T + Q

        # Update
        jH = jacob_h()
        zPred = observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + R
        K = PPred @ jH.T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        return xEst, PEst

    time = 0.0
    history = {
        "time": [],
        "x_true": [],
        "y_true": [],
        "yaw_true": [],
        "v_true": [],
        "x_est": [],
        "y_est": [],
        "yaw_est": [],
        "v_est": [],
        "x_dr": [],
        "y_dr": [],
        "z_x": [],
        "z_y": [],
        "pos_cov_trace": []
    }

    while time <= sim_time:
        time += dt
        u = calc_input()

        # True motion
        x_true = motion_model(x_true, u)

        # Observations
        z = observation_model(x_true) + gps_noise_mat @ rng.randn(2, 1)
        ud = u + input_noise_mat @ rng.randn(2, 1)

        # Dead reckoning (odometry only)
        x_dr = motion_model(x_dr, ud)

        # EKF
        x_est, P_est = ekf_estimation(x_est, P_est, z, ud)

        # Log
        history["time"].append(time)
        history["x_true"].append(x_true[0, 0])
        history["y_true"].append(x_true[1, 0])
        history["yaw_true"].append(x_true[2, 0])
        history["v_true"].append(x_true[3, 0])
        history["x_est"].append(x_est[0, 0])
        history["y_est"].append(x_est[1, 0])
        history["yaw_est"].append(x_est[2, 0])
        history["v_est"].append(x_est[3, 0])
        history["x_dr"].append(x_dr[0, 0])
        history["y_dr"].append(x_dr[1, 0])
        history["z_x"].append(z[0, 0])
        history["z_y"].append(z[1, 0])
        history["pos_cov_trace"].append(np.trace(P_est[:2, :2]))

    # Convert to arrays
    for k in history:
        history[k] = np.array(history[k])

    # Compute errors
    pos_err_ekf = np.hypot(history["x_est"] - history["x_true"], history["y_est"] - history["y_true"])
    pos_err_dr = np.hypot(history["x_dr"] - history["x_true"], history["y_dr"] - history["y_true"])

    history["rmse_pos_ekf"] = float(np.sqrt(np.mean(pos_err_ekf ** 2)))
    history["rmse_pos_dr"] = float(np.sqrt(np.mean(pos_err_dr ** 2)))
    history["max_err_pos_ekf"] = float(np.max(pos_err_ekf))
    history["max_err_pos_dr"] = float(np.max(pos_err_dr))

    return history


def main():
    print("=" * 70)
    print("REPRODUCING AND AUDITING PYTHONROBOTICS 2D EKF LOCALIZATION")
    print("=" * 70)

    # 1. Run exact upstream configuration
    res_upstream = run_upstream_ekf(sim_time=50.0, dt=0.1, seed=42, fix_noise_math=False)
    print(f"[Upstream Exact]   Dead Reckoning RMSE : {res_upstream['rmse_pos_dr']:.4f} m")
    print(f"[Upstream Exact]   EKF Estimated RMSE  : {res_upstream['rmse_pos_ekf']:.4f} m")
    print(f"[Upstream Exact]   Max Error EKF       : {res_upstream['max_err_pos_ekf']:.4f} m")

    # 2. Run statistically corrected configuration
    res_corrected = run_upstream_ekf(sim_time=50.0, dt=0.1, seed=42, fix_noise_math=True)
    print(f"[Corrected Math]   Dead Reckoning RMSE : {res_corrected['rmse_pos_dr']:.4f} m")
    print(f"[Corrected Math]   EKF Estimated RMSE  : {res_corrected['rmse_pos_ekf']:.4f} m")
    print(f"[Corrected Math]   Max Error EKF       : {res_corrected['max_err_pos_ekf']:.4f} m")

    # Save results to CSV / NPZ
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / "reproduce_reference_results.npz",
        time=res_upstream["time"],
        x_true=res_upstream["x_true"],
        y_true=res_upstream["y_true"],
        x_dr=res_upstream["x_dr"],
        y_dr=res_upstream["y_dr"],
        x_est=res_upstream["x_est"],
        y_est=res_upstream["y_est"],
        z_x=res_upstream["z_x"],
        z_y=res_upstream["z_y"],
        rmse_ekf_upstream=res_upstream["rmse_pos_ekf"],
        rmse_dr_upstream=res_upstream["rmse_pos_dr"],
        rmse_ekf_corrected=res_corrected["rmse_pos_ekf"],
        rmse_dr_corrected=res_corrected["rmse_pos_dr"],
    )
    print(f"Saved numerical data to {results_dir / 'reproduce_reference_results.npz'}")

    # Plot figure
    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Plot A: Upstream Trajectory
    ax1 = axes[0]
    ax1.plot(res_upstream["z_x"], res_upstream["z_y"], ".g", markersize=3, alpha=0.3, label="GPS observations")
    ax1.plot(res_upstream["x_true"], res_upstream["y_true"], "-b", linewidth=1.5, label="Ground Truth")
    ax1.plot(res_upstream["x_dr"], res_upstream["y_dr"], "--k", linewidth=1.2, label=f"Dead Reckoning (RMSE={res_upstream['rmse_pos_dr']:.2f}m)")
    ax1.plot(res_upstream["x_est"], res_upstream["y_est"], "-r", linewidth=1.5, label=f"EKF (RMSE={res_upstream['rmse_pos_ekf']:.2f}m)")
    ax1.set_title("(a) PythonRobotics Verbatim Reproduction", fontsize=10)
    ax1.set_xlabel("X Position [m]", fontsize=9)
    ax1.set_ylabel("Y Position [m]", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.axis("equal")
    ax1.legend(loc="best", fontsize=7.5)

    # Plot B: Error vs Time comparison
    ax2 = axes[1]
    err_dr = np.hypot(res_upstream["x_dr"] - res_upstream["x_true"], res_upstream["y_dr"] - res_upstream["y_true"])
    err_ekf = np.hypot(res_upstream["x_est"] - res_upstream["x_true"], res_upstream["y_est"] - res_upstream["y_true"])
    ax2.plot(res_upstream["time"], err_dr, "--k", label="Dead Reckoning Error [m]", linewidth=1.2)
    ax2.plot(res_upstream["time"], err_ekf, "-r", label="EKF Position Error [m]", linewidth=1.5)
    ax2.set_title("(b) Localization Error over Time", fontsize=10)
    ax2.set_xlabel("Time [s]", fontsize=9)
    ax2.set_ylabel("2D Position Error [m]", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig_path = fig_dir / "fig_e0_pythonrobotics_reproduction.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated validation figure: {fig_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
