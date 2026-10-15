"""
robot_localization.visualization.illustrations

Educational and research vector illustrations I1 through I6:
- I1: Robot Sensor Layout & Base Frames
- I2: Four-Wheel Skid-Steer Odometry Geometry
- I3: EKF Predict-Update Cycle & Information Flow
- I4: Wheel Slip Mechanics vs. Encoder Measurement
- I5: GNSS Outage Drift & Convergence Dynamics
- I6: Mapped LiDAR Range-Bearing Geometry
"""

from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from robot_localization.visualization.ieee_style import set_ieee_style


def draw_illustration_i1(output_dir: Path):
    """I1: Detailed 4WD platform layout with sensor frames."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)

    # Base chassis
    ax.add_patch(patches.Rectangle((-0.275, -0.20), 0.55, 0.40, facecolor="#f0f0f0", edgecolor="#222222", lw=1.5))

    # Wheels
    w_locs = [
        (0.10, 0.17, "FL (Enc #1)"),
        (0.10, -0.23, "FR (Enc #2)"),
        (-0.25, 0.17, "RL (Enc #3)"),
        (-0.25, -0.23, "RR (Enc #4)"),
    ]
    for wx, wy, label in w_locs:
        ax.add_patch(patches.Rectangle((wx, wy), 0.15, 0.06, facecolor="#444444", edgecolor="#111111", lw=1.0))
        ax.text(wx + 0.075, wy + 0.03, label, color="white", fontsize=6, ha="center", va="center")

    # Center origin
    ax.plot(0, 0, "k+", markersize=10, markeredgewidth=1.5)
    ax.text(0.02, -0.05, "Chassis Origin $(0,0)$", fontsize=6.5)

    # Sensor frames
    # IMU Frame
    ax.plot(0.0, 0.0, "ro", markersize=5)
    ax.text(0.0, 0.08, "IMU Frame $\\{B\\}$", color="red", fontsize=7, ha="center")

    # GNSS Antenna Offset
    ax.plot(0.05, 0.0, "g^", markersize=6)
    ax.annotate("", xy=(0.05, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.0))
    ax.text(0.06, -0.08, "GNSS Lever-Arm $l=[l_x, l_y]^T$", color="green", fontsize=6.5)

    # LiDAR
    ax.plot(0.20, 0.0, "b*", markersize=8)
    ax.text(0.20, 0.05, "LiDAR Frame $\\{L\\}$", color="blue", fontsize=7, ha="center")

    # Base axes
    ax.annotate("", xy=(0.35, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))
    ax.text(0.37, 0.0, "$+X_B$", color="red", fontsize=8, va="center")

    ax.annotate("", xy=(0.0, 0.28), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=1.5))
    ax.text(0.0, 0.30, "$+Y_B$", color="blue", fontsize=8, ha="center")

    ax.set_xlim(-0.40, 0.45)
    ax.set_ylim(-0.35, 0.38)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Illustration I1: 4WD Robot Sensors & Reference Frames", fontsize=8.5, pad=8)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"illustration_i1_robot_layout.{ext}")
    plt.close()


def draw_illustration_i2(output_dir: Path):
    """I2: Wheel odometry geometry & effective track width."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)

    # Robot body outline
    ax.add_patch(patches.Rectangle((-0.2, -0.18), 0.4, 0.36, facecolor="none", edgecolor="#444444", lw=1.2, ls="--"))

    # Left and right wheel tracks
    ax.plot([-0.15, 0.15], [0.18, 0.18], "b-", lw=3.0, label="Left Wheel Track ($v_L$)")
    ax.plot([-0.15, 0.15], [-0.18, -0.18], "r-", lw=3.0, label="Right Wheel Track ($v_R$)")

    # Velocities
    ax.annotate("", xy=(0.28, 0.18), xytext=(0.15, 0.18),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=1.8))
    ax.text(0.30, 0.18, "$v_L = \\frac{1}{2}(v_{FL} + v_{RL})$", color="blue", fontsize=7.5, va="center")

    ax.annotate("", xy=(0.35, -0.18), xytext=(0.15, -0.18),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.8))
    ax.text(0.37, -0.18, "$v_R = \\frac{1}{2}(v_{FR} + v_{RR})$", color="red", fontsize=7.5, va="center")

    # Center forward speed and rotation
    ax.annotate("", xy=(0.31, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="purple", lw=1.8))
    ax.text(0.33, 0.0, "$v = \\frac{1}{2}(v_L + v_R)$", color="purple", fontsize=7.5, va="center")

    # Track width annotation
    ax.plot([-0.05, -0.05], [-0.18, 0.18], "k|-", lw=1.0)
    ax.text(-0.08, 0.0, "Effective Track Width $b_{\\text{eff}}$\n(Scrubbing Offset: $b_{\\text{eff}} > b$)",
            ha="right", va="center", fontsize=7)

    # Angular rate arc
    arc = patches.Arc((0, 0), 0.25, 0.25, angle=0, theta1=20, theta2=160, color="purple", lw=1.2)
    ax.add_patch(arc)
    ax.annotate("", xy=(-0.11, 0.06), xytext=(-0.11, 0.04),
                arrowprops=dict(arrowstyle="-|>", color="purple", lw=1.2))
    ax.text(0.0, 0.15, "$\\omega = \\frac{v_R - v_L}{b_{\\text{eff}}}$", color="purple", fontsize=7.5, ha="center")

    ax.set_xlim(-0.35, 0.65)
    ax.set_ylim(-0.30, 0.30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Illustration I2: Skid-Steer Kinematics Geometry", fontsize=8.5, pad=8)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"illustration_i2_odometry_geometry.{ext}")
    plt.close()


def draw_illustration_i3(output_dir: Path):
    """I3: EKF Predict-Update cycle."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=300)

    # Prediction Box
    pred_box = patches.FancyBboxPatch((0.05, 0.45), 0.38, 0.45, boxstyle="round,pad=0.03",
                                       facecolor="#e8f0fe", edgecolor="#1a73e8", lw=1.5)
    ax.add_patch(pred_box)
    ax.text(0.24, 0.82, "TIME UPDATE (Prediction)", ha="center", fontsize=8, fontweight="bold", color="#174ea6")
    ax.text(0.24, 0.70, "$\\hat{x}_{k|k-1} = f(\\hat{x}_{k-1|k-1}, u_k)$", ha="center", fontsize=7.5)
    ax.text(0.24, 0.58, "$P_{k|k-1} = F_k P_{k-1} F_k^T + Q_k$", ha="center", fontsize=7.5)
    ax.text(0.24, 0.49, "State Propagates & Uncertainty Grows", ha="center", fontsize=6.5, color="#555555")

    # Correction Box
    corr_box = patches.FancyBboxPatch((0.57, 0.45), 0.38, 0.45, boxstyle="round,pad=0.03",
                                       facecolor="#fce8e6", edgecolor="#d93025", lw=1.5)
    ax.add_patch(corr_box)
    ax.text(0.76, 0.82, "MEASUREMENT UPDATE", ha="center", fontsize=8, fontweight="bold", color="#a50e0e")
    ax.text(0.76, 0.70, "Innovation: $y_k = z_k - h(\\hat{x})$", ha="center", fontsize=7.5)
    ax.text(0.76, 0.59, "Kalman Gain: $K_k = P H^T S_k^{-1}$", ha="center", fontsize=7.5)
    ax.text(0.76, 0.49, "Joseph: $P = (I-KH)P(I-KH)^T + KRK^T$", ha="center", fontsize=6.5)

    # Flow arrows
    ax.annotate("", xy=(0.57, 0.68), xytext=(0.43, 0.68),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5))
    ax.text(0.50, 0.71, "Prior $\\hat{x}^-, P^-$", ha="center", fontsize=7)

    # Return loop
    ax.annotate("", xy=(0.24, 0.45), xytext=(0.76, 0.45),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.35", color="#333333", lw=1.5))
    ax.text(0.50, 0.22, "Posterior $\\hat{x}_k, P_k$ to Next Step", ha="center", fontsize=7.5)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.15, 0.95)
    ax.axis("off")
    ax.set_title("Illustration I3: Discrete Extended Kalman Filter State Flow", fontsize=8.5, pad=8)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"illustration_i3_ekf_cycle.{ext}")
    plt.close()


def draw_illustration_i4(output_dir: Path):
    """I4: Wheel slip mechanics vs encoder rotation."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=300)

    # Wheel rolling on ground
    ground_y = 0.20
    ax.plot([-0.2, 1.2], [ground_y, ground_y], "k-", lw=2.0)
    ax.text(1.22, ground_y, "Ground Plane", fontsize=7, va="center")

    # Wheel at x = 0.3
    r = 0.18
    wheel = patches.Circle((0.3, ground_y + r), r, facecolor="#cccccc", edgecolor="#222222", lw=1.5)
    ax.add_patch(wheel)

    # Wheel rotation arrow
    rot_arc = patches.Arc((0.3, ground_y + r), 0.22, 0.22, angle=0, theta1=45, theta2=315, color="red", lw=1.5)
    ax.add_patch(rot_arc)
    ax.annotate("", xy=(0.3 + 0.08, ground_y + r - 0.07), xytext=(0.3 + 0.06, ground_y + r - 0.08),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))
    ax.text(0.3, ground_y + r, "$\\omega$", ha="center", va="center", color="red", fontsize=9, fontweight="bold")

    # Expected travel vs actual travel
    ax.annotate("", xy=(0.95, ground_y - 0.08), xytext=(0.3, ground_y - 0.08),
                arrowprops=dict(arrowstyle="|-|", color="red", lw=1.2))
    ax.text(0.625, ground_y - 0.14, "Encoder Assumed Travel: $s_{\\text{enc}} = r \\cdot \\theta$", color="red", ha="center", fontsize=7)

    ax.annotate("", xy=(0.70, ground_y + 0.05), xytext=(0.3, ground_y + 0.05),
                arrowprops=dict(arrowstyle="|-|", color="blue", lw=1.2))
    ax.text(0.50, ground_y + 0.08, "Actual Ground Travel: $s_{\\text{ground}} = (1-s) \\cdot r \\cdot \\theta$", color="blue", ha="center", fontsize=7)

    # Slip zone under contact patch
    ax.plot([0.25, 0.35], [ground_y, ground_y], "r-", lw=3.5)
    ax.text(0.3, ground_y - 0.04, "Tire Scrub / Slip Zone ($s > 0$)", color="red", ha="center", fontsize=6.5)

    ax.set_xlim(-0.1, 1.35)
    ax.set_ylim(0.0, 0.60)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Illustration I4: Longitudinal Slip vs. Encoder Measurement", fontsize=8.5, pad=8)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"illustration_i4_wheel_slip.{ext}")
    plt.close()


def draw_illustration_i5(output_dir: Path):
    """I5: GNSS dropout drift and step recovery."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=300)

    t = np.linspace(0, 50, 200)
    # Drift starts at t=15, recovers at t=35
    err = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < 15.0:
            err[i] = 0.15 + 0.05 * np.sin(ti)
        elif ti <= 35.0:
            dt = ti - 15.0
            err[i] = 0.15 + 0.08 * (dt ** 1.3)
        else:
            dt = ti - 35.0
            err[i] = 0.15 + 1.8 * np.exp(-1.5 * dt) + 0.05 * np.sin(ti)

    cov_bound = 0.35 * np.ones_like(t)
    cov_bound[t > 15.0] = 0.35 + 0.12 * (t[t > 15.0] - 15.0)
    cov_bound[t > 35.0] = 0.35 + (cov_bound[t > 35.0][0] - 0.35) * np.exp(-1.2 * (t[t > 35.0] - 35.0))

    ax.plot(t, err, "-", color="#d62728", lw=1.5, label="Actual Position Error")
    ax.plot(t, cov_bound, "--", color="#1f77b4", lw=1.2, label="$2\\sigma$ Filter Uncertainty Bound")
    ax.axvspan(15.0, 35.0, color="#ffdddd", alpha=0.6, label="GNSS Outage Window")

    ax.annotate("Outage Begins:\nUncertainty Grows", xy=(15.0, 0.5), xytext=(8.0, 1.2),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9), fontsize=6.8)
    ax.annotate("GNSS Fix Restored:\nRapid Convergence", xy=(35.5, 0.6), xytext=(37.0, 1.4),
                arrowprops=dict(arrowstyle="->", color="#cc0000", lw=0.9), fontsize=6.8, color="#cc0000")

    ax.set_title("Illustration I5: Filter Uncertainty During GNSS Denial", fontsize=8.5)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel("Error & Uncertainty [m]", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"illustration_i5_gnss_outage_dynamics.{ext}")
    plt.close()


def draw_illustration_i6(output_dir: Path):
    """I6: Mapped LiDAR ranging geometry and bearing."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)

    # Robot at (1, 1)
    rx, ry = 1.0, 1.0
    theta = math.radians(35.0)

    # Landmark at (4, 3)
    lx, ly = 4.0, 3.0
    ax.plot(lx, ly, "s", color="#ff7f0e", markersize=7, zorder=5)
    ax.text(lx + 0.15, ly, "Known Landmark $L_i=[L_x, L_y]^T$", fontsize=7.5, color="#d95f02", va="center")

    # Robot body
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    p_box = np.array([[-0.4, 0.4, 0.4, -0.4, -0.4], [-0.25, -0.25, 0.25, 0.25, -0.25]])
    p_rot = R @ p_box
    ax.plot(rx + p_rot[0, :], ry + p_rot[1, :], "k-", lw=1.2)

    # Sensor position (with offset)
    offset = np.array([0.25, 0.0])
    s_pos = np.array([rx, ry]) + R @ offset
    ax.plot(s_pos[0], s_pos[1], "bo", markersize=5)
    ax.text(s_pos[0] - 0.2, s_pos[1] - 0.3, "LiDAR Origin", fontsize=6.5, color="blue")

    # Range vector to landmark
    ax.plot([s_pos[0], lx], [s_pos[1], ly], "--", color="#2ca02c", lw=1.5)
    mid_x, mid_y = (s_pos[0] + lx)/2, (s_pos[1] + ly)/2
    ax.text(mid_x - 0.2, mid_y + 0.15, "Range $r = \\|L_i - p_{\\text{sensor}}\\|$", color="#2ca02c", fontsize=7.5)

    # Heading reference line
    head_len = 1.8
    ax.annotate("", xy=(s_pos[0] + head_len*math.cos(theta), s_pos[1] + head_len*math.sin(theta)),
                xytext=(s_pos[0], s_pos[1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax.text(s_pos[0] + head_len*math.cos(theta) + 0.1, s_pos[1] + head_len*math.sin(theta),
            "Heading $\\theta$", color="red", fontsize=7.5)

    # Bearing angle arc
    bearing_true = math.atan2(ly - s_pos[1], lx - s_pos[0]) - theta
    arc = patches.Arc(s_pos, 1.2, 1.2, angle=math.degrees(theta),
                      theta1=0, theta2=math.degrees(bearing_true), color="#2ca02c", lw=1.2)
    ax.add_patch(arc)
    ax.text(s_pos[0] + 0.7*math.cos(theta + bearing_true/2),
            s_pos[1] + 0.7*math.sin(theta + bearing_true/2), "$\\beta$", color="#2ca02c", fontsize=8, fontweight="bold")

    ax.set_xlim(0.2, 5.2)
    ax.set_ylim(0.2, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Illustration I6: Mapped Range-Bearing Geometry", fontsize=8.5, pad=8)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(output_dir / f"illustration_i6_lidar_geometry.{ext}")
    plt.close()
