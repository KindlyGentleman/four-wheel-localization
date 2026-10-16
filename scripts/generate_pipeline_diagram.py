#!/usr/bin/env python3
"""
scripts/generate_pipeline_diagram.py

Generates publication-quality, editable vector architecture diagrams of the
multi-sensor fusion and state estimation pipeline:
- assets/pipeline.svg
- assets/pipeline.pdf
- assets/pipeline.png (300 DPI)

Strict scientific rules:
1. Ground truth flows ONLY into Physical Simulation and Evaluation, NEVER into estimators.
2. Two distinct estimator branches:
   - Branch A: Filter-Based Multi-Sensor Fusion (EKF, UKF, Robust EKF)
   - Branch B: Known-Map Particle Localization (Fixed PF, Adaptive AMCL)
3. Zero arrow crossings over component boxes.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def draw_card(ax, x, y, w, h, title, items, header_bg, body_bg, border_color, title_color="#0F172A"):
    """Draw a clean publication card with header banner and structured bullet items."""
    # Outer box
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        facecolor=body_bg,
        edgecolor=border_color,
        linewidth=1.1,
        zorder=2
    )
    ax.add_patch(box)

    # Header banner
    h_height = 0.20 * h
    header_box = FancyBboxPatch(
        (x, y + h - h_height), w, h_height,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        facecolor=header_bg,
        edgecolor="none",
        zorder=3
    )
    ax.add_patch(header_box)

    # Header text
    ax.text(
        x + 0.5 * w, y + h - 0.5 * h_height,
        title,
        ha="center", va="center",
        fontsize=8.5, fontweight="bold",
        color=title_color,
        zorder=4
    )

    # Body bullet text
    text_y = y + h - h_height - 0.10 * h
    spacing = (h - h_height - 0.12 * h) / max(len(items), 1)

    for item in items:
        ax.text(
            x + 0.06 * w, text_y,
            f"• {item}",
            ha="left", va="center",
            fontsize=7.2,
            color="#334155",
            zorder=4
        )
        text_y -= spacing


def draw_arrow(ax, p1, p2, color="#475569", lw=1.2, style="->", rad=0.0, dashed=False):
    """Draw a smooth connecting arrow between components."""
    ls = "--" if dashed else "-"
    arrow = FancyArrowPatch(
        p1, p2,
        arrowstyle=ArrowStyle("Simple", head_length=4.5, head_width=4.0, tail_width=1.0),
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=lw,
        linestyle=ls,
        zorder=5
    )
    ax.add_patch(arrow)


def main():
    fig = plt.figure(figsize=(13.2, 7.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # Title & Subtitle
    ax.text(
        50, 58.0,
        "Sensor Fusion and State Estimation Pipeline for 4WD Mobile Robot",
        ha="center", va="center",
        fontsize=13.5, fontweight="bold",
        color="#0F172A"
    )
    ax.text(
        50, 56.0,
        "Physical Modeling, Asynchronous Preprocessing, Dual Estimator Branches, and Statistical Consistency Verification",
        ha="center", va="center",
        fontsize=8.5, fontstyle="italic",
        color="#475569"
    )

    # -------------------------------------------------------------------------
    # Column 1: Physical Simulation & Sensor Simulation
    # -------------------------------------------------------------------------
    draw_card(
        ax, 3, 31, 18, 22,
        "1. Physical Simulation",
        [
            "Skid-steer 4WD drive command",
            "Left/Right motor speeds",
            "Longitudinal wheel slip s_i",
            "Midpoint pose integration",
            "Ground-Truth Pose x_true",
            "Volcanic obstacle environment"
        ],
        header_bg="#E2E8F0", body_bg="#F8FAFC", border_color="#94A3B8", title_color="#0F172A"
    )

    draw_card(
        ax, 3, 5, 18, 22,
        "2. Sensor Simulation",
        [
            "4x Optical Encoders (50 Hz)",
            "  - Discrete ticks & jitter",
            "MPU-9250 IMU Gyro/Accel (50 Hz)",
            "  - Gyro bias drift & noise",
            "CMPS12 Magnetic Compass (10 Hz)",
            "GNSS Receiver (1 Hz, lever arm)",
            "2D LiDAR Scanner (5 Hz)"
        ],
        header_bg="#DBEAFE", body_bg="#EFF6FF", border_color="#60A5FA", title_color="#1E3A8A"
    )

    # Connect Box 1 -> Box 2
    draw_arrow(ax, (12, 31), (12, 27), color="#0284C7", lw=1.4)
    ax.text(12.8, 29.0, "Physical Motion", ha="left", va="center", fontsize=6.8, color="#0369A1")

    # -------------------------------------------------------------------------
    # Column 2: Preprocessing & Synchronization
    # -------------------------------------------------------------------------
    draw_card(
        ax, 25, 18, 19, 26,
        "3. Preprocessing & Sync",
        [
            "Chronological event scheduler",
            "Encoder tick difference to speed",
            "Kinematic 4-wheel averaging",
            "Disagreement |v_FL - v_RL|",
            "Gyro-wheel slip delta |w_enc - w_g|",
            "Antenna lever-arm transform",
            "Dynamic Q_u noise covariance",
            "Magnetic declination correction"
        ],
        header_bg="#FEF3C7", body_bg="#FFFBEB", border_color="#FBBF24", title_color="#92400E"
    )

    # Connect Box 2 -> Box 3
    draw_arrow(ax, (21, 16), (25, 25), color="#D97706", lw=1.4)
    ax.text(22.3, 22.8, "Raw Packets", ha="center", va="bottom", fontsize=6.5, color="#B45309", rotation=32)

    # -------------------------------------------------------------------------
    # Column 3: Dual Estimator Branches
    # -------------------------------------------------------------------------
    # Branch A: Filter-Based Fusion
    draw_card(
        ax, 48, 31, 24, 22,
        "4. Branch A: Filter Fusion",
        [
            "B0: Kinematic Wheel Dead-Reckoning",
            "B1: Wheel Odometry + Gyro Dead-Reckoning",
            "B5: Bias-Aware EKF [x, y, theta, b_g]^T",
            "A4: Manifold UKF (Circular S^1 averaging)",
            "A5: Robust Innovation-Gated EKF",
            "  - Chi2 innovation validation gates",
            "  - Slip-adaptive process covariance",
            "  - Outage drift inflation"
        ],
        header_bg="#EDE9FE", body_bg="#F5F3FF", border_color="#A78BFA", title_color="#4C1D95"
    )

    # Branch B: Known-Map Particle Localization
    draw_card(
        ax, 48, 5, 24, 22,
        "5. Branch B: Known-Map AMCL",
        [
            "Volcanic Obstacle Testbed Map",
            "Euclidean Distance Transform Field",
            "  - O(1) distance lookup for raycasting",
            "A6: Fixed Particle Filter (N = 200)",
            "  - Low-variance systematic resampling",
            "A7: Adaptive AMCL Filter",
            "  - KLD sampling (N in [60, 300])",
            "  - Dual-rate w_slow/w_fast recovery"
        ],
        header_bg="#D1FAE5", body_bg="#ECFDF5", border_color="#34D399", title_color="#065F46"
    )

    # Connect Box 3 -> Branch A
    draw_arrow(ax, (44, 35), (48, 41), color="#7C3AED", lw=1.4)
    ax.text(45.6, 39.8, "Odom+IMU+GNSS", ha="center", va="bottom", fontsize=6.0, color="#6D28D9", rotation=28)

    # Connect Box 3 -> Branch B
    draw_arrow(ax, (44, 23), (48, 17), color="#059669", lw=1.4)
    ax.text(45.6, 18.2, "Odom+LiDAR", ha="center", va="top", fontsize=6.0, color="#047857", rotation=-28)

    # -------------------------------------------------------------------------
    # Column 4: Statistical Evaluation & Verification Ledger
    # -------------------------------------------------------------------------
    draw_card(
        ax, 76, 5, 21, 48,
        "6. Verification & Evaluation",
        [
            "Ground-Truth Temporal Sync",
            "ATE Position RMSE & MAE [m]",
            "Heading Tracking RMSE [deg]",
            "NEES Statistical Consistency",
            "  - 2D position chi2 check",
            "  - 3D pose chi2 check",
            "NIS Innovation Consistency",
            "  - GNSS 2-DOF health check",
            "  - Compass 1-DOF check",
            "Empirical 95% Coverage",
            "  - Ellipse containment [%]",
            "Outage Decomposition",
            "  - Pre-, in-, and post-outage",
            "Execution Efficiency",
            "  - Per-update CPU time [ms]",
            "  - Real-time performance ratio"
        ],
        header_bg="#FFE4E6", body_bg="#FFF1F2", border_color="#FB7185", title_color="#9F1239"
    )

    # Connect Branch A -> Evaluation
    draw_arrow(ax, (72, 42), (76, 42), color="#BE185D", lw=1.4)
    ax.text(74.0, 43.2, "Pose & Cov", ha="center", va="bottom", fontsize=6.5, color="#9D174D")

    # Connect Branch B -> Evaluation
    draw_arrow(ax, (72, 16), (76, 16), color="#BE185D", lw=1.4)
    ax.text(74.0, 17.2, "Particles", ha="center", va="bottom", fontsize=6.5, color="#9D174D")

    # Ground Truth Connection: From Physical Sim to Evaluation (routed cleanly around bottom)
    draw_arrow(ax, (3, 42), (1.5, 42), color="#475569", lw=1.2, dashed=True)
    draw_arrow(ax, (1.5, 42), (1.5, 2.0), color="#475569", lw=1.2, dashed=True)
    draw_arrow(ax, (1.5, 2.0), (86.5, 2.0), color="#475569", lw=1.2, dashed=True)
    draw_arrow(ax, (86.5, 2.0), (86.5, 5.0), color="#475569", lw=1.2, dashed=True)
    ax.text(45.0, 2.8, "Ground-Truth Pose x_true (Strict Evaluation Only - Zero Estimator Leakage)",
            ha="center", va="bottom", fontsize=7.0, color="#475569", fontstyle="italic")

    # Save outputs
    svg_path = ASSETS_DIR / "pipeline.svg"
    pdf_path = ASSETS_DIR / "pipeline.pdf"
    png_path = ASSETS_DIR / "pipeline.png"

    plt.tight_layout()
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    fw_assets = PROJECT_ROOT / "four-wheel-localization" / "assets"
    if fw_assets.exists():
        import shutil
        shutil.copy2(svg_path, fw_assets / "pipeline.svg")
        shutil.copy2(pdf_path, fw_assets / "pipeline.pdf")
        shutil.copy2(png_path, fw_assets / "pipeline.png")

    print(f"Pipeline architecture diagram successfully generated:")
    print(f"  - SVG: {svg_path}")
    print(f"  - PDF: {pdf_path}")
    print(f"  - PNG: {png_path} (300 DPI)")


if __name__ == "__main__":
    main()
