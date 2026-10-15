"""
robot_localization.visualization.ieee_style

IEEE Transactions and Journal publication graphics styling configuration.
Provides seamless scienceplots integration with an automatic robust standalone
Matplotlib fallback when LaTeX or system fonts are unavailable.
"""

import matplotlib.pyplot as plt


def set_ieee_style(use_latex: bool = False):
    """
    Apply IEEE publication styling rules:
    - Column widths: Single column ~3.5 in, Double column ~7.16 in
    - Font size: 8-9pt for labels, 7-8pt for ticks and legends, 10pt for titles
    - Clean axes with subtle grid lines
    - Colorblind-safe publication palette
    """
    try:
        if use_latex:
            plt.style.use(["science", "ieee"])
        else:
            plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        # Robust standalone fallback without scienceplots dependency
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": True,
            "legend.edgecolor": "0.8",
            "legend.fancybox": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "text.usetex": False,
        })


# Consistent research palette
IEEE_PALETTE = {
    "ground_truth": "#1f77b4",       # Solid Blue
    "odometry": "#2ca02c",           # Green / Charcoal
    "b0_dr": "#7f7f7f",              # Gray
    "b1_gyro": "#ff7f0e",            # Orange
    "b2_gnss": "#17becf",            # Cyan
    "b3_ekf": "#d62728",             # Crimson Red
    "b4_compass": "#9467bd",         # Purple
    "b5_bias": "#8c564b",            # Brown
    "b6_lidar": "#e377c2",           # Pink / Magenta
    "gnss_meas": "#2ca02c",          # Forest Green
    "outage_shade": "#ffdddd",       # Soft red shading
    "confidence_bound": "#aec7e8",   # Soft blue
}
