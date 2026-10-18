#!/usr/bin/env python3
"""
scripts/verify_figures.py

Automated figure quality, format parity, and data integrity verification script.
Verifies:
1. File existence and format parity (.png, .pdf, .svg) across assets, figures, and mirror dirs.
2. Non-zero file sizes and valid raster dimensions.
3. Synchronized underlying benchmark artifacts (CSV, NPZ).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

FIGURE_NAMES = ["figure_main", "figure_realdata", "figure_tradeoffs"]
FORMATS = [".png", ".pdf", ".svg"]
DIRECTORIES = [
    REPO_ROOT / "assets",
    REPO_ROOT / "figures",
    REPO_ROOT / "four-wheel-localization" / "assets",
]

REQUIRED_BENCHMARK_FILES = [
    REPO_ROOT / "results" / "realdata_summary.csv",
    REPO_ROOT / "results" / "realdata_traces.npz",
    REPO_ROOT / "results" / "map_benchmark_summary.csv",
    REPO_ROOT / "results" / "quick_metrics.csv",
    REPO_ROOT / "results" / "saved_benchmark_traces.npz",
    REPO_ROOT / "results" / "data_manifest.json",
]


def verify_figure_files() -> list[str]:
    errors = []
    for fig_name in FIGURE_NAMES:
        for fmt in FORMATS:
            for directory in DIRECTORIES:
                target_file = directory / f"{fig_name}{fmt}"
                if not target_file.exists():
                    errors.append(f"Missing file: {target_file}")
                    continue
                size_bytes = target_file.stat().st_size
                if size_bytes < 1024:
                    errors.append(f"File too small ({size_bytes} bytes): {target_file}")
                
                # Check raster image dimensions
                if fmt == ".png":
                    try:
                        with Image.open(target_file) as img:
                            width, height = img.size
                            if width < 1200 or height < 800:
                                errors.append(f"Low resolution ({width}x{height}): {target_file}")
                    except Exception as exc:
                        errors.append(f"Corrupt PNG image {target_file}: {exc}")
    return errors


def verify_benchmark_data() -> list[str]:
    errors = []
    for req_file in REQUIRED_BENCHMARK_FILES:
        if not req_file.exists():
            errors.append(f"Missing benchmark artifact: {req_file}")
            continue
        if req_file.stat().st_size < 100:
            errors.append(f"Benchmark file suspiciously small: {req_file}")

    # Inspect realdata_traces.npz
    npz_path = REPO_ROOT / "results" / "realdata_traces.npz"
    if npz_path.exists():
        try:
            data = np.load(npz_path)
            required_keys = ["timestamps", "gt_poses", "est_R0_dead_reckoning", "sup_states_R4_recoverable_temporal"]
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing key '{key}' in {npz_path}")
        except Exception as exc:
            errors.append(f"Error loading {npz_path}: {exc}")

    # Inspect realdata_summary.csv
    csv_path = REPO_ROOT / "results" / "realdata_summary.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if len(df) < 10:
                errors.append(f"Expected >=10 rows in {csv_path}, found {len(df)}")
            required_cols = ["configuration", "pos_rmse_m", "recovery_success", "recovery_time_s"]
            for col in required_cols:
                if col not in df.columns:
                    errors.append(f"Missing column '{col}' in {csv_path}")
        except Exception as exc:
            errors.append(f"Error reading {csv_path}: {exc}")

    return errors


def main() -> int:
    print("=" * 80)
    print("AUTOMATED FIGURE QUALITY AND ARTIFACT INTEGRITY VERIFICATION")
    print("=" * 80)

    fig_errors = verify_figure_files()
    data_errors = verify_benchmark_data()
    all_errors = fig_errors + data_errors

    if all_errors:
        print(f"FAILED: Found {len(all_errors)} verification errors:")
        for err in all_errors:
            print(f"  [ERROR] {err}")
        return 1

    print("SUCCESS: All figures exist with full format parity (.png, .pdf, .svg).")
    print("SUCCESS: All raster images meet publication-grade resolution requirements.")
    print("SUCCESS: All benchmark summaries, traces, and manifests are verified intact.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
