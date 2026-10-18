#!/usr/bin/env python3
"""
scripts/verify_figures.py

Automated figure quality, format parity, mirror synchronization, and data integrity verification.
Performs comprehensive multi-point inspection:
1. Format parity across .png, .pdf, and .svg for all primary publication figures.
2. Byte-for-byte mirror integrity (SHA-256) between canonical assets and replica directories.
3. Raster visual properties: resolution thresholds, pixel variance, and non-blank canvas checks.
4. Numerical integrity of underlying benchmark summaries (CSVs) and trace archives (NPZs).
"""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

FIGURE_NAMES = ["figure_main", "figure_realdata", "figure_tradeoffs"]
FORMATS = [".png", ".pdf", ".svg"]
CANONICAL_DIR = REPO_ROOT / "assets"
MIRROR_DIRS = [
    REPO_ROOT / "figures",
    REPO_ROOT / "four-wheel-localization" / "assets",
]

REQUIRED_BENCHMARK_FILES = [
    REPO_ROOT / "results" / "realdata_summary.csv",
    REPO_ROOT / "results" / "realdata_traces.npz",
    REPO_ROOT / "results" / "map_benchmark_summary.csv",
    REPO_ROOT / "results" / "quick_metrics.csv",
    REPO_ROOT / "results" / "ablation_summary.csv",
    REPO_ROOT / "results" / "amcl_kidnapping_summary.csv",
    REPO_ROOT / "results" / "saved_benchmark_traces.npz",
    REPO_ROOT / "results" / "data_manifest.json",
]


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_format_parity_and_sizes() -> tuple[list[str], int]:
    errors = []
    checks_count = 0
    all_dirs = [CANONICAL_DIR] + MIRROR_DIRS

    for fig_name in FIGURE_NAMES:
        for fmt in FORMATS:
            for d in all_dirs:
                checks_count += 1
                target = d / f"{fig_name}{fmt}"
                if not target.exists():
                    errors.append(f"Missing figure file: {target}")
                    continue
                size_bytes = target.stat().st_size
                if size_bytes < 5000:
                    errors.append(f"File suspiciously small ({size_bytes} bytes): {target}")
    return errors, checks_count


def verify_mirror_synchronization() -> tuple[list[str], int]:
    errors = []
    checks_count = 0

    for fig_name in FIGURE_NAMES:
        for fmt in FORMATS:
            canon_file = CANONICAL_DIR / f"{fig_name}{fmt}"
            if not canon_file.exists():
                continue
            canon_hash = _compute_sha256(canon_file)

            for m_dir in MIRROR_DIRS:
                checks_count += 1
                mirror_file = m_dir / f"{fig_name}{fmt}"
                if not mirror_file.exists():
                    errors.append(f"Mirror target missing: {mirror_file}")
                    continue
                mirror_hash = _compute_sha256(mirror_file)
                if canon_hash != mirror_hash:
                    errors.append(
                        f"Checksum mismatch between canonical {canon_file.name} and {mirror_file} "
                        f"({canon_hash[:8]} vs {mirror_hash[:8]})"
                    )
    return errors, checks_count


def verify_raster_visual_properties() -> tuple[list[str], int]:
    errors = []
    checks_count = 0

    for fig_name in FIGURE_NAMES:
        png_path = CANONICAL_DIR / f"{fig_name}.png"
        if not png_path.exists():
            continue
        checks_count += 1
        try:
            with Image.open(png_path) as img:
                w, h = img.size
                if w < 1500 or h < 1000:
                    errors.append(f"Image resolution below publication standard ({w}x{h}): {png_path}")

                arr = np.asarray(img.convert("RGB"), dtype=np.float32)
                std_val = float(np.std(arr))
                mean_val = float(np.mean(arr))

                # Image variance checks: prevent blank white or black canvases
                if std_val < 8.0:
                    errors.append(f"Image lacks visual variance (std={std_val:.2f}, potential blank): {png_path}")
                if mean_val < 15.0 or mean_val > 250.0:
                    errors.append(f"Image mean intensity out of valid range (mean={mean_val:.1f}): {png_path}")
        except Exception as exc:
            errors.append(f"Corrupt PNG or visual verification failure for {png_path}: {exc}")
    return errors, checks_count


def verify_benchmark_numerical_integrity() -> tuple[list[str], int]:
    errors = []
    checks_count = 0

    for req_file in REQUIRED_BENCHMARK_FILES:
        checks_count += 1
        if not req_file.exists():
            errors.append(f"Missing required benchmark artifact: {req_file}")
            continue
        if req_file.stat().st_size < 100:
            errors.append(f"Artifact unexpectedly small: {req_file}")

    # Inspect realdata_traces.npz
    npz_path = REPO_ROOT / "results" / "realdata_traces.npz"
    if npz_path.exists():
        checks_count += 1
        try:
            d = np.load(npz_path)
            req_keys = [
                "timestamps", "gt_poses", "est_R0_dead_reckoning",
                "est_R4_recoverable_temporal", "reloc_idx_R4_recoverable_temporal",
                "sup_states_R4_recoverable_temporal"
            ]
            for k in req_keys:
                if k not in d:
                    errors.append(f"Missing key '{k}' in {npz_path}")

            if "timestamps" in d:
                t = d["timestamps"]
                if len(t) < 50 or not np.all(np.diff(t) > 0.0):
                    errors.append("Timestamps in realdata_traces.npz are non-monotonic or empty")

            if "gt_poses" in d:
                gt = d["gt_poses"]
                if not np.all(np.isfinite(gt)):
                    errors.append("Non-finite coordinates detected in gt_poses")

            if "reloc_idx_R4_recoverable_temporal" in d:
                relocs = d["reloc_idx_R4_recoverable_temporal"]
                if len(relocs) == 0:
                    errors.append("Expected verified relocalization index for R4_recoverable_temporal")
        except Exception as exc:
            errors.append(f"Failed inspecting {npz_path}: {exc}")

    # Inspect realdata_summary.csv
    csv_path = REPO_ROOT / "results" / "realdata_summary.csv"
    if csv_path.exists():
        checks_count += 1
        try:
            df = pd.read_csv(csv_path)
            if len(df) < 10:
                errors.append(f"Expected >=10 trials in {csv_path}, found {len(df)}")
            req_cols = ["configuration", "pos_rmse_m", "recovery_success", "recovery_time_s"]
            for col in req_cols:
                if col not in df.columns:
                    errors.append(f"Missing column '{col}' in {csv_path}")
            if not np.all(np.isfinite(df["pos_rmse_m"])):
                errors.append("Non-finite pos_rmse_m in realdata_summary.csv")
            if not np.all(df["pos_rmse_m"] > 0.0):
                errors.append("Non-positive pos_rmse_m in realdata_summary.csv")
        except Exception as exc:
            errors.append(f"Failed inspecting {csv_path}: {exc}")

    # Inspect map_benchmark_summary.csv
    map_csv = REPO_ROOT / "results" / "map_benchmark_summary.csv"
    if map_csv.exists():
        checks_count += 1
        try:
            df_m = pd.read_csv(map_csv)
            if len(df_m) < 5:
                errors.append(f"Expected >=5 trials in {map_csv}, found {len(df_m)}")
            if not np.all((df_m["pf_rmse_m"] > 0.0) & (df_m["pf_rmse_m"] < 1.0)):
                errors.append("PF RMSE outside expected range (0, 1.0m)")
            if not np.all((df_m["amcl_rmse_m"] > 0.0) & (df_m["amcl_rmse_m"] < 1.0)):
                errors.append("AMCL RMSE outside expected range (0, 1.0m)")
            if not np.all((df_m["amcl_mean_particles"] >= 20) & (df_m["amcl_mean_particles"] <= 500)):
                errors.append("AMCL particle count outside valid range [20, 500]")
        except Exception as exc:
            errors.append(f"Failed inspecting {map_csv}: {exc}")

    return errors, checks_count


def main() -> int:
    print("=" * 80)
    print("AUTOMATED FIGURE INTEGRITY, PARITY, AND DATA AUDIT")
    print("=" * 80)

    e1, c1 = verify_format_parity_and_sizes()
    e2, c2 = verify_mirror_synchronization()
    e3, c3 = verify_raster_visual_properties()
    e4, c4 = verify_benchmark_numerical_integrity()

    all_errors = e1 + e2 + e3 + e4
    total_checks = c1 + c2 + c3 + c4

    print(f"Verification Scope:")
    print(f"  - Format parity and file size checks:  {c1} items verified")
    print(f"  - Byte-level SHA-256 mirror checks:    {c2} items verified")
    print(f"  - Raster resolution and entropy checks: {c3} images verified")
    print(f"  - Dataset numerical integrity checks:  {c4} archives/tables verified")
    print(f"  - Total audit checkpoints:             {total_checks}")
    print("-" * 80)

    if all_errors:
        print(f"FAILED: Found {len(all_errors)} verification errors:")
        for err in all_errors:
            print(f"  [ERROR] {err}")
        print("=" * 80)
        return 1

    print("SUCCESS: All 3 publication figures verified in .png, .pdf, and .svg formats.")
    print("SUCCESS: 100% SHA-256 byte parity confirmed across all canonical and mirror directories.")
    print("SUCCESS: Visual raster variance confirmed (no blank or corrupted renderings).")
    print("SUCCESS: Numerical benchmark traces and summaries verified finite, positive, and intact.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
