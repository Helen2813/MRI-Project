#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_external_phase_subset_home.py

Windows-friendly analysis script with NO command-line parameters.
Edit only the CONFIG block below, then run:

    python analyze_external_phase_subset_home.py

Goal:
  Check whether external validation performance is better on cases whose DCE
  phase timing is compatible with the I-SPY2 training protocol.

This script DOES NOT train a model and DOES NOT run nnU-Net inference.
It only reads already-generated prediction masks and expert masks, then computes
Dice/precision/recall for:
  - v1 raw predictions
  - v1 + largest connected component
  - v2 timing-aware raw predictions
  - v2 timing-aware + largest connected component

It then summarizes results by:
  - cohort: DUKE / ISPY1 / NACT
  - phase quality: good / acceptable / poor / no_times
  - stricter phase group: both_good / compatible_not_both_good / poor_or_missing
  - cohort x phase quality

Use this on your home computer. It is CPU-only and should run in minutes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

warnings.filterwarnings("ignore")

try:
    from scipy import ndimage
    from scipy.stats import spearmanr
except Exception:  # keep script usable even if scipy has issues
    ndimage = None
    spearmanr = None


# =============================================================================
# CONFIG — EDIT ONLY THIS BLOCK IF YOUR PATHS ARE DIFFERENT
# =============================================================================

# v1: original external predictions, usually file-order phase selection
PRED_V1_DIR = Path(r"C:\nnw\mama_mia_output")

# v2: timing-aware external predictions
PRED_V2_DIR = Path(r"C:\nnw\mama_mia_output_v2")

# Expert MAMA-MIA masks
GT_DIR = Path(r"C:\Users\olegk\Desktop\MRI Project\segmentations_2\expert")

# Phase-selection report created by preprocess_mama_mia_nnunet_v2.py
PHASE_REPORT = Path(r"C:\Users\olegk\Desktop\MRI Project\external_manifest\phase_selection_report.csv")

# Output folder
OUT_DIR = Path(r"C:\Users\olegk\Desktop\MRI Project\external_manifest")

# If you only want a fast test, set LIMIT = 20. For full analysis, keep None.
LIMIT: Optional[int] = None

# Largest connected component postprocessing
APPLY_LCC = True


# =============================================================================
# HELPERS
# =============================================================================

def case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def cohort_from_case(case_id: str) -> str:
    c = case_id.upper()
    if c.startswith("DUKE"):
        return "DUKE"
    if c.startswith("ISPY1"):
        return "ISPY1"
    if c.startswith("NACT"):
        return "NACT"
    return "OTHER"


def load_mask(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return (arr > 0).astype(np.uint8)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    denom = pred_sum + gt_sum
    if denom == 0:
        return 1.0
    inter = int(np.logical_and(pred, gt).sum())
    return float(2.0 * inter / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    tp = int(np.logical_and(pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    precision = float(tp / pred_sum) if pred_sum > 0 else 0.0
    recall = float(tp / gt_sum) if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)
    return precision, recall


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if ndimage is None:
        return mask
    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest_label = int(np.argmax(sizes) + 1)
    return (labeled == largest_label).astype(np.uint8)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def classify_strict_phase(row: pd.Series) -> str:
    early_q = str(row.get("early_quality", "")).lower()
    late_q = str(row.get("late_quality", "")).lower()
    overall = str(row.get("overall", "")).lower()

    # Strict: both phases within the "good" timing range.
    if early_q == "good" and late_q == "good":
        return "both_good"

    # Compatible but not perfect: original preprocessing called it good/acceptable.
    if overall in {"good", "acceptable"}:
        return "compatible_not_both_good"

    return "poor_or_missing"


def add_phase_group_columns(phase: pd.DataFrame) -> pd.DataFrame:
    phase = phase.copy()
    phase["case_id"] = phase["case_id"].astype(str).str.upper()

    if "overall" not in phase.columns:
        phase["overall"] = "unknown"

    # Normalize strings
    for col in ["overall", "early_quality", "late_quality"]:
        if col in phase.columns:
            phase[col] = phase[col].astype(str).str.lower()
        else:
            phase[col] = "unknown"

    phase["strict_phase_group"] = phase.apply(classify_strict_phase, axis=1)

    # Numeric timing deviations if present
    for col in ["early_dev_s", "late_dev_s", "early_t_s", "late_t_s"]:
        if col in phase.columns:
            phase[col] = phase[col].apply(safe_float)
        else:
            phase[col] = np.nan

    return phase


def compute_one_case(case_id: str, pred_dir: Path, gt_path: Path, prefix: str) -> Dict[str, float]:
    pred_path = pred_dir / f"{case_id}.nii.gz"
    if not pred_path.exists():
        return {
            f"{prefix}_exists": 0,
            f"{prefix}_raw_dice": np.nan,
            f"{prefix}_raw_precision": np.nan,
            f"{prefix}_raw_recall": np.nan,
            f"{prefix}_lcc_dice": np.nan,
            f"{prefix}_lcc_precision": np.nan,
            f"{prefix}_lcc_recall": np.nan,
        }

    gt = load_mask(gt_path)
    pred = load_mask(pred_path)

    raw_d = dice(pred, gt)
    raw_p, raw_r = precision_recall(pred, gt)

    if APPLY_LCC:
        pred_lcc = keep_largest_component(pred)
    else:
        pred_lcc = pred

    lcc_d = dice(pred_lcc, gt)
    lcc_p, lcc_r = precision_recall(pred_lcc, gt)

    return {
        f"{prefix}_exists": 1,
        f"{prefix}_raw_dice": raw_d,
        f"{prefix}_raw_precision": raw_p,
        f"{prefix}_raw_recall": raw_r,
        f"{prefix}_lcc_dice": lcc_d,
        f"{prefix}_lcc_precision": lcc_p,
        f"{prefix}_lcc_recall": lcc_r,
    }


def summarize(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [("ALL", df)]

    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row["n"] = len(sub)
        for m in metric_cols:
            vals = sub[m].dropna()
            row[f"{m}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{m}_std"] = vals.std() if len(vals) > 1 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(title: str, table: pd.DataFrame, metric_cols: List[str]) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)
    if table.empty:
        print("No rows")
        return

    display = table.copy()
    for col in display.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    print(display.to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading phase report...")
    if not PHASE_REPORT.exists():
        raise FileNotFoundError(f"Phase report not found: {PHASE_REPORT}")
    phase = pd.read_csv(PHASE_REPORT)
    phase = add_phase_group_columns(phase)

    print("Finding prediction files...")
    pred_files = sorted(PRED_V2_DIR.glob("*.nii.gz"))
    if not pred_files:
        print(f"No v2 predictions found in {PRED_V2_DIR}")
        print("Trying v1 folder instead, but v2-specific columns will be missing.")
        pred_files = sorted(PRED_V1_DIR.glob("*.nii.gz"))

    case_ids = [case_id_from_path(p).upper() for p in pred_files]
    if LIMIT is not None:
        case_ids = case_ids[:LIMIT]

    print(f"Cases to evaluate: {len(case_ids)}")

    rows = []
    missing_gt = 0

    for i, case_id in enumerate(case_ids, start=1):
        gt_path = GT_DIR / f"{case_id.lower()}.nii.gz"
        if not gt_path.exists():
            missing_gt += 1
            continue

        row: Dict[str, object] = {
            "case_id": case_id,
            "cohort": cohort_from_case(case_id),
        }
        row.update(compute_one_case(case_id, PRED_V1_DIR, gt_path, "v1"))
        row.update(compute_one_case(case_id, PRED_V2_DIR, gt_path, "v2"))
        rows.append(row)

        if i % 50 == 0:
            print(f"  evaluated {i}/{len(case_ids)}")

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No cases evaluated. Check paths in CONFIG.")

    merged = metrics.merge(
        phase,
        on="case_id",
        how="left",
        suffixes=("", "_phase"),
    )

    # Fill missing phase labels
    merged["overall"] = merged["overall"].fillna("no_phase_report")
    merged["strict_phase_group"] = merged["strict_phase_group"].fillna("no_phase_report")

    # Case-level improvements
    merged["delta_v2raw_minus_v1raw"] = merged["v2_raw_dice"] - merged["v1_raw_dice"]
    merged["delta_v2lcc_minus_v1lcc"] = merged["v2_lcc_dice"] - merged["v1_lcc_dice"]
    merged["delta_v2lcc_minus_v2raw"] = merged["v2_lcc_dice"] - merged["v2_raw_dice"]

    case_csv = OUT_DIR / "external_phase_subset_case_metrics.csv"
    merged.to_csv(case_csv, index=False)

    metric_cols = [
        "v1_raw_dice",
        "v1_lcc_dice",
        "v2_raw_dice",
        "v2_lcc_dice",
        "v2_raw_precision",
        "v2_raw_recall",
        "v2_lcc_precision",
        "v2_lcc_recall",
        "delta_v2raw_minus_v1raw",
        "delta_v2lcc_minus_v1lcc",
        "delta_v2lcc_minus_v2raw",
    ]

    tables = {
        "overall": summarize(merged, [], metric_cols),
        "by_overall_phase_quality": summarize(merged, ["overall"], metric_cols),
        "by_strict_phase_group": summarize(merged, ["strict_phase_group"], metric_cols),
        "by_cohort": summarize(merged, ["cohort"], metric_cols),
        "by_cohort_and_phase_quality": summarize(merged, ["cohort", "overall"], metric_cols),
        "by_cohort_and_strict_phase": summarize(merged, ["cohort", "strict_phase_group"], metric_cols),
    }

    for name, table in tables.items():
        out = OUT_DIR / f"external_phase_subset_summary_{name}.csv"
        table.to_csv(out, index=False)

    print()
    print("DONE")
    print(f"Missing GT masks: {missing_gt}")
    print(f"Case-level metrics saved: {case_csv}")
    print(f"Summary tables saved in: {OUT_DIR}")

    print_summary("OVERALL", tables["overall"], metric_cols)
    print_summary("BY PHASE QUALITY: good / acceptable / poor", tables["by_overall_phase_quality"], metric_cols)
    print_summary("BY STRICT PHASE GROUP", tables["by_strict_phase_group"], metric_cols)
    print_summary("BY COHORT", tables["by_cohort"], metric_cols)
    print_summary("BY COHORT x PHASE QUALITY", tables["by_cohort_and_phase_quality"], metric_cols)

    # Spearman correlations: do larger timing deviations correspond to lower Dice?
    print()
    print("=" * 90)
    print("TIMING-DEVIATION CORRELATION WITH v2+LCC DICE")
    print("=" * 90)
    if spearmanr is None:
        print("scipy.stats.spearmanr not available, skipping correlation.")
    else:
        for dev_col in ["early_dev_s", "late_dev_s"]:
            if dev_col in merged.columns:
                sub = merged[[dev_col, "v2_lcc_dice"]].dropna()
                sub = sub[np.isfinite(sub[dev_col]) & np.isfinite(sub["v2_lcc_dice"])]
                if len(sub) >= 10:
                    rho, p = spearmanr(sub[dev_col], sub["v2_lcc_dice"])
                    print(f"{dev_col}: Spearman rho={rho:.3f}, p={p:.4g}, n={len(sub)}")

    print()
    print("How to read this:")
    print("  1) If 'both_good' or 'good' cases have clearly higher Dice than poor cases, timing mismatch likely matters.")
    print("  2) Check COHORT x PHASE QUALITY to avoid confounding: DUKE/ISPY1/NACT differ strongly.")
    print("  3) v2_lcc_dice is the current best no-retraining external pipeline.")


if __name__ == "__main__":
    main()
