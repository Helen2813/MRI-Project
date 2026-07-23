# run_08_failure_attribution_analysis.py
# ------------------------------------------------------------
# External failure attribution analysis for breast DCE-MRI segmentation.
#
# Purpose:
#   Analyze WHY zero-shot external validation performance dropped.
#   This script does NOT train a model and does NOT need A100/cloud.
#   It uses existing predictions, expert masks, phase metadata, and optional
#   contrast-kinetic features from folder 07.
#
# Run from PowerShell/CMD:
#   python run_08_failure_attribution_analysis.py
#
# Edit only the CONFIG block below if your paths are different.
# ------------------------------------------------------------

from __future__ import annotations

import ast
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import SimpleITK as sitk

# Matplotlib is used only for saving paper-style figures.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import ndimage
    from scipy.stats import spearmanr, pearsonr
except Exception:
    ndimage = None
    spearmanr = None
    pearsonr = None

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
except Exception:
    RandomForestRegressor = None
    RandomForestClassifier = None
    SimpleImputer = None
    KFold = None
    cross_val_score = None
    make_pipeline = None


# =============================================================================
# CONFIG — EDIT ONLY HERE
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")

# External expert masks
GT_DIR = PROJECT_ROOT / "segmentations_2" / "expert"

# Existing external predictions
PRED_V1_DIR = Path(r"C:\nnw\mama_mia_output")       # baseline phase selection
PRED_V2_DIR = Path(r"C:\nnw\mama_mia_output_v2")    # timing-aware phase selection

# Original external DCE images, used only for optional preview overlays
IMAGES_DIR = PROJECT_ROOT / "images_2"

# Phase-selection report produced by 06_external_validation/preprocess_mama_mia_nnunet_v2.py
PHASE_REPORT_CSV = PROJECT_ROOT / "external_manifest" / "phase_selection_report.csv"

# Optional table with acquisition metadata. Used if available.
CLINICAL_TABLE = PROJECT_ROOT / "tables" / "clinical_and_imaging_info.xlsx"

# Optional outputs from folder 07. Used if the script exists/has already been run.
KINETIC_CSV = PROJECT_ROOT / "07_contrast_kinetic_shift_analysis" / "metrics" / "contrast_kinetic_case_features_and_metrics.csv"

# Output folder for this analysis
OUT_ROOT = PROJECT_ROOT / "08_external_failure_attribution_analysis"
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"
PREVIEWS_DIR = OUT_ROOT / "previews"

# Debug option. Use None for all cases. Use e.g. 20 if you only want a fast test.
LIMIT: Optional[int] = None

# Compute expensive surface metrics? Usually not needed because this script focuses on failure attribution.
# Set True only if you want approximate HD95/ASSD recomputed for v2+LCC; it will be slower.
COMPUTE_SURFACE_METRICS = False

# Save overlay previews for worst cases. This is useful for qualitative review.
SAVE_PREVIEWS = True
MAX_PREVIEWS_TOTAL = 36

# Exploratory machine-learning factor attribution.
# This does NOT imply causality; it is only to rank variables associated with Dice.
RUN_EXPLORATORY_RF = True

# If nnU-Net probability .npz files are present, threshold sensitivity can be explored.
# This is skipped automatically if .npz files are missing.
RUN_THRESHOLD_SENSITIVITY_IF_PROBS_EXIST = True
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    for d in [OUT_ROOT, METRICS_DIR, FIGURES_DIR, PREVIEWS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7].upper()
    return path.stem.upper()


def cohort_from_case(case_id: str) -> str:
    u = case_id.upper()
    if u.startswith("DUKE"):
        return "DUKE"
    if u.startswith("ISPY1"):
        return "ISPY1"
    if u.startswith("NACT"):
        return "NACT"
    return "OTHER"


def load_binary(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return (arr > 0).astype(np.uint8)


def load_image_array(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def get_spacing(path: Path) -> Tuple[float, float, float]:
    img = sitk.ReadImage(str(path))
    # SimpleITK order is x,y,z; array order is z,y,x. For volume product, order does not matter.
    sp = img.GetSpacing()
    if len(sp) >= 3:
        return float(sp[0]), float(sp[1]), float(sp[2])
    return 1.0, 1.0, 1.0


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    prec = float(tp / pred.sum()) if pred.sum() > 0 else 0.0
    rec = float(tp / gt.sum()) if gt.sum() > 0 else (1.0 if pred.sum() == 0 else 0.0)
    return prec, rec


def mask_volume_cm3(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    return float(mask.sum() * spacing[0] * spacing[1] * spacing[2] / 1000.0)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if ndimage is None or mask.sum() == 0:
        return mask.astype(np.uint8)
    labeled, n = ndimage.label(mask > 0)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask > 0, labeled, index=np.arange(1, n + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return (labeled == largest_label).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if ndimage is None or mask.sum() == 0:
        return mask.astype(np.uint8)
    labeled, n = ndimage.label(mask > 0)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n + 1):
        comp = labeled == i
        if int(comp.sum()) >= min_voxels:
            out[comp] = 1
    return out


def component_stats(mask: np.ndarray) -> Dict[str, float]:
    total = int(mask.sum())
    if total == 0 or ndimage is None:
        return {
            "n_components": 0,
            "n_components_ge50": 0,
            "n_components_ge100": 0,
            "largest_component_voxels": 0,
            "largest_component_fraction": np.nan,
            "second_largest_component_fraction": np.nan,
        }
    labeled, n = ndimage.label(mask > 0)
    if n == 0:
        return {
            "n_components": 0,
            "n_components_ge50": 0,
            "n_components_ge100": 0,
            "largest_component_voxels": 0,
            "largest_component_fraction": np.nan,
            "second_largest_component_fraction": np.nan,
        }
    sizes = np.asarray(ndimage.sum(mask > 0, labeled, index=np.arange(1, n + 1)), dtype=float)
    sizes_sorted = np.sort(sizes)[::-1]
    largest = float(sizes_sorted[0]) if len(sizes_sorted) else 0.0
    second = float(sizes_sorted[1]) if len(sizes_sorted) > 1 else 0.0
    return {
        "n_components": int(n),
        "n_components_ge50": int((sizes >= 50).sum()),
        "n_components_ge100": int((sizes >= 100).sum()),
        "largest_component_voxels": largest,
        "largest_component_fraction": float(largest / total) if total > 0 else np.nan,
        "second_largest_component_fraction": float(second / total) if total > 0 else np.nan,
    }


def surface_metrics_optional(pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, float]:
    if not COMPUTE_SURFACE_METRICS:
        return {"hd95_mm": np.nan, "assd_mm": np.nan}
    if pred.sum() == 0 or gt.sum() == 0:
        return {"hd95_mm": np.nan, "assd_mm": np.nan}
    try:
        pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
        gt_img = sitk.GetImageFromArray(gt.astype(np.uint8))
        pred_img.SetSpacing(spacing)
        gt_img.SetSpacing(spacing)
        pred_surface = sitk.LabelContour(pred_img) > 0
        gt_surface = sitk.LabelContour(gt_img) > 0
        pred_surf_arr = sitk.GetArrayFromImage(pred_surface).astype(bool)
        gt_surf_arr = sitk.GetArrayFromImage(gt_surface).astype(bool)
        dist_to_gt = sitk.GetArrayFromImage(
            sitk.Abs(sitk.SignedMaurerDistanceMap(gt_img, squaredDistance=False, useImageSpacing=True))
        )
        dist_to_pred = sitk.GetArrayFromImage(
            sitk.Abs(sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True))
        )
        d1 = dist_to_gt[pred_surf_arr]
        d2 = dist_to_pred[gt_surf_arr]
        if len(d1) == 0 or len(d2) == 0:
            return {"hd95_mm": np.nan, "assd_mm": np.nan}
        all_d = np.concatenate([d1, d2])
        return {"hd95_mm": float(np.percentile(all_d, 95)), "assd_mm": float(np.mean(all_d))}
    except Exception:
        return {"hd95_mm": np.nan, "assd_mm": np.nan}


def metrics_for_mask(prefix: str, pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, float]:
    p, r = precision_recall(pred, gt)
    out = {
        f"{prefix}_dice": dice_score(pred, gt),
        f"{prefix}_iou": iou_score(pred, gt),
        f"{prefix}_precision": p,
        f"{prefix}_recall": r,
        f"{prefix}_pred_vol_cm3": mask_volume_cm3(pred, spacing),
        f"{prefix}_pred_voxels": int(pred.sum()),
        f"{prefix}_empty_pred": int(pred.sum() == 0),
    }
    surf = surface_metrics_optional(pred, gt, spacing)
    out[f"{prefix}_hd95_mm"] = surf["hd95_mm"]
    out[f"{prefix}_assd_mm"] = surf["assd_mm"]
    return out


def safe_divide(a: float, b: float) -> float:
    if b is None or b == 0 or pd.isna(b):
        return np.nan
    return float(a / b)


def parse_times_maybe(s) -> List[float]:
    try:
        vals = ast.literal_eval(str(s))
        return [float(x) for x in vals]
    except Exception:
        return []


def add_acquisition_timing_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "times" in out.columns:
        times_col = "times"
    elif "acquisition_times" in out.columns:
        times_col = "acquisition_times"
    else:
        return out

    first_posts = []
    last_posts = []
    acq_spans = []
    n_post = []
    for s in out[times_col]:
        times = parse_times_maybe(s)
        posts = [t for t in times if t > 0]
        first_posts.append(min(posts) if posts else np.nan)
        last_posts.append(max(posts) if posts else np.nan)
        acq_spans.append((max(posts) - min(posts)) if len(posts) >= 2 else np.nan)
        n_post.append(len(posts))
    out["first_post_time_s"] = first_posts
    out["last_post_time_s"] = last_posts
    out["post_contrast_span_s"] = acq_spans
    out["n_post_phases"] = n_post
    return out


def add_phase_group_columns(phase: pd.DataFrame) -> pd.DataFrame:
    out = phase.copy()
    if "case_id" in out.columns:
        out["case_id"] = out["case_id"].astype(str).str.upper()
    if "overall" not in out.columns:
        out["overall"] = "no_phase_report"
    if "early_quality" not in out.columns:
        out["early_quality"] = "unknown"
    if "late_quality" not in out.columns:
        out["late_quality"] = "unknown"

    def strict(row) -> str:
        e = str(row.get("early_quality", "unknown"))
        l = str(row.get("late_quality", "unknown"))
        if e == "good" and l == "good":
            return "both_good"
        if e in ["good", "acceptable"] and l in ["good", "acceptable"]:
            return "compatible_not_both_good"
        return "poor_or_missing"

    out["strict_phase_group"] = out.apply(strict, axis=1)
    out = add_acquisition_timing_columns(out)
    return out


def read_phase_report() -> pd.DataFrame:
    if not PHASE_REPORT_CSV.exists():
        print(f"WARNING: phase report not found: {PHASE_REPORT_CSV}")
        return pd.DataFrame(columns=["case_id", "overall", "strict_phase_group"])
    phase = pd.read_csv(PHASE_REPORT_CSV)
    return add_phase_group_columns(phase)


def read_clinical_metadata() -> pd.DataFrame:
    if not CLINICAL_TABLE.exists():
        return pd.DataFrame()
    try:
        clinical = pd.read_excel(CLINICAL_TABLE, dtype=str)
    except Exception as e:
        print(f"WARNING: could not read clinical table, skipping metadata merge: {e}")
        return pd.DataFrame()
    if "patient_id" not in clinical.columns:
        return pd.DataFrame()
    clinical["case_id"] = clinical["patient_id"].astype(str).str.upper()
    keep_cols = [c for c in clinical.columns if c in [
        "case_id", "dataset", "num_phases", "acquisition_times", "scanner_manufacturer",
        "scanner_model", "field_strength", "slice_thickness", "pixel_spacing", "patient_id"
    ]]
    clinical = clinical[keep_cols].copy()
    clinical = add_acquisition_timing_columns(clinical)
    return clinical


def read_kinetic_features() -> pd.DataFrame:
    if not KINETIC_CSV.exists():
        print(f"Optional kinetic CSV not found, skipping merge: {KINETIC_CSV}")
        return pd.DataFrame()
    try:
        kin = pd.read_csv(KINETIC_CSV)
    except Exception as e:
        print(f"WARNING: could not read kinetic CSV: {e}")
        return pd.DataFrame()
    case_col = None
    for c in ["case_id", "pid", "patient_id"]:
        if c in kin.columns:
            case_col = c
            break
    if case_col is None:
        print("WARNING: kinetic CSV has no case_id/pid/patient_id column, skipping.")
        return pd.DataFrame()
    kin = kin.rename(columns={case_col: "case_id"})
    kin["case_id"] = kin["case_id"].astype(str).str.upper()

    # Prefix kinetic columns to avoid accidental overwrites, except case_id and cohort-like columns.
    rename = {}
    for c in kin.columns:
        if c == "case_id":
            continue
        if c.lower() in ["cohort", "dataset"]:
            rename[c] = f"kin_{c}"
        elif not c.startswith("kin_"):
            rename[c] = f"kin_{c}"
    kin = kin.rename(columns=rename)
    return kin


def evaluate_case(case_id: str) -> Dict[str, object]:
    gt_path = GT_DIR / f"{case_id.lower()}.nii.gz"
    if not gt_path.exists():
        return {"case_id": case_id, "error": "missing_gt"}

    gt = load_binary(gt_path)
    spacing = get_spacing(gt_path)
    gt_vol = mask_volume_cm3(gt, spacing)

    row: Dict[str, object] = {
        "case_id": case_id,
        "cohort": cohort_from_case(case_id),
        "gt_voxels": int(gt.sum()),
        "gt_vol_cm3": gt_vol,
        "gt_empty": int(gt.sum() == 0),
        "voxel_volume_mm3": float(spacing[0] * spacing[1] * spacing[2]),
    }

    for version, pred_dir in [("v1", PRED_V1_DIR), ("v2", PRED_V2_DIR)]:
        pred_path = pred_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            # Try lowercase fallback
            pred_path = pred_dir / f"{case_id.lower()}.nii.gz"
        if not pred_path.exists():
            row[f"{version}_missing_pred"] = 1
            continue

        raw = load_binary(pred_path)
        lcc = keep_largest_component(raw)
        small50 = remove_small_components(raw, 50)
        small100 = remove_small_components(raw, 100)

        row[f"{version}_missing_pred"] = 0
        row.update(metrics_for_mask(f"{version}_raw", raw, gt, spacing))
        row.update(metrics_for_mask(f"{version}_lcc", lcc, gt, spacing))
        row.update(metrics_for_mask(f"{version}_small50", small50, gt, spacing))
        row.update(metrics_for_mask(f"{version}_small100", small100, gt, spacing))

        # Component features for raw mask
        cs = component_stats(raw)
        for k, v in cs.items():
            row[f"{version}_raw_{k}"] = v

    return row


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Volume ratios and deltas
    for version in ["v1", "v2"]:
        for variant in ["raw", "lcc", "small50", "small100"]:
            vol_col = f"{version}_{variant}_pred_vol_cm3"
            dice_col = f"{version}_{variant}_dice"
            if vol_col in out.columns:
                out[f"{version}_{variant}_pred_to_gt_vol_ratio"] = out.apply(
                    lambda r: safe_divide(r.get(vol_col, np.nan), r.get("gt_vol_cm3", np.nan)), axis=1
                )
            if dice_col in out.columns:
                out[f"{version}_{variant}_error_1_minus_dice"] = 1.0 - out[dice_col]

    # Improvements
    if "v2_raw_dice" in out and "v1_raw_dice" in out:
        out["delta_v2_raw_minus_v1_raw_dice"] = out["v2_raw_dice"] - out["v1_raw_dice"]
    if "v2_lcc_dice" in out and "v1_lcc_dice" in out:
        out["delta_v2_lcc_minus_v1_lcc_dice"] = out["v2_lcc_dice"] - out["v1_lcc_dice"]
    if "v2_lcc_dice" in out and "v2_raw_dice" in out:
        out["delta_v2_lcc_minus_v2_raw_dice"] = out["v2_lcc_dice"] - out["v2_raw_dice"]
    if "v1_lcc_dice" in out and "v1_raw_dice" in out:
        out["delta_v1_lcc_minus_v1_raw_dice"] = out["v1_lcc_dice"] - out["v1_raw_dice"]

    # log volume helps correlations/plots
    out["log_gt_vol_cm3"] = np.log10(out["gt_vol_cm3"].astype(float).clip(lower=1e-6))

    # Volume quintiles
    try:
        out["gt_volume_quintile"] = pd.qcut(out["gt_vol_cm3"], 5, labels=["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"], duplicates="drop")
    except Exception:
        out["gt_volume_quintile"] = "unknown"

    # Clinically useful bins
    out["gt_volume_bin"] = pd.cut(
        out["gt_vol_cm3"].astype(float),
        bins=[-np.inf, 1, 5, 20, np.inf],
        labels=["<1 cm3", "1-5 cm3", "5-20 cm3", ">20 cm3"],
    )

    if "v2_raw_pred_to_gt_vol_ratio" in out.columns:
        out["v2_raw_volume_ratio_bin"] = pd.cut(
            out["v2_raw_pred_to_gt_vol_ratio"].astype(float),
            bins=[-np.inf, 0.5, 1.5, 3.0, 10.0, np.inf],
            labels=["under <0.5x", "near 0.5-1.5x", "over 1.5-3x", "severe 3-10x", "extreme >10x"],
        )

    if "v2_raw_n_components" in out.columns:
        out["v2_raw_component_count_bin"] = pd.cut(
            out["v2_raw_n_components"].astype(float),
            bins=[-np.inf, 1, 3, 10, np.inf],
            labels=["1", "2-3", "4-10", ">10"],
        )

    # Failure type based on v2+LCC as current best no-retraining pipeline.
    out["failure_type_v2_lcc"] = out.apply(classify_failure, axis=1)
    out["issue_flags"] = out.apply(issue_flags, axis=1)
    return out


def classify_failure(row: pd.Series) -> str:
    d = row.get("v2_lcc_dice", np.nan)
    p = row.get("v2_lcc_precision", np.nan)
    r = row.get("v2_lcc_recall", np.nan)
    raw_p = row.get("v2_raw_precision", np.nan)
    raw_r = row.get("v2_raw_recall", np.nan)
    ratio = row.get("v2_raw_pred_to_gt_vol_ratio", np.nan)
    gt_vol = row.get("gt_vol_cm3", np.nan)
    ncomp = row.get("v2_raw_n_components", np.nan)
    lcc_gain = row.get("delta_v2_lcc_minus_v2_raw_dice", np.nan)

    if pd.notna(d) and d >= 0.75:
        return "high_success"
    if pd.notna(d) and d >= 0.60:
        return "moderate_success"
    if pd.notna(gt_vol) and gt_vol < 1.0:
        return "small_lesion_failure"
    if pd.notna(r) and r < 0.35 and pd.notna(p) and p >= 0.35:
        return "undersegmentation_or_missed_tumor"
    if pd.notna(p) and p < 0.35 and pd.notna(r) and r >= 0.60:
        return "oversegmentation_false_positive"
    if pd.notna(ratio) and ratio > 3.0 and pd.notna(raw_p) and raw_p < 0.55:
        return "oversegmentation_volume"
    if pd.notna(ncomp) and ncomp > 5 and pd.notna(lcc_gain) and lcc_gain > 0.05:
        return "multi_component_false_positives"
    if pd.notna(p) and p < 0.45 and pd.notna(r) and r < 0.45:
        return "poor_localization"
    if pd.notna(d) and d < 0.50:
        return "low_overlap_domain_boundary"
    return "moderate_unclear"


def issue_flags(row: pd.Series) -> str:
    flags: List[str] = []
    if row.get("gt_vol_cm3", np.inf) < 1.0:
        flags.append("small_gt_volume")
    if row.get("v2_raw_pred_to_gt_vol_ratio", 0) > 3.0:
        flags.append("raw_oversegmentation_volume")
    if row.get("v2_raw_pred_to_gt_vol_ratio", np.inf) < 0.5:
        flags.append("raw_undersegmentation_volume")
    if row.get("v2_raw_n_components", 0) > 5:
        flags.append("many_raw_components")
    if row.get("delta_v2_lcc_minus_v2_raw_dice", 0) > 0.05:
        flags.append("lcc_helped")
    if row.get("delta_v2_lcc_minus_v2_raw_dice", 0) < -0.05:
        flags.append("lcc_hurt")
    if row.get("v2_lcc_precision", 1) < 0.4:
        flags.append("low_precision")
    if row.get("v2_lcc_recall", 1) < 0.4:
        flags.append("low_recall")
    # Kinetic flags, if 07 columns are present. Names are intentionally broad; actual columns may vary.
    for col in row.index:
        lc = col.lower()
        if ("tumor_to" in lc or "tumor_ring" in lc or "tumor_minus" in lc) and pd.notna(row[col]):
            try:
                if float(row[col]) <= 0:
                    flags.append("weak_tumor_to_background_contrast")
                    break
            except Exception:
                pass
    return ";".join(flags) if flags else "none"


def summarize(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    if group_cols:
        iterator = df.groupby(group_cols, dropna=False)
    else:
        iterator = [((), df)]
    for key, sub in iterator:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: key[i] for i, col in enumerate(group_cols)}
        row["n"] = len(sub)
        for c in metric_cols:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce").dropna()
                row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{c}_std"] = float(vals.std()) if len(vals) > 1 else np.nan
                row[f"{c}_median"] = float(vals.median()) if len(vals) else np.nan
                row[f"{c}_q25"] = float(vals.quantile(0.25)) if len(vals) else np.nan
                row[f"{c}_q75"] = float(vals.quantile(0.75)) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_columns_for_correlation(df: pd.DataFrame) -> List[str]:
    excluded = {
        "v1_raw_dice", "v1_lcc_dice", "v2_raw_dice", "v2_lcc_dice",
        "v2_raw_precision", "v2_raw_recall", "v2_lcc_precision", "v2_lcc_recall",
        "v1_raw_precision", "v1_raw_recall", "v1_lcc_precision", "v1_lcc_recall",
    }
    cols = []
    for c in df.columns:
        if c in excluded:
            continue
        if c in ["case_id"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # skip mostly empty columns
            if df[c].notna().sum() >= 30:
                cols.append(c)
    return cols


def correlation_table(df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    rows = []
    if spearmanr is None:
        return pd.DataFrame()
    factors = numeric_columns_for_correlation(df)
    for target in targets:
        if target not in df.columns:
            continue
        for f in factors:
            sub = df[[f, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 30 or sub[f].nunique() < 3 or sub[target].nunique() < 3:
                continue
            try:
                rho, p = spearmanr(sub[f], sub[target])
                pr, pp = pearsonr(sub[f], sub[target]) if pearsonr is not None else (np.nan, np.nan)
                rows.append({
                    "target": target,
                    "factor": f,
                    "n": len(sub),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "abs_spearman_rho": abs(float(rho)),
                })
            except Exception:
                continue
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "abs_spearman_rho"], ascending=[True, False])
    return out


def exploratory_rf(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not RUN_EXPLORATORY_RF or RandomForestRegressor is None:
        return pd.DataFrame(), pd.DataFrame()
    target = "v2_lcc_dice"
    if target not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    candidate_cols = numeric_columns_for_correlation(df)
    # Remove direct leakage / derived metrics too close to Dice. Keep failure descriptors/factors.
    banned_fragments = [
        "dice", "iou", "precision", "recall", "error_1_minus", "hd95", "assd",
        "delta_v2", "delta_v1", "empty_pred",
    ]
    feature_cols = [c for c in candidate_cols if not any(b in c.lower() for b in banned_fragments)]
    if len(feature_cols) < 3:
        return pd.DataFrame(), pd.DataFrame()

    data = df[feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    # Drop columns that are almost fully missing or constant.
    keep = []
    for c in feature_cols:
        if data[c].notna().sum() >= max(30, int(0.25 * len(data))) and data[c].nunique(dropna=True) >= 3:
            keep.append(c)
    if len(keep) < 3:
        return pd.DataFrame(), pd.DataFrame()

    X = data[keep]
    y = data[target].astype(float)

    reg = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=8,
        n_jobs=-1,
    )
    pipe = make_pipeline(SimpleImputer(strategy="median"), reg)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=-1)
    except Exception:
        scores = np.array([np.nan])
    pipe.fit(X, y)
    fitted_reg = pipe.named_steps["randomforestregressor"]
    importances = fitted_reg.feature_importances_
    imp = pd.DataFrame({"feature": keep, "importance": importances}).sort_values("importance", ascending=False)
    perf = pd.DataFrame({
        "target": [target],
        "n_cases": [len(y)],
        "n_features": [len(keep)],
        "cv_r2_mean": [float(np.nanmean(scores))],
        "cv_r2_std": [float(np.nanstd(scores))],
        "note": ["Exploratory association model only; not causal and not for model selection."],
    })

    # Classifier for bottom quartile failures
    try:
        q25 = float(y.quantile(0.25))
        y_cls = (y <= q25).astype(int)
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=43,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf_pipe = make_pipeline(SimpleImputer(strategy="median"), clf)
        clf_pipe.fit(X, y_cls)
        fitted_clf = clf_pipe.named_steps["randomforestclassifier"]
        clf_imp = pd.DataFrame({
            "feature": keep,
            "bottom_quartile_failure_importance": fitted_clf.feature_importances_,
        }).sort_values("bottom_quartile_failure_importance", ascending=False)
        imp = imp.merge(clf_imp, on="feature", how="left")
    except Exception:
        pass

    return imp, perf


# =============================================================================
# Plotting
# =============================================================================

def save_current(fig_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_overall_method_bars(df: pd.DataFrame) -> None:
    methods = [
        ("v1 raw", "v1_raw_dice"),
        ("v1 + LCC", "v1_lcc_dice"),
        ("v2 raw", "v2_raw_dice"),
        ("v2 + LCC", "v2_lcc_dice"),
        ("v2 small50", "v2_small50_dice"),
        ("v2 small100", "v2_small100_dice"),
    ]
    labels, means, stds = [], [], []
    for label, col in methods:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            labels.append(label)
            means.append(vals.mean())
            stds.append(vals.std())
    plt.figure(figsize=(8.5, 4.6))
    plt.bar(labels, means)
    plt.ylabel("Dice")
    plt.title("External segmentation performance by no-retraining pipeline")
    plt.ylim(0, max(0.75, max(means) + 0.08 if means else 0.75))
    for i, m in enumerate(means):
        plt.text(i, m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    save_current(FIGURES_DIR / "fig01_external_pipeline_dice_comparison.png")


def plot_by_cohort_methods(df: pd.DataFrame) -> None:
    methods = ["v1_raw_dice", "v1_lcc_dice", "v2_raw_dice", "v2_lcc_dice"]
    labels = ["v1 raw", "v1 + LCC", "v2 raw", "v2 + LCC"]
    cohorts = [c for c in ["DUKE", "ISPY1", "NACT"] if c in set(df["cohort"])]
    x = np.arange(len(cohorts))
    width = 0.18
    plt.figure(figsize=(9, 5))
    for j, col in enumerate(methods):
        means = []
        for c in cohorts:
            means.append(pd.to_numeric(df.loc[df["cohort"] == c, col], errors="coerce").mean())
        plt.bar(x + (j - 1.5) * width, means, width, label=labels[j])
    plt.xticks(x, cohorts)
    plt.ylabel("Dice")
    plt.title("External Dice by cohort and pipeline")
    plt.legend(frameon=False, fontsize=8)
    save_current(FIGURES_DIR / "fig02_external_dice_by_cohort_and_pipeline.png")


def plot_box_by_group(df: pd.DataFrame, group_col: str, value_col: str, out_name: str, title: str) -> None:
    if group_col not in df.columns or value_col not in df.columns:
        return
    groups = [g for g in df[group_col].dropna().unique()]
    groups = sorted(groups, key=lambda x: str(x))
    vals = [pd.to_numeric(df.loc[df[group_col] == g, value_col], errors="coerce").dropna().values for g in groups]
    vals = [v for v in vals if len(v) > 0]
    if not vals:
        return
    labels = [str(g) for g in groups if len(pd.to_numeric(df.loc[df[group_col] == g, value_col], errors="coerce").dropna()) > 0]
    plt.figure(figsize=(max(6, len(labels) * 1.2), 4.8))
    plt.boxplot(vals, labels=labels, showfliers=False)
    plt.ylabel(value_col.replace("_", " "))
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    save_current(FIGURES_DIR / out_name)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, out_name: str, title: str, x_log: bool = False) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return
    sub = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5:
        return
    plt.figure(figsize=(6, 5))
    x = sub[x_col].astype(float)
    y = sub[y_col].astype(float)
    plt.scatter(x, y, alpha=0.55, s=18)
    if x_log:
        plt.xscale("log")
    plt.xlabel(x_col.replace("_", " "))
    plt.ylabel(y_col.replace("_", " "))
    plt.title(title)
    save_current(FIGURES_DIR / out_name)


def plot_failure_counts(df: pd.DataFrame) -> None:
    if "failure_type_v2_lcc" not in df.columns:
        return
    counts = df["failure_type_v2_lcc"].value_counts().sort_values(ascending=True)
    plt.figure(figsize=(9, max(4, 0.45 * len(counts))))
    plt.barh(counts.index.astype(str), counts.values)
    plt.xlabel("Number of cases")
    plt.title("Failure taxonomy using v2 + LCC predictions")
    for i, v in enumerate(counts.values):
        plt.text(v + 1, i, str(v), va="center", fontsize=8)
    save_current(FIGURES_DIR / "fig07_failure_type_counts.png")


def plot_correlation_bar(corr: pd.DataFrame, target: str, out_name: str) -> None:
    if corr.empty:
        return
    sub = corr[corr["target"] == target].copy()
    if sub.empty:
        return
    sub = sub.sort_values("abs_spearman_rho", ascending=False).head(15).sort_values("spearman_rho")
    plt.figure(figsize=(9, max(4, 0.4 * len(sub))))
    plt.barh(sub["factor"], sub["spearman_rho"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Spearman correlation with " + target)
    plt.title("Top univariable associations with external Dice")
    save_current(FIGURES_DIR / out_name)


def plot_rf_importance(imp: pd.DataFrame) -> None:
    if imp.empty:
        return
    sub = imp.head(20).sort_values("importance")
    plt.figure(figsize=(9, max(4, 0.42 * len(sub))))
    plt.barh(sub["feature"], sub["importance"])
    plt.xlabel("Random forest importance")
    plt.title("Exploratory multivariable factor ranking for v2 + LCC Dice")
    save_current(FIGURES_DIR / "fig11_exploratory_rf_feature_importance.png")


def save_preview(case_id: str, gt: np.ndarray, pred_raw: np.ndarray, pred_lcc: np.ndarray, out_path: Path) -> None:
    # Choose slice with largest GT area, fallback to largest pred area.
    if gt.sum() > 0:
        z = int(np.argmax(gt.reshape(gt.shape[0], -1).sum(axis=1)))
    elif pred_lcc.sum() > 0:
        z = int(np.argmax(pred_lcc.reshape(pred_lcc.shape[0], -1).sum(axis=1)))
    else:
        z = gt.shape[0] // 2

    bg = None
    case_dir = IMAGES_DIR / case_id
    if case_dir.exists():
        files = sorted(case_dir.glob("*.nii.gz"))
        if files:
            try:
                arr = load_image_array(files[0])
                if arr.shape == gt.shape:
                    bg = arr[z]
            except Exception:
                bg = None
    if bg is None:
        bg = np.zeros_like(gt[z], dtype=float)

    plt.figure(figsize=(7, 7))
    plt.imshow(bg, cmap="gray")
    try:
        if gt[z].sum() > 0:
            plt.contour(gt[z], levels=[0.5], linewidths=1.5)
        if pred_raw[z].sum() > 0:
            plt.contour(pred_raw[z], levels=[0.5], linewidths=1.0)
        if pred_lcc[z].sum() > 0:
            plt.contour(pred_lcc[z], levels=[0.5], linewidths=1.0, linestyles="dashed")
    except Exception:
        pass
    plt.title(f"{case_id} | GT / raw / LCC contours | z={z}")
    plt.axis("off")
    save_current(out_path)


def make_previews(df: pd.DataFrame) -> None:
    if not SAVE_PREVIEWS:
        return
    # Select cases: worst, strongest oversegmentation, strongest LCC help, strongest LCC harm.
    selected = []
    if "v2_lcc_dice" in df.columns:
        selected += list(df.sort_values("v2_lcc_dice").head(12)["case_id"])
    if "v2_raw_pred_to_gt_vol_ratio" in df.columns:
        selected += list(df.sort_values("v2_raw_pred_to_gt_vol_ratio", ascending=False).head(8)["case_id"])
    if "delta_v2_lcc_minus_v2_raw_dice" in df.columns:
        selected += list(df.sort_values("delta_v2_lcc_minus_v2_raw_dice", ascending=False).head(8)["case_id"])
        selected += list(df.sort_values("delta_v2_lcc_minus_v2_raw_dice", ascending=True).head(8)["case_id"])

    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for c in selected:
        cu = str(c).upper()
        if cu not in seen:
            seen.add(cu)
            ordered.append(cu)
    ordered = ordered[:MAX_PREVIEWS_TOTAL]

    for case_id in ordered:
        try:
            gt_path = GT_DIR / f"{case_id.lower()}.nii.gz"
            pred_path = PRED_V2_DIR / f"{case_id}.nii.gz"
            if not pred_path.exists():
                pred_path = PRED_V2_DIR / f"{case_id.lower()}.nii.gz"
            if not gt_path.exists() or not pred_path.exists():
                continue
            gt = load_binary(gt_path)
            raw = load_binary(pred_path)
            lcc = keep_largest_component(raw)
            save_preview(case_id, gt, raw, lcc, PREVIEWS_DIR / f"preview_{case_id}_gt_raw_lcc.png")
        except Exception:
            continue


# =============================================================================
# Threshold sensitivity if probability maps exist
# =============================================================================

def read_probability_npz(case_id: str, pred_dir: Path) -> Optional[np.ndarray]:
    # nnU-Net typically writes CASE.npz next to CASE.nii.gz when --save_probabilities is used.
    candidates = [pred_dir / f"{case_id}.npz", pred_dir / f"{case_id.lower()}.npz"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    try:
        data = np.load(path)
        key = "probabilities" if "probabilities" in data.files else data.files[0]
        arr = data[key]
        # Common shape: C, Z, Y, X. Tumor channel is usually 1 for binary task.
        if arr.ndim == 4:
            if arr.shape[0] >= 2:
                return arr[1].astype(np.float32)
            return arr[0].astype(np.float32)
        if arr.ndim == 3:
            return arr.astype(np.float32)
    except Exception:
        return None
    return None


def threshold_sensitivity(case_ids: Iterable[str]) -> pd.DataFrame:
    rows = []
    if not RUN_THRESHOLD_SENSITIVITY_IF_PROBS_EXIST:
        return pd.DataFrame()
    # Quick check: do any probability files exist?
    any_npz = any(PRED_V2_DIR.glob("*.npz"))
    if not any_npz:
        return pd.DataFrame()

    print("Probability files found. Running threshold sensitivity for v2 probabilities...")
    for i, case_id in enumerate(case_ids, start=1):
        gt_path = GT_DIR / f"{case_id.lower()}.nii.gz"
        if not gt_path.exists():
            continue
        prob = read_probability_npz(case_id, PRED_V2_DIR)
        if prob is None:
            continue
        gt = load_binary(gt_path)
        spacing = get_spacing(gt_path)
        for t in THRESHOLDS:
            pred = (prob >= t).astype(np.uint8)
            lcc = keep_largest_component(pred)
            rows.append({
                "case_id": case_id,
                "cohort": cohort_from_case(case_id),
                "threshold": t,
                "raw_dice": dice_score(pred, gt),
                "lcc_dice": dice_score(lcc, gt),
                "raw_precision": precision_recall(pred, gt)[0],
                "raw_recall": precision_recall(pred, gt)[1],
                "lcc_precision": precision_recall(lcc, gt)[0],
                "lcc_recall": precision_recall(lcc, gt)[1],
                "raw_pred_vol_cm3": mask_volume_cm3(pred, spacing),
                "lcc_pred_vol_cm3": mask_volume_cm3(lcc, spacing),
            })
        if i % 50 == 0:
            print(f"  threshold sensitivity {i} cases")
    return pd.DataFrame(rows)


def plot_threshold_sensitivity(thr: pd.DataFrame) -> None:
    if thr.empty:
        return
    summary = thr.groupby("threshold", as_index=False).agg(
        raw_dice_mean=("raw_dice", "mean"),
        lcc_dice_mean=("lcc_dice", "mean"),
        raw_precision_mean=("raw_precision", "mean"),
        raw_recall_mean=("raw_recall", "mean"),
        lcc_precision_mean=("lcc_precision", "mean"),
        lcc_recall_mean=("lcc_recall", "mean"),
    )
    summary.to_csv(METRICS_DIR / "threshold_sensitivity_summary.csv", index=False)

    plt.figure(figsize=(7, 4.8))
    plt.plot(summary["threshold"], summary["raw_dice_mean"], marker="o", label="raw Dice")
    plt.plot(summary["threshold"], summary["lcc_dice_mean"], marker="o", label="LCC Dice")
    plt.xlabel("Probability threshold")
    plt.ylabel("Mean Dice")
    plt.title("v2 probability threshold sensitivity")
    plt.legend(frameon=False)
    save_current(FIGURES_DIR / "fig12_threshold_sensitivity_dice.png")


# =============================================================================
# Article takeaways
# =============================================================================

def write_article_takeaways(df: pd.DataFrame, corr: pd.DataFrame, rf_perf: pd.DataFrame) -> None:
    def mean(col: str) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").mean()) if col in df.columns else float("nan")

    lines = []
    lines.append("08 External failure attribution analysis — article takeaways")
    lines.append("===========================================================")
    lines.append("")
    lines.append(f"Cases evaluated: {len(df)}")
    lines.append(f"v1 raw Dice mean: {mean('v1_raw_dice'):.4f}")
    lines.append(f"v1 + LCC Dice mean: {mean('v1_lcc_dice'):.4f}")
    lines.append(f"v2 raw Dice mean: {mean('v2_raw_dice'):.4f}")
    lines.append(f"v2 + LCC Dice mean: {mean('v2_lcc_dice'):.4f}")
    lines.append(f"v2 raw precision/recall: {mean('v2_raw_precision'):.4f} / {mean('v2_raw_recall'):.4f}")
    lines.append(f"v2 + LCC precision/recall: {mean('v2_lcc_precision'):.4f} / {mean('v2_lcc_recall'):.4f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- LCC/postprocessing increases precision but usually lowers recall, consistent with oversegmentation and false-positive components.")
    lines.append("- Timing-aware phase selection improves performance only modestly; timing deviation alone should not be presented as the sole driver of external degradation.")
    lines.append("- Use this section as a failure-attribution/domain-shift analysis, not as proof of clinical deployment readiness.")
    lines.append("")

    if "failure_type_v2_lcc" in df.columns:
        lines.append("Failure taxonomy counts:")
        counts = df["failure_type_v2_lcc"].value_counts()
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    if not corr.empty:
        lines.append("Top Spearman associations with v2+LCC Dice:")
        sub = corr[corr["target"] == "v2_lcc_dice"].head(10)
        for _, r in sub.iterrows():
            lines.append(f"- {r['factor']}: rho={r['spearman_rho']:.3f}, p={r['spearman_p']:.4g}, n={int(r['n'])}")
        lines.append("")

    if not rf_perf.empty:
        lines.append("Exploratory RF factor model:")
        for _, r in rf_perf.iterrows():
            lines.append(f"- target={r['target']}; CV R2={r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}; features={int(r['n_features'])}")
        lines.append("  Note: association model only; not causal.")
        lines.append("")

    lines.append("Suggested manuscript wording:")
    lines.append("External failure attribution showed that the cross-cohort performance drop was multifactorial. Timing-aware phase selection and connected-component postprocessing produced modest improvements, with the largest effect coming from reduction of false-positive components rather than from timing correction alone. The combination of low precision, relatively preserved recall, high predicted-to-ground-truth volume ratios in failure cases, and cohort-dependent behavior supports oversegmentation under domain shift as a major failure mode. These findings motivate site-specific adaptation or acquisition-time/contrast-kinetic modeling rather than relying on zero-shot deployment.")

    (OUT_ROOT / "article_takeaways_08.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("08 External failure attribution analysis")
    print("========================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root : {OUT_ROOT}")
    print()

    # Check paths
    missing = []
    for p in [GT_DIR, PRED_V1_DIR, PRED_V2_DIR]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: required folders not found:")
        for m in missing:
            print("  ", m)
        raise FileNotFoundError("Required folder(s) missing. Edit CONFIG at top of script.")

    pred_files = sorted(PRED_V2_DIR.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"No v2 predictions found in {PRED_V2_DIR}")
    case_ids = [case_id_from_path(p) for p in pred_files]
    case_ids = sorted(set(case_ids))
    if LIMIT is not None:
        case_ids = case_ids[:LIMIT]
    print(f"Cases to evaluate: {len(case_ids)}")

    rows = []
    for i, case_id in enumerate(case_ids, start=1):
        row = evaluate_case(case_id)
        rows.append(row)
        if i % 50 == 0:
            print(f"  evaluated {i}/{len(case_ids)}")

    df = pd.DataFrame(rows)
    df = df[df.get("error", "") != "missing_gt"].copy() if "error" in df.columns else df.copy()
    if df.empty:
        raise RuntimeError("No cases evaluated. Check GT_DIR/PRED paths.")

    # Merge metadata
    phase = read_phase_report()
    if not phase.empty:
        df = df.merge(phase, on="case_id", how="left", suffixes=("", "_phase"))
    if "overall" not in df.columns:
        df["overall"] = "no_phase_report"
    if "strict_phase_group" not in df.columns:
        df["strict_phase_group"] = "no_phase_report"

    clinical = read_clinical_metadata()
    if not clinical.empty:
        # Avoid duplicate columns already in phase report.
        dup = [c for c in clinical.columns if c != "case_id" and c in df.columns]
        clinical = clinical.rename(columns={c: f"clinical_{c}" for c in dup})
        df = df.merge(clinical, on="case_id", how="left")

    kinetic = read_kinetic_features()
    if not kinetic.empty:
        dup = [c for c in kinetic.columns if c != "case_id" and c in df.columns]
        kinetic = kinetic.rename(columns={c: f"kin_extra_{c}" for c in dup})
        df = df.merge(kinetic, on="case_id", how="left")

    df = add_derived_columns(df)

    # Save case-level table
    case_path = METRICS_DIR / "failure_attribution_case_table.csv"
    df.to_csv(case_path, index=False)
    print(f"Case-level table saved: {case_path}")

    # Summary tables
    metric_cols = [
        "v1_raw_dice", "v1_lcc_dice", "v2_raw_dice", "v2_lcc_dice",
        "v2_raw_precision", "v2_raw_recall", "v2_lcc_precision", "v2_lcc_recall",
        "delta_v2_raw_minus_v1_raw_dice", "delta_v2_lcc_minus_v2_raw_dice",
        "v2_raw_pred_to_gt_vol_ratio", "v2_raw_n_components", "v2_raw_largest_component_fraction",
        "gt_vol_cm3",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    summary_specs = {
        "summary_overall_methods.csv": ([], metric_cols),
        "summary_by_cohort.csv": (["cohort"], metric_cols),
        "summary_by_phase_quality.csv": (["overall"], metric_cols),
        "summary_by_cohort_and_phase_quality.csv": (["cohort", "overall"], metric_cols),
        "summary_by_gt_volume_quintile.csv": (["gt_volume_quintile"], metric_cols),
        "summary_by_gt_volume_bin.csv": (["gt_volume_bin"], metric_cols),
        "summary_by_failure_type.csv": (["failure_type_v2_lcc"], metric_cols),
    }
    if "v2_raw_volume_ratio_bin" in df.columns:
        summary_specs["summary_by_v2_raw_volume_ratio_bin.csv"] = (["v2_raw_volume_ratio_bin"], metric_cols)
    if "v2_raw_component_count_bin" in df.columns:
        summary_specs["summary_by_component_count_bin.csv"] = (["v2_raw_component_count_bin"], metric_cols)

    for filename, (groups, metrics) in summary_specs.items():
        summarize(df, groups, metrics).to_csv(METRICS_DIR / filename, index=False)

    # Useful case lists
    if "v2_lcc_dice" in df.columns:
        df.sort_values("v2_lcc_dice").head(50).to_csv(METRICS_DIR / "top_50_worst_v2_lcc_cases.csv", index=False)
        df.sort_values("v2_lcc_dice", ascending=False).head(50).to_csv(METRICS_DIR / "top_50_best_v2_lcc_cases.csv", index=False)
    if "delta_v2_lcc_minus_v2_raw_dice" in df.columns:
        df.sort_values("delta_v2_lcc_minus_v2_raw_dice", ascending=False).head(50).to_csv(METRICS_DIR / "top_50_cases_where_lcc_helped.csv", index=False)
        df.sort_values("delta_v2_lcc_minus_v2_raw_dice").head(50).to_csv(METRICS_DIR / "top_50_cases_where_lcc_hurt.csv", index=False)
    if "v2_raw_pred_to_gt_vol_ratio" in df.columns:
        df.sort_values("v2_raw_pred_to_gt_vol_ratio", ascending=False).head(50).to_csv(METRICS_DIR / "top_50_oversegmentation_by_volume_ratio.csv", index=False)
        df.sort_values("v2_raw_pred_to_gt_vol_ratio").head(50).to_csv(METRICS_DIR / "top_50_undersegmentation_by_volume_ratio.csv", index=False)

    # Correlation analysis
    corr = correlation_table(df, targets=["v2_lcc_dice", "v2_lcc_precision", "v2_lcc_recall", "delta_v2_lcc_minus_v2_raw_dice"])
    if not corr.empty:
        corr.to_csv(METRICS_DIR / "correlations_with_external_metrics.csv", index=False)

    # Exploratory Random Forest factor attribution
    rf_imp, rf_perf = exploratory_rf(df)
    if not rf_imp.empty:
        rf_imp.to_csv(METRICS_DIR / "exploratory_rf_feature_importance.csv", index=False)
    if not rf_perf.empty:
        rf_perf.to_csv(METRICS_DIR / "exploratory_rf_performance.csv", index=False)

    # Threshold sensitivity if probabilities exist
    thr = threshold_sensitivity(case_ids)
    if not thr.empty:
        thr.to_csv(METRICS_DIR / "threshold_sensitivity_case_level.csv", index=False)
        plot_threshold_sensitivity(thr)

    # Figures
    plot_overall_method_bars(df)
    plot_by_cohort_methods(df)
    plot_box_by_group(df, "cohort", "v2_lcc_dice", "fig03_v2_lcc_dice_by_cohort_boxplot.png", "v2 + LCC Dice by cohort")
    plot_box_by_group(df, "overall", "v2_lcc_dice", "fig04_v2_lcc_dice_by_phase_quality_boxplot.png", "v2 + LCC Dice by phase quality")
    plot_box_by_group(df, "gt_volume_quintile", "v2_lcc_dice", "fig05_v2_lcc_dice_by_gt_volume_quintile.png", "v2 + LCC Dice by tumor-volume quintile")
    plot_scatter(df, "gt_vol_cm3", "v2_lcc_dice", "fig06_gt_volume_vs_v2_lcc_dice.png", "Ground-truth tumor volume vs v2 + LCC Dice", x_log=True)
    plot_scatter(df, "v2_raw_pred_to_gt_vol_ratio", "v2_lcc_dice", "fig08_predicted_to_gt_volume_ratio_vs_dice.png", "Predicted/GT volume ratio vs v2 + LCC Dice", x_log=True)
    plot_scatter(df, "v2_raw_n_components", "delta_v2_lcc_minus_v2_raw_dice", "fig09_components_vs_lcc_gain.png", "Raw connected components vs LCC Dice gain")
    plot_scatter(df, "v2_raw_largest_component_fraction", "delta_v2_lcc_minus_v2_raw_dice", "fig10_largest_component_fraction_vs_lcc_gain.png", "Largest component fraction vs LCC Dice gain")
    if "first_post_time_s" in df.columns:
        plot_scatter(df, "first_post_time_s", "v2_lcc_dice", "fig13_first_post_time_vs_dice.png", "First post-contrast time vs v2 + LCC Dice")
    if "early_dev_s" in df.columns:
        plot_scatter(df, "early_dev_s", "v2_lcc_dice", "fig14_early_timing_deviation_vs_dice.png", "Early timing deviation vs v2 + LCC Dice")
    if "late_dev_s" in df.columns:
        plot_scatter(df, "late_dev_s", "v2_lcc_dice", "fig15_late_timing_deviation_vs_dice.png", "Late timing deviation vs v2 + LCC Dice")

    # Kinetic scatter plots if 07 columns exist. Choose columns by name patterns.
    kinetic_candidates = [c for c in df.columns if c.startswith("kin_") and pd.api.types.is_numeric_dtype(df[c])]
    # Save up to a few interpretable kinetic scatter plots.
    for c in kinetic_candidates:
        lc = c.lower()
        if any(key in lc for key in ["tumor_to", "ring", "auc", "max", "uptake", "wash", "slope"]):
            safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in c)[:80]
            plot_scatter(df, c, "v2_lcc_dice", f"fig16_kinetic_{safe_name}_vs_dice.png", f"{c} vs v2 + LCC Dice")
            # avoid too many figures
            if len(list(FIGURES_DIR.glob("fig16_kinetic_*_vs_dice.png"))) >= 8:
                break

    plot_failure_counts(df)
    plot_correlation_bar(corr, "v2_lcc_dice", "fig17_top_correlations_with_v2_lcc_dice.png")
    plot_rf_importance(rf_imp)
    make_previews(df)

    write_article_takeaways(df, corr, rf_perf)

    print()
    print("DONE")
    print(f"Metrics saved in: {METRICS_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Previews saved in: {PREVIEWS_DIR}")
    print(f"Takeaways: {OUT_ROOT / 'article_takeaways_08.txt'}")
    print()
    print("Open article_takeaways_08.txt first, then inspect:")
    print("  metrics/failure_attribution_case_table.csv")
    print("  metrics/correlations_with_external_metrics.csv")
    print("  figures/fig01_external_pipeline_dice_comparison.png")
    print("  figures/fig07_failure_type_counts.png")


if __name__ == "__main__":
    main()
