# run_11_catastrophic_case_review.py
# Final sanity-check and visual-review packet for catastrophic external failures.
# No command-line parameters. Edit only CONFIG below if your paths differ.

from __future__ import annotations

import math
import shutil
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================================================================
# CONFIG — EDIT ONLY HERE IF NEEDED
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")

# Input table from folder 10. This should already exist after running script 10.
CASE_TABLE_CSV = PROJECT_ROOT / r"10_final_external_sanity_checks\metrics\final_sanity_case_table_with_volume_bounds.csv"

# External data/prediction locations.
IMAGES_DIR = PROJECT_ROOT / "images_2"
GT_DIR = PROJECT_ROOT / r"segmentations_2\expert"
PRED_V2_RAW_DIR = Path(r"C:\nnw\mama_mia_output_v2")

# Output folder for this review step.
OUTPUT_ROOT = PROJECT_ROOT / "11_catastrophic_case_review"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REVIEW_PACKET_DIR = OUTPUT_ROOT / "review_packet"
PREVIEW_DIR = REVIEW_PACKET_DIR / "previews"
NIFTI_REVIEW_DIR = REVIEW_PACKET_DIR / "nifti_cases"

# Which cases to save for manual visual review.
# All cases in the catastrophic group will be included.
CATASTROPHIC_GROUP_NAME = "spatial_or_boundary_mismatch_beyond_volume"

# Also save a few comparison examples from other groups, so the qualitative figure
# can show normal and non-catastrophic failures side-by-side.
SAVE_COMPARISON_EXAMPLES = True
N_COMPARISON_PER_GROUP = 6

# If True, save NIfTI copies for the selected review cases.
# This makes it easy to open each case in 3D Slicer / ITK-SNAP.
SAVE_NIFTI_REVIEW_PACKET = True

# If True, also save PNG previews for selected cases.
SAVE_PREVIEW_PNGS = True

# Margin for cropped preview around mask region.
CROP_MARGIN_PIXELS = 35

# Save a full technical table for all 526 cases, plus a selected review packet.
RANDOM_SEED = 42


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_dirs() -> None:
    for d in [OUTPUT_ROOT, METRICS_DIR, FIGURES_DIR, REVIEW_PACKET_DIR, PREVIEW_DIR, NIFTI_REVIEW_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return one column even if duplicate column names exist."""
    out = df[col]
    if isinstance(out, pd.DataFrame):
        return out.iloc[:, 0]
    return out


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_case_id_col(df: pd.DataFrame) -> str:
    c = first_existing_col(df, ["case_id", "patient_id", "pid", "id"])
    if c is None:
        raise KeyError("Could not find a case identifier column. Expected case_id/patient_id/pid/id.")
    return c


def infer_group_col(df: pd.DataFrame) -> Optional[str]:
    exact_candidates = [
        "volume_bound_diagnostic_group",
        "v2_lcc_volume_bound_diagnostic_group",
        "diagnostic_group",
        "volume_bound_group",
        "group",
    ]
    c = first_existing_col(df, exact_candidates)
    if c is not None:
        return c
    for col in df.columns:
        low = col.lower()
        if "group" in low and ("volume" in low or "bound" in low or "diagnostic" in low):
            return col
    return None


def numeric(df: pd.DataFrame, col: Optional[str], default: float = np.nan) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(safe_series(df, col), errors="coerce")


def load_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def sitk_to_array(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


def load_binary(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = load_image(path)
    arr = (sitk_to_array(img) > 0).astype(np.uint8)
    return arr, img


def get_case_paths(case_id: str) -> Dict[str, Optional[Path]]:
    case_upper = case_id.upper()
    case_lower = case_id.lower()

    gt_path = GT_DIR / f"{case_lower}.nii.gz"
    pred_path = PRED_V2_RAW_DIR / f"{case_upper}.nii.gz"

    case_dir = IMAGES_DIR / case_upper
    if not case_dir.exists():
        # Some folders may preserve original casing.
        possible = [p for p in IMAGES_DIR.glob("*") if p.is_dir() and p.name.upper() == case_upper]
        case_dir = possible[0] if possible else case_dir

    phase_files = sorted(case_dir.glob("*.nii.gz")) if case_dir.exists() else []
    pre_path = phase_files[0] if phase_files else None

    return {
        "gt": gt_path if gt_path.exists() else None,
        "pred_v2_raw": pred_path if pred_path.exists() else None,
        "case_dir": case_dir if case_dir.exists() else None,
        "pre": pre_path,
    }


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(np.uint8)
    labeled, n = ndimage.label(mask > 0)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    largest_label = int(np.argmax(sizes) + 1)
    return (labeled == largest_label).astype(np.uint8)


def save_mask_like(mask: np.ndarray, reference_img: sitk.Image, out_path: Path) -> None:
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(reference_img)
    sitk.WriteImage(out, str(out_path), useCompression=True)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred_b, gt_b).sum() / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred_b = pred > 0
    gt_b = gt > 0
    tp = np.logical_and(pred_b, gt_b).sum()
    precision = float(tp / pred_b.sum()) if pred_b.sum() > 0 else 0.0
    recall = float(tp / gt_b.sum()) if gt_b.sum() > 0 else 0.0
    return precision, recall


def n_components(mask: np.ndarray) -> int:
    if mask.sum() == 0:
        return 0
    _, n = ndimage.label(mask > 0)
    return int(n)


def component_stats(mask: np.ndarray) -> Dict[str, float]:
    if mask.sum() == 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0,
            "second_largest_component_voxels": 0,
            "largest_component_fraction": np.nan,
            "second_largest_component_fraction": np.nan,
        }
    labeled, n = ndimage.label(mask > 0)
    if n == 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0,
            "second_largest_component_voxels": 0,
            "largest_component_fraction": np.nan,
            "second_largest_component_fraction": np.nan,
        }
    sizes = np.asarray(ndimage.sum(mask > 0, labeled, index=np.arange(1, n + 1)), dtype=float)
    sizes_sorted = np.sort(sizes)[::-1]
    total = float((mask > 0).sum())
    largest = float(sizes_sorted[0]) if len(sizes_sorted) > 0 else 0.0
    second = float(sizes_sorted[1]) if len(sizes_sorted) > 1 else 0.0
    return {
        "n_components": int(n),
        "largest_component_voxels": largest,
        "second_largest_component_voxels": second,
        "largest_component_fraction": largest / total if total > 0 else np.nan,
        "second_largest_component_fraction": second / total if total > 0 else 0.0,
    }


def mask_centroid_voxel(mask: np.ndarray) -> Tuple[float, float, float]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (np.nan, np.nan, np.nan)
    zyx = coords.mean(axis=0)
    return (float(zyx[0]), float(zyx[1]), float(zyx[2]))


def voxel_to_physical(img: sitk.Image, centroid_zyx: Tuple[float, float, float]) -> Tuple[float, float, float]:
    z, y, x = centroid_zyx
    if any(np.isnan([z, y, x])):
        return (np.nan, np.nan, np.nan)
    try:
        # SimpleITK continuous index order is x, y, z.
        p = img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
        return (float(p[0]), float(p[1]), float(p[2]))
    except Exception:
        return (np.nan, np.nan, np.nan)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if np.any(~np.isfinite(aa)) or np.any(~np.isfinite(bb)):
        return np.nan
    return float(np.linalg.norm(aa - bb))


def image_info(prefix: str, img: Optional[sitk.Image], arr: Optional[np.ndarray] = None) -> Dict[str, object]:
    if img is None:
        return {
            f"{prefix}_exists": False,
            f"{prefix}_shape_zyx": "",
            f"{prefix}_spacing_xyz": "",
            f"{prefix}_origin_xyz": "",
            f"{prefix}_direction": "",
            f"{prefix}_voxels": np.nan,
        }
    if arr is None:
        arr = sitk_to_array(img)
    return {
        f"{prefix}_exists": True,
        f"{prefix}_shape_zyx": str(tuple(arr.shape)),
        f"{prefix}_spacing_xyz": str(tuple(round(float(x), 6) for x in img.GetSpacing())),
        f"{prefix}_origin_xyz": str(tuple(round(float(x), 6) for x in img.GetOrigin())),
        f"{prefix}_direction": str(tuple(round(float(x), 6) for x in img.GetDirection())),
        f"{prefix}_voxels": int((arr > 0).sum()) if arr is not None else np.nan,
    }


def allclose_tuple(a: Sequence[float], b: Sequence[float], tol: float) -> bool:
    try:
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0))
    except Exception:
        return False


def compare_image_geometry(gt_img: sitk.Image, pred_img: sitk.Image) -> Dict[str, object]:
    gt_size = tuple(gt_img.GetSize())
    pred_size = tuple(pred_img.GetSize())
    gt_spacing = tuple(gt_img.GetSpacing())
    pred_spacing = tuple(pred_img.GetSpacing())
    gt_origin = tuple(gt_img.GetOrigin())
    pred_origin = tuple(pred_img.GetOrigin())
    gt_direction = tuple(gt_img.GetDirection())
    pred_direction = tuple(pred_img.GetDirection())

    size_match = gt_size == pred_size
    spacing_match = allclose_tuple(gt_spacing, pred_spacing, tol=1e-4)
    origin_match = allclose_tuple(gt_origin, pred_origin, tol=1e-2)
    direction_match = allclose_tuple(gt_direction, pred_direction, tol=1e-4)

    return {
        "gt_size_xyz": str(gt_size),
        "pred_size_xyz": str(pred_size),
        "size_match": size_match,
        "spacing_match": spacing_match,
        "origin_match": origin_match,
        "direction_match": direction_match,
        "any_geometry_mismatch": not (size_match and spacing_match and origin_match and direction_match),
    }


def normalize_slice_for_display(sl: np.ndarray) -> np.ndarray:
    x = sl.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    vals = x[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return x


def choose_display_slice(gt: np.ndarray, pred_raw: np.ndarray, pred_lcc: np.ndarray) -> int:
    if gt.shape != pred_raw.shape or gt.shape != pred_lcc.shape:
        return int(gt.shape[0] // 2)
    combined = (gt > 0).astype(np.uint8) + (pred_raw > 0).astype(np.uint8) + (pred_lcc > 0).astype(np.uint8)
    area = combined.reshape(combined.shape[0], -1).sum(axis=1)
    if area.max() <= 0:
        return int(combined.shape[0] // 2)
    return int(np.argmax(area))


def crop_bounds_from_masks(masks: Sequence[np.ndarray], z: int, margin: int, shape_yx: Tuple[int, int]) -> Tuple[slice, slice]:
    y_max, x_max = shape_yx
    coords_list = []
    for m in masks:
        if m is None or m.ndim != 3 or z < 0 or z >= m.shape[0]:
            continue
        coords = np.argwhere(m[z] > 0)
        if coords.size > 0:
            coords_list.append(coords)
    if not coords_list:
        return slice(0, y_max), slice(0, x_max)
    coords = np.vstack(coords_list)
    y0 = max(int(coords[:, 0].min()) - margin, 0)
    y1 = min(int(coords[:, 0].max()) + margin + 1, y_max)
    x0 = max(int(coords[:, 1].min()) - margin, 0)
    x1 = min(int(coords[:, 1].max()) + margin + 1, x_max)
    if y1 <= y0 or x1 <= x0:
        return slice(0, y_max), slice(0, x_max)
    return slice(y0, y1), slice(x0, x1)


def overlay_contours(ax, base2d: np.ndarray, gt2d: np.ndarray, raw2d: np.ndarray, lcc2d: np.ndarray, title: str) -> None:
    ax.imshow(normalize_slice_for_display(base2d), cmap="gray")
    try:
        if gt2d.max() > 0:
            ax.contour(gt2d.astype(float), levels=[0.5], colors=["lime"], linewidths=1.5)
        if raw2d.max() > 0:
            ax.contour(raw2d.astype(float), levels=[0.5], colors=["red"], linewidths=1.0)
        if lcc2d.max() > 0:
            ax.contour(lcc2d.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.2)
    except Exception:
        pass
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def make_preview_png(
    case_id: str,
    row: pd.Series,
    pre_arr: Optional[np.ndarray],
    gt: np.ndarray,
    pred_raw: np.ndarray,
    pred_lcc: np.ndarray,
    out_path: Path,
) -> None:
    if plt is None:
        return

    if pred_raw.shape != gt.shape or pred_lcc.shape != gt.shape:
        # Cannot overlay if masks are not in the same array frame.
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        z_gt = int(gt.shape[0] // 2)
        z_pred = int(pred_raw.shape[0] // 2)
        axes[0].imshow(gt[z_gt] > 0, cmap="gray")
        axes[0].set_title("GT mask")
        axes[1].imshow(pred_raw[z_pred] > 0, cmap="gray")
        axes[1].set_title("Pred raw")
        axes[2].imshow(pred_lcc[z_pred] > 0, cmap="gray")
        axes[2].set_title("Pred LCC")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{case_id}: SHAPE MISMATCH - no overlay", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    z = choose_display_slice(gt, pred_raw, pred_lcc)
    if pre_arr is not None and pre_arr.shape == gt.shape:
        base = pre_arr
        base_label = "pre-contrast"
    else:
        base = np.zeros_like(gt, dtype=np.float32)
        base_label = "blank background"

    ysl, xsl = crop_bounds_from_masks([gt, pred_raw, pred_lcc], z, CROP_MARGIN_PIXELS, gt.shape[1:])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title_short = (
        f"{case_id} | Dice={row.get('v2_lcc_dice', np.nan):.3f} | "
        f"maxDice={row.get('v2_lcc_theoretical_max_dice_by_volume_ratio', np.nan):.3f} | "
        f"gap={row.get('v2_lcc_dice_gap_to_volume_bound', np.nan):.3f}"
    )
    overlay_contours(
        axes[0], base[z], gt[z], pred_raw[z], pred_lcc[z],
        f"Full axial slice z={z} ({base_label})"
    )
    overlay_contours(
        axes[1], base[z][ysl, xsl], gt[z][ysl, xsl], pred_raw[z][ysl, xsl], pred_lcc[z][ysl, xsl],
        "Cropped overlay"
    )
    fig.suptitle(title_short + "\nContours: GT=green, raw=red, LCC=yellow", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def clean_case_id(x: object) -> str:
    return str(x).strip().replace(".nii.gz", "").replace(".nii", "").upper()


# =============================================================================
# TABLE PREPARATION
# =============================================================================

def add_or_reconstruct_volume_bound_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize expected key columns.
    case_col = infer_case_id_col(df)
    df["case_id_std"] = safe_series(df, case_col).map(clean_case_id)

    if "cohort" not in df.columns:
        df["cohort"] = df["case_id_std"].str.extract(r"^(DUKE|ISPY1|NACT)", expand=False).fillna("OTHER")

    dice_col = first_existing_col(df, ["v2_lcc_dice", "primary_dice", "dice"])
    ratio_col = first_existing_col(df, ["v2_lcc_pred_to_gt_vol_ratio", "pred_to_gt_vol_ratio", "volume_ratio"])
    max_col = first_existing_col(df, [
        "v2_lcc_theoretical_max_dice_by_volume_ratio",
        "theoretical_max_dice_by_volume_ratio",
        "max_dice_by_volume_ratio",
    ])
    util_col = first_existing_col(df, [
        "v2_lcc_dice_utilization_of_volume_bound",
        "dice_utilization_of_volume_bound",
        "utilization_of_volume_bound",
    ])
    gap_col = first_existing_col(df, [
        "v2_lcc_dice_gap_to_volume_bound",
        "dice_gap_to_volume_bound",
        "gap_to_volume_bound",
    ])

    if dice_col and dice_col != "v2_lcc_dice":
        df["v2_lcc_dice"] = numeric(df, dice_col)
    elif "v2_lcc_dice" not in df.columns:
        df["v2_lcc_dice"] = np.nan

    if ratio_col and ratio_col != "v2_lcc_pred_to_gt_vol_ratio":
        df["v2_lcc_pred_to_gt_vol_ratio"] = numeric(df, ratio_col)
    elif "v2_lcc_pred_to_gt_vol_ratio" not in df.columns:
        df["v2_lcc_pred_to_gt_vol_ratio"] = np.nan

    # Reconstruct theoretical maximum Dice if needed.
    if max_col and max_col != "v2_lcc_theoretical_max_dice_by_volume_ratio":
        df["v2_lcc_theoretical_max_dice_by_volume_ratio"] = numeric(df, max_col)
    elif "v2_lcc_theoretical_max_dice_by_volume_ratio" not in df.columns:
        r = numeric(df, "v2_lcc_pred_to_gt_vol_ratio")
        denom = 1.0 + r
        with np.errstate(divide="ignore", invalid="ignore"):
            max_dice = 2.0 * np.minimum(r, 1.0) / denom
        max_dice = max_dice.where(np.isfinite(max_dice), np.nan)
        df["v2_lcc_theoretical_max_dice_by_volume_ratio"] = max_dice

    if util_col and util_col != "v2_lcc_dice_utilization_of_volume_bound":
        df["v2_lcc_dice_utilization_of_volume_bound"] = numeric(df, util_col)
    elif "v2_lcc_dice_utilization_of_volume_bound" not in df.columns:
        max_dice = numeric(df, "v2_lcc_theoretical_max_dice_by_volume_ratio")
        dice = numeric(df, "v2_lcc_dice")
        with np.errstate(divide="ignore", invalid="ignore"):
            util = dice / max_dice
        df["v2_lcc_dice_utilization_of_volume_bound"] = util.where(np.isfinite(util), np.nan)

    if gap_col and gap_col != "v2_lcc_dice_gap_to_volume_bound":
        df["v2_lcc_dice_gap_to_volume_bound"] = numeric(df, gap_col)
    elif "v2_lcc_dice_gap_to_volume_bound" not in df.columns:
        df["v2_lcc_dice_gap_to_volume_bound"] = numeric(df, "v2_lcc_theoretical_max_dice_by_volume_ratio") - numeric(df, "v2_lcc_dice")

    group_col = infer_group_col(df)
    if group_col and group_col != "volume_bound_diagnostic_group":
        df["volume_bound_diagnostic_group"] = safe_series(df, group_col).astype(str)
    elif "volume_bound_diagnostic_group" not in df.columns:
        # Conservative fallback grouping; this may not match the original 10-script groups exactly.
        max_dice = numeric(df, "v2_lcc_theoretical_max_dice_by_volume_ratio")
        gap = numeric(df, "v2_lcc_dice_gap_to_volume_bound")
        util = numeric(df, "v2_lcc_dice_utilization_of_volume_bound")
        conditions = [
            (max_dice < 0.50),
            (max_dice >= 0.70) & (gap >= 0.60),
            (max_dice >= 0.70) & (gap >= 0.20),
            (util >= 0.80),
        ]
        choices = [
            "volume_ratio_limits_dice",
            "spatial_or_boundary_mismatch_beyond_volume",
            "mixed_volume_and_spatial_error",
            "near_volume_bound",
        ]
        df["volume_bound_diagnostic_group"] = np.select(conditions, choices, default="other_uncategorized")

    return df


def select_review_cases(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    selected_parts = []

    cat = df[df["volume_bound_diagnostic_group"] == CATASTROPHIC_GROUP_NAME].copy()
    cat["review_reason"] = "all_catastrophic_spatial_mismatch"
    selected_parts.append(cat)

    if SAVE_COMPARISON_EXAMPLES:
        for group, grp in df.groupby("volume_bound_diagnostic_group", dropna=False):
            if str(group) == CATASTROPHIC_GROUP_NAME:
                continue
            g = grp.copy()
            # Pick worst Dice cases for each group; these are more useful for visual review.
            g = g.sort_values("v2_lcc_dice", ascending=True).head(N_COMPARISON_PER_GROUP)
            g["review_reason"] = f"comparison_{group}"
            selected_parts.append(g)

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        selected = df.sort_values("v2_lcc_dice", ascending=True).head(24).copy()
        selected["review_reason"] = "fallback_worst_dice"

    selected = selected.drop_duplicates(subset=["case_id_std"]).copy()
    selected = selected.sort_values(["volume_bound_diagnostic_group", "v2_lcc_dice"], ascending=[True, True])
    return selected


# =============================================================================
# CASE REVIEW
# =============================================================================

def review_single_case(row: pd.Series) -> Dict[str, object]:
    case_id = clean_case_id(row["case_id_std"])
    paths = get_case_paths(case_id)
    out: Dict[str, object] = {
        "case_id": case_id,
        "cohort": row.get("cohort", ""),
        "review_reason": row.get("review_reason", ""),
        "volume_bound_diagnostic_group": row.get("volume_bound_diagnostic_group", ""),
        "table_v2_lcc_dice": row.get("v2_lcc_dice", np.nan),
        "table_theoretical_max_dice": row.get("v2_lcc_theoretical_max_dice_by_volume_ratio", np.nan),
        "table_dice_gap_to_volume_bound": row.get("v2_lcc_dice_gap_to_volume_bound", np.nan),
        "table_dice_utilization": row.get("v2_lcc_dice_utilization_of_volume_bound", np.nan),
        "table_pred_to_gt_volume_ratio": row.get("v2_lcc_pred_to_gt_vol_ratio", np.nan),
        "gt_path": str(paths["gt"]) if paths["gt"] else "",
        "pred_v2_raw_path": str(paths["pred_v2_raw"]) if paths["pred_v2_raw"] else "",
        "pre_path": str(paths["pre"]) if paths["pre"] else "",
        "case_dir": str(paths["case_dir"]) if paths["case_dir"] else "",
        "error": "",
    }

    if paths["gt"] is None or paths["pred_v2_raw"] is None:
        out["error"] = "missing_gt_or_prediction"
        return out

    try:
        gt, gt_img = load_binary(paths["gt"])
        pred_raw, pred_img = load_binary(paths["pred_v2_raw"])
        pred_lcc = keep_largest_component(pred_raw)
        pre_arr = None
        pre_img = None
        if paths["pre"] is not None:
            try:
                pre_img = load_image(paths["pre"])
                pre_arr = sitk_to_array(pre_img)
            except Exception:
                pre_arr = None
                pre_img = None

        out.update(image_info("gt", gt_img, gt))
        out.update(image_info("pred_raw", pred_img, pred_raw))
        out.update(compare_image_geometry(gt_img, pred_img))

        if gt.shape == pred_raw.shape:
            dice_raw = dice_score(pred_raw, gt)
            dice_lcc = dice_score(pred_lcc, gt)
            prec_lcc, rec_lcc = precision_recall(pred_lcc, gt)
            raw_stats = component_stats(pred_raw)
            lcc_stats = component_stats(pred_lcc)

            voxel_vol = float(np.prod(gt_img.GetSpacing()))
            gt_vol_cm3 = float(gt.sum() * voxel_vol / 1000.0)
            pred_raw_vol_cm3 = float(pred_raw.sum() * voxel_vol / 1000.0)
            pred_lcc_vol_cm3 = float(pred_lcc.sum() * voxel_vol / 1000.0)

            gt_centroid_v = mask_centroid_voxel(gt)
            pred_lcc_centroid_v = mask_centroid_voxel(pred_lcc)
            gt_centroid_p = voxel_to_physical(gt_img, gt_centroid_v)
            pred_lcc_centroid_p = voxel_to_physical(pred_img, pred_lcc_centroid_v)
            centroid_dist_mm = euclidean_distance(gt_centroid_p, pred_lcc_centroid_p)

            out.update({
                "computed_v2_raw_dice": dice_raw,
                "computed_v2_lcc_dice": dice_lcc,
                "computed_v2_lcc_precision": prec_lcc,
                "computed_v2_lcc_recall": rec_lcc,
                "gt_vol_cm3_computed": gt_vol_cm3,
                "pred_raw_vol_cm3_computed": pred_raw_vol_cm3,
                "pred_lcc_vol_cm3_computed": pred_lcc_vol_cm3,
                "pred_lcc_to_gt_vol_ratio_computed": pred_lcc_vol_cm3 / gt_vol_cm3 if gt_vol_cm3 > 0 else np.nan,
                "gt_centroid_voxel_zyx": str(tuple(round(x, 3) for x in gt_centroid_v)),
                "pred_lcc_centroid_voxel_zyx": str(tuple(round(x, 3) for x in pred_lcc_centroid_v)),
                "gt_centroid_physical_xyz_mm": str(tuple(round(x, 3) for x in gt_centroid_p)),
                "pred_lcc_centroid_physical_xyz_mm": str(tuple(round(x, 3) for x in pred_lcc_centroid_p)),
                "gt_to_pred_lcc_centroid_distance_mm": centroid_dist_mm,
                "raw_n_components_computed": raw_stats["n_components"],
                "raw_largest_component_fraction_computed": raw_stats["largest_component_fraction"],
                "raw_second_largest_component_fraction_computed": raw_stats["second_largest_component_fraction"],
                "lcc_n_components_computed": lcc_stats["n_components"],
                "lcc_largest_component_fraction_computed": lcc_stats["largest_component_fraction"],
            })

            # Basic suspicion flags for technical review.
            out["large_centroid_distance_gt_50mm"] = bool(np.isfinite(centroid_dist_mm) and centroid_dist_mm > 50.0)
            out["large_centroid_distance_gt_100mm"] = bool(np.isfinite(centroid_dist_mm) and centroid_dist_mm > 100.0)
            out["computed_dice_matches_table_tol_0p02"] = bool(
                np.isfinite(dice_lcc) and np.isfinite(row.get("v2_lcc_dice", np.nan)) and abs(dice_lcc - float(row.get("v2_lcc_dice"))) <= 0.02
            )
            out["suspect_case_pairing_or_geometry"] = bool(
                out.get("any_geometry_mismatch", False)
                or out.get("large_centroid_distance_gt_100mm", False)
                or not out.get("computed_dice_matches_table_tol_0p02", True)
            )
        else:
            out.update({
                "computed_v2_raw_dice": np.nan,
                "computed_v2_lcc_dice": np.nan,
                "computed_v2_lcc_precision": np.nan,
                "computed_v2_lcc_recall": np.nan,
                "suspect_case_pairing_or_geometry": True,
                "error": "shape_mismatch_gt_pred",
            })

        # Save review packet.
        case_out_dir = NIFTI_REVIEW_DIR / case_id
        if SAVE_NIFTI_REVIEW_PACKET:
            case_out_dir.mkdir(parents=True, exist_ok=True)
            if paths["pre"] is not None:
                shutil.copy2(paths["pre"], case_out_dir / f"{case_id}_image_pre.nii.gz")
            shutil.copy2(paths["gt"], case_out_dir / f"{case_id}_gt.nii.gz")
            shutil.copy2(paths["pred_v2_raw"], case_out_dir / f"{case_id}_pred_v2_raw.nii.gz")
            save_mask_like(pred_lcc, pred_img, case_out_dir / f"{case_id}_pred_v2_lcc.nii.gz")
            out["review_nifti_folder"] = str(case_out_dir)
            out["review_pred_lcc_path"] = str(case_out_dir / f"{case_id}_pred_v2_lcc.nii.gz")

        if SAVE_PREVIEW_PNGS:
            group_name = str(row.get("volume_bound_diagnostic_group", "unknown_group")).replace("/", "_")
            group_dir = PREVIEW_DIR / group_name
            group_dir.mkdir(parents=True, exist_ok=True)
            png_path = group_dir / f"{case_id}_overlay.png"
            try:
                make_preview_png(case_id, row, pre_arr, gt, pred_raw, pred_lcc, png_path)
                out["preview_png_path"] = str(png_path)
            except Exception as e:
                out["preview_error"] = repr(e)

        return out

    except Exception as e:
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
        return out


# =============================================================================
# CLEANED INTERCORRELATION TABLE
# =============================================================================

def base_feature_name(name: str) -> str:
    s = str(name).lower()
    # Remove common transform/pipeline prefixes.
    for prefix in ["log_", "sqrt_", "kin_", "v1_raw_", "v2_raw_", "v1_lcc_", "v2_lcc_", "v1_", "v2_"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Normalize common suffixes.
    for suffix in ["_computed", "_cm3", "_norm", "_ratio"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # Treat transformed versions as same family.
    s = s.replace("gt_vol", "tumor_volume")
    s = s.replace("pred_to_gt_vol", "pred_volume_relative_to_gt")
    return s


def is_trivial_pair(a: str, b: str) -> bool:
    aa = str(a).lower()
    bb = str(b).lower()
    if aa == bb:
        return True
    if base_feature_name(aa) == base_feature_name(bb):
        return True
    # Remove obvious pairs: raw-vs-raw across v1/v2 same metric.
    def strip_pipeline(x: str) -> str:
        for p in ["v1_raw_", "v2_raw_", "v1_lcc_", "v2_lcc_", "kin_"]:
            if x.startswith(p):
                return x[len(p):]
        return x
    if strip_pipeline(aa) == strip_pipeline(bb):
        return True
    # Treat max enhancement and AUC from same region as same kinetic family for cleaned display.
    families = ["fg", "gt", "ring", "tumor_to_ring", "tumor_to_fg"]
    for fam in families:
        if fam in aa and fam in bb and (("max_enh" in aa and "auc_enh" in bb) or ("auc_enh" in aa and "max_enh" in bb)):
            return True
    return False


def compute_cleaned_intercorrelations(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    # Candidate diagnostic predictors only. Exclude target outcomes and duplicate transforms where possible.
    candidate_substrings = [
        "n_components",
        "largest_component_fraction",
        "second_largest_component_fraction",
        "pred_to_gt_vol_ratio",
        "pred_vol_cm3",
        "gt_vol_cm3",
        "theoretical_max",
        "dice_gap_to_volume_bound",
        "dice_utilization",
        "first_post_s",
        "early_dev_s",
        "late_dev_s",
        "max_enh",
        "auc_enh",
        "washout",
        "slope",
        "tumor_to_ring",
        "tumor_to_fg",
    ]
    exclude_substrings = ["dice_mean", "precision_mean", "recall_mean"]
    num_cols = []
    for col in df.columns:
        low = str(col).lower()
        if any(x in low for x in candidate_substrings) and not any(x in low for x in exclude_substrings):
            s = pd.to_numeric(safe_series(df, col), errors="coerce")
            if s.notna().sum() >= 30 and s.nunique(dropna=True) >= 3:
                num_cols.append(col)

    rows = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            if is_trivial_pair(a, b):
                continue
            aa = pd.to_numeric(safe_series(df, a), errors="coerce")
            bb = pd.to_numeric(safe_series(df, b), errors="coerce")
            sub = pd.DataFrame({"a": aa, "b": bb}).dropna()
            if len(sub) < 30 or sub["a"].nunique() < 3 or sub["b"].nunique() < 3:
                continue
            rho = sub["a"].corr(sub["b"], method="spearman")
            if np.isfinite(rho):
                rows.append({"feature_a": a, "feature_b": b, "spearman_rho": rho, "abs_rho": abs(rho), "n": len(sub)})
    out = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    out.to_csv(out_path, index=False)
    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dirs()
    print("11 Catastrophic external case review")
    print("=====================================")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print(f"Case table   : {CASE_TABLE_CSV}")

    if not CASE_TABLE_CSV.exists():
        raise FileNotFoundError(f"Missing case table: {CASE_TABLE_CSV}")

    df = pd.read_csv(CASE_TABLE_CSV)
    print(f"Cases loaded : {len(df)}")
    df = add_or_reconstruct_volume_bound_columns(df)

    # Save standardized full table used by this script.
    standardized_path = METRICS_DIR / "case_table_standardized_for_11.csv"
    df.to_csv(standardized_path, index=False)

    selected = select_review_cases(df)
    selected_path = METRICS_DIR / "selected_cases_for_visual_review.csv"
    selected.to_csv(selected_path, index=False)

    print()
    print("Volume-bound diagnostic groups in full table:")
    group_counts = df["volume_bound_diagnostic_group"].value_counts(dropna=False).reset_index()
    group_counts.columns = ["volume_bound_diagnostic_group", "n"]
    print(group_counts.to_string(index=False))
    group_counts.to_csv(METRICS_DIR / "volume_bound_group_counts.csv", index=False)

    print()
    print("Selected cases for review:")
    print(selected[["case_id_std", "cohort", "volume_bound_diagnostic_group", "v2_lcc_dice", "v2_lcc_theoretical_max_dice_by_volume_ratio", "v2_lcc_dice_gap_to_volume_bound", "review_reason"]].head(50).to_string(index=False))
    print(f"Selected total: {len(selected)}")

    rows = []
    for idx, (_, row) in enumerate(selected.iterrows(), start=1):
        res = review_single_case(row)
        rows.append(res)
        if idx % 10 == 0 or idx == len(selected):
            print(f"  reviewed {idx}/{len(selected)}")

    review_df = pd.DataFrame(rows)
    review_path = METRICS_DIR / "technical_geometry_and_pairing_review.csv"
    review_df.to_csv(review_path, index=False)

    # Catastrophic-only table for quick inspection.
    cat_review = review_df[review_df["volume_bound_diagnostic_group"] == CATASTROPHIC_GROUP_NAME].copy()
    cat_review_path = METRICS_DIR / "catastrophic_spatial_mismatch_review.csv"
    cat_review.to_csv(cat_review_path, index=False)

    # Cleaned intercorrelation table to avoid self/log duplicate pairs.
    cleaned_corr = compute_cleaned_intercorrelations(df, METRICS_DIR / "cleaned_predictor_intercorrelations_nontrivial.csv")

    # Short report for manuscript notes.
    report_path = OUTPUT_ROOT / "article_takeaways_11.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("11 CATASTROPHIC CASE REVIEW — TAKEAWAYS\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Cases loaded: {len(df)}\n")
        f.write(f"Selected for visual/technical review: {len(selected)}\n")
        f.write(f"Catastrophic group name: {CATASTROPHIC_GROUP_NAME}\n")
        f.write(f"Catastrophic cases selected: {len(cat_review)}\n\n")

        if len(cat_review) > 0:
            f.write("Catastrophic technical flags:\n")
            for col in ["any_geometry_mismatch", "large_centroid_distance_gt_50mm", "large_centroid_distance_gt_100mm", "suspect_case_pairing_or_geometry", "computed_dice_matches_table_tol_0p02"]:
                if col in cat_review.columns:
                    f.write(f"  {col}: {cat_review[col].sum() if cat_review[col].dtype != object else cat_review[col].astype(str).value_counts().to_dict()}\n")
            if "gt_to_pred_lcc_centroid_distance_mm" in cat_review.columns:
                dist = pd.to_numeric(cat_review["gt_to_pred_lcc_centroid_distance_mm"], errors="coerce")
                f.write(f"  centroid distance median/mean/max mm: {dist.median():.2f} / {dist.mean():.2f} / {dist.max():.2f}\n")
            if "computed_v2_lcc_dice" in cat_review.columns:
                d = pd.to_numeric(cat_review["computed_v2_lcc_dice"], errors="coerce")
                f.write(f"  recomputed Dice median/mean: {d.median():.4f} / {d.mean():.4f}\n")
            f.write("\n")

        f.write("Cleaned high intercorrelations excluding trivial transform/self pairs:\n")
        if len(cleaned_corr) > 0:
            top = cleaned_corr.head(15)
            for _, r in top.iterrows():
                f.write(f"  {r['feature_a']} vs {r['feature_b']}: rho={r['spearman_rho']:.3f}, n={int(r['n'])}\n")
        else:
            f.write("  None computed.\n")
        f.write("\nSuggested wording:\n")
        f.write("  Use this review to verify that catastrophic spatial-mismatch cases are genuine model/domain failures rather than case-ID, geometry, or coordinate-frame artifacts.\n")
        f.write("  If geometry is consistent but centroid distances are large, describe these as severe spatial localization/boundary failures.\n")
        f.write("  If geometry mismatches are found, fix preprocessing/evaluation before finalizing any external metrics.\n")

    # Also create a simple Markdown index for opening in 3D Slicer.
    index_md = REVIEW_PACKET_DIR / "OPEN_THESE_CASES_IN_3D_SLICER.md"
    with open(index_md, "w", encoding="utf-8") as f:
        f.write("# 11 visual review packet\n\n")
        f.write("Open the NIfTI files inside each case folder in 3D Slicer or ITK-SNAP.\n\n")
        f.write("Recommended layers:\n")
        f.write("- image_pre as background\n")
        f.write("- gt as green label\n")
        f.write("- pred_v2_raw as red label\n")
        f.write("- pred_v2_lcc as yellow/orange label\n\n")
        for _, r in review_df.iterrows():
            f.write(f"## {r.get('case_id', '')}\n")
            f.write(f"- group: {r.get('volume_bound_diagnostic_group', '')}\n")
            f.write(f"- Dice: {r.get('table_v2_lcc_dice', np.nan)}\n")
            f.write(f"- folder: `{r.get('review_nifti_folder', '')}`\n")
            f.write(f"- preview: `{r.get('preview_png_path', '')}`\n\n")

    print()
    print("DONE — 11 catastrophic case review complete.")
    print(f"Metrics      : {METRICS_DIR}")
    print(f"Review packet: {REVIEW_PACKET_DIR}")
    print(f"Takeaways    : {report_path}")
    print()
    print("Quick results:")
    print(f"  selected cases: {len(selected)}")
    print(f"  catastrophic cases: {len(cat_review)}")
    if len(cat_review) > 0:
        for col in ["any_geometry_mismatch", "suspect_case_pairing_or_geometry", "large_centroid_distance_gt_100mm"]:
            if col in cat_review.columns:
                try:
                    print(f"  {col}: {int(pd.to_numeric(cat_review[col], errors='coerce').fillna(0).sum())}")
                except Exception:
                    print(f"  {col}: {cat_review[col].astype(str).value_counts().to_dict()}")
        if "gt_to_pred_lcc_centroid_distance_mm" in cat_review.columns:
            dist = pd.to_numeric(cat_review["gt_to_pred_lcc_centroid_distance_mm"], errors="coerce")
            print(f"  catastrophic centroid distance median/mean/max: {dist.median():.1f}/{dist.mean():.1f}/{dist.max():.1f} mm")
    if len(cleaned_corr) > 0:
        print("  cleaned top nontrivial intercorrelations:")
        print(cleaned_corr.head(8).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR")
        print("===========")
        print(repr(e))
        print(traceback.format_exc())
        print("\nIf partial outputs were saved, send the last lines and I will make a finish-from-saved-CSV helper.")
        raise
