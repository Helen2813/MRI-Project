# contrast_kinetic_shift_analysis.py
# Folder: 07_contrast_kinetic_shift_analysis
#
# Purpose:
#   Adds a protocol/contrast-kinetic shift analysis for external MAMA-MIA validation.
#   This is CPU-friendly and can run on a home Windows machine if the external images,
#   expert masks, and prediction folders already exist.
#
# What it computes:
#   1) acquisition-time summaries by cohort and phase quality
#   2) tumor and background contrast-enhancement kinetics from ALL available DCE phases
#   3) association between kinetics/timing and external segmentation performance
#   4) v2 raw vs v2 largest-connected-component vs exploratory kinetic-component postprocessing
#   5) publication-ready CSV tables and PNG figures
#
# Run:
#   python contrast_kinetic_shift_analysis.py
#
# Edit only the CONFIG block below if your folders differ.

from __future__ import annotations

import ast
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    from scipy import ndimage
    from scipy.stats import spearmanr
except Exception as e:  # pragma: no cover
    raise ImportError("This script needs scipy. Install with: pip install scipy") from e

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise ImportError("This script needs matplotlib. Install with: pip install matplotlib") from e


# =============================================================================
# CONFIG — EDIT ONLY HERE
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")

# Original external MAMA-MIA images: one folder per case, several DCE phase files per case.
IMAGES_DIR = PROJECT_ROOT / "images_2"

# MAMA-MIA table containing patient_id, dataset, acquisition_times, etc.
CLINICAL_TABLE = PROJECT_ROOT / "tables" / "clinical_and_imaging_info.xlsx"

# Expert tumor masks, expected names like duke_001.nii.gz / ispy1_001.nii.gz / nact_001.nii.gz.
GT_DIR = PROJECT_ROOT / "segmentations_2" / "expert"

# Existing prediction folders from earlier external validation experiments.
# v1 = original phase-order preprocessing; v2 = acquisition-time-aware phase selection.
PRED_V1_DIR = Path(r"C:\nnw\mama_mia_output")
PRED_V2_DIR = Path(r"C:\nnw\mama_mia_output_v2")

# Phase selection report produced by preprocess_mama_mia_nnunet_v2.py.
PHASE_REPORT = PROJECT_ROOT / "external_manifest" / "phase_selection_report.csv"

# Output folder for this new analysis.
OUT_DIR = PROJECT_ROOT / "07_contrast_kinetic_shift_analysis"
METRICS_DIR = OUT_DIR / "metrics"
FIGURES_DIR = OUT_DIR / "figures"
PREVIEW_DIR = OUT_DIR / "previews"

# For full analysis use None. For a quick test use 10 or 20.
LIMIT_CASES: Optional[int] = None

# Preview PNGs are useful for sanity checking but can be slow. Keep small.
SAVE_PREVIEWS = True
PREVIEW_LIMIT = 12

# Tumor/peritumor ring settings. This is voxel-based, simple, and CPU-friendly.
PERITUMOR_RING_ITERATIONS = 5
MIN_COMPONENT_VOXELS = 50

# Timing targets from the current paper/external v2 preprocessing.
TARGET_EARLY_S = 90
TARGET_LATE_S = 420
GOOD_DEV_S = 120
ACCEPTABLE_DEV_S = 300

# Relative enhancement denominator: using a robust small epsilon avoids exploding division by near-zero voxels.
REL_ENH_CLIP_LOW = -5.0
REL_ENH_CLIP_HIGH = 10.0

# =============================================================================
# Utilities
# =============================================================================


def ensure_dirs() -> None:
    for d in [OUT_DIR, METRICS_DIR, FIGURES_DIR, PREVIEW_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def parse_times(value) -> List[float]:
    """Parse acquisition_times strings like '[0, 165, 288, 411]'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return [float(x) for x in parsed]
    except Exception:
        return []


def case_to_cohort(case_id: str) -> str:
    u = case_id.upper()
    if u.startswith("DUKE"):
        return "DUKE"
    if u.startswith("ISPY1"):
        return "ISPY1"
    if u.startswith("NACT"):
        return "NACT"
    return "OTHER"


def read_img(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def img_arr(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)


def resample_to_reference(img: sitk.Image, ref: sitk.Image, is_mask: bool) -> sitk.Image:
    """Resample image/mask to ref geometry if needed."""
    same_size = img.GetSize() == ref.GetSize()
    same_spacing = np.allclose(img.GetSpacing(), ref.GetSpacing())
    same_origin = np.allclose(img.GetOrigin(), ref.GetOrigin())
    same_direction = np.allclose(img.GetDirection(), ref.GetDirection())
    if same_size and same_spacing and same_origin and same_direction:
        return img

    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    default_value = 0
    return sitk.Resample(img, ref, sitk.Transform(), interpolator, default_value, img.GetPixelID())


def load_mask(mask_path: Path, ref_img: sitk.Image) -> Optional[np.ndarray]:
    if not mask_path.exists():
        return None
    m = read_img(mask_path)
    m = resample_to_reference(m, ref_img, is_mask=True)
    return (img_arr(m) > 0).astype(bool)


def foreground_mask(pre: np.ndarray) -> np.ndarray:
    finite = np.isfinite(pre)
    nz = finite & (np.abs(pre) > 1e-6)
    if nz.sum() > 1000:
        return nz
    vals = pre[finite]
    if vals.size == 0:
        return np.ones_like(pre, dtype=bool)
    threshold = np.percentile(vals, 1)
    fg = finite & (pre > threshold)
    return fg if fg.sum() > 1000 else finite


def robust_epsilon(pre: np.ndarray, fg: np.ndarray) -> float:
    vals = np.abs(pre[fg & np.isfinite(pre)])
    vals = vals[vals > 1e-6]
    if vals.size == 0:
        return 1.0
    return float(max(1.0, np.percentile(vals, 5)))


def safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or mask.sum() == 0:
        return np.nan
    vals = arr[mask & np.isfinite(arr)]
    return float(np.mean(vals)) if vals.size else np.nan


def safe_median(arr: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or mask.sum() == 0:
        return np.nan
    vals = arr[mask & np.isfinite(arr)]
    return float(np.median(vals)) if vals.size else np.nan


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    prec = float(tp / pred.sum()) if pred.sum() > 0 else 0.0
    rec = float(tp / gt.sum()) if gt.sum() > 0 else (1.0 if pred.sum() == 0 else 0.0)
    return prec, rec


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    largest = int(np.argmax(sizes) + 1)
    return labels == largest


def mask_volume_vox(mask: np.ndarray) -> int:
    return int(mask.astype(bool).sum())


def peritumor_ring(mask: np.ndarray, fg: np.ndarray, iterations: int = PERITUMOR_RING_ITERATIONS) -> np.ndarray:
    if mask is None or mask.sum() == 0:
        return np.zeros_like(fg, dtype=bool)
    dil = ndimage.binary_dilation(mask.astype(bool), iterations=iterations)
    return dil & (~mask.astype(bool)) & fg


def build_relative_enhancement_stack(phase_arrays: List[np.ndarray], times: List[float], fg: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """
    Returns rel enhancement stack for post-contrast phases only.
    Shape: [n_post, z, y, x]. Values are clipped relative enhancement.
    """
    if len(phase_arrays) < 2:
        return np.empty((0,) + phase_arrays[0].shape, dtype=np.float32), []

    pre = phase_arrays[0]
    eps = robust_epsilon(pre, fg)

    post_enh = []
    post_times = []
    for i in range(1, len(phase_arrays)):
        t = times[i] if i < len(times) else float(i)
        enh = (phase_arrays[i] - pre) / (np.abs(pre) + eps)
        enh = np.clip(enh, REL_ENH_CLIP_LOW, REL_ENH_CLIP_HIGH).astype(np.float32)
        post_enh.append(enh)
        post_times.append(float(t))

    return np.stack(post_enh, axis=0), post_times


def auc_trapezoid(times: List[float], values: List[float]) -> float:
    if len(times) < 2 or len(values) < 2:
        return np.nan
    order = np.argsort(times)
    x = np.asarray(times, dtype=float)[order]
    y = np.asarray(values, dtype=float)[order]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    if x[-1] == x[0]:
        return np.nan
    return float(np.trapezoid(y, x))


@dataclass
class KineticFeatures:
    first_post_s: float = np.nan
    last_post_s: float = np.nan
    time_span_s: float = np.nan
    mean_first_enh: float = np.nan
    mean_last_enh: float = np.nan
    max_enh: float = np.nan
    time_to_peak_s: float = np.nan
    auc_enh: float = np.nan
    auc_enh_norm: float = np.nan
    observed_uptake_rate: float = np.nan
    late_slope: float = np.nan
    washout_index: float = np.nan

    def as_dict(self, prefix: str) -> Dict[str, float]:
        return {f"{prefix}_{k}": v for k, v in self.__dict__.items()}


def compute_kinetics(rel_stack: np.ndarray, post_times: List[float], mask: np.ndarray) -> KineticFeatures:
    if rel_stack.size == 0 or len(post_times) == 0 or mask is None or mask.sum() == 0:
        return KineticFeatures()

    curve = []
    for j in range(rel_stack.shape[0]):
        curve.append(safe_mean(rel_stack[j], mask))

    times = list(map(float, post_times))
    vals = np.asarray(curve, dtype=float)
    finite = np.isfinite(vals) & np.isfinite(times)
    if finite.sum() == 0:
        return KineticFeatures()

    times_arr = np.asarray(times, dtype=float)[finite]
    vals_arr = vals[finite]
    order = np.argsort(times_arr)
    times_arr = times_arr[order]
    vals_arr = vals_arr[order]

    first_t = float(times_arr[0])
    last_t = float(times_arr[-1])
    first_v = float(vals_arr[0])
    last_v = float(vals_arr[-1])

    peak_i = int(np.nanargmax(vals_arr))
    peak_v = float(vals_arr[peak_i])
    peak_t = float(times_arr[peak_i])

    full_times = [0.0] + list(times_arr)
    full_vals = [0.0] + list(vals_arr)
    auc = auc_trapezoid(full_times, full_vals)
    span = float(last_t - 0.0) if np.isfinite(last_t) else np.nan
    auc_norm = float(auc / span) if np.isfinite(auc) and span > 0 else np.nan
    uptake = float(first_v / first_t) if first_t > 0 else np.nan

    if last_t > peak_t:
        late_slope = float((last_v - peak_v) / (last_t - peak_t))
    else:
        late_slope = np.nan

    return KineticFeatures(
        first_post_s=first_t,
        last_post_s=last_t,
        time_span_s=span,
        mean_first_enh=first_v,
        mean_last_enh=last_v,
        max_enh=peak_v,
        time_to_peak_s=peak_t,
        auc_enh=auc,
        auc_enh_norm=auc_norm,
        observed_uptake_rate=uptake,
        late_slope=late_slope,
        washout_index=float(peak_v - last_v),
    )


def kinetic_component_selection(pred_mask: np.ndarray, rel_stack: np.ndarray, post_times: List[float], fg: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Exploratory postprocessing: among predicted connected components, keep the component
    with the strongest enhancement consistency, while penalizing tiny components.

    This is NOT tuned to labels. It is only based on image kinetics and predicted components.
    It should be reported as exploratory, not as primary unless validated separately.
    """
    pred_mask = pred_mask.astype(bool)
    if pred_mask.sum() == 0:
        return pred_mask, {"kin_n_components": 0, "kin_selected_score": np.nan, "kin_selected_voxels": 0}

    labels, n = ndimage.label(pred_mask)
    if n <= 1:
        return pred_mask, {"kin_n_components": int(n), "kin_selected_score": np.nan, "kin_selected_voxels": int(pred_mask.sum())}

    # Background reference: median max enhancement in foreground.
    if rel_stack.size > 0:
        max_enh_map = np.nanmax(rel_stack, axis=0)
        fg_ref = safe_median(max_enh_map, fg)
        if not np.isfinite(fg_ref):
            fg_ref = 0.0
    else:
        max_enh_map = None
        fg_ref = 0.0

    best_label = None
    best_score = -np.inf
    best_info = {}

    for lab in range(1, n + 1):
        comp = labels == lab
        vox = int(comp.sum())
        if vox < MIN_COMPONENT_VOXELS:
            continue
        k = compute_kinetics(rel_stack, post_times, comp)
        # Component score: stronger max enhancement above global foreground + normalized AUC + mild size term.
        enh_signal = (k.max_enh - fg_ref) if np.isfinite(k.max_enh) else 0.0
        auc_signal = k.auc_enh_norm if np.isfinite(k.auc_enh_norm) else 0.0
        size_term = math.log1p(vox)
        score = float(enh_signal + 0.25 * auc_signal + 0.03 * size_term)
        if score > best_score:
            best_score = score
            best_label = lab
            best_info = {
                "kin_selected_score": score,
                "kin_selected_voxels": vox,
                "kin_selected_max_enh": k.max_enh,
                "kin_selected_auc_norm": k.auc_enh_norm,
                "kin_selected_time_to_peak_s": k.time_to_peak_s,
            }

    if best_label is None:
        # fallback to LCC if all components are too small
        selected = keep_largest_component(pred_mask)
        best_info = {"kin_selected_score": np.nan, "kin_selected_voxels": int(selected.sum())}
    else:
        selected = labels == best_label

    best_info["kin_n_components"] = int(n)
    return selected, best_info


def metrics_for_prediction(prefix: str, pred: Optional[np.ndarray], gt: Optional[np.ndarray]) -> Dict[str, float]:
    if pred is None or gt is None:
        return {
            f"{prefix}_dice": np.nan,
            f"{prefix}_precision": np.nan,
            f"{prefix}_recall": np.nan,
            f"{prefix}_voxels": np.nan,
        }
    p, r = precision_recall(pred, gt)
    return {
        f"{prefix}_dice": dice(pred, gt),
        f"{prefix}_precision": p,
        f"{prefix}_recall": r,
        f"{prefix}_voxels": int(pred.sum()),
    }


def find_prediction(pred_dir: Path, case_id: str) -> Optional[Path]:
    candidates = [
        pred_dir / f"{case_id}.nii.gz",
        pred_dir / f"{case_id.lower()}.nii.gz",
        pred_dir / f"{case_id.upper()}.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: case-insensitive search
    if pred_dir.exists():
        target = f"{case_id}.nii.gz".lower()
        for p in pred_dir.glob("*.nii.gz"):
            if p.name.lower() == target:
                return p
    return None


def phase_quality_from_devs(early_dev: float, late_dev: float) -> str:
    vals = [early_dev, late_dev]
    if any((not np.isfinite(v)) for v in vals):
        return "unknown"
    if all(v <= GOOD_DEV_S for v in vals):
        return "good"
    if all(v <= ACCEPTABLE_DEV_S for v in vals):
        return "acceptable"
    return "poor"


def load_phase_report() -> pd.DataFrame:
    if not PHASE_REPORT.exists():
        print(f"WARNING: phase report not found: {PHASE_REPORT}")
        return pd.DataFrame()
    df = pd.read_csv(PHASE_REPORT)
    df["case_id_upper"] = df["case_id"].astype(str).str.upper()
    return df


def load_clinical_table() -> pd.DataFrame:
    df = pd.read_excel(CLINICAL_TABLE, dtype=str)
    df = df[~df["patient_id"].astype(str).str.upper().str.startswith("ISPY2")].copy()
    df["case_id_upper"] = df["patient_id"].astype(str).str.upper()
    df["cohort"] = df["case_id_upper"].apply(case_to_cohort)
    df["times_parsed"] = df["acquisition_times"].apply(parse_times) if "acquisition_times" in df.columns else [[] for _ in range(len(df))]
    return df


def plot_case_preview(case_id: str, phase_arrays: List[np.ndarray], rel_stack: np.ndarray, gt: Optional[np.ndarray], pred_raw: Optional[np.ndarray], pred_lcc: Optional[np.ndarray], out_path: Path) -> None:
    if len(phase_arrays) == 0:
        return
    if gt is not None and gt.sum() > 0:
        z = int(np.round(np.mean(np.where(gt)[0])))
    else:
        z = phase_arrays[0].shape[0] // 2

    pre = phase_arrays[0][z]
    first_post = phase_arrays[1][z] if len(phase_arrays) > 1 else pre
    last_post = phase_arrays[-1][z] if len(phase_arrays) > 1 else pre
    max_enh = np.nanmax(rel_stack[:, z], axis=0) if rel_stack.size else np.zeros_like(pre)

    panels = [
        (pre, "pre", "gray"),
        (first_post, "first post", "gray"),
        (last_post, "last post", "gray"),
        (max_enh, "max relative enhancement", "inferno"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, (arr, title, cmap) in zip(axes[0], panels):
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # overlays
    base = last_post
    overlay_items = [
        (gt, "expert mask"),
        (pred_raw, "v2 raw pred"),
        (pred_lcc, "v2 LCC pred"),
        (None, "enhancement curve")
    ]
    for ax, (mask, title) in zip(axes[1], overlay_items):
        if mask is None and title != "enhancement curve":
            ax.imshow(base, cmap="gray")
            ax.set_title(title + " missing", fontsize=9)
            ax.axis("off")
        elif title == "enhancement curve":
            ax.axis("on")
            # curve is filled by caller in main if needed; keep placeholder readable
            ax.text(0.5, 0.5, "Curve saved\nin CSV", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=9)
        else:
            ax.imshow(base, cmap="gray")
            ax.imshow(np.ma.masked_where(~mask[z].astype(bool), mask[z]), alpha=0.35)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.suptitle(case_id, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    if not group_cols:
        work = df.copy()
        work["group"] = "ALL"
        group_cols = ["group"]
    rows = []
    for key, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row["n"] = len(grp)
        for col in metric_cols:
            if col in grp.columns:
                vals = pd.to_numeric(grp[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std())
                row[f"{col}_median"] = float(vals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_box_by_cohort(df: pd.DataFrame) -> None:
    cohorts = [c for c in ["DUKE", "ISPY1", "NACT"] if c in set(df["cohort"])]
    data_first = [df.loc[df["cohort"] == c, "first_post_s"].dropna().values for c in cohorts]
    data_last = [df.loc[df["cohort"] == c, "last_post_s"].dropna().values for c in cohorts]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions_first = np.arange(len(cohorts)) * 2.0
    positions_last = positions_first + 0.7
    ax.boxplot(data_first, positions=positions_first, widths=0.55)
    ax.boxplot(data_last, positions=positions_last, widths=0.55)
    ax.axhline(TARGET_EARLY_S, linestyle="--", linewidth=1, label="target early 90s")
    ax.axhline(TARGET_LATE_S, linestyle=":", linewidth=1, label="target late 420s")
    ax.set_xticks(positions_first + 0.35)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Acquisition time (s)")
    ax.set_title("External DCE acquisition timing by cohort")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_acquisition_timing_by_cohort.png", dpi=300)
    plt.close(fig)


def plot_mean_curves(curve_df: pd.DataFrame) -> None:
    if curve_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for cohort, grp in curve_df.groupby("cohort"):
        # bin time for smoother cohort-level average
        tmp = grp.dropna(subset=["time_s", "tumor_rel_enh"])
        if tmp.empty:
            continue
        tmp = tmp.copy()
        tmp["time_bin"] = (tmp["time_s"] / 60).round() * 60
        agg = tmp.groupby("time_bin")["tumor_rel_enh"].mean().reset_index()
        ax.plot(agg["time_bin"], agg["tumor_rel_enh"], marker="o", label=cohort)
    ax.axvline(TARGET_EARLY_S, linestyle="--", linewidth=1)
    ax.axvline(TARGET_LATE_S, linestyle=":", linewidth=1)
    ax.set_xlabel("Acquisition time (s)")
    ax.set_ylabel("Mean tumor relative enhancement")
    ax.set_title("Observed tumor enhancement curves by external cohort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_observed_enhancement_curves_by_cohort.png", dpi=300)
    plt.close(fig)


def plot_bar_metric(df: pd.DataFrame, group_col: str, metric: str, filename: str, title: str, ylabel: str) -> None:
    sub = df.dropna(subset=[metric])
    if sub.empty or group_col not in sub.columns:
        return
    order = [x for x in ["good", "acceptable", "poor", "unknown"] if x in set(sub[group_col].astype(str))]
    if not order:
        order = sorted(sub[group_col].dropna().astype(str).unique())
    means = [sub.loc[sub[group_col].astype(str) == g, metric].mean() for g in order]
    errs = [sub.loc[sub[group_col].astype(str) == g, metric].std() for g in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(order, means, yerr=errs, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x: str, y: str, filename: str, title: str, xlabel: str, ylabel: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    sub = df.dropna(subset=[x, y])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    cohorts = sorted(sub["cohort"].dropna().unique()) if "cohort" in sub.columns else ["ALL"]
    for c in cohorts:
        g = sub[sub["cohort"] == c] if "cohort" in sub.columns else sub
        ax.scatter(g[x], g[y], s=18, alpha=0.65, label=c)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(cohorts) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def write_takeaways(df: pd.DataFrame, corr_df: pd.DataFrame, post_df: pd.DataFrame) -> None:
    lines = []
    lines.append("CONTRAST-KINETIC SHIFT ANALYSIS — QUICK TAKEAWAYS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Cases analyzed: {len(df)}")
    if "cohort" in df.columns:
        lines.append("Cohort counts:")
        for c, n in df["cohort"].value_counts().items():
            lines.append(f"  {c}: {n}")
    lines.append("")
    if "first_post_s" in df.columns:
        lines.append("First post-contrast timing by cohort:")
        for c, grp in df.groupby("cohort"):
            vals = grp["first_post_s"].dropna()
            if len(vals):
                lines.append(f"  {c}: mean={vals.mean():.1f}s, median={vals.median():.1f}s, n={len(vals)}")
    lines.append("")
    if "v2_lcc_dice" in df.columns:
        lines.append(f"Overall v2+LCC Dice: {df['v2_lcc_dice'].mean():.4f}")
        if "overall" in df.columns:
            lines.append("v2+LCC Dice by phase quality:")
            for q, grp in df.groupby("overall"):
                vals = grp["v2_lcc_dice"].dropna()
                if len(vals):
                    lines.append(f"  {q}: mean={vals.mean():.4f}, n={len(vals)}")
    lines.append("")
    if not post_df.empty:
        lines.append("Postprocessing comparison, overall mean Dice:")
        for col in ["v2_raw_dice", "v2_lcc_dice", "v2_kinetic_cc_dice"]:
            if col in df.columns:
                lines.append(f"  {col}: {df[col].mean():.4f}")
    lines.append("")
    if not corr_df.empty:
        lines.append("Strongest absolute Spearman correlations with v2+LCC Dice:")
        tmp = corr_df[corr_df["metric"] == "v2_lcc_dice"].copy()
        if not tmp.empty:
            tmp["abs_rho"] = tmp["spearman_rho"].abs()
            tmp = tmp.sort_values("abs_rho", ascending=False).head(8)
            for _, r in tmp.iterrows():
                lines.append(f"  {r['feature']}: rho={r['spearman_rho']:.3f}, p={r['p_value']:.4g}, n={int(r['n'])}")
    lines.append("")
    lines.append("Suggested manuscript framing:")
    lines.append("  External degradation should be framed as measured acquisition/contrast-kinetic shift,")
    lines.append("  not simply as an unexplained low external Dice. This script produces the tables and")
    lines.append("  figures needed to support that statement.")
    lines.append("")
    lines.append("Important caution:")
    lines.append("  Kinetic-component postprocessing is exploratory and image-only. It should not be")
    lines.append("  described as tuned unless separately validated. The primary no-retraining result")
    lines.append("  should remain v2 + largest connected component if that is the pre-specified rule.")

    (OUT_DIR / "article_takeaways.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main analysis
# =============================================================================


def main() -> None:
    ensure_dirs()
    print("Loading clinical table and phase report...")
    clinical = load_clinical_table()
    phase_report = load_phase_report()

    phase_lookup = {}
    if not phase_report.empty:
        phase_lookup = {r["case_id_upper"]: r.to_dict() for _, r in phase_report.iterrows()}

    case_dirs = sorted([p for p in IMAGES_DIR.iterdir() if p.is_dir()])
    case_dirs = [p for p in case_dirs if not p.name.upper().startswith("ISPY2")]
    if LIMIT_CASES is not None:
        case_dirs = case_dirs[: int(LIMIT_CASES)]

    print(f"Cases to analyze: {len(case_dirs)}")
    print(f"Output folder: {OUT_DIR}")

    clinical_lookup = {r["case_id_upper"]: r.to_dict() for _, r in clinical.iterrows()}

    rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    for idx, case_dir in enumerate(case_dirs, start=1):
        case_id = case_dir.name.upper()
        cohort = case_to_cohort(case_id)
        phase_files = sorted(case_dir.glob("*.nii.gz"))
        if len(phase_files) < 2:
            errors.append({"case_id": case_id, "error": "too few phase files"})
            continue

        clinical_row = clinical_lookup.get(case_id, {})
        times = clinical_row.get("times_parsed", [])
        if not isinstance(times, list):
            times = parse_times(clinical_row.get("acquisition_times", ""))
        if len(times) != len(phase_files):
            # fallback: keep ordering and use indices as pseudo-time; still process but mark mismatch
            times = [float(i) for i in range(len(phase_files))]
            time_mismatch = True
        else:
            time_mismatch = False

        try:
            ref_img = read_img(phase_files[0])
            phase_imgs = [read_img(p) for p in phase_files]
            phase_imgs = [resample_to_reference(im, ref_img, is_mask=False) for im in phase_imgs]
            phase_arrays = [img_arr(im) for im in phase_imgs]
            pre = phase_arrays[0]
            fg = foreground_mask(pre)
            rel_stack, post_times = build_relative_enhancement_stack(phase_arrays, times, fg)

            gt_path = GT_DIR / f"{case_id.lower()}.nii.gz"
            gt = load_mask(gt_path, ref_img)
            if gt is None:
                errors.append({"case_id": case_id, "error": f"missing GT mask: {gt_path.name}"})
                continue

            # Prediction masks if available.
            pred_v1 = None
            pred_v2 = None
            p1 = find_prediction(PRED_V1_DIR, case_id)
            p2 = find_prediction(PRED_V2_DIR, case_id)
            if p1 is not None:
                pred_v1 = load_mask(p1, ref_img)
            if p2 is not None:
                pred_v2 = load_mask(p2, ref_img)

            pred_v1_lcc = keep_largest_component(pred_v1) if pred_v1 is not None else None
            pred_v2_lcc = keep_largest_component(pred_v2) if pred_v2 is not None else None
            pred_v2_kin = None
            kin_comp_info: Dict[str, float] = {}
            if pred_v2 is not None:
                pred_v2_kin, kin_comp_info = kinetic_component_selection(pred_v2, rel_stack, post_times, fg)

            # Masks for kinetic analysis.
            ring = peritumor_ring(gt, fg)
            gt_k = compute_kinetics(rel_stack, post_times, gt)
            ring_k = compute_kinetics(rel_stack, post_times, ring)
            fg_k = compute_kinetics(rel_stack, post_times, fg)

            # Tumor-to-background kinetic ratios/differences.
            tumor_to_ring_max_diff = gt_k.max_enh - ring_k.max_enh if np.isfinite(gt_k.max_enh) and np.isfinite(ring_k.max_enh) else np.nan
            tumor_to_ring_auc_diff = gt_k.auc_enh_norm - ring_k.auc_enh_norm if np.isfinite(gt_k.auc_enh_norm) and np.isfinite(ring_k.auc_enh_norm) else np.nan
            tumor_to_fg_max_diff = gt_k.max_enh - fg_k.max_enh if np.isfinite(gt_k.max_enh) and np.isfinite(fg_k.max_enh) else np.nan

            pr = phase_lookup.get(case_id, {})
            early_t = float(pr.get("early_t_s", np.nan)) if str(pr.get("early_t_s", "")).strip() not in ["", "nan"] else np.nan
            late_t = float(pr.get("late_t_s", np.nan)) if str(pr.get("late_t_s", "")).strip() not in ["", "nan"] else np.nan
            early_dev = float(pr.get("early_dev_s", np.nan)) if str(pr.get("early_dev_s", "")).strip() not in ["", "nan"] else np.nan
            late_dev = float(pr.get("late_dev_s", np.nan)) if str(pr.get("late_dev_s", "")).strip() not in ["", "nan"] else np.nan
            overall_quality = str(pr.get("overall", phase_quality_from_devs(early_dev, late_dev)))

            row: Dict[str, object] = {
                "case_id": case_id,
                "cohort": cohort,
                "n_phases": len(phase_files),
                "times": str(times),
                "time_mismatch_flag": int(time_mismatch),
                "first_post_s": float(times[1]) if len(times) > 1 else np.nan,
                "last_post_s": float(times[-1]) if len(times) > 1 else np.nan,
                "acquisition_time_span_s": float(times[-1] - times[0]) if len(times) > 1 else np.nan,
                "early_t_s": early_t,
                "late_t_s": late_t,
                "early_dev_s": early_dev,
                "late_dev_s": late_dev,
                "overall": overall_quality,
                "early_quality": pr.get("early_quality", "unknown"),
                "late_quality": pr.get("late_quality", "unknown"),
                "gt_voxels": int(gt.sum()),
                "fg_voxels": int(fg.sum()),
                "ring_voxels": int(ring.sum()),
                "tumor_to_ring_max_enh_diff": tumor_to_ring_max_diff,
                "tumor_to_ring_auc_norm_diff": tumor_to_ring_auc_diff,
                "tumor_to_fg_max_enh_diff": tumor_to_fg_max_diff,
            }
            row.update(gt_k.as_dict("gt"))
            row.update(ring_k.as_dict("ring"))
            row.update(fg_k.as_dict("fg"))
            row.update(metrics_for_prediction("v1_raw", pred_v1, gt))
            row.update(metrics_for_prediction("v1_lcc", pred_v1_lcc, gt))
            row.update(metrics_for_prediction("v2_raw", pred_v2, gt))
            row.update(metrics_for_prediction("v2_lcc", pred_v2_lcc, gt))
            row.update(metrics_for_prediction("v2_kinetic_cc", pred_v2_kin, gt))
            row.update(kin_comp_info)

            # useful deltas
            if np.isfinite(row.get("v2_lcc_dice", np.nan)) and np.isfinite(row.get("v2_raw_dice", np.nan)):
                row["delta_v2_lcc_minus_raw_dice"] = row["v2_lcc_dice"] - row["v2_raw_dice"]
            else:
                row["delta_v2_lcc_minus_raw_dice"] = np.nan
            if np.isfinite(row.get("v2_kinetic_cc_dice", np.nan)) and np.isfinite(row.get("v2_lcc_dice", np.nan)):
                row["delta_kineticcc_minus_lcc_dice"] = row["v2_kinetic_cc_dice"] - row["v2_lcc_dice"]
            else:
                row["delta_kineticcc_minus_lcc_dice"] = np.nan

            rows.append(row)

            # case curve rows
            if rel_stack.size > 0:
                for j, t in enumerate(post_times):
                    curve_rows.append({
                        "case_id": case_id,
                        "cohort": cohort,
                        "overall": overall_quality,
                        "time_s": float(t),
                        "tumor_rel_enh": safe_mean(rel_stack[j], gt),
                        "ring_rel_enh": safe_mean(rel_stack[j], ring),
                        "fg_rel_enh": safe_mean(rel_stack[j], fg),
                    })

            if SAVE_PREVIEWS and len(rows) <= PREVIEW_LIMIT:
                plot_case_preview(
                    case_id=case_id,
                    phase_arrays=phase_arrays,
                    rel_stack=rel_stack,
                    gt=gt,
                    pred_raw=pred_v2,
                    pred_lcc=pred_v2_lcc,
                    out_path=PREVIEW_DIR / f"{case_id}_kinetic_preview.png",
                )

            if idx % 25 == 0:
                print(f"  processed {idx}/{len(case_dirs)}")

        except Exception as e:
            errors.append({"case_id": case_id, "error": repr(e)})
            print(f"ERROR {case_id}: {e}")

    df = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curve_rows)
    err_df = pd.DataFrame(errors)

    case_csv = METRICS_DIR / "contrast_kinetic_case_features_and_metrics.csv"
    curve_csv = METRICS_DIR / "contrast_kinetic_curves_long_format.csv"
    err_csv = METRICS_DIR / "contrast_kinetic_errors.csv"
    df.to_csv(case_csv, index=False)
    curve_df.to_csv(curve_csv, index=False)
    err_df.to_csv(err_csv, index=False)

    # Summary tables.
    metric_cols = [
        "first_post_s", "last_post_s", "early_dev_s", "late_dev_s",
        "gt_max_enh", "gt_auc_enh_norm", "gt_observed_uptake_rate", "gt_late_slope", "gt_washout_index",
        "tumor_to_ring_max_enh_diff", "tumor_to_ring_auc_norm_diff", "tumor_to_fg_max_enh_diff",
        "v2_raw_dice", "v2_raw_precision", "v2_raw_recall",
        "v2_lcc_dice", "v2_lcc_precision", "v2_lcc_recall",
        "v2_kinetic_cc_dice", "v2_kinetic_cc_precision", "v2_kinetic_cc_recall",
        "delta_v2_lcc_minus_raw_dice", "delta_kineticcc_minus_lcc_dice",
    ]
    for name, groups in {
        "overall": [],
        "by_cohort": ["cohort"],
        "by_phase_quality": ["overall"],
        "by_cohort_and_phase_quality": ["cohort", "overall"],
    }.items():
        summary = summarize(df, groups, metric_cols)
        summary.to_csv(METRICS_DIR / f"summary_{name}.csv", index=False)

    # Correlation analysis.
    corr_features = [
        "first_post_s", "last_post_s", "acquisition_time_span_s", "early_dev_s", "late_dev_s",
        "gt_max_enh", "gt_auc_enh_norm", "gt_observed_uptake_rate", "gt_late_slope", "gt_washout_index",
        "ring_max_enh", "ring_auc_enh_norm", "fg_max_enh", "fg_auc_enh_norm",
        "tumor_to_ring_max_enh_diff", "tumor_to_ring_auc_norm_diff", "tumor_to_fg_max_enh_diff",
    ]
    corr_metrics = ["v2_raw_dice", "v2_lcc_dice", "v2_lcc_precision", "v2_lcc_recall"]
    corr_rows = []
    for feat in corr_features:
        if feat not in df.columns:
            continue
        for met in corr_metrics:
            if met not in df.columns:
                continue
            sub = df[[feat, met]].apply(pd.to_numeric, errors="coerce").dropna()
            sub = sub[np.isfinite(sub[feat]) & np.isfinite(sub[met])]
            if len(sub) >= 10:
                rho, p = spearmanr(sub[feat], sub[met])
                corr_rows.append({
                    "feature": feat,
                    "metric": met,
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n": int(len(sub)),
                })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(METRICS_DIR / "correlations_kinetic_timing_vs_external_metrics.csv", index=False)

    # Postprocessing comparison table.
    post_cols = [c for c in ["v2_raw_dice", "v2_lcc_dice", "v2_kinetic_cc_dice", "v2_raw_precision", "v2_lcc_precision", "v2_kinetic_cc_precision", "v2_raw_recall", "v2_lcc_recall", "v2_kinetic_cc_recall"] if c in df.columns]
    post_df = summarize(df, [], post_cols)
    post_df.to_csv(METRICS_DIR / "postprocessing_comparison_overall.csv", index=False)

    # Figures.
    if len(df):
        plot_box_by_cohort(df)
        plot_mean_curves(curve_df)
        plot_bar_metric(df, "overall", "v2_lcc_dice", "fig03_v2_lcc_dice_by_phase_quality.png", "v2+LCC Dice by phase quality", "Dice")
        plot_bar_metric(df, "cohort", "v2_lcc_dice", "fig04_v2_lcc_dice_by_cohort.png", "v2+LCC Dice by cohort", "Dice")
        plot_scatter(df, "first_post_s", "v2_lcc_dice", "fig05_first_post_time_vs_v2_lcc_dice.png", "First post-contrast time vs external Dice", "First post-contrast time (s)", "v2+LCC Dice")
        plot_scatter(df, "tumor_to_ring_max_enh_diff", "v2_lcc_dice", "fig06_tumor_to_ring_enhancement_vs_dice.png", "Tumor-to-ring enhancement contrast vs external Dice", "Tumor-to-ring max enhancement difference", "v2+LCC Dice")
        plot_scatter(df, "gt_observed_uptake_rate", "v2_lcc_dice", "fig07_observed_uptake_rate_vs_dice.png", "Observed uptake rate vs external Dice", "Observed uptake rate", "v2+LCC Dice")

        # postprocessing method figure
        means = []
        labels = []
        for col, label in [("v2_raw_dice", "v2 raw"), ("v2_lcc_dice", "v2 + LCC"), ("v2_kinetic_cc_dice", "v2 + kinetic component")]:
            if col in df.columns:
                means.append(float(pd.to_numeric(df[col], errors="coerce").mean()))
                labels.append(label)
        if means:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(labels, means)
            ax.set_ylabel("Dice")
            ax.set_title("No-retraining external postprocessing comparison")
            ax.set_ylim(0, max(0.75, max(means) + 0.05))
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "fig08_postprocessing_comparison.png", dpi=300)
            plt.close(fig)

    write_takeaways(df, corr_df, post_df)

    print()
    print("DONE")
    print(f"Cases analyzed: {len(df)}")
    print(f"Errors: {len(err_df)}")
    print(f"Case metrics: {case_csv}")
    print(f"Curve CSV: {curve_csv}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Previews: {PREVIEW_DIR}")
    print(f"Takeaways: {OUT_DIR / 'article_takeaways.txt'}")

    if len(df) and "v2_lcc_dice" in df.columns:
        print()
        print("Key quick numbers:")
        print(f"  v2 raw Dice mean: {df['v2_raw_dice'].mean():.4f}" if "v2_raw_dice" in df.columns else "")
        print(f"  v2 + LCC Dice mean: {df['v2_lcc_dice'].mean():.4f}")
        if "v2_kinetic_cc_dice" in df.columns:
            print(f"  v2 + kinetic component Dice mean: {df['v2_kinetic_cc_dice'].mean():.4f}")
        print("  By phase quality:")
        for q, grp in df.groupby("overall"):
            print(f"    {q}: n={len(grp)}, v2+LCC Dice={grp['v2_lcc_dice'].mean():.4f}")


if __name__ == "__main__":
    main()
