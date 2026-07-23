# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    from scipy import ndimage
except Exception as e:
    raise RuntimeError(
        "scipy is required for connected-component postprocessing. Install with: pip install scipy"
    ) from e

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

PRED_V1_DIR = Path(r"C:\nnw\mama_mia_output")
PRED_V2_DIR = Path(r"C:\nnw\mama_mia_output_v2")
GT_DIR = PROJECT_ROOT / "segmentations_2" / "expert"
PHASE_REPORT = PROJECT_ROOT / "external_manifest" / "phase_selection_report.csv"

OUTPUT_ROOT = PROJECT_ROOT / "09_external_postprocessing_devtest"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURES_DIR = OUTPUT_ROOT / "figures"
PREVIEWS_DIR = OUTPUT_ROOT / "previews"

# Stratified split for honest tuning/evaluation.
DEV_FRACTION = 0.30
RANDOM_SEED = 2813

# Candidate postprocessing thresholds.
SMALL_COMPONENT_THRESHOLDS_VOXELS = [50, 100, 250, 500, 1000]
CONDITIONAL_LCC_LARGEST_FRACTION_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
CONDITIONAL_LCC_N_COMPONENTS_THRESHOLDS = [2, 3, 4, 5]

# Candidate volume thresholds are derived from DEV raw-v2 predicted volumes.
# This uses no ground truth for applying the rule, but the threshold is selected
# using DEV metrics, so TEST is the honest evaluation.
USE_DEV_VOLUME_QUANTILE_RULES = True
PRED_VOLUME_QUANTILES = [0.75, 0.85, 0.90, 0.95]

# Selection score. Higher is better.
# A small recall penalty prevents selecting very conservative masks that boost
# Dice only by destroying recall.
RECALL_DROP_PENALTY_WEIGHT = 0.25
MIN_REPORT_DELTA = 0.005

# Save a small number of visual overlay previews for helpful/harmful methods.
SAVE_PREVIEWS = True
N_PREVIEWS_PER_GROUP = 8

# If True, save processed NIfTI masks for dev-selected method on TEST only.
# Keep False during exploration to avoid large outputs.
SAVE_SELECTED_TEST_MASKS = False
SELECTED_MASK_DIR = OUTPUT_ROOT / "selected_test_masks"


# =============================================================================
# Basic utilities
# =============================================================================


def ensure_dirs() -> None:
    for d in [OUTPUT_ROOT, METRICS_DIR, FIGURES_DIR, PREVIEWS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if SAVE_SELECTED_TEST_MASKS:
        SELECTED_MASK_DIR.mkdir(parents=True, exist_ok=True)


def strip_nii_gz(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
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


def load_binary(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return (arr > 0).astype(np.uint8)


def read_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def get_spacing_from_image(path: Path) -> Tuple[float, float, float]:
    img = sitk.ReadImage(str(path))
    sp = img.GetSpacing()  # x, y, z
    return float(sp[0]), float(sp[1]), float(sp[2])


def save_binary_like(mask: np.ndarray, reference_path: Path, output_path: Path) -> None:
    ref = sitk.ReadImage(str(reference_path))
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(ref)
    sitk.WriteImage(out, str(output_path), useCompression=True)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    denom = pred_sum + gt_sum
    if denom == 0:
        return 1.0
    inter = int(np.logical_and(pred, gt).sum())
    return float(2.0 * inter / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(pred, gt).sum())
    return float(inter / union)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    tp = int(np.logical_and(pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    precision = float(tp / pred_sum) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
    recall = float(tp / gt_sum) if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)
    return precision, recall


def volume_cm3(mask: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> float:
    voxel_mm3 = spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]
    return float(mask.sum() * voxel_mm3 / 1000.0)


def safe_ratio(num: float, den: float) -> float:
    if den <= 0 or not np.isfinite(den):
        return np.nan
    return float(num / den)


# =============================================================================
# Connected-component operations
# =============================================================================

STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.uint8)


def component_labels(mask: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.int32), 0, np.array([], dtype=np.float64)
    labels, n = ndimage.label(mask.astype(bool), structure=STRUCTURE_26)
    if n == 0:
        return labels.astype(np.int32), 0, np.array([], dtype=np.float64)
    sizes = ndimage.sum(mask.astype(np.uint8), labels, index=np.arange(1, n + 1))
    return labels.astype(np.int32), int(n), np.asarray(sizes, dtype=np.float64)


def component_features(mask: np.ndarray) -> Dict[str, float]:
    total = float(mask.sum())
    labels, n, sizes = component_labels(mask)
    if n == 0 or total <= 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0.0,
            "second_largest_component_voxels": 0.0,
            "largest_component_fraction": 0.0,
            "second_largest_component_fraction": 0.0,
        }
    sorted_sizes = np.sort(sizes)[::-1]
    largest = float(sorted_sizes[0])
    second = float(sorted_sizes[1]) if len(sorted_sizes) > 1 else 0.0
    return {
        "n_components": float(n),
        "largest_component_voxels": largest,
        "second_largest_component_voxels": second,
        "largest_component_fraction": float(largest / total),
        "second_largest_component_fraction": float(second / total),
    }


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n, sizes = component_labels(mask)
    if n <= 1:
        return mask.astype(np.uint8)
    largest_label = int(np.argmax(sizes) + 1)
    return (labels == largest_label).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    labels, n, sizes = component_labels(mask)
    if n == 0:
        return mask.astype(np.uint8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i, s in enumerate(sizes, start=1):
        if float(s) >= float(min_voxels):
            out[labels == i] = 1
    return out


def fill_holes_3d(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(np.uint8)
    try:
        return ndimage.binary_fill_holes(mask.astype(bool)).astype(np.uint8)
    except Exception:
        return mask.astype(np.uint8)


def intersection(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    return np.logical_and(mask_a > 0, mask_b > 0).astype(np.uint8)


def union(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    return np.logical_or(mask_a > 0, mask_b > 0).astype(np.uint8)


# =============================================================================
# Candidate method definitions
# =============================================================================

@dataclass
class Candidate:
    name: str
    family: str
    description: str
    apply: Callable[[Dict[str, Any]], np.ndarray]
    uses_dev_tuned_param: bool = False
    paper_safe_if_selected_on_dev: bool = True


def build_base_candidates(pred_volume_thresholds: Optional[Dict[str, float]] = None) -> List[Candidate]:
    """Build candidate methods. pred_volume_thresholds contains dev-derived cm3 cutoffs."""
    cands: List[Candidate] = []

    # Raw and standard baselines.
    cands.append(Candidate(
        name="v2_raw",
        family="baseline",
        description="Timing-aware v2 raw binary prediction.",
        apply=lambda x: x["v2"],
    ))
    cands.append(Candidate(
        name="v2_lcc",
        family="standard_lcc",
        description="Always keep largest connected component from v2 prediction.",
        apply=lambda x: keep_largest_component(x["v2"]),
    ))
    cands.append(Candidate(
        name="v2_lcc_fillholes",
        family="standard_lcc",
        description="v2 largest connected component followed by 3D hole filling.",
        apply=lambda x: fill_holes_3d(keep_largest_component(x["v2"])),
    ))

    # Small-component removal.
    for min_vox in SMALL_COMPONENT_THRESHOLDS_VOXELS:
        cands.append(Candidate(
            name=f"v2_remove_small_{min_vox}",
            family="small_component_removal",
            description=f"Remove v2 connected components smaller than {min_vox} voxels.",
            apply=lambda x, mv=min_vox: remove_small_components(x["v2"], mv),
        ))
        cands.append(Candidate(
            name=f"v2_remove_small_{min_vox}_fillholes",
            family="small_component_removal",
            description=f"Remove v2 components <{min_vox} voxels, then fill holes.",
            apply=lambda x, mv=min_vox: fill_holes_3d(remove_small_components(x["v2"], mv)),
        ))

    # Conditional LCC by raw-v2 component structure.
    for t in CONDITIONAL_LCC_LARGEST_FRACTION_THRESHOLDS:
        label = str(t).replace(".", "p")
        cands.append(Candidate(
            name=f"v2_cond_lcc_largestfrac_le_{label}",
            family="conditional_lcc_largest_fraction",
            description=f"Apply LCC only if v2 largest-component fraction <= {t:.2f}.",
            apply=lambda x, thr=t: keep_largest_component(x["v2"]) if x["v2_features"]["largest_component_fraction"] <= thr and x["v2_features"]["n_components"] >= 2 else x["v2"],
        ))

    for n_thr in CONDITIONAL_LCC_N_COMPONENTS_THRESHOLDS:
        cands.append(Candidate(
            name=f"v2_cond_lcc_ncomp_ge_{n_thr}",
            family="conditional_lcc_n_components",
            description=f"Apply LCC only if v2 has at least {n_thr} connected components.",
            apply=lambda x, nt=n_thr: keep_largest_component(x["v2"]) if x["v2_features"]["n_components"] >= nt else x["v2"],
        ))

    # Dev-derived predicted-volume conditional LCC.
    if pred_volume_thresholds:
        for key, cutoff in pred_volume_thresholds.items():
            safe_key = key.replace(".", "p")
            cands.append(Candidate(
                name=f"v2_cond_lcc_predvol_gt_{safe_key}",
                family="conditional_lcc_predicted_volume",
                description=f"Apply LCC only if v2 predicted volume > {cutoff:.3f} cm3 ({key} dev quantile).",
                apply=lambda x, cv=cutoff: keep_largest_component(x["v2"]) if x["v2_pred_vol_cm3"] > cv else x["v2"],
                uses_dev_tuned_param=True,
            ))

    # v1/v2 consensus variants.
    cands.append(Candidate(
        name="v1v2_intersection",
        family="v1_v2_consensus",
        description="Intersection of v1 and v2 predictions; conservative consensus.",
        apply=lambda x: intersection(x["v1"], x["v2"]),
    ))
    cands.append(Candidate(
        name="v1v2_intersection_lcc",
        family="v1_v2_consensus",
        description="Intersection of v1 and v2 predictions followed by LCC.",
        apply=lambda x: keep_largest_component(intersection(x["v1"], x["v2"])),
    ))
    cands.append(Candidate(
        name="v1v2_union_lcc",
        family="v1_v2_consensus",
        description="Union of v1 and v2 predictions followed by LCC.",
        apply=lambda x: keep_largest_component(union(x["v1"], x["v2"])),
    ))
    cands.append(Candidate(
        name="v2_lcc_intersect_v1",
        family="v1_v2_consensus",
        description="v2 LCC intersected with v1 raw prediction.",
        apply=lambda x: intersection(keep_largest_component(x["v2"]), x["v1"]),
    ))
    cands.append(Candidate(
        name="v2_lcc_union_v1_lcc",
        family="v1_v2_consensus",
        description="Union of v2 LCC and v1 LCC.",
        apply=lambda x: union(keep_largest_component(x["v2"]), keep_largest_component(x["v1"])),
    ))

    # v1 baselines for context, not expected final.
    cands.append(Candidate(
        name="v1_raw",
        family="context_v1",
        description="Baseline v1 raw prediction for reference.",
        apply=lambda x: x["v1"],
        paper_safe_if_selected_on_dev=False,
    ))
    cands.append(Candidate(
        name="v1_lcc",
        family="context_v1",
        description="v1 largest connected component for reference.",
        apply=lambda x: keep_largest_component(x["v1"]),
        paper_safe_if_selected_on_dev=False,
    ))

    return cands


# =============================================================================
# Case discovery and splitting
# =============================================================================


def discover_cases() -> pd.DataFrame:
    if not PRED_V1_DIR.exists():
        raise FileNotFoundError(f"PRED_V1_DIR not found: {PRED_V1_DIR}")
    if not PRED_V2_DIR.exists():
        raise FileNotFoundError(f"PRED_V2_DIR not found: {PRED_V2_DIR}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT_DIR not found: {GT_DIR}")

    v1_files = {strip_nii_gz(p).upper(): p for p in PRED_V1_DIR.glob("*.nii.gz")}
    v2_files = {strip_nii_gz(p).upper(): p for p in PRED_V2_DIR.glob("*.nii.gz")}

    rows = []
    for case_id in sorted(set(v1_files) & set(v2_files)):
        gt_path = GT_DIR / f"{case_id.lower()}.nii.gz"
        if not gt_path.exists():
            continue
        rows.append({
            "case_id": case_id,
            "cohort": cohort_from_case(case_id),
            "v1_path": str(v1_files[case_id]),
            "v2_path": str(v2_files[case_id]),
            "gt_path": str(gt_path),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No cases found with v1, v2, and GT masks.")

    # Merge phase report if available.
    if PHASE_REPORT.exists():
        try:
            ph = pd.read_csv(PHASE_REPORT)
            if "case_id" in ph.columns:
                ph["case_id"] = ph["case_id"].astype(str).str.upper()
                keep_cols = [c for c in [
                    "case_id", "overall", "early_quality", "late_quality",
                    "early_t_s", "late_t_s", "early_dev_s", "late_dev_s",
                    "n_phases"
                ] if c in ph.columns]
                df = df.merge(ph[keep_cols], on="case_id", how="left")
        except Exception as e:
            print(f"WARNING: could not read phase report: {e}")

    if "overall" not in df.columns:
        df["overall"] = "unknown"
    df["overall"] = df["overall"].fillna("unknown").astype(str)

    df["split_group"] = df["cohort"].astype(str) + "__" + df["overall"].astype(str)
    return df


def assign_dev_test_split(df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    out = df.copy()
    out["split"] = "test"

    for _, idxs in out.groupby("split_group").groups.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        if n >= 4:
            n_dev = max(1, int(round(n * DEV_FRACTION)))
        elif n >= 2:
            n_dev = 1
        else:
            n_dev = 0
        dev_idxs = idxs[:n_dev]
        out.loc[dev_idxs, "split"] = "dev"

    return out


# =============================================================================
# Metrics and summaries
# =============================================================================


def evaluate_prediction(case_meta: Dict[str, Any], method: str, mask: np.ndarray, gt: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> Dict[str, Any]:
    pred_vol = volume_cm3(mask, spacing_xyz)
    gt_vol = volume_cm3(gt, spacing_xyz)
    comp = component_features(mask)
    precision, recall = precision_recall(mask, gt)
    return {
        "case_id": case_meta["case_id"],
        "cohort": case_meta.get("cohort", "OTHER"),
        "overall": case_meta.get("overall", "unknown"),
        "split": case_meta.get("split", "unknown"),
        "method": method,
        "dice": dice_score(mask, gt),
        "iou": iou_score(mask, gt),
        "precision": precision,
        "recall": recall,
        "pred_vol_cm3": pred_vol,
        "gt_vol_cm3": gt_vol,
        "pred_to_gt_vol_ratio": safe_ratio(pred_vol, gt_vol),
        "pred_voxels": int(mask.sum()),
        "gt_voxels": int(gt.sum()),
        "n_components": comp["n_components"],
        "largest_component_fraction": comp["largest_component_fraction"],
        "second_largest_component_fraction": comp["second_largest_component_fraction"],
    }


def aggregate_summary(long_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = ["dice", "iou", "precision", "recall", "pred_vol_cm3", "gt_vol_cm3", "pred_to_gt_vol_ratio", "n_components", "largest_component_fraction"]
    rows = []
    for key, grp in long_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row["n_cases"] = int(grp["case_id"].nunique())
        row["n_rows"] = int(len(grp))
        for m in metric_cols:
            if m in grp.columns:
                vals = pd.to_numeric(grp[m], errors="coerce")
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_median"] = float(vals.median())
                row[f"{m}_std"] = float(vals.std())
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_on_dev(summary_by_split_method: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """Select best method using DEV only with recall-drop penalty."""
    dev = summary_by_split_method[summary_by_split_method["split"] == "dev"].copy()
    if dev.empty:
        raise RuntimeError("No dev rows available for selection.")

    baseline = dev[dev["method"] == "v2_lcc"]
    baseline_recall = float(baseline["recall_mean"].iloc[0]) if len(baseline) else float(dev["recall_mean"].mean())
    baseline_dice = float(baseline["dice_mean"].iloc[0]) if len(baseline) else float(dev["dice_mean"].mean())

    dev["recall_drop_vs_v2_lcc"] = baseline_recall - dev["recall_mean"]
    dev["recall_penalty"] = dev["recall_drop_vs_v2_lcc"].clip(lower=0.0) * RECALL_DROP_PENALTY_WEIGHT
    dev["selection_score"] = dev["dice_mean"] - dev["recall_penalty"]
    dev["dice_delta_vs_v2_lcc"] = dev["dice_mean"] - baseline_dice
    dev = dev.sort_values(["selection_score", "dice_mean"], ascending=False)
    return str(dev.iloc[0]["method"]), dev


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = RANDOM_SEED) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(values[idx])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def add_ci_to_method_summary(summary: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        method = r["method"]
        split = r["split"]
        vals = long_df[(long_df["method"] == method) & (long_df["split"] == split)]["dice"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(vals, n_boot=1000)
        rr = r.to_dict()
        rr["dice_ci95_low"] = ci_low
        rr["dice_ci95_high"] = ci_high
        rows.append(rr)
    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================


def save_bar(summary: pd.DataFrame, split: str, out_path: Path, top_n: int = 20) -> None:
    if plt is None:
        return
    sub = summary[summary["split"] == split].copy()
    if sub.empty:
        return
    sub = sub.sort_values("dice_mean", ascending=False).head(top_n)
    plt.figure(figsize=(10, max(4, 0.35 * len(sub))))
    y = np.arange(len(sub))
    plt.barh(y, sub["dice_mean"].values)
    plt.yticks(y, sub["method"].values, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Dice")
    plt.title(f"Top postprocessing candidates on {split.upper()} split")
    plt.xlim(0, max(0.8, float(sub["dice_mean"].max()) + 0.05))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_precision_recall_scatter(summary: pd.DataFrame, split: str, out_path: Path) -> None:
    if plt is None:
        return
    sub = summary[summary["split"] == split].copy()
    if sub.empty:
        return
    plt.figure(figsize=(7, 6))
    plt.scatter(sub["recall_mean"], sub["precision_mean"], s=50, alpha=0.8)
    for _, r in sub.iterrows():
        if r["method"] in ["v2_raw", "v2_lcc"] or r["dice_mean"] >= sub["dice_mean"].quantile(0.9):
            plt.annotate(r["method"], (r["recall_mean"], r["precision_mean"]), fontsize=7)
    plt.xlabel("Mean recall")
    plt.ylabel("Mean precision")
    plt.title(f"Precision/recall trade-off on {split.upper()} split")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_delta_vs_baseline(summary: pd.DataFrame, split: str, out_path: Path, baseline_method: str = "v2_lcc") -> None:
    if plt is None:
        return
    sub = summary[summary["split"] == split].copy()
    base = sub[sub["method"] == baseline_method]
    if sub.empty or base.empty:
        return
    baseline = float(base["dice_mean"].iloc[0])
    sub["delta_dice"] = sub["dice_mean"] - baseline
    sub = sub.sort_values("delta_dice", ascending=False).head(25)
    plt.figure(figsize=(10, max(4, 0.35 * len(sub))))
    y = np.arange(len(sub))
    plt.barh(y, sub["delta_dice"].values)
    plt.axvline(0, linewidth=1)
    plt.yticks(y, sub["method"].values, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Δ Dice vs {baseline_method}")
    plt.title(f"Postprocessing change relative to {baseline_method} on {split.upper()}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_by_cohort_plot(summary_by_cohort: pd.DataFrame, split: str, selected_methods: List[str], out_path: Path) -> None:
    if plt is None:
        return
    sub = summary_by_cohort[(summary_by_cohort["split"] == split) & (summary_by_cohort["method"].isin(selected_methods))].copy()
    if sub.empty:
        return
    pivot = sub.pivot(index="cohort", columns="method", values="dice_mean")
    pivot = pivot.reindex([c for c in ["DUKE", "ISPY1", "NACT", "OTHER"] if c in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Mean Dice")
    ax.set_title(f"Selected methods by cohort on {split.upper()} split")
    ax.set_ylim(0, max(0.8, float(np.nanmax(pivot.values)) + 0.05))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_preview(case_row: pd.Series, masks: Dict[str, np.ndarray], gt: np.ndarray, out_path: Path, title: str) -> None:
    if plt is None:
        return
    try:
        # choose slice with maximum GT, fallback to prediction.
        if gt.sum() > 0:
            z = int(np.argmax(gt.reshape(gt.shape[0], -1).sum(axis=1)))
        else:
            any_pred = next(iter(masks.values()))
            z = int(np.argmax(any_pred.reshape(any_pred.shape[0], -1).sum(axis=1)))

        panels = []
        labels = []
        for name, m in masks.items():
            panels.append(m[z])
            labels.append(name)

        n = len(panels)
        plt.figure(figsize=(3.5 * n, 3.6))
        for i, (p, lab) in enumerate(zip(panels, labels), start=1):
            plt.subplot(1, n, i)
            # Show combined mask classes: GT=1, pred=2, overlap=3 if possible.
            gt2 = gt[z].astype(np.uint8)
            pred2 = p.astype(np.uint8)
            combo = gt2 + 2 * pred2
            plt.imshow(combo, cmap="viridis", interpolation="nearest")
            plt.title(lab, fontsize=9)
            plt.axis("off")
        plt.suptitle(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except Exception:
        pass


# =============================================================================
# Main evaluation
# =============================================================================


def precompute_dev_volume_thresholds(cases_df: pd.DataFrame) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    if not USE_DEV_VOLUME_QUANTILE_RULES:
        return thresholds
    vols = []
    for _, row in cases_df[cases_df["split"] == "dev"].iterrows():
        try:
            v2 = load_binary(Path(row["v2_path"]))
            gt_path = Path(row["gt_path"])
            spacing = get_spacing_from_image(gt_path)
            vols.append(volume_cm3(v2, spacing))
        except Exception:
            continue
    if len(vols) >= 10:
        arr = np.asarray(vols, dtype=float)
        for q in PRED_VOLUME_QUANTILES:
            thresholds[f"q{int(q*100)}"] = float(np.quantile(arr, q))
    return thresholds


def run_evaluation(cases_df: pd.DataFrame, candidates: List[Candidate]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    errors = []
    n = len(cases_df)

    for ii, (_, row) in enumerate(cases_df.iterrows(), start=1):
        case_id = row["case_id"]
        try:
            gt_path = Path(row["gt_path"])
            v1_path = Path(row["v1_path"])
            v2_path = Path(row["v2_path"])
            gt = load_binary(gt_path)
            v1 = load_binary(v1_path)
            v2 = load_binary(v2_path)

            # Defensive shape check. If mismatch, skip case rather than crash.
            if gt.shape != v1.shape or gt.shape != v2.shape:
                errors.append({"case_id": case_id, "error": f"shape mismatch gt={gt.shape} v1={v1.shape} v2={v2.shape}"})
                continue

            spacing = get_spacing_from_image(gt_path)
            v2_features = component_features(v2)
            v1_features = component_features(v1)
            v2_pred_vol_cm3 = volume_cm3(v2, spacing)
            v1_pred_vol_cm3 = volume_cm3(v1, spacing)

            context = {
                "v1": v1,
                "v2": v2,
                "gt": gt,
                "v1_features": v1_features,
                "v2_features": v2_features,
                "v1_pred_vol_cm3": v1_pred_vol_cm3,
                "v2_pred_vol_cm3": v2_pred_vol_cm3,
            }
            meta = row.to_dict()

            for cand in candidates:
                try:
                    pred = cand.apply(context).astype(np.uint8)
                    if pred.shape != gt.shape:
                        errors.append({"case_id": case_id, "method": cand.name, "error": "candidate shape mismatch"})
                        continue
                    rec = evaluate_prediction(meta, cand.name, pred, gt, spacing)
                    rec["family"] = cand.family
                    rec["description"] = cand.description
                    rec["uses_dev_tuned_param"] = cand.uses_dev_tuned_param
                    rec["paper_safe_if_selected_on_dev"] = cand.paper_safe_if_selected_on_dev
                    rows.append(rec)
                except Exception as e:
                    errors.append({"case_id": case_id, "method": cand.name, "error": repr(e)})
                    continue
        except Exception as e:
            errors.append({"case_id": case_id, "error": repr(e)})
            continue

        if ii % 25 == 0 or ii == n:
            print(f"  evaluated {ii}/{n}")

    long_df = pd.DataFrame(rows)
    err_df = pd.DataFrame(errors)
    return long_df, err_df


def write_takeaways(
    cases_df: pd.DataFrame,
    long_df: pd.DataFrame,
    method_summary: pd.DataFrame,
    dev_selection_table: pd.DataFrame,
    selected_method: str,
    out_path: Path,
) -> None:
    lines = []
    lines.append("09 EXTERNAL POSTPROCESSING DEV/TEST ANALYSIS — QUICK TAKEAWAYS")
    lines.append("=" * 78)
    lines.append(f"Cases evaluated: {cases_df['case_id'].nunique()}")
    lines.append(f"Dev fraction: {DEV_FRACTION:.2f}  seed={RANDOM_SEED}")
    lines.append("")
    lines.append("Split counts:")
    for split, n in cases_df["split"].value_counts().to_dict().items():
        lines.append(f"  {split}: {n}")
    lines.append("")

    # Key baselines.
    lines.append("Main method performance by split:")
    for split in ["dev", "test"]:
        lines.append(f"  {split.upper()}:")
        sub = method_summary[method_summary["split"] == split].copy()
        for m in ["v2_raw", "v2_lcc", selected_method]:
            rr = sub[sub["method"] == m]
            if len(rr):
                r = rr.iloc[0]
                lines.append(
                    f"    {m:35s} Dice={r['dice_mean']:.4f} "
                    f"Prec={r['precision_mean']:.4f} Rec={r['recall_mean']:.4f} "
                    f"CI95=[{r.get('dice_ci95_low', np.nan):.4f}, {r.get('dice_ci95_high', np.nan):.4f}]"
                )
    lines.append("")

    lines.append(f"Dev-selected candidate: {selected_method}")
    base_test = method_summary[(method_summary["split"] == "test") & (method_summary["method"] == "v2_lcc")]
    sel_test = method_summary[(method_summary["split"] == "test") & (method_summary["method"] == selected_method)]
    if len(base_test) and len(sel_test):
        delta = float(sel_test.iloc[0]["dice_mean"] - base_test.iloc[0]["dice_mean"])
        lines.append(f"Held-out TEST ΔDice vs v2_lcc: {delta:+.4f}")
        if delta >= MIN_REPORT_DELTA:
            lines.append("Interpretation: dev-selected rule improved held-out test Dice enough to consider for main/supplementary reporting.")
        else:
            lines.append("Interpretation: dev-selected rule did not clearly improve over standard v2+LCC on held-out test.")
    lines.append("")

    lines.append("Top 10 methods on DEV by selection score:")
    for _, r in dev_selection_table.head(10).iterrows():
        lines.append(
            f"  {r['method']:40s} score={r['selection_score']:.4f} "
            f"Dice={r['dice_mean']:.4f} Prec={r['precision_mean']:.4f} Rec={r['recall_mean']:.4f}"
        )
    lines.append("")

    lines.append("Top 10 methods on TEST by Dice (screening only; not selection-safe):")
    test_top = method_summary[method_summary["split"] == "test"].sort_values("dice_mean", ascending=False).head(10)
    for _, r in test_top.iterrows():
        lines.append(
            f"  {r['method']:40s} Dice={r['dice_mean']:.4f} "
            f"Prec={r['precision_mean']:.4f} Rec={r['recall_mean']:.4f}"
        )
    lines.append("")

    lines.append("Important caution:")
    lines.append("  Use DEV to choose a rule and TEST for the honest estimate. Do not describe the TEST-best")
    lines.append("  method as tuned/final unless it was selected before viewing TEST results.")
    lines.append("  If no candidate clearly beats v2+LCC on TEST, keep v2+LCC as the paper-safe no-training pipeline.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Preview generation
# =============================================================================


def save_previews_if_requested(cases_df: pd.DataFrame, long_df: pd.DataFrame, selected_method: str, candidates_by_name: Dict[str, Candidate]) -> None:
    if not SAVE_PREVIEWS or plt is None:
        return

    try:
        # Find helpful and harmful cases for selected method vs v2_lcc on test.
        pivot = long_df[long_df["method"].isin(["v2_lcc", selected_method])].pivot_table(
            index="case_id", columns="method", values="dice", aggfunc="first"
        ).reset_index()
        if "v2_lcc" not in pivot.columns or selected_method not in pivot.columns:
            return
        pivot["delta_selected_vs_lcc"] = pivot[selected_method] - pivot["v2_lcc"]
        meta = cases_df[["case_id", "split", "v1_path", "v2_path", "gt_path"]].copy()
        pivot = pivot.merge(meta, on="case_id", how="left")
        pivot = pivot[pivot["split"] == "test"]

        helpful = pivot.sort_values("delta_selected_vs_lcc", ascending=False).head(N_PREVIEWS_PER_GROUP)
        harmful = pivot.sort_values("delta_selected_vs_lcc", ascending=True).head(N_PREVIEWS_PER_GROUP)

        for group_name, sub in [("helped", helpful), ("hurt", harmful)]:
            out_dir = PREVIEWS_DIR / f"selected_{selected_method}_{group_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for _, r in sub.iterrows():
                try:
                    gt = load_binary(Path(r["gt_path"]))
                    v1 = load_binary(Path(r["v1_path"]))
                    v2 = load_binary(Path(r["v2_path"]))
                    spacing = get_spacing_from_image(Path(r["gt_path"]))
                    ctx = {
                        "v1": v1,
                        "v2": v2,
                        "gt": gt,
                        "v1_features": component_features(v1),
                        "v2_features": component_features(v2),
                        "v1_pred_vol_cm3": volume_cm3(v1, spacing),
                        "v2_pred_vol_cm3": volume_cm3(v2, spacing),
                    }
                    sel_mask = candidates_by_name[selected_method].apply(ctx).astype(np.uint8)
                    v2_lcc = keep_largest_component(v2)
                    masks = {"GT+v2_lcc": v2_lcc, f"GT+{selected_method}": sel_mask}
                    title = f"{r['case_id']} delta={r['delta_selected_vs_lcc']:+.3f}"
                    make_preview(r, masks, gt, out_dir / f"{r['case_id']}_{group_name}.png", title)
                except Exception:
                    continue
    except Exception as e:
        print(f"WARNING: preview generation failed: {e}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ensure_dirs()
    print("09 External postprocessing dev/test analysis")
    print("============================================")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print(f"PRED v1      : {PRED_V1_DIR}")
    print(f"PRED v2      : {PRED_V2_DIR}")
    print(f"GT           : {GT_DIR}")
    print()

    cases = discover_cases()
    cases = assign_dev_test_split(cases)
    cases.to_csv(METRICS_DIR / "dev_test_split_cases.csv", index=False)

    print(f"Cases found: {len(cases)}")
    print("Split counts:")
    print(cases["split"].value_counts().to_string())
    print("Cohort x split counts:")
    print(pd.crosstab(cases["cohort"], cases["split"]).to_string())
    print()

    pred_vol_thresholds = precompute_dev_volume_thresholds(cases)
    if pred_vol_thresholds:
        (METRICS_DIR / "dev_predicted_volume_thresholds.json").write_text(json.dumps(pred_vol_thresholds, indent=2), encoding="utf-8")
        print("Dev-derived predicted-volume thresholds:")
        for k, v in pred_vol_thresholds.items():
            print(f"  {k}: {v:.3f} cm3")
        print()

    candidates = build_base_candidates(pred_vol_thresholds)
    candidates_by_name = {c.name: c for c in candidates}
    pd.DataFrame([{
        "method": c.name,
        "family": c.family,
        "description": c.description,
        "uses_dev_tuned_param": c.uses_dev_tuned_param,
        "paper_safe_if_selected_on_dev": c.paper_safe_if_selected_on_dev,
    } for c in candidates]).to_csv(METRICS_DIR / "candidate_methods_manifest.csv", index=False)

    print(f"Candidate methods: {len(candidates)}")
    for c in candidates[:8]:
        print(f"  {c.name}")
    if len(candidates) > 8:
        print(f"  ... {len(candidates)-8} more")
    print()

    print("Evaluating candidates...")
    long_df, err_df = run_evaluation(cases, candidates)
    long_df.to_csv(METRICS_DIR / "all_candidate_case_metrics_long.csv", index=False)
    if not err_df.empty:
        err_df.to_csv(METRICS_DIR / "errors.csv", index=False)
        print(f"WARNING: {len(err_df)} errors saved to metrics/errors.csv")
    print(f"Metric rows saved: {len(long_df)}")
    print()

    # Summary tables.
    print("Writing summary tables...")
    summary_method_split = aggregate_summary(long_df, ["split", "method"])
    summary_method_split = add_ci_to_method_summary(summary_method_split, long_df)
    summary_method_split.to_csv(METRICS_DIR / "summary_by_split_and_method.csv", index=False)

    summary_family_split = aggregate_summary(long_df, ["split", "family"])
    summary_family_split.to_csv(METRICS_DIR / "summary_by_split_and_family.csv", index=False)

    summary_cohort = aggregate_summary(long_df, ["split", "cohort", "method"])
    summary_cohort.to_csv(METRICS_DIR / "summary_by_split_cohort_method.csv", index=False)

    summary_phase = aggregate_summary(long_df, ["split", "overall", "method"])
    summary_phase.to_csv(METRICS_DIR / "summary_by_split_phasequality_method.csv", index=False)

    # Method comparison relative to v2_lcc within each split.
    rel_rows = []
    for split, sub in summary_method_split.groupby("split"):
        base = sub[sub["method"] == "v2_lcc"]
        raw = sub[sub["method"] == "v2_raw"]
        base_dice = float(base["dice_mean"].iloc[0]) if len(base) else np.nan
        raw_dice = float(raw["dice_mean"].iloc[0]) if len(raw) else np.nan
        for _, r in sub.iterrows():
            rr = r.to_dict()
            rr["delta_dice_vs_v2_lcc"] = float(r["dice_mean"] - base_dice) if np.isfinite(base_dice) else np.nan
            rr["delta_dice_vs_v2_raw"] = float(r["dice_mean"] - raw_dice) if np.isfinite(raw_dice) else np.nan
            rel_rows.append(rr)
    rel = pd.DataFrame(rel_rows)
    rel.sort_values(["split", "dice_mean"], ascending=[True, False]).to_csv(METRICS_DIR / "method_comparison_relative_to_baselines.csv", index=False)

    selected_method, dev_selection_table = select_best_on_dev(summary_method_split)
    dev_selection_table.to_csv(METRICS_DIR / "dev_selection_ranking.csv", index=False)

    # Test report for selected method.
    selected_report_rows = []
    for split in ["dev", "test"]:
        sub = summary_method_split[summary_method_split["split"] == split]
        base = sub[sub["method"] == "v2_lcc"]
        sel = sub[sub["method"] == selected_method]
        raw = sub[sub["method"] == "v2_raw"]
        if len(sel):
            row = {
                "split": split,
                "selected_method": selected_method,
                "selected_dice_mean": float(sel["dice_mean"].iloc[0]),
                "selected_precision_mean": float(sel["precision_mean"].iloc[0]),
                "selected_recall_mean": float(sel["recall_mean"].iloc[0]),
                "v2_lcc_dice_mean": float(base["dice_mean"].iloc[0]) if len(base) else np.nan,
                "v2_raw_dice_mean": float(raw["dice_mean"].iloc[0]) if len(raw) else np.nan,
            }
            row["delta_selected_vs_v2_lcc"] = row["selected_dice_mean"] - row["v2_lcc_dice_mean"]
            row["delta_selected_vs_v2_raw"] = row["selected_dice_mean"] - row["v2_raw_dice_mean"]
            selected_report_rows.append(row)
    selected_report = pd.DataFrame(selected_report_rows)
    selected_report.to_csv(METRICS_DIR / "paper_safe_dev_selected_method_report.csv", index=False)

    # Figures.
    print("Creating figures...")
    try:
        save_bar(summary_method_split, "dev", FIGURES_DIR / "fig01_top_methods_dev.png")
        save_bar(summary_method_split, "test", FIGURES_DIR / "fig02_top_methods_test_screening.png")
        save_delta_vs_baseline(summary_method_split, "dev", FIGURES_DIR / "fig03_delta_vs_v2_lcc_dev.png")
        save_delta_vs_baseline(summary_method_split, "test", FIGURES_DIR / "fig04_delta_vs_v2_lcc_test_screening.png")
        save_precision_recall_scatter(summary_method_split, "dev", FIGURES_DIR / "fig05_precision_recall_dev.png")
        save_precision_recall_scatter(summary_method_split, "test", FIGURES_DIR / "fig06_precision_recall_test.png")
        selected_for_plots = ["v2_raw", "v2_lcc", selected_method]
        # Add screening-best test method for visual context if different.
        test_best = summary_method_split[summary_method_split["split"] == "test"].sort_values("dice_mean", ascending=False).head(1)
        if len(test_best):
            tb = str(test_best.iloc[0]["method"])
            if tb not in selected_for_plots:
                selected_for_plots.append(tb)
        save_by_cohort_plot(summary_cohort, "test", selected_for_plots, FIGURES_DIR / "fig07_selected_methods_by_cohort_test.png")
    except Exception as e:
        print(f"WARNING: some figures failed: {e}")

    # Optional previews for selected method.
    save_previews_if_requested(cases, long_df, selected_method, candidates_by_name)

    # Optional masks for selected method on test.
    if SAVE_SELECTED_TEST_MASKS and selected_method in candidates_by_name:
        print("Saving selected-method masks for TEST split...")
        cand = candidates_by_name[selected_method]
        for _, row in cases[cases["split"] == "test"].iterrows():
            try:
                gt_path = Path(row["gt_path"])
                v1 = load_binary(Path(row["v1_path"]))
                v2 = load_binary(Path(row["v2_path"]))
                spacing = get_spacing_from_image(gt_path)
                ctx = {
                    "v1": v1,
                    "v2": v2,
                    "gt": load_binary(gt_path),
                    "v1_features": component_features(v1),
                    "v2_features": component_features(v2),
                    "v1_pred_vol_cm3": volume_cm3(v1, spacing),
                    "v2_pred_vol_cm3": volume_cm3(v2, spacing),
                }
                pred = cand.apply(ctx).astype(np.uint8)
                save_binary_like(pred, Path(row["v2_path"]), SELECTED_MASK_DIR / f"{row['case_id']}.nii.gz")
            except Exception:
                continue

    write_takeaways(cases, long_df, summary_method_split, dev_selection_table, selected_method, OUTPUT_ROOT / "article_takeaways_09.txt")

    print()
    print("DONE — 09 dev/test postprocessing analysis complete.")
    print(f"Metrics : {METRICS_DIR}")
    print(f"Figures : {FIGURES_DIR}")
    print(f"Takeaways: {OUTPUT_ROOT / 'article_takeaways_09.txt'}")
    print()
    print("Quick results:")
    print(f"  Dev-selected method: {selected_method}")
    display_cols = ["split", "method", "dice_mean", "precision_mean", "recall_mean", "dice_ci95_low", "dice_ci95_high"]
    quick = summary_method_split[summary_method_split["method"].isin(["v2_raw", "v2_lcc", selected_method])][display_cols].copy()
    quick = quick.sort_values(["split", "dice_mean"], ascending=[True, False])
    with pd.option_context("display.max_rows", 50, "display.max_columns", 20, "display.width", 180):
        print(quick.to_string(index=False))

    print()
    print("Top TEST methods by Dice (screening only — do not treat as tuned final unless selected on dev):")
    top_test = summary_method_split[summary_method_split["split"] == "test"].sort_values("dice_mean", ascending=False).head(12)
    with pd.option_context("display.max_rows", 20, "display.max_columns", 20, "display.width", 180):
        print(top_test[["method", "family", "dice_mean", "precision_mean", "recall_mean", "dice_ci95_low", "dice_ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print()
        print("FATAL ERROR")
        print("===========")
        print(repr(e))
        print()
        print("If case-level outputs were already saved, send me the last lines and I will make a finish-from-saved-CSV helper instead of recomputing.")
        raise
