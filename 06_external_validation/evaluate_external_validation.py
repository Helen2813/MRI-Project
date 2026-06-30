# evaluate_external_validation.py
# Computes segmentation metrics (Dice, HD95, precision, recall) for external validation.
# Compares nnU-Net ResEncL predictions vs expert MAMA-MIA segmentations.
# Stratifies results by cohort (DUKE / ISPY1 / NACT).
# Run: python evaluate_external_validation.py

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

PRED_DIR = Path(r"C:\nnw\mama_mia_output_v2")
GT_DIR   = Path(r"C:\Users\olegk\Desktop\MRI Project\segmentations_2\expert")
OUT_PATH = Path(r"C:\Users\olegk\Desktop\MRI Project\external_manifest\external_validation_metrics.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_binary(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    return (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)


def get_spacing(path: Path) -> tuple:
    img = sitk.ReadImage(str(path))
    return img.GetSpacing()  # (x, y, z) in mm


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else (1.0 if pred.sum() == 0 else 0.0)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple:
    tp = (pred & gt).sum()
    prec = float(tp / pred.sum()) if pred.sum() > 0 else 0.0
    rec  = float(tp / gt.sum())   if gt.sum()   > 0 else (1.0 if pred.sum() == 0 else 0.0)
    return prec, rec


def hausdorff_95(pred: np.ndarray, gt: np.ndarray, spacing: tuple) -> float:
    """
    Approximate HD95 using surface distance computation via SimpleITK.
    Returns HD95 in mm. Returns NaN if either mask is empty.
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    try:
        pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
        gt_img   = sitk.GetImageFromArray(gt.astype(np.uint8))
        # spacing: SimpleITK uses (x,y,z); array is (z,y,x)
        pred_img.SetSpacing((spacing[0], spacing[1], spacing[2]))
        gt_img.SetSpacing((spacing[0], spacing[1], spacing[2]))

        filter_hd = sitk.HausdorffDistanceImageFilter()
        filter_hd.Execute(pred_img, gt_img)
        # Use average surface distance as HD95 proxy if direct HD95 not available
        # For true HD95, compute from surface distance maps
        hausdorff = filter_hd.GetHausdorffDistance()

        # surface distance for 95th percentile
        dist_map_pred = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True))
        dist_map_gt   = sitk.Abs(sitk.SignedMaurerDistanceMap(gt_img,   squaredDistance=False, useImageSpacing=True))

        # distances from pred surface to GT and vice versa
        pred_surface = sitk.LabelContour(pred_img) > 0
        gt_surface   = sitk.LabelContour(gt_img)   > 0

        pred_surf_arr = sitk.GetArrayFromImage(pred_surface).astype(bool)
        gt_surf_arr   = sitk.GetArrayFromImage(gt_surface).astype(bool)
        dist_p2g_arr  = sitk.GetArrayFromImage(dist_map_gt)
        dist_g2p_arr  = sitk.GetArrayFromImage(dist_map_pred)

        d_p2g = dist_p2g_arr[pred_surf_arr]
        d_g2p = dist_g2p_arr[gt_surf_arr]

        if len(d_p2g) == 0 or len(d_g2p) == 0:
            return float(hausdorff)

        all_dists = np.concatenate([d_p2g, d_g2p])
        return float(np.percentile(all_dists, 95))

    except Exception:
        return np.nan


def cohort_from_name(name: str) -> str:
    n = name.upper()
    if n.startswith("DUKE"):  return "DUKE"
    if n.startswith("ISPY1"): return "ISPY1"
    if n.startswith("NACT"):  return "NACT"
    return "OTHER"


# ── FIND PREDICTION FILES ─────────────────────────────────────────────────────

pred_files = sorted(PRED_DIR.glob("*.nii.gz"))
print(f"Predictions found: {len(pred_files)}")

# ── COMPUTE METRICS ───────────────────────────────────────────────────────────

rows = []

for pred_path in pred_files:
    case_id = pred_path.stem.replace(".nii", "")  # strip .nii.gz
    cohort  = cohort_from_name(case_id)

    # find matching GT mask: GT files are lowercase (e.g. duke_001.nii.gz)
    gt_name = case_id.lower() + ".nii.gz"
    gt_path = GT_DIR / gt_name
    if not gt_path.exists():
        print(f"  GT not found: {gt_name}")
        continue

    try:
        pred_arr = load_binary(pred_path)
        gt_arr   = load_binary(gt_path)
        spacing  = get_spacing(gt_path)

        dice     = dice_score(pred_arr, gt_arr)
        prec, rec = precision_recall(pred_arr, gt_arr)
        hd95     = hausdorff_95(pred_arr, gt_arr, spacing)
        gt_vol   = float(gt_arr.sum() * spacing[0] * spacing[1] * spacing[2] / 1000)  # cm³

        rows.append({
            "case_id":    case_id,
            "cohort":     cohort,
            "dice":       round(dice, 4),
            "hd95_mm":    round(hd95, 2) if not np.isnan(hd95) else np.nan,
            "precision":  round(prec, 4),
            "recall":     round(rec, 4),
            "gt_vol_cm3": round(gt_vol, 2),
            "pred_empty": int(load_binary(pred_path).sum() == 0),
            "gt_empty":   int(gt_arr.sum() == 0),
        })

    except Exception as e:
        print(f"  ERROR {case_id}: {e}")

# ── SAVE ──────────────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)

# ── REPORT ────────────────────────────────────────────────────────────────────

print(f"\nEvaluated: {len(df)} cases")
print(f"Saved: {OUT_PATH}")
print()

sep = "=" * 60
print(sep)
print("  EXTERNAL VALIDATION RESULTS")
print(sep)

for cohort in ["DUKE", "ISPY1", "NACT", "ALL"]:
    sub = df if cohort == "ALL" else df[df["cohort"] == cohort]
    if len(sub) == 0:
        continue
    valid = sub[sub["gt_empty"] == 0]  # exclude cases with no GT mask

    print(f"\n  {cohort} (n={len(valid)} non-empty GT):")
    print(f"    Dice      : {valid['dice'].mean():.4f} ± {valid['dice'].std():.4f}")
    print(f"    HD95 (mm) : {valid['hd95_mm'].dropna().mean():.2f} ± {valid['hd95_mm'].dropna().std():.2f}")
    print(f"    Precision : {valid['precision'].mean():.4f} ± {valid['precision'].std():.4f}")
    print(f"    Recall    : {valid['recall'].mean():.4f} ± {valid['recall'].std():.4f}")
    print(f"    Empty pred: {sub['pred_empty'].sum()}")

# small lesion analysis
print()
print(sep)
print("  SMALL LESION ANALYSIS (GT volume < 1 cm³)")
print(sep)
small = df[(df["gt_vol_cm3"] < 1.0) & (df["gt_empty"] == 0)]
print(f"  Small cases : {len(small)}")
if len(small) > 0:
    print(f"  Dice        : {small['dice'].mean():.4f} ± {small['dice'].std():.4f}")
    print(f"  Recall      : {small['recall'].mean():.4f} ± {small['recall'].std():.4f}")
print(sep)
