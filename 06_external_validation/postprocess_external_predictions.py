# postprocess_external_predictions.py
# Applies postprocessing variants to external validation predictions.
# Tests: largest connected component, small component removal, threshold variants.
# Pre-specified sensitivity analysis — not tuned on test set.
# Run: python postprocess_external_predictions.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from scipy import ndimage

# ── CONFIG ────────────────────────────────────────────────────────────────────

PRED_DIR_V1  = Path(r"C:\nnw\mama_mia_output_v2")
PRED_DIR = Path(r"C:\nnw\mama_mia_output_v2")
GT_DIR       = Path(r"C:\Users\olegk\Desktop\MRI Project\segmentations_2\expert")
PHASE_REPORT = Path(r"C:\Users\olegk\Desktop\MRI Project\external_manifest\phase_selection_report.csv")
OUT_CSV      = Path(r"C:\Users\olegk\Desktop\MRI Project\external_manifest\postprocessing_sensitivity.csv")

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_binary(path: Path) -> np.ndarray:
    return (sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0).astype(np.uint8)


def get_spacing(path: Path) -> tuple:
    return sitk.ReadImage(str(path)).GetSpacing()


def dice(pred, gt):
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else (1.0 if pred.sum() == 0 else 0.0)


def precision_recall(pred, gt):
    tp = (pred & gt).sum()
    prec = float(tp / pred.sum()) if pred.sum() > 0 else 0.0
    rec  = float(tp / gt.sum())   if gt.sum()   > 0 else (1.0 if pred.sum() == 0 else 0.0)
    return prec, rec


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the single largest connected component."""
    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    return (labeled == largest).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_voxels: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_voxels."""
    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask)
    out = np.zeros_like(mask)
    for i in range(1, n + 1):
        if (labeled == i).sum() >= min_voxels:
            out[labeled == i] = 1
    return out


def cohort_from(name: str) -> str:
    n = name.upper()
    if n.startswith("DUKE"):  return "DUKE"
    if n.startswith("ISPY1"): return "ISPY1"
    if n.startswith("NACT"):  return "NACT"
    return "OTHER"


# ── LOAD PHASE QUALITY REPORT ─────────────────────────────────────────────────

phase_df = None
if PHASE_REPORT.exists():
    phase_df = pd.read_csv(PHASE_REPORT)
    phase_df["case_id_upper"] = phase_df["case_id"].str.upper()
    quality_lookup = dict(zip(phase_df["case_id_upper"], phase_df["overall"]))
else:
    quality_lookup = {}

# ── MAIN ──────────────────────────────────────────────────────────────────────

pred_files = sorted(PRED_DIR.glob("*.nii.gz"))
print(f"Predictions: {len(pred_files)}")

rows = []

for pred_path in pred_files:
    case_id = pred_path.stem.replace(".nii", "")
    cohort  = cohort_from(case_id)
    quality = quality_lookup.get(case_id.upper(), "unknown")

    gt_path = GT_DIR / (case_id.lower() + ".nii.gz")
    if not gt_path.exists():
        continue

    try:
        pred_raw = load_binary(pred_path)
        gt       = load_binary(gt_path)
        if gt.sum() == 0:
            continue

        spacing = get_spacing(gt_path)
        gt_vol  = float(gt.sum() * spacing[0] * spacing[1] * spacing[2] / 1000)

        # postprocessing variants
        variants = {
            "raw":             pred_raw,
            "largest_cc":      keep_largest_component(pred_raw),
            "remove_small50":  remove_small_components(pred_raw, min_voxels=50),
            "remove_small100": remove_small_components(pred_raw, min_voxels=100),
            "lcc_plus_small50": remove_small_components(
                                    keep_largest_component(pred_raw), min_voxels=50),
        }

        row = {
            "case_id":    case_id,
            "cohort":     cohort,
            "phase_quality": quality,
            "gt_vol_cm3": round(gt_vol, 2),
        }

        for name, pred in variants.items():
            d = dice(pred, gt)
            p, r = precision_recall(pred, gt)
            row[f"{name}_dice"]      = round(d, 4)
            row[f"{name}_precision"] = round(p, 4)
            row[f"{name}_recall"]    = round(r, 4)

        rows.append(row)

    except Exception as e:
        print(f"  ERROR {case_id}: {e}")

# ── SAVE & REPORT ─────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"\nEvaluated: {len(df)} cases")
print(f"Saved: {OUT_CSV}")

sep = "=" * 65
variants = ["raw", "largest_cc", "remove_small50", "remove_small100", "lcc_plus_small50"]

print()
print(sep)
print("  POSTPROCESSING SENSITIVITY — ALL COHORTS")
print(sep)
for v in variants:
    d = df[f"{v}_dice"].mean()
    p = df[f"{v}_precision"].mean()
    r = df[f"{v}_recall"].mean()
    print(f"  {v:25s}: Dice={d:.4f}  Prec={p:.4f}  Rec={r:.4f}")

print()
print(sep)
print("  BY COHORT (best postprocessing per cohort)")
print(sep)
for cohort in ["DUKE", "ISPY1", "NACT"]:
    sub = df[df["cohort"] == cohort]
    if len(sub) == 0:
        continue
    print(f"\n  {cohort} (n={len(sub)}):")
    for v in variants:
        d = sub[f"{v}_dice"].mean()
        print(f"    {v:25s}: Dice={d:.4f}")

print()
print(sep)
print("  BY PHASE QUALITY (raw vs best postprocessing)")
print(sep)
for q in ["good", "acceptable", "poor"]:
    sub = df[df["phase_quality"] == q]
    if len(sub) == 0:
        continue
    raw  = sub["raw_dice"].mean()
    best = max(sub[f"{v}_dice"].mean() for v in variants)
    best_name = max(variants, key=lambda v: sub[f"{v}_dice"].mean())
    print(f"  {q:12s} (n={len(sub):3d}): raw={raw:.4f}  best={best:.4f} ({best_name})")
print(sep)
