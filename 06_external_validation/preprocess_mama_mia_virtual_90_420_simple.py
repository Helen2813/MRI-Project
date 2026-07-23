#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_mama_mia_virtual_90_420_simple.py

Simple Windows-friendly debug script.
No command-line parameters are needed.
Edit only the CONFIG block below, then run:

    python preprocess_mama_mia_virtual_90_420_simple.py

Goal:
  Use ALL available MAMA-MIA DCE timepoints to create virtual standardized
  early/late images at 90 s and 420 s, then save the usual 5-channel nnU-Net
  input format:

    CASE_0000.nii.gz = pre-contrast
    CASE_0001.nii.gz = virtual early image at 90 s
    CASE_0002.nii.gz = virtual late image at 420 s
    CASE_0003.nii.gz = virtual early - pre
    CASE_0004.nii.gz = virtual late - pre

This script is for preprocessing/debugging on a laptop. It does NOT run nnU-Net
inference and it does NOT train a model.
"""

from __future__ import annotations

import ast
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG — EDIT ONLY THIS BLOCK
# =============================================================================

# Folder with one folder per external MAMA-MIA case, for example:
# images_2/DUKE_001/*.nii.gz
# images_2/DUKE_002/*.nii.gz
IMAGES_DIR = Path(r"C:\Users\olegk\Desktop\MRI Project\images_2")

# MAMA-MIA clinical/imaging table. Must contain columns:
#   patient_id
#   acquisition_times
CLINICAL_TABLE = Path(r"C:\Users\olegk\Desktop\MRI Project\tables\clinical_and_imaging_info.xlsx")

# Output folder for new 5-channel files.
# Keep this separate from the old mama_mia_input_v2 folder.
OUTPUT_DIR = Path(r"C:\nnw\mama_mia_input_virtual_90_420_debug")

# CSV report describing exactly what was done for each case.
REPORT_CSV = Path(r"C:\Users\olegk\Desktop\MRI Project\external_manifest\virtual_90_420_report_debug.csv")

# Optional preview PNGs for visual checking. Set SAVE_PREVIEWS=False if not needed.
SAVE_PREVIEWS = True
PREVIEW_DIR = Path(r"C:\Users\olegk\Desktop\MRI Project\external_manifest\virtual_90_420_previews")

# Process only first N cases for laptop debug.
# Use LIMIT = None only later, when you want all external cases.
LIMIT = 10

# If you want specific cases only, put them here, e.g. ["DUKE_001", "DUKE_002"].
# Leave empty to use the first LIMIT non-I-SPY2 cases.
CASE_IDS: List[str] = []

# I-SPY2 training target times used in the paper's timing-aware external setup.
TARGET_EARLY_S = 90.0
TARGET_LATE_S = 420.0

# If target 90 s lies between pre-contrast time 0 and the first post-contrast
# acquisition, allow interpolation between them. If False, the script uses the
# nearest acquired phase instead.
ALLOW_PRE_TO_POST_INTERPOLATION = True

# Skip I-SPY2 folders/cases from MAMA-MIA external preprocessing.
SKIP_ISPY2 = True

# For debug folders, True is usually convenient. For full runs, False is safer.
OVERWRITE = True

# If True, the script will write the report but will not save NIfTI files.
DRY_RUN = False


# =============================================================================
# BASIC HELPERS
# =============================================================================


def read_clinical_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clinical table not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    raise ValueError(f"Unsupported clinical table format: {path.suffix}")



def parse_times(value: Any) -> List[float]:
    """Parse values like '[0, 165, 288, 411]' into a list of seconds."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            times = [float(x) for x in parsed]
        else:
            times = [float(parsed)]
    except Exception:
        # Fallback for strings like: 0, 165, 288, 411
        cleaned = text.replace("[", "").replace("]", "").replace(";", ",")
        try:
            times = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
        except Exception:
            return []

    # Most examples in the repo are seconds, e.g. 165, 411, 584.
    # If a table ever stores minutes like [0, 1.5, 7], convert likely-minute values.
    if times and max(times) <= 20 and len(times) > 2:
        times = [t * 60.0 for t in times]

    return times



def build_times_lookup(clinical: pd.DataFrame) -> Dict[str, List[float]]:
    required = {"patient_id", "acquisition_times"}
    missing = required - set(clinical.columns)
    if missing:
        raise ValueError(f"Clinical table missing required columns: {sorted(missing)}")

    df = clinical.copy()
    df["patient_id_upper"] = df["patient_id"].astype(str).str.upper()
    if SKIP_ISPY2:
        df = df[~df["patient_id_upper"].str.startswith("ISPY2")].copy()

    df["times_parsed"] = df["acquisition_times"].apply(parse_times)
    return dict(zip(df["patient_id_upper"], df["times_parsed"]))



def load_img(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))



def save_img(img: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)



def same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )



def resample_to_reference(img: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """Resample image to the pre-contrast image grid if needed."""
    if same_grid(img, ref):
        return img
    return sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, 0.0, img.GetPixelID())



def to_array(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)



def image_like(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(arr.astype(np.float32, copy=False))
    out.CopyInformation(ref)
    return out


# =============================================================================
# VIRTUAL 90s / 420s INTERPOLATION
# =============================================================================


def nearest_index(times: Sequence[float], target_s: float) -> int:
    return int(np.argmin([abs(float(t) - target_s) for t in times]))



def find_bracket(times: Sequence[float], target_s: float) -> Optional[Tuple[int, int]]:
    """Return indices of the two acquired phases around target_s, or None."""
    ordered = sorted([(i, float(t)) for i, t in enumerate(times)], key=lambda x: x[1])

    for (left_idx, left_t), (right_idx, right_t) in zip(ordered[:-1], ordered[1:]):
        if left_t <= target_s <= right_t:
            if not ALLOW_PRE_TO_POST_INTERPOLATION and left_t <= 0 < target_s:
                return None
            return left_idx, right_idx

    return None



def virtual_phase(
    phase_files: Sequence[Path],
    times: Sequence[float],
    target_s: float,
    ref_img: sitk.Image,
) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Build a virtual phase at target_s.

    If target is between two acquired phases, use linear interpolation:
      I_virtual = (1-w) * I_left + w * I_right

    If target is outside the available time range, use nearest acquired phase
    and flag it in the report.
    """
    if len(phase_files) != len(times):
        raise ValueError(f"phase_files/times mismatch: {len(phase_files)} files vs {len(times)} times")

    # Exact acquired phase.
    for idx, t in enumerate(times):
        if abs(float(t) - target_s) < 1e-6:
            img = resample_to_reference(load_img(phase_files[idx]), ref_img)
            return img, {
                "method": "exact",
                "target_s": target_s,
                "left_idx": idx,
                "right_idx": idx,
                "left_t_s": float(t),
                "right_t_s": float(t),
                "weight_right": 0.0,
                "gap_s": 0.0,
                "nearest_idx": idx,
                "nearest_t_s": float(t),
                "nearest_dev_s": 0.0,
                "quality": "good",
                "range_status": "inside",
            }

    bracket = find_bracket(times, target_s)
    if bracket is not None:
        left_idx, right_idx = bracket
        left_t = float(times[left_idx])
        right_t = float(times[right_idx])
        gap = right_t - left_t
        if gap <= 0:
            raise ValueError(f"Bad acquisition times around target {target_s}: {left_t}, {right_t}")

        w = float((target_s - left_t) / gap)

        left_img = resample_to_reference(load_img(phase_files[left_idx]), ref_img)
        right_img = resample_to_reference(load_img(phase_files[right_idx]), ref_img)

        arr = (1.0 - w) * to_array(left_img) + w * to_array(right_img)
        out = image_like(arr, ref_img)

        # This is only interpolation quality, not segmentation quality.
        if gap <= 180:
            quality = "good"
        elif gap <= 360:
            quality = "acceptable"
        else:
            quality = "poor_gap"

        nearest_idx = nearest_index(times, target_s)
        return out, {
            "method": "linear_interpolation",
            "target_s": target_s,
            "left_idx": left_idx,
            "right_idx": right_idx,
            "left_t_s": left_t,
            "right_t_s": right_t,
            "weight_right": w,
            "gap_s": gap,
            "nearest_idx": nearest_idx,
            "nearest_t_s": float(times[nearest_idx]),
            "nearest_dev_s": abs(float(times[nearest_idx]) - target_s),
            "quality": quality,
            "range_status": "inside",
        }

    # Outside range: nearest fallback.
    idx = nearest_index(times, target_s)
    t = float(times[idx])
    img = resample_to_reference(load_img(phase_files[idx]), ref_img)

    if target_s < min(times):
        range_status = "before_first_acquired"
    elif target_s > max(times):
        range_status = "after_last_acquired"
    else:
        range_status = "inside_no_bracket"

    return img, {
        "method": "nearest_fallback",
        "target_s": target_s,
        "left_idx": idx,
        "right_idx": idx,
        "left_t_s": t,
        "right_t_s": t,
        "weight_right": 0.0,
        "gap_s": np.nan,
        "nearest_idx": idx,
        "nearest_t_s": t,
        "nearest_dev_s": abs(t - target_s),
        "quality": "poor_outside_range" if range_status != "inside_no_bracket" else "fallback",
        "range_status": range_status,
    }



def make_subtraction(img_a: sitk.Image, img_b: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    a = resample_to_reference(img_a, ref_img)
    b = resample_to_reference(img_b, ref_img)
    return image_like(to_array(a) - to_array(b), ref_img)


# =============================================================================
# PREVIEW PNGS
# =============================================================================


def robust_window(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)



def save_preview(case_id: str, images: Dict[str, sitk.Image]) -> str:
    if not SAVE_PREVIEWS:
        return ""

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("  Preview skipped: matplotlib is not installed")
        return ""

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    names = ["pre", "early90", "late420", "early90_minus_pre", "late420_minus_pre"]
    arrays = [to_array(images[name]) for name in names]
    z = arrays[0].shape[0] // 2

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, name, arr in zip(axes, names, arrays):
        ax.imshow(robust_window(arr[z]), cmap="gray")
        ax.set_title(name, fontsize=8)
        ax.axis("off")

    fig.suptitle(case_id, fontsize=10)
    fig.tight_layout()

    out_path = PREVIEW_DIR / f"{case_id}_virtual_90_420.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


# =============================================================================
# MAIN PROCESSING
# =============================================================================


def completed_case(case_id: str) -> bool:
    return all((OUTPUT_DIR / f"{case_id}_{i:04d}.nii.gz").exists() for i in range(5))



def add_prefix(prefix: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in meta.items()}



def select_case_dirs() -> List[Path]:
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

    case_dirs = sorted([p for p in IMAGES_DIR.iterdir() if p.is_dir()])

    if SKIP_ISPY2:
        case_dirs = [p for p in case_dirs if not p.name.upper().startswith("ISPY2")]

    if CASE_IDS:
        wanted = {x.upper() for x in CASE_IDS}
        case_dirs = [p for p in case_dirs if p.name.upper() in wanted]

    if LIMIT is not None:
        case_dirs = case_dirs[:LIMIT]

    return case_dirs



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_PREVIEWS:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clinical/imaging table...")
    clinical = read_clinical_table(CLINICAL_TABLE)
    times_lookup = build_times_lookup(clinical)

    case_dirs = select_case_dirs()

    print(f"Cases selected: {len(case_dirs)}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Report CSV: {REPORT_CSV}")
    print(f"Targets: early={TARGET_EARLY_S:.0f}s, late={TARGET_LATE_S:.0f}s")
    print()

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0

    for case_dir in case_dirs:
        case_id = case_dir.name.upper()
        phase_files = sorted(case_dir.glob("*.nii.gz"))

        row: Dict[str, Any] = {
            "case_id": case_id,
            "status": "started",
            "n_phase_files": len(phase_files),
            "phase_files": json.dumps([p.name for p in phase_files]),
        }

        try:
            if not OVERWRITE and completed_case(case_id):
                row["status"] = "skipped_existing"
                rows.append(row)
                skipped += 1
                print(f"SKIP {case_id}: already exists")
                continue

            if len(phase_files) < 2:
                raise ValueError(f"Too few phase files: {len(phase_files)}")

            times = times_lookup.get(case_id, [])
            row["times"] = json.dumps(times)

            if not times:
                raise ValueError("No acquisition_times found for case")

            if len(times) != len(phase_files):
                raise ValueError(
                    f"Number of acquisition times ({len(times)}) does not match "
                    f"number of NIfTI phase files ({len(phase_files)})"
                )

            pre_idx = nearest_index(times, 0.0)
            pre_img = load_img(phase_files[pre_idx])

            early_img, early_meta = virtual_phase(phase_files, times, TARGET_EARLY_S, pre_img)
            late_img, late_meta = virtual_phase(phase_files, times, TARGET_LATE_S, pre_img)

            early_sub = make_subtraction(early_img, pre_img, pre_img)
            late_sub = make_subtraction(late_img, pre_img, pre_img)

            images = {
                "pre": pre_img,
                "early90": early_img,
                "late420": late_img,
                "early90_minus_pre": early_sub,
                "late420_minus_pre": late_sub,
            }

            if not DRY_RUN:
                save_img(pre_img, OUTPUT_DIR / f"{case_id}_0000.nii.gz")
                save_img(early_img, OUTPUT_DIR / f"{case_id}_0001.nii.gz")
                save_img(late_img, OUTPUT_DIR / f"{case_id}_0002.nii.gz")
                save_img(early_sub, OUTPUT_DIR / f"{case_id}_0003.nii.gz")
                save_img(late_sub, OUTPUT_DIR / f"{case_id}_0004.nii.gz")

                preview_path = save_preview(case_id, images)
            else:
                preview_path = ""

            row.update({
                "status": "processed",
                "pre_idx": pre_idx,
                "pre_t_s": float(times[pre_idx]),
                "preview_path": preview_path,
            })
            row.update(add_prefix("early", early_meta))
            row.update(add_prefix("late", late_meta))

            qualities = [str(early_meta["quality"]), str(late_meta["quality"])]
            if all(q == "good" for q in qualities):
                row["overall_virtual_quality"] = "good"
            elif any(q.startswith("poor") for q in qualities):
                row["overall_virtual_quality"] = "poor"
            else:
                row["overall_virtual_quality"] = "acceptable"

            rows.append(row)
            processed += 1

            print(
                f"OK {case_id}: "
                f"early={early_meta['method']} {early_meta['quality']}, "
                f"late={late_meta['method']} {late_meta['quality']}"
            )

        except Exception as e:
            row["status"] = "error"
            row["error"] = str(e)
            rows.append(row)
            errors.append({"case_id": case_id, "error": str(e)})
            print(f"ERROR {case_id}: {e}")

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_CSV, index=False)

    if errors:
        error_csv = REPORT_CSV.with_name(REPORT_CSV.stem + "_errors.csv")
        pd.DataFrame(errors).to_csv(error_csv, index=False)
    else:
        error_csv = None

    print()
    print("DONE")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {len(errors)}")
    print(f"NIfTI output files now in folder: {len(list(OUTPUT_DIR.glob('*.nii.gz')))}")
    print(f"Report saved: {REPORT_CSV}")
    if error_csv is not None:
        print(f"Errors saved: {error_csv}")
    if SAVE_PREVIEWS:
        print(f"Preview folder: {PREVIEW_DIR}")

    if len(report) > 0 and "overall_virtual_quality" in report.columns:
        print()
        print("Virtual phase quality summary:")
        print(report["overall_virtual_quality"].value_counts(dropna=False).to_string())

    print()
    print("Next check:")
    print("  Open the preview PNGs and confirm the virtual 90s/420s and subtraction images look reasonable.")
    print("  This script only prepares inputs; nnU-Net inference should be run later on GPU/cloud/HPC.")


if __name__ == "__main__":
    main()
