# preprocess_mama_mia_nnunet_v2.py
# Improved preprocessing using acquisition_times from MAMA-MIA clinical table.
# Selects early/late phases closest to ISPY2 training protocol timing.
# ISPY2 targets: early ~90-120s, late ~420s post-contrast.
# Run: python preprocess_mama_mia_nnunet_v2.py

import ast
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGES_DIR   = Path(r"C:\Users\olegk\Desktop\MRI Project\images_2")
CLINICAL_CSV = Path(r"C:\Users\olegk\Desktop\MRI Project\tables\clinical_and_imaging_info.xlsx")
OUTPUT_DIR   = Path(r"C:\nnw\mama_mia_input_v2")
REPORT_PATH  = Path(r"C:\Users\olegk\Desktop\MRI Project\external_manifest\phase_selection_report.csv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target acquisition times from ISPY2 training protocol (seconds)
TARGET_EARLY_S = 90    # ~1.5 min: first post-contrast peak
TARGET_LATE_S  = 420   # ~7 min: late/washout phase

# Maximum acceptable deviation from target (seconds)
# If no phase is within this range, flag as "protocol_mismatch"
MAX_EARLY_DEV = 300    # allow up to 5 min deviation for early
MAX_LATE_DEV  = 300    # allow up to 5 min deviation for late

# ── HELPERS ───────────────────────────────────────────────────────────────────

def parse_times(s: str) -> list:
    """Parse acquisition_times string like '[0, 584, 714]' to list of ints."""
    try:
        return [int(x) for x in ast.literal_eval(str(s))]
    except Exception:
        return []


def select_phases(times: list) -> dict:
    """
    Given list of acquisition times (seconds, first=0=pre-contrast),
    return dict with selected phase indices and quality flags.
    """
    if len(times) < 2:
        return None

    post_times = [(i, t) for i, t in enumerate(times) if t > 0]
    if not post_times:
        return None

    # pre is always index 0 (time=0)
    pre_idx = times.index(0)

    # early: post-contrast phase closest to TARGET_EARLY_S
    early_idx, early_t = min(post_times, key=lambda x: abs(x[1] - TARGET_EARLY_S))
    early_dev = abs(early_t - TARGET_EARLY_S)

    # late: post-contrast phase closest to TARGET_LATE_S (must be different from early)
    late_candidates = [(i, t) for i, t in post_times if i != early_idx]
    if late_candidates:
        late_idx, late_t = min(late_candidates, key=lambda x: abs(x[1] - TARGET_LATE_S))
    else:
        late_idx, late_t = early_idx, early_t  # fallback
    late_dev = abs(late_t - TARGET_LATE_S)

    # quality assessment
    if early_dev <= 120:
        early_quality = "good"
    elif early_dev <= MAX_EARLY_DEV:
        early_quality = "acceptable"
    else:
        early_quality = "poor"

    if late_dev <= 120:
        late_quality = "good"
    elif late_dev <= MAX_LATE_DEV:
        late_quality = "acceptable"
    else:
        late_quality = "poor"

    overall = "good" if early_quality == "good" and late_quality in ("good", "acceptable") \
        else "acceptable" if "poor" not in [early_quality, late_quality] \
        else "poor"

    return {
        "pre_idx":      pre_idx,
        "early_idx":    early_idx,
        "early_t":      early_t,
        "early_dev":    early_dev,
        "early_quality": early_quality,
        "late_idx":     late_idx,
        "late_t":       late_t,
        "late_dev":     late_dev,
        "late_quality": late_quality,
        "overall":      overall,
    }


def load_img(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def save_img(img: sitk.Image, path: Path):
    sitk.WriteImage(img, str(path), useCompression=True)


def make_subtraction(img_a: sitk.Image, img_b: sitk.Image) -> sitk.Image:
    arr = (sitk.GetArrayFromImage(img_a).astype(np.float32) -
           sitk.GetArrayFromImage(img_b).astype(np.float32))
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img_a)
    return out


# ── LOAD CLINICAL TABLE ───────────────────────────────────────────────────────

print("Loading clinical table...")
clinical = pd.read_excel(CLINICAL_CSV, dtype=str)
clinical = clinical[~clinical["patient_id"].str.upper().str.startswith("ISPY2")].copy()
clinical["times_parsed"] = clinical["acquisition_times"].apply(parse_times)
clinical["patient_id_upper"] = clinical["patient_id"].str.upper()

# build lookup: patient_id_upper -> times
times_lookup = dict(zip(clinical["patient_id_upper"], clinical["times_parsed"]))
print(f"  Non-ISPY2 cases in table: {len(clinical)}")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

case_dirs = sorted([d for d in IMAGES_DIR.iterdir() if d.is_dir()])
print(f"  Image folders: {len(case_dirs)}")

rows    = []
done    = 0
errors  = []

# timing quality counters
quality_counts = {"good": 0, "acceptable": 0, "poor": 0, "no_times": 0}

for case_dir in case_dirs:
    case_id = case_dir.name.upper()

    # skip if already done
    if (OUTPUT_DIR / f"{case_id}_0000.nii.gz").exists():
        done += 1
        continue

    phase_files = sorted(case_dir.glob("*.nii.gz"))
    if len(phase_files) < 2:
        errors.append({"case_id": case_id, "error": "too few phase files"})
        continue

    # get acquisition times
    times = times_lookup.get(case_id, [])

    if times and len(times) == len(phase_files):
        # use timing-based selection
        sel = select_phases(times)
        if sel is None:
            sel = {"pre_idx": 0, "early_idx": 1, "late_idx": len(phase_files)-1,
                   "overall": "fallback", "early_t": -1, "late_t": -1,
                   "early_dev": -1, "late_dev": -1,
                   "early_quality": "fallback", "late_quality": "fallback"}
    else:
        # fallback: first=pre, second=early, last=late
        sel = {"pre_idx": 0, "early_idx": 1, "late_idx": len(phase_files)-1,
               "overall": "no_times", "early_t": -1, "late_t": -1,
               "early_dev": -1, "late_dev": -1,
               "early_quality": "no_times", "late_quality": "no_times"}
        quality_counts["no_times"] += 1

    if sel["overall"] in quality_counts:
        quality_counts[sel["overall"]] += 1

    try:
        pre_img   = load_img(phase_files[sel["pre_idx"]])
        early_img = load_img(phase_files[sel["early_idx"]])
        late_img  = load_img(phase_files[sel["late_idx"]])

        sub_early = make_subtraction(early_img, pre_img)
        sub_late  = make_subtraction(late_img,  pre_img)

        save_img(pre_img,   OUTPUT_DIR / f"{case_id}_0000.nii.gz")
        save_img(early_img, OUTPUT_DIR / f"{case_id}_0001.nii.gz")
        save_img(late_img,  OUTPUT_DIR / f"{case_id}_0002.nii.gz")
        save_img(sub_early, OUTPUT_DIR / f"{case_id}_0003.nii.gz")
        save_img(sub_late,  OUTPUT_DIR / f"{case_id}_0004.nii.gz")

        done += 1

    except Exception as e:
        errors.append({"case_id": case_id, "error": str(e)})
        continue

    rows.append({
        "case_id":       case_id,
        "n_phases":      len(phase_files),
        "times":         str(times),
        "pre_idx":       sel["pre_idx"],
        "early_idx":     sel["early_idx"],
        "early_t_s":     sel["early_t"],
        "early_dev_s":   sel["early_dev"],
        "early_quality": sel["early_quality"],
        "late_idx":      sel["late_idx"],
        "late_t_s":      sel["late_t"],
        "late_dev_s":    sel["late_dev"],
        "late_quality":  sel["late_quality"],
        "overall":       sel["overall"],
    })

    if done % 50 == 0:
        print(f"  processed: {done}/{len(case_dirs)}")

# ── REPORT ────────────────────────────────────────────────────────────────────

report = pd.DataFrame(rows)
report.to_csv(REPORT_PATH, index=False)

print(f"\nDone: {done}  Errors: {len(errors)}")
print(f"Files: {len(list(OUTPUT_DIR.glob('*.nii.gz')))}")
print()
print("Phase selection quality summary:")
for q, n in quality_counts.items():
    print(f"  {q:12s}: {n}")

# show timing statistics per dataset
if len(report) > 0:
    report["dataset"] = report["case_id"].apply(
        lambda x: "DUKE" if x.startswith("DUKE") else "ISPY1" if x.startswith("ISPY1") else "NACT"
    )
    print()
    print("Early phase timing by dataset:")
    for ds, grp in report[report["early_t_s"] > 0].groupby("dataset"):
        print(f"  {ds}: mean early={grp['early_t_s'].mean():.0f}s  "
              f"mean late={grp['late_t_s'].mean():.0f}s  "
              f"good={( grp['overall']=='good').sum()}  "
              f"poor={(grp['overall']=='poor').sum()}")

print()
print("Next: run inference on mama_mia_input_v2")
print('  nnUNetv2_predict -i "C:\\nnw\\mama_mia_input_v2" -o "C:\\nnw\\mama_mia_output_v2" '
      '-d 501 -c 3d_fullres -tr nnUNetTrainer_200epochs -p nnUNetResEncUNetLPlans '
      '-f 0 1 2 3 4 -chk checkpoint_best.pth -device cuda')