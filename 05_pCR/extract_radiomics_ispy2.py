# extract_radiomics_ispy2.py
# Extracts PyRadiomics features from BreastDCEDL_ISPY2 dataset.
# Shape features extracted once (from pre-contrast mask).
# Firstorder + texture extracted from: early post-contrast,
# late post-contrast, and early-pre subtraction map.
# Supports resume: skips already-processed cases.
# Run: python extract_radiomics_ispy2.py

import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("radiomics").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR = r"C:\Users\olegk\Desktop\MRI Project\data_main\BreastDCEDL_ISPY2"
TSV_PATH = r"C:\Users\olegk\Desktop\MRI Project\data_main\BreastDCEDL_ISPY2_full_metadata.tsv"
OUT_PATH = r"C:\Users\olegk\Desktop\MRI Project\data_main\radiomics_features_ispy2.csv"

SAVE_EVERY      = 25
MIN_MASK_VOXELS = 10

# ── EXTRACTORS ────────────────────────────────────────────────────────────────
# shape_extractor: shape features only (geometry, no intensity)
shape_extractor = featureextractor.RadiomicsFeatureExtractor()
shape_extractor.disableAllFeatures()
shape_extractor.enableFeatureClassByName("shape")

# intensity_extractor: firstorder + texture (no shape, applied per image)
intensity_extractor = featureextractor.RadiomicsFeatureExtractor()
intensity_extractor.disableAllFeatures()
intensity_extractor.enableFeatureClassByName("firstorder")
intensity_extractor.enableFeatureClassByName("glcm")
intensity_extractor.enableFeatureClassByName("glrlm")
intensity_extractor.enableFeatureClassByName("glszm")

for ext in [shape_extractor, intensity_extractor]:
    ext.settings["binWidth"]               = 25
    ext.settings["resampledPixelSpacing"]  = None
    ext.settings["minimumROIDimensions"]   = 1
    ext.settings["minimumROISize"]         = MIN_MASK_VOXELS
    ext.settings["label"]                  = 1
    ext.settings["verbose"]                = False

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_nifti(path: str) -> sitk.Image:
    return sitk.ReadImage(str(path))


def to_float_dict(result: dict, prefix: str) -> dict:
    """Convert PyRadiomics result to float dict with given prefix."""
    out = {}
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            out[f"{prefix}_{k.replace('original_', '')}"] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def build_dce_path(pid: str, phase_idx: int) -> str:
    return os.path.join(DATA_DIR, pid, "dce", f"{pid}_spy2_vis1_dce_aqc_{phase_idx}.nii.gz")


def build_mask_path(pid: str) -> str:
    return os.path.join(DATA_DIR, pid, "mask", f"{pid}_spy2_vis1_mask.nii.gz")


def make_subtraction(early: sitk.Image, pre: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(early).astype(np.float32) - \
          sitk.GetArrayFromImage(pre).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(early)
    return img


# ── LOAD MANIFEST ─────────────────────────────────────────────────────────────

tsv = pd.read_csv(TSV_PATH, sep="\t")
tsv = tsv.dropna(subset=["pid"]).copy()
tsv["pre"]        = tsv["pre"].astype(int)
tsv["post_early"] = tsv["post_early"].astype(int)
tsv["post_late"]  = tsv["post_late"].astype(int)

# resume from checkpoint
processed = set()
if os.path.exists(OUT_PATH):
    existing   = pd.read_csv(OUT_PATH)
    processed  = set(existing["pid"].astype(str))
    print(f"Resuming — already done: {len(processed)}/{len(tsv)}")
else:
    print(f"Starting fresh — {len(tsv)} patients")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

rows   = []
errors = []

for _, row in tsv.iterrows():
    pid       = str(row["pid"])
    pre_idx   = row["pre"]
    early_idx = row["post_early"]
    late_idx  = row["post_late"]

    if pid in processed:
        continue

    pre_path   = build_dce_path(pid, pre_idx)
    early_path = build_dce_path(pid, early_idx)
    late_path  = build_dce_path(pid, late_idx)
    mask_path  = build_mask_path(pid)

    missing = [p for p in [pre_path, early_path, late_path, mask_path]
               if not os.path.exists(p)]
    if missing:
        errors.append({"pid": pid, "error": f"missing {[os.path.basename(p) for p in missing]}"})
        continue

    try:
        pre_img   = load_nifti(pre_path)
        early_img = load_nifti(early_path)
        late_img  = load_nifti(late_path)
        mask_raw  = load_nifti(mask_path)
        mask_img  = sitk.Cast(mask_raw > 0, sitk.sitkUInt8)

        # skip empty masks
        if sitk.GetArrayFromImage(mask_img).sum() < MIN_MASK_VOXELS:
            errors.append({"pid": pid, "error": "mask too small"})
            continue

        sub_img = make_subtraction(early_img, pre_img)

        feats = {"pid": pid}

        # shape features once from pre-contrast (geometry only)
        feats.update(to_float_dict(shape_extractor.execute(pre_img, mask_img), "shape"))

        # intensity + texture per image
        feats.update(to_float_dict(intensity_extractor.execute(pre_img,   mask_img), "pre"))
        feats.update(to_float_dict(intensity_extractor.execute(early_img, mask_img), "early"))
        feats.update(to_float_dict(intensity_extractor.execute(late_img,  mask_img), "late"))
        feats.update(to_float_dict(intensity_extractor.execute(sub_img,   mask_img), "sub"))

        rows.append(feats)
        processed.add(pid)

    except Exception as e:
        errors.append({"pid": pid, "error": str(e)})
        continue

    # checkpoint save
    if len(rows) % SAVE_EVERY == 0 and rows:
        chunk = pd.DataFrame(rows)
        if os.path.exists(OUT_PATH):
            chunk = pd.concat([pd.read_csv(OUT_PATH), chunk], ignore_index=True)
        chunk.to_csv(OUT_PATH, index=False)
        print(f"  checkpoint: {len(chunk)}/{len(tsv)} saved")
        rows = []

# final save
if rows:
    final = pd.DataFrame(rows)
    if os.path.exists(OUT_PATH):
        final = pd.concat([pd.read_csv(OUT_PATH), final], ignore_index=True)
    final.to_csv(OUT_PATH, index=False)

# ── SUMMARY ───────────────────────────────────────────────────────────────────

if os.path.exists(OUT_PATH):
    result_df = pd.read_csv(OUT_PATH)
    print(f"\nDone. Cases: {len(result_df)}, features: {len(result_df.columns)-1}, errors: {len(errors)}")
    print(f"Saved: {OUT_PATH}")

if errors:
    err_df   = pd.DataFrame(errors)
    err_path = OUT_PATH.replace(".csv", "_errors.csv")
    err_df.to_csv(err_path, index=False)
    print(f"Errors: {err_path}")