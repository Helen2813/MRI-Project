# setup_nnunet_local.py
# Copies ResEncL model weights to short path C:\nnw\ to avoid Windows 260-char path limit.
# Run: python setup_nnunet_local.py

import os
import shutil
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

# source: extracted model from instance backup
SRC_BASE = Path(r"C:\Users\olegk\Desktop\MRI Project\extracted\02_full_models_results\home\ubuntu\data\breast_mri\segmentation\nnunet_results\Dataset501_BreastDCE_ISPY2\nnUNetTrainer_200epochs__nnUNetResEncUNetLPlans__3d_fullres")

# destination: short path for nnU-Net
DST_BASE = Path(r"C:\nnw\Dataset501_BreastDCE_ISPY2\nnUNetTrainer_200epochs__nnUNetResEncUNetLPlans__3d_fullres")

# files needed at model root level
ROOT_FILES = ["plans.json", "dataset.json", "dataset_fingerprint.json",
              "postprocessing.json", "postprocessing.pkl"]

# ── COPY ──────────────────────────────────────────────────────────────────────

DST_BASE.mkdir(parents=True, exist_ok=True)

# copy root-level config files
for fname in ROOT_FILES:
    src = SRC_BASE / fname
    dst = DST_BASE / fname
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  copied: {fname}")
    else:
        print(f"  MISSING: {fname}")

# copy fold checkpoints
for fold in range(5):
    src_fold = SRC_BASE / f"fold_{fold}" / "checkpoint_best.pth"
    dst_fold = DST_BASE / f"fold_{fold}" / "checkpoint_best.pth"
    dst_fold.parent.mkdir(parents=True, exist_ok=True)
    if src_fold.exists():
        shutil.copy2(src_fold, dst_fold)
        print(f"  copied: fold_{fold}/checkpoint_best.pth")
    else:
        print(f"  MISSING: fold_{fold}/checkpoint_best.pth")

# ── VERIFY ────────────────────────────────────────────────────────────────────

print("\nVerification:")
print(f"  plans.json   : {(DST_BASE / 'plans.json').exists()}")
print(f"  dataset.json : {(DST_BASE / 'dataset.json').exists()}")
for fold in range(5):
    ckpt = DST_BASE / f"fold_{fold}" / "checkpoint_best.pth"
    size_mb = ckpt.stat().st_size / 1e6 if ckpt.exists() else 0
    print(f"  fold_{fold}/checkpoint_best.pth : {ckpt.exists()} ({size_mb:.0f} MB)")

print("\nnnU-Net environment variables to set:")
print(f'  $env:nnUNet_results = "C:\\nnw"')
print(f'  $env:nnUNet_raw = "C:\\nnw\\raw"')
print(f'  $env:nnUNet_preprocessed = "C:\\nnw\\preprocessed"')
print("\nDone. Next: run preprocess_mama_mia_nnunet.py")