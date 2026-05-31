# preprocess_mama_mia_nnunet.py
# Converts MAMA-MIA DCE-MRI images to nnU-Net 5-channel input format.
# Channels: 0=pre, 1=early, 2=late, 3=early-pre subtraction, 4=late-pre subtraction
# Output naming: {CASE_ID}_0000.nii.gz ... {CASE_ID}_0004.nii.gz
# Run: python preprocess_mama_mia_nnunet.py

import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGES_DIR = Path(r"C:\Users\olegk\Desktop\MRI Project\images_2")
OUTPUT_DIR = Path(r"C:\nnw\mama_mia_input")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_phase_files(case_dir: Path) -> list:
    """Return sorted list of .nii.gz files in case folder."""
    files = sorted(case_dir.glob("*.nii.gz"))
    return files


def load_img(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def save_img(img: sitk.Image, path: Path):
    sitk.WriteImage(img, str(path), useCompression=True)


def make_subtraction(img_a: sitk.Image, img_b: sitk.Image) -> sitk.Image:
    """Compute img_a - img_b preserving image metadata."""
    arr = (sitk.GetArrayFromImage(img_a).astype(np.float32) -
           sitk.GetArrayFromImage(img_b).astype(np.float32))
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img_a)
    return out


# ── MAIN ──────────────────────────────────────────────────────────────────────

case_dirs = sorted([d for d in IMAGES_DIR.iterdir() if d.is_dir()])
print(f"Found {len(case_dirs)} cases")

# check how many are already done
already = len(list(OUTPUT_DIR.glob("*_0000.nii.gz")))
if already:
    print(f"Already preprocessed: {already} cases — will skip")

done   = 0
errors = []

for case_dir in case_dirs:
    case_id = case_dir.name.upper()  # e.g. DUKE_001, ISPY1_1001, NACT_01

    # skip if already preprocessed
    if (OUTPUT_DIR / f"{case_id}_0000.nii.gz").exists():
        done += 1
        continue

    phase_files = get_phase_files(case_dir)
    if len(phase_files) < 3:
        errors.append({"case": case_id, "error": f"only {len(phase_files)} phase files"})
        continue

    try:
        # phase assignment: first=pre, second=early, last=late
        pre_img   = load_img(phase_files[0])
        early_img = load_img(phase_files[1])
        late_img  = load_img(phase_files[-1])  # last file = late

        # compute subtraction channels
        sub_early = make_subtraction(early_img, pre_img)
        sub_late  = make_subtraction(late_img,  pre_img)

        # save 5 channels in nnU-Net naming convention
        save_img(pre_img,   OUTPUT_DIR / f"{case_id}_0000.nii.gz")
        save_img(early_img, OUTPUT_DIR / f"{case_id}_0001.nii.gz")
        save_img(late_img,  OUTPUT_DIR / f"{case_id}_0002.nii.gz")
        save_img(sub_early, OUTPUT_DIR / f"{case_id}_0003.nii.gz")
        save_img(sub_late,  OUTPUT_DIR / f"{case_id}_0004.nii.gz")

        done += 1

    except Exception as e:
        errors.append({"case": case_id, "error": str(e)})

    if done % 50 == 0:
        print(f"  processed: {done}/{len(case_dirs)}")

print(f"\nDone: {done}  Errors: {len(errors)}")
for e in errors:
    print(f"  ERROR {e['case']}: {e['error']}")

print(f"\nOutput: {OUTPUT_DIR}")
print(f"Files: {len(list(OUTPUT_DIR.glob('*.nii.gz')))}")
print("\nNext steps:")
print("  1. Set env vars in PowerShell:")
print('     $env:nnUNet_results = "C:\\nnw"')
print('     $env:nnUNet_raw = "C:\\nnw\\raw"')
print('     $env:nnUNet_preprocessed = "C:\\nnw\\preprocessed"')
print("  2. Run inference:")
print('     nnUNetv2_predict -i "C:\\nnw\\mama_mia_input" -o "C:\\nnw\\mama_mia_output" -d 501 -c 3d_fullres -tr nnUNetTrainer_200epochs -p nnUNetResEncUNetLPlans -f 0 1 2 3 4 --save_probabilities')