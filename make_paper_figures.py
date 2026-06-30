# make_paper_figures.py
# Generates two figures for the paper:
#   1. Preprocessing figure: pre/early/late + early-sub/late-sub (6 panels)
#   2. Uncertainty overlay figure: segmentation + per-voxel uncertainty heatmap
# Run: python make_paper_figures.py

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

CASE_DIR  = Path(r"C:\Users\olegk\Desktop\MRI Project\data_main\BreastDCEDL_ISPY2\ACRIN-6698-102212")
DCE_DIR   = CASE_DIR / "dce"
MASK_PATH = CASE_DIR / "mask" / "ACRIN-6698-102212_spy2_vis1_mask.nii.gz"

OUT_DIR = Path(r"C:\Users\olegk\Desktop\MRI Project\05_pCR\paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASE_ID = "ACRIN-6698-102212"

# phase indices: 0=pre, 1=early, 7=late (per cases_all.csv for this case)
PRE_IDX   = 0
EARLY_IDX = 1
LATE_IDX  = 7

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_arr(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def normalize_for_display(arr: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
    lo, hi = np.percentile(arr, [p_low, p_high])
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo + 1e-6)


def get_best_slice(mask: np.ndarray) -> int:
    """Return axial slice index with largest tumor area."""
    areas = mask.sum(axis=(1, 2))
    return int(np.argmax(areas))


def crop_around_mask(arr_2d: np.ndarray, mask_2d: np.ndarray, margin: int = 25) -> tuple:
    """
    Crop image to a guaranteed SQUARE region centered on the mask, with margin.
    """
    if mask_2d.sum() == 0:
        return arr_2d, (0, arr_2d.shape[0], 0, arr_2d.shape[1])

    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    r_center = (rmin + rmax) / 2
    c_center = (cmin + cmax) / 2

    bbox_h = rmax - rmin
    bbox_w = cmax - cmin
    side = int(max(bbox_h, bbox_w) + 2 * margin)

    H, W = arr_2d.shape
    side = min(side, H, W)  # cannot exceed image size
    half = side // 2

    r0 = int(np.clip(r_center - half, 0, H - side))
    c0 = int(np.clip(c_center - half, 0, W - side))
    r1 = r0 + side
    c1 = c0 + side

    return arr_2d[r0:r1, c0:c1], (r0, r1, c0, c1)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("Loading DCE phases...")
pre   = load_arr(DCE_DIR / f"{CASE_ID}_spy2_vis1_dce_aqc_{PRE_IDX}.nii.gz")
early = load_arr(DCE_DIR / f"{CASE_ID}_spy2_vis1_dce_aqc_{EARLY_IDX}.nii.gz")
late  = load_arr(DCE_DIR / f"{CASE_ID}_spy2_vis1_dce_aqc_{LATE_IDX}.nii.gz")
mask  = load_arr(MASK_PATH) > 0

slice_idx = get_best_slice(mask)
print(f"Selected slice: {slice_idx} (largest tumor area)")

# extract slices
pre_s   = pre[slice_idx]
early_s = early[slice_idx]
late_s  = late[slice_idx]
mask_s  = mask[slice_idx]

# subtraction images
sub_early = early - pre
sub_late  = late - pre
sub_early_s = sub_early[slice_idx]
sub_late_s  = sub_late[slice_idx]

# crop all panels to a consistent bounding box around the tumor
_, bbox = crop_around_mask(pre_s, mask_s, margin=25)
r0, r1, c0, c1 = bbox

pre_s       = pre_s[r0:r1, c0:c1]
early_s     = early_s[r0:r1, c0:c1]
late_s      = late_s[r0:r1, c0:c1]
mask_s      = mask_s[r0:r1, c0:c1]
sub_early_s = sub_early_s[r0:r1, c0:c1]
sub_late_s  = sub_late_s[r0:r1, c0:c1]

print(f"Cropped to bounding box: rows[{r0}:{r1}] cols[{c0}:{c1}]  shape={pre_s.shape}")

# ── FIGURE 1: PREPROCESSING / 5-CHANNEL INPUT ────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

panels = [
    (pre_s,       "(a) Pre-contrast",       axes[0, 0]),
    (early_s,     "(b) Early post-contrast", axes[0, 1]),
    (late_s,      "(c) Late post-contrast",  axes[0, 2]),
    (sub_early_s, "(d) Early subtraction",   axes[1, 0]),
    (sub_late_s,  "(e) Late subtraction",    axes[1, 1]),
]

for img, title, ax in panels:
    disp = normalize_for_display(img)
    cmap = "gray" if "subtraction" not in title.lower() else "RdBu_r"
    vmin, vmax = (0, 1) if cmap == "gray" else (-0.5, 0.5)
    ax.imshow(disp, cmap=cmap, vmin=vmin if cmap=="gray" else None,
              vmax=vmax if cmap=="gray" else None)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

# panel f: tumor mask overlay on pre-contrast
ax = axes[1, 2]
disp_pre = normalize_for_display(pre_s)
ax.imshow(disp_pre, cmap="gray")
mask_overlay = np.ma.masked_where(~mask_s, mask_s)
ax.imshow(mask_overlay, cmap="autumn", alpha=0.5)
ax.set_title("(f) Expert tumor annotation", fontsize=11)
ax.axis("off")

plt.tight_layout()
out1 = OUT_DIR / "fig_preprocessing_5channel.png"
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── FIGURE 2: UNCERTAINTY OVERLAY (using real model softmax probabilities) ──

print("\nLoading real softmax probabilities from fold predictions...")

NPZ_PATH = Path(
    r"C:\Users\olegk\Desktop\MRI Project\extracted\02_full_models_results\home\ubuntu\data\breast_mri\segmentation"
    r"\nnunet_results\Dataset501_BreastDCE_ISPY2"
    r"\nnUNetTrainer_200epochs__nnUNetResEncUNetLPlans__3d_fullres\fold_0\validation\ACRIN6698102212.npz"
)

npz_data = np.load(NPZ_PATH)
print("Keys in npz:", list(npz_data.keys()))

# nnU-Net typically stores softmax probabilities under 'probabilities'
prob_key = "probabilities" if "probabilities" in npz_data else list(npz_data.keys())[0]
probs = npz_data[prob_key]  # shape: (C, Z, Y, X) or (Z, Y, X) depending on version
print("Probability array shape:", probs.shape)

# foreground (tumor class) probability - assume channel 1 is tumor if multi-channel
if probs.ndim == 4:
    prob_tumor = probs[1]  # class 1 = tumor
else:
    prob_tumor = probs

# uncertainty = entropy of binary prediction: -p*log(p) - (1-p)*log(1-p)
eps = 1e-7
p = np.clip(prob_tumor, eps, 1 - eps)
entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
entropy = entropy / np.log(2)  # normalize to [0, 1]

# extract same slice as preprocessing figure
uncertainty = entropy[slice_idx]
prob_slice  = prob_tumor[slice_idx]

# apply same crop as figure 1
uncertainty = uncertainty[r0:r1, c0:c1]
prob_slice  = prob_slice[r0:r1, c0:c1]

print(f"Uncertainty range on selected slice: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
print(f"Mean uncertainty in tumor region: {uncertainty[mask_s].mean():.4f}")
print(f"Mean uncertainty in background: {uncertainty[~mask_s].mean():.4f}")

# light gaussian smoothing purely for visualization clarity (does not affect underlying values used in text)
from scipy.ndimage import gaussian_filter
uncertainty_smooth = gaussian_filter(uncertainty, sigma=0.8)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# panel a: predicted segmentation boundary (from thresholded probability)
pred_mask_s = prob_slice > 0.5
ax = axes[0]
disp_pre = normalize_for_display(pre_s)
ax.imshow(disp_pre, cmap="gray")
ax.contour(mask_s, colors="lime", linewidths=2.0)
ax.contour(pred_mask_s, colors="magenta", linewidths=1.5, linestyles="--")
ax.plot([], [], color="lime", linewidth=2.0, label="Expert")
ax.plot([], [], color="magenta", linewidth=1.5, linestyle="--", label="Predicted")
ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
ax.set_title("(a) Expert vs predicted boundary", fontsize=10)
ax.axis("off")

# panel b: real voxel-wise entropy uncertainty map (smoothed for visibility)
ax = axes[1]
ax.imshow(disp_pre, cmap="gray")
im = ax.imshow(uncertainty_smooth, cmap="hot", alpha=0.7,
                vmin=0, vmax=max(uncertainty_smooth.max(), 0.05))
ax.set_title("(b) Voxel-wise predictive entropy", fontsize=10)
ax.axis("off")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Normalized entropy", fontsize=9)

# panel c: combined overlay - segmentation + uncertainty
ax = axes[2]
ax.imshow(disp_pre, cmap="gray")
mask_overlay = np.ma.masked_where(~mask_s, mask_s)
ax.imshow(mask_overlay, cmap="Greens", alpha=0.2)
im2 = ax.imshow(np.ma.masked_where(uncertainty_smooth < uncertainty_smooth.max()*0.05,
                                     uncertainty_smooth),
                 cmap="hot", alpha=0.75, vmin=0, vmax=max(uncertainty_smooth.max(), 0.05))
ax.set_title("(c) Segmentation + uncertainty overlay", fontsize=10)
ax.axis("off")

plt.tight_layout()
out2 = OUT_DIR / "fig_uncertainty_overlay.png"
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out2}")
print(f"\nThis case (ACRIN6698102212): Dice=0.835 (from resenc_case_level_uncertainty.csv)")
print("Figure uses REAL model softmax probabilities, not a proxy.")