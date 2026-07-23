# 11 Catastrophic case review

This folder contains the final visual/technical sanity-check script for the external validation failure analysis.

## Purpose

Script 10 found cases where theoretical Dice based on predicted/GT volume ratio was high, but actual Dice was very low. These cases can represent true severe spatial/boundary failures, but they can also reveal technical problems such as wrong case pairing, orientation mismatch, or coordinate-frame mismatch.

This script checks those cases before the manuscript is finalized.

## What it reads

Default paths are set at the top of the script:

- `10_final_external_sanity_checks/metrics/final_sanity_case_table_with_volume_bounds.csv`
- external images: `images_2/`
- expert masks: `segmentations_2/expert/`
- v2 raw predictions: `C:\nnw\mama_mia_output_v2`

## How to run

Put this folder here:

```text
C:\Users\olegk\Desktop\MRI Project\11_catastrophic_case_review
```

Run:

```text
python run_11_catastrophic_case_review.py
```

No command-line parameters are needed.

## Outputs

The script creates:

```text
11_catastrophic_case_review/
  metrics/
    case_table_standardized_for_11.csv
    selected_cases_for_visual_review.csv
    technical_geometry_and_pairing_review.csv
    catastrophic_spatial_mismatch_review.csv
    cleaned_predictor_intercorrelations_nontrivial.csv
    volume_bound_group_counts.csv
  review_packet/
    OPEN_THESE_CASES_IN_3D_SLICER.md
    previews/
    nifti_cases/
  article_takeaways_11.txt
```

## How to interpret

The most important checks are:

- `any_geometry_mismatch`
- `size_match`, `spacing_match`, `origin_match`, `direction_match`
- `computed_dice_matches_table_tol_0p02`
- `gt_to_pred_lcc_centroid_distance_mm`
- `suspect_case_pairing_or_geometry`

If geometry is consistent but centroid distances are very large, this supports genuine severe spatial localization/boundary failure.

If geometry mismatches or case-pairing issues are found, external metrics should be corrected before final manuscript writing.

## Visual review

Open the PNG previews first. For suspicious cases, open the NIfTI files in 3D Slicer or ITK-SNAP:

- background: `*_image_pre.nii.gz`
- GT mask: `*_gt.nii.gz`
- raw prediction: `*_pred_v2_raw.nii.gz`
- LCC prediction: `*_pred_v2_lcc.nii.gz`

Contours in preview PNGs:

- GT = green
- raw prediction = red
- LCC prediction = yellow
