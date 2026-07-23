# 08 External Failure Attribution Analysis

This folder contains a home-computer analysis script for understanding why external segmentation performance drops on non-I-SPY2 MAMA-MIA cohorts.

It does **not** train a model. It uses existing predictions and expert masks.

## Files

```text
08_external_failure_attribution_analysis/
  run_08_failure_attribution_analysis.py
  README_08_failure_attribution.md
  requirements_08.txt
```

## How to run

Put this folder inside:

```text
C:\Users\olegk\Desktop\MRI Project\
```

Then run:

```text
python run_08_failure_attribution_analysis.py
```

No command-line parameters are needed. Edit the CONFIG block at the top of the script only if your paths changed.

## Required existing folders

```text
C:\nnw\mama_mia_output
C:\nnw\mama_mia_output_v2
C:\Users\olegk\Desktop\MRI Project\segmentations_2\expert
C:\Users\olegk\Desktop\MRI Project\external_manifest\phase_selection_report.csv
```

Optional, but useful:

```text
C:\Users\olegk\Desktop\MRI Project\07_contrast_kinetic_shift_analysis\metrics\contrast_kinetic_case_features_and_metrics.csv
```

If folder 07 has already been run, this script merges those kinetic features into the attribution analysis.

## Outputs

The script creates:

```text
08_external_failure_attribution_analysis/
  metrics/
  figures/
  previews/
  article_takeaways_08.txt
```

Important CSVs:

```text
metrics/failure_attribution_case_table.csv
metrics/summary_overall_methods.csv
metrics/summary_by_cohort.csv
metrics/summary_by_phase_quality.csv
metrics/summary_by_gt_volume_quintile.csv
metrics/summary_by_failure_type.csv
metrics/correlations_with_external_metrics.csv
metrics/top_50_worst_v2_lcc_cases.csv
metrics/top_50_cases_where_lcc_helped.csv
metrics/top_50_oversegmentation_by_volume_ratio.csv
```

Important figures:

```text
fig01_external_pipeline_dice_comparison.png
fig02_external_dice_by_cohort_and_pipeline.png
fig03_v2_lcc_dice_by_cohort_boxplot.png
fig05_v2_lcc_dice_by_gt_volume_quintile.png
fig06_gt_volume_vs_v2_lcc_dice.png
fig07_failure_type_counts.png
fig08_predicted_to_gt_volume_ratio_vs_dice.png
fig09_components_vs_lcc_gain.png
fig17_top_correlations_with_v2_lcc_dice.png
```

## What this analysis tests

The script checks whether the external failure is associated with:

- tumor size;
- oversegmentation / undersegmentation;
- predicted-to-ground-truth volume ratio;
- number of connected components;
- largest connected component effect;
- cohort;
- phase timing quality;
- timing deviation;
- optional contrast-kinetic features from folder 07.

## How to interpret

This is not causal inference. It is failure attribution / exploratory association analysis.

Good manuscript framing:

> External degradation was multifactorial. Timing-aware phase selection and largest connected-component postprocessing improved performance modestly, with the largest gain coming from reduction of false-positive components. Failure attribution showed that poor external performance was associated with oversegmentation, tumor-size effects, component fragmentation, cohort differences, and contrast/protocol shift rather than acquisition timing alone.

Avoid claiming:

> Timing mismatch fully explains the external performance drop.

The previous phase-only analysis already suggested that timing deviation alone has weak case-level correlation with Dice.
