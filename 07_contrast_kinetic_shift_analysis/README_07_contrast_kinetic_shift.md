# 07_contrast_kinetic_shift_analysis

This folder adds a CPU-friendly contrast-kinetic shift analysis for the external MAMA-MIA validation.

## Run

Put the folder inside your project:

`C:\Users\olegk\Desktop\MRI Project\07_contrast_kinetic_shift_analysis`

Then from that folder run:

`python contrast_kinetic_shift_analysis.py`

No command-line parameters are needed. Edit only the CONFIG block at the top of the script if your paths differ.

## Outputs

The script creates:

- `metrics/contrast_kinetic_case_features_and_metrics.csv`
- `metrics/contrast_kinetic_curves_long_format.csv`
- `metrics/summary_overall.csv`
- `metrics/summary_by_cohort.csv`
- `metrics/summary_by_phase_quality.csv`
- `metrics/summary_by_cohort_and_phase_quality.csv`
- `metrics/correlations_kinetic_timing_vs_external_metrics.csv`
- `metrics/postprocessing_comparison_overall.csv`
- `figures/*.png`
- `previews/*.png`
- `article_takeaways.txt`

## What the analysis supports in the paper

It helps support a stronger domain-shift story:

> External degradation is associated with measurable DCE acquisition-timing and contrast-kinetic mismatch rather than being an unexplained low external Dice.

It also tests an exploratory no-retraining kinetic-component postprocessing rule. This should be treated as exploratory unless independently validated.
