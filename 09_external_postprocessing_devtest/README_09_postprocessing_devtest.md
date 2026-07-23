# 09 External postprocessing dev/test analysis

This folder tests no-training postprocessing variants for the external breast DCE-MRI segmentation predictions.

## Goal

Evaluate whether any simple postprocessing can improve the current best no-training external result:

- `v2_raw`
- `v2 + largest connected component (LCC)`
- small component removal
- conditional LCC
- v1/v2 consensus/intersection/union variants

The analysis uses a deterministic stratified dev/test split:

- DEV: used to choose a rule
- TEST: held-out evaluation of the dev-selected rule

This is important because tuning postprocessing on all 526 external cases and reporting the same cases as final results would be leakage.

## How to run

Place this folder at:

```text
C:\Users\olegk\Desktop\MRI Project\09_external_postprocessing_devtest
```

Then run from inside the folder:

```text
python run_09_postprocessing_devtest.py
```

No command-line parameters are needed. Edit the `CONFIG` block at the top of the Python file only if paths differ.

## Expected input folders

```text
C:\nnw\mama_mia_output
C:\nnw\mama_mia_output_v2
C:\Users\olegk\Desktop\MRI Project\segmentations_2\expert
C:\Users\olegk\Desktop\MRI Project\external_manifest\phase_selection_report.csv
```

## Main outputs

```text
metrics/dev_test_split_cases.csv
metrics/candidate_methods_manifest.csv
metrics/all_candidate_case_metrics_long.csv
metrics/summary_by_split_and_method.csv
metrics/method_comparison_relative_to_baselines.csv
metrics/dev_selection_ranking.csv
metrics/paper_safe_dev_selected_method_report.csv
figures/fig01_top_methods_dev.png
figures/fig02_top_methods_test_screening.png
figures/fig03_delta_vs_v2_lcc_dev.png
figures/fig04_delta_vs_v2_lcc_test_screening.png
figures/fig07_selected_methods_by_cohort_test.png
article_takeaways_09.txt
```

## Interpretation

Use the DEV-selected rule and its TEST performance as the safest postprocessing result.

If the best TEST method was not selected by DEV, treat it as screening-only, not as a final tuned method.

If nothing beats `v2 + LCC` on held-out TEST, keep `v2 + LCC` as the paper-safe no-training external pipeline and describe the other variants as negative/supplementary.
