# finish_09_from_saved_csv.py
# Reads already saved outputs from 09_external_postprocessing_devtest and prints final tables.
# No MRI files are loaded and no postprocessing is recomputed.

from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project\09_external_postprocessing_devtest")
METRICS = ROOT / "metrics"

def infer_family(method: str) -> str:
    m = str(method).lower()
    if m in {"v2_raw", "v1_raw"}:
        return "baseline_raw"
    if "fillholes" in m and "lcc" in m:
        return "lcc_fillholes"
    if "lcc" in m and "conditional" not in m:
        return "lcc"
    if "remove_small" in m or "small" in m:
        return "small_component_removal"
    if "conditional" in m or "cond" in m:
        return "conditional_lcc"
    if "intersect" in m or "intersection" in m:
        return "v1_v2_intersection"
    if "union" in m:
        return "v1_v2_union"
    if "v1" in m and "v2" in m:
        return "v1_v2_consensus"
    return "other"

def load_csv(name: str) -> pd.DataFrame | None:
    path = METRICS / name
    if not path.exists():
        print(f"Missing: {path}")
        return None
    print(f"Loaded: {path}")
    return pd.read_csv(path)

def main():
    print("09 finish-from-saved-CSV")
    print("=========================")
    print(f"Metrics folder: {METRICS}")
    print()

    summary = load_csv("summary_by_split_and_method.csv")
    selected = load_csv("paper_safe_dev_selected_method_report.csv")
    ranking = load_csv("dev_selection_ranking.csv")

    if summary is None:
        print("Cannot continue: summary_by_split_and_method.csv not found.")
        return

    if "family" not in summary.columns:
        summary["family"] = summary["method"].apply(infer_family)
        out = METRICS / "summary_by_split_and_method_with_family.csv"
        summary.to_csv(out, index=False)
        print(f"Saved with family column: {out}")

    # Determine selected method
    selected_method = None
    if selected is not None and "method" in selected.columns and len(selected) > 0:
        # Usually contains multiple rows for selected/v2/raw; take first non-baseline if possible
        methods = list(selected["method"].astype(str).unique())
        for m in methods:
            if m not in {"v2_raw", "v2_lcc"}:
                selected_method = m
                break
        if selected_method is None:
            selected_method = methods[0]
    elif ranking is not None and "method" in ranking.columns and len(ranking) > 0:
        selected_method = str(ranking.iloc[0]["method"])

    print()
    print("Quick comparison: selected method vs baselines")
    print("------------------------------------------------")
    compare_methods = ["v2_raw", "v2_lcc"]
    if selected_method and selected_method not in compare_methods:
        compare_methods.append(selected_method)
    quick = summary[summary["method"].isin(compare_methods)].copy()
    if len(quick):
        cols = [c for c in ["split","method","family","dice_mean","precision_mean","recall_mean","dice_ci95_low","dice_ci95_high"] if c in quick.columns]
        quick = quick.sort_values(["split","dice_mean"], ascending=[True, False])
        print(quick[cols].to_string(index=False))
    else:
        print("No baseline/selected rows found.")

    print()
    print("Top TEST methods by Dice — screening only")
    print("-----------------------------------------")
    test = summary[summary["split"].astype(str).str.lower() == "test"].copy()
    if len(test):
        top = test.sort_values("dice_mean", ascending=False).head(15)
        cols = [c for c in ["method","family","dice_mean","precision_mean","recall_mean","dice_ci95_low","dice_ci95_high"] if c in top.columns]
        print(top[cols].to_string(index=False))
    else:
        print("No TEST rows found.")

    # Paired-looking delta from test means only (not a statistical paired test)
    print()
    print("Mean Dice deltas on held-out TEST")
    print("---------------------------------")
    test_lookup = {str(r["method"]): float(r["dice_mean"]) for _, r in test.iterrows()} if len(test) else {}
    base_lcc = test_lookup.get("v2_lcc")
    base_raw = test_lookup.get("v2_raw")
    for m in compare_methods:
        if m in test_lookup:
            d_lcc = test_lookup[m] - base_lcc if base_lcc is not None else float("nan")
            d_raw = test_lookup[m] - base_raw if base_raw is not None else float("nan")
            print(f"{m:30s} Dice={test_lookup[m]:.4f}  Δvs v2_lcc={d_lcc:+.4f}  Δvs v2_raw={d_raw:+.4f}")

    print()
    print("Interpretation guide")
    print("--------------------")
    print("Use only the dev-selected method as paper-safe. Top TEST methods are screening only and should not be reported as tuned final methods unless revalidated.")
    print("If selected method improves held-out TEST by <0.01 Dice or CI strongly overlaps v2_lcc, treat it as a sensitivity analysis, not a new method.")

if __name__ == "__main__":
    main()
