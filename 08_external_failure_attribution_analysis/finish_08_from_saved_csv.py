# finish_08_from_saved_csv.py
# ------------------------------------------------------------
# Finishes the 08 external failure-attribution analysis from the
# already saved case-level CSV, without recomputing MRI masks/images.
#
# Put this file in:
#   C:\Users\olegk\Desktop\MRI Project\08_external_failure_attribution_analysis
# Run:
#   python finish_08_from_saved_csv.py
# ------------------------------------------------------------

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from e

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# =========================
# CONFIG — EDIT ONLY IF NEEDED
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_DIR = SCRIPT_DIR / "metrics"
FIGURES_DIR = SCRIPT_DIR / "figures"
PREVIEWS_DIR = SCRIPT_DIR / "previews"
TAKEAWAYS_PATH = SCRIPT_DIR / "article_takeaways_08.txt"

CASE_TABLE_PATH = METRICS_DIR / "failure_attribution_case_table.csv"

# Current best no-retraining external pipeline from prior analysis.
PRIMARY_METHOD = "v2_lcc"
BASELINE_METHOD = "v2_raw"
RAW_V1_METHOD = "v1_raw"
LCC_V1_METHOD = "v1_lcc"

# Failure-type thresholds. These are descriptive, not tuned for final performance.
GOOD_DICE_THRESHOLD = 0.70
VERY_LOW_DICE_THRESHOLD = 0.30
LOW_PRECISION_THRESHOLD = 0.50
LOW_RECALL_THRESHOLD = 0.50
SEVERE_LOW_PRECISION_THRESHOLD = 0.30
SEVERE_LOW_RECALL_THRESHOLD = 0.30
HIGH_VOLUME_RATIO_THRESHOLD = 2.0
LOW_VOLUME_RATIO_THRESHOLD = 0.5

# Top correlations to show in figure/table.
TOP_N_CORR = 20


# =========================
# UTILITIES
# =========================

def ensure_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)


def read_case_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Case-level table not found:\n  {path}\n\n"
            "The previous 08 script must have created this before it crashed."
        )
    df = pd.read_csv(path)

    # Critical fix: avoid ambiguous Series/DataFrame behavior from duplicate column names.
    duplicate_count = int(pd.Index(df.columns).duplicated().sum())
    if duplicate_count:
        print(f"WARNING: found {duplicate_count} duplicate column name(s). Keeping first copy only.")
        df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    # Try to convert numeric-looking object columns to numeric, leaving IDs/categories intact.
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            # Convert if at least 80% of non-empty values look numeric.
            non_empty = df[col].notna().sum()
            if non_empty > 0 and converted.notna().sum() / non_empty >= 0.80:
                df[col] = converted
    return df


def available_method_prefixes(df: pd.DataFrame) -> list[str]:
    preferred = ["v1_raw", "v1_lcc", "v2_raw", "v2_lcc", "v2_kinetic_cc"]
    found = []
    for p in preferred:
        if f"{p}_dice" in df.columns:
            found.append(p)
    # Also detect any other *_dice prefixes.
    for col in df.columns:
        if col.endswith("_dice"):
            p = col[:-5]
            if p not in found and not p.startswith("delta"):
                found.append(p)
    return found


def metric_summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "median": float(s.median()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def summarize_methods(df: pd.DataFrame, methods: list[str], group_col: str | None = None) -> pd.DataFrame:
    metric_suffixes = ["dice", "iou", "precision", "recall", "pred_vol_cm3", "gt_vol_cm3", "pred_to_gt_vol_ratio", "num_components", "largest_component_fraction"]
    rows = []

    if group_col is None:
        groups = [("ALL", df)]
        group_name = "group"
    else:
        if group_col not in df.columns:
            return pd.DataFrame()
        groups = list(df.groupby(group_col, dropna=False))
        group_name = group_col

    for key, grp in groups:
        for method in methods:
            row = {group_name: key, "method": method, "n_cases": int(len(grp))}
            for suffix in metric_suffixes:
                col = f"{method}_{suffix}"
                # gt_vol_cm3 may be shared rather than method-specific.
                if suffix == "gt_vol_cm3" and col not in grp.columns and "gt_vol_cm3" in grp.columns:
                    col = "gt_vol_cm3"
                if col in grp.columns:
                    sm = metric_summary(grp[col])
                    row[f"{suffix}_mean"] = sm["mean"]
                    row[f"{suffix}_std"] = sm["std"]
                    row[f"{suffix}_median"] = sm["median"]
            rows.append(row)
    return pd.DataFrame(rows)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Common deltas.
    if "v2_lcc_dice" in df.columns and "v2_raw_dice" in df.columns and "delta_v2_lcc_minus_v2_raw_dice" not in df.columns:
        df["delta_v2_lcc_minus_v2_raw_dice"] = df["v2_lcc_dice"] - df["v2_raw_dice"]
    if "v2_raw_dice" in df.columns and "v1_raw_dice" in df.columns and "delta_v2_raw_minus_v1_raw_dice" not in df.columns:
        df["delta_v2_raw_minus_v1_raw_dice"] = df["v2_raw_dice"] - df["v1_raw_dice"]
    if "v2_lcc_dice" in df.columns and "v1_lcc_dice" in df.columns and "delta_v2_lcc_minus_v1_lcc_dice" not in df.columns:
        df["delta_v2_lcc_minus_v1_lcc_dice"] = df["v2_lcc_dice"] - df["v1_lcc_dice"]

    # Method-specific predicted/GT volume ratio.
    for method in available_method_prefixes(df):
        pred_col_candidates = [f"{method}_pred_vol_cm3", f"{method}_pred_volume_cm3", f"{method}_volume_cm3"]
        pred_col = next((c for c in pred_col_candidates if c in df.columns), None)
        gt_col = "gt_vol_cm3" if "gt_vol_cm3" in df.columns else ("gt_volume_cm3" if "gt_volume_cm3" in df.columns else None)
        ratio_col = f"{method}_pred_to_gt_vol_ratio"
        if pred_col and gt_col and ratio_col not in df.columns:
            gt = pd.to_numeric(df[gt_col], errors="coerce")
            pred = pd.to_numeric(df[pred_col], errors="coerce")
            df[ratio_col] = np.where(gt > 0, pred / gt, np.nan)

    # Generic best ratio for current primary method.
    primary_ratio = f"{PRIMARY_METHOD}_pred_to_gt_vol_ratio"
    if primary_ratio in df.columns and "pred_to_gt_vol_ratio" not in df.columns:
        df["pred_to_gt_vol_ratio"] = df[primary_ratio]

    # Tumor-volume quintiles.
    gt_col = "gt_vol_cm3" if "gt_vol_cm3" in df.columns else ("gt_volume_cm3" if "gt_volume_cm3" in df.columns else None)
    if gt_col and "gt_volume_quintile" not in df.columns:
        gt = pd.to_numeric(df[gt_col], errors="coerce")
        try:
            df["gt_volume_quintile"] = pd.qcut(
                gt.rank(method="first"),
                q=5,
                labels=["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"],
            )
        except Exception:
            df["gt_volume_quintile"] = np.nan

    # Coarse predicted-volume-ratio groups.
    ratio_col = "pred_to_gt_vol_ratio" if "pred_to_gt_vol_ratio" in df.columns else primary_ratio
    if ratio_col in df.columns and "volume_ratio_group" not in df.columns:
        r = pd.to_numeric(df[ratio_col], errors="coerce")
        df["volume_ratio_group"] = pd.cut(
            r,
            bins=[-np.inf, 0.5, 1.0, 2.0, 5.0, np.inf],
            labels=["<0.5 under", "0.5-1.0", "1.0-2.0", "2.0-5.0 over", ">5.0 severe over"],
        )

    # Failure taxonomy from primary method.
    dice = pd.to_numeric(df.get(f"{PRIMARY_METHOD}_dice", pd.Series(np.nan, index=df.index)), errors="coerce")
    prec = pd.to_numeric(df.get(f"{PRIMARY_METHOD}_precision", pd.Series(np.nan, index=df.index)), errors="coerce")
    rec = pd.to_numeric(df.get(f"{PRIMARY_METHOD}_recall", pd.Series(np.nan, index=df.index)), errors="coerce")
    ratio = pd.to_numeric(df.get(ratio_col, pd.Series(np.nan, index=df.index)), errors="coerce") if ratio_col in df.columns else pd.Series(np.nan, index=df.index)

    small_mask = pd.Series(False, index=df.index)
    if "gt_volume_quintile" in df.columns:
        small_mask = df["gt_volume_quintile"].astype(str).eq("Q1 smallest")

    failure = pd.Series("mixed_or_boundary_error", index=df.index, dtype=object)
    failure[dice >= GOOD_DICE_THRESHOLD] = "good_segmentation"
    failure[(dice < VERY_LOW_DICE_THRESHOLD) & (rec < SEVERE_LOW_RECALL_THRESHOLD)] = "severe_miss_or_undersegmentation"
    failure[(dice < GOOD_DICE_THRESHOLD) & (prec < SEVERE_LOW_PRECISION_THRESHOLD) & (rec >= LOW_RECALL_THRESHOLD)] = "severe_oversegmentation"
    failure[(dice < GOOD_DICE_THRESHOLD) & (prec < LOW_PRECISION_THRESHOLD) & (rec >= LOW_RECALL_THRESHOLD)] = "oversegmentation"
    failure[(dice < GOOD_DICE_THRESHOLD) & (rec < LOW_RECALL_THRESHOLD) & (prec >= LOW_PRECISION_THRESHOLD)] = "undersegmentation"
    failure[(dice < GOOD_DICE_THRESHOLD) & small_mask] = "small_lesion_failure"
    failure[(dice < GOOD_DICE_THRESHOLD) & (ratio > HIGH_VOLUME_RATIO_THRESHOLD) & (rec >= LOW_RECALL_THRESHOLD)] = "volume_inflation_oversegmentation"
    failure[(dice < GOOD_DICE_THRESHOLD) & (ratio < LOW_VOLUME_RATIO_THRESHOLD) & (rec < LOW_RECALL_THRESHOLD)] = "low_volume_undersegmentation"
    df["primary_failure_type"] = failure

    return df


def correlation_table(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    # De-duplicate again defensively.
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    numeric_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 30 and s.nunique(dropna=True) >= 3:
            numeric_cols.append(c)

    # Exclude direct segmentation metrics as predictors to avoid circular trivial correlations.
    direct_metric_tokens = ["_dice", "_iou", "_precision", "_recall"]
    exclude_exact = set(targets)
    feature_cols = []
    for c in numeric_cols:
        if c in exclude_exact:
            continue
        if any(tok in c for tok in direct_metric_tokens):
            continue
        # Keep derived explanatory columns like volume ratio, components, timing, kinetic features.
        feature_cols.append(c)

    rows = []
    for target in targets:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce")
        for f in feature_cols:
            x = pd.to_numeric(df[f], errors="coerce")
            sub = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 30 or sub["x"].nunique() < 3 or sub["y"].nunique() < 3:
                continue
            if spearmanr is not None:
                rho, p = spearmanr(sub["x"], sub["y"])
            else:
                rho = sub["x"].rank().corr(sub["y"].rank())
                p = np.nan
            rows.append({
                "target": target,
                "feature": f,
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "p_value": float(p) if pd.notna(p) else np.nan,
                "n": int(len(sub)),
                "abs_rho": abs(float(rho)) if pd.notna(rho) else np.nan,
            })
    corr = pd.DataFrame(rows)
    if len(corr) == 0:
        return corr

    # Benjamini-Hochberg FDR per target, if p-values exist.
    corr["q_value_bh"] = np.nan
    for target, g_idx in corr.groupby("target").groups.items():
        idx = list(g_idx)
        pvals = corr.loc[idx, "p_value"].astype(float)
        valid = pvals.notna()
        if valid.sum() == 0:
            continue
        valid_idx = pvals[valid].index.tolist()
        p = pvals.loc[valid_idx].values
        order = np.argsort(p)
        ranked = p[order]
        m = len(ranked)
        q = ranked * m / (np.arange(m) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0, 1)
        q_original = np.empty_like(q)
        q_original[order] = q
        corr.loc[valid_idx, "q_value_bh"] = q_original

    return corr.sort_values(["target", "abs_rho"], ascending=[True, False])


def save_group_summaries(df: pd.DataFrame, methods: list[str]) -> None:
    summaries = {
        "summary_overall_methods.csv": summarize_methods(df, methods, None),
        "summary_by_cohort.csv": summarize_methods(df, methods, "cohort"),
        "summary_by_phase_quality.csv": summarize_methods(df, methods, "overall"),
        "summary_by_strict_phase_group.csv": summarize_methods(df, methods, "strict_phase_group"),
        "summary_by_gt_volume_quintile.csv": summarize_methods(df, methods, "gt_volume_quintile"),
        "summary_by_volume_ratio_group.csv": summarize_methods(df, methods, "volume_ratio_group"),
        "summary_by_failure_type.csv": summarize_methods(df, methods, "primary_failure_type"),
    }
    for name, table in summaries.items():
        if table is not None and len(table) > 0:
            table.to_csv(METRICS_DIR / name, index=False)
            print(f"  saved {name}")

    # Cohort x failure type compact summary for primary method.
    if "cohort" in df.columns and "primary_failure_type" in df.columns:
        rows = []
        for (cohort, ftype), grp in df.groupby(["cohort", "primary_failure_type"], dropna=False):
            row = {"cohort": cohort, "primary_failure_type": ftype, "n": int(len(grp))}
            for col in [f"{PRIMARY_METHOD}_dice", f"{PRIMARY_METHOD}_precision", f"{PRIMARY_METHOD}_recall", "pred_to_gt_vol_ratio"]:
                if col in grp.columns:
                    row[f"{col}_mean"] = pd.to_numeric(grp[col], errors="coerce").mean()
                    row[f"{col}_median"] = pd.to_numeric(grp[col], errors="coerce").median()
            rows.append(row)
        pd.DataFrame(rows).to_csv(METRICS_DIR / "summary_by_cohort_and_failure_type.csv", index=False)
        print("  saved summary_by_cohort_and_failure_type.csv")


def fig_save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_external_pipeline_dice(df: pd.DataFrame, methods: list[str]) -> None:
    means = []
    labels = []
    for m in methods:
        col = f"{m}_dice"
        if col in df.columns:
            means.append(pd.to_numeric(df[col], errors="coerce").mean())
            labels.append(m.replace("_", " + "))
    if not means:
        return
    plt.figure(figsize=(9, 5))
    plt.bar(labels, means)
    plt.ylabel("Mean Dice")
    plt.title("External Dice by pipeline")
    plt.ylim(0, max(0.75, np.nanmax(means) + 0.05))
    plt.xticks(rotation=25, ha="right")
    for i, v in enumerate(means):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    fig_save(FIGURES_DIR / "fig01_external_pipeline_dice_comparison.png")


def plot_dice_by_cohort(df: pd.DataFrame, methods: list[str]) -> None:
    if "cohort" not in df.columns:
        return
    cohorts = list(df["cohort"].dropna().astype(str).unique())
    cohorts = [c for c in ["DUKE", "ISPY1", "NACT"] if c in cohorts] + [c for c in cohorts if c not in ["DUKE", "ISPY1", "NACT"]]
    valid_methods = [m for m in methods if f"{m}_dice" in df.columns]
    if not cohorts or not valid_methods:
        return
    x = np.arange(len(cohorts))
    width = 0.8 / len(valid_methods)
    plt.figure(figsize=(10, 5.5))
    for j, m in enumerate(valid_methods):
        vals = [pd.to_numeric(df[df["cohort"].astype(str) == c][f"{m}_dice"], errors="coerce").mean() for c in cohorts]
        plt.bar(x + (j - (len(valid_methods)-1)/2) * width, vals, width=width, label=m.replace("_", " + "))
    plt.xticks(x, cohorts)
    plt.ylabel("Mean Dice")
    plt.title("External Dice by cohort and pipeline")
    plt.legend(fontsize=8)
    plt.ylim(0, 0.75)
    fig_save(FIGURES_DIR / "fig02_external_dice_by_cohort_and_pipeline.png")


def plot_precision_recall(df: pd.DataFrame, methods: list[str]) -> None:
    valid = [m for m in methods if f"{m}_precision" in df.columns and f"{m}_recall" in df.columns]
    if not valid:
        return
    labels = [m.replace("_", " + ") for m in valid]
    prec = [pd.to_numeric(df[f"{m}_precision"], errors="coerce").mean() for m in valid]
    rec = [pd.to_numeric(df[f"{m}_recall"], errors="coerce").mean() for m in valid]
    x = np.arange(len(valid))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, prec, width, label="Precision")
    plt.bar(x + width/2, rec, width, label="Recall")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Mean value")
    plt.title("External precision-recall trade-off")
    plt.ylim(0, 1.0)
    plt.legend()
    fig_save(FIGURES_DIR / "fig03_external_precision_recall_by_pipeline.png")


def plot_dice_by_volume_quintile(df: pd.DataFrame) -> None:
    if "gt_volume_quintile" not in df.columns or f"{PRIMARY_METHOD}_dice" not in df.columns:
        return
    order = ["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"]
    groups = [pd.to_numeric(df[df["gt_volume_quintile"].astype(str) == q][f"{PRIMARY_METHOD}_dice"], errors="coerce").dropna().values for q in order]
    groups = [g for g in groups if len(g) > 0]
    labels = [q for q in order if (df["gt_volume_quintile"].astype(str) == q).any()]
    if not groups:
        return
    plt.figure(figsize=(9, 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.ylabel("v2 + LCC Dice")
    plt.title("External Dice by ground-truth tumor-volume quintile")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.0)
    fig_save(FIGURES_DIR / "fig04_v2_lcc_dice_by_gt_volume_quintile.png")


def plot_failure_counts(df: pd.DataFrame) -> None:
    if "primary_failure_type" not in df.columns:
        return
    counts = df["primary_failure_type"].value_counts().sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(counts.index.astype(str), counts.values)
    plt.xlabel("Number of cases")
    plt.title("External failure-type taxonomy")
    for i, v in enumerate(counts.values):
        plt.text(v + 1, i, str(int(v)), va="center")
    fig_save(FIGURES_DIR / "fig05_failure_type_counts.png")


def plot_volume_ratio_vs_dice(df: pd.DataFrame) -> None:
    if "pred_to_gt_vol_ratio" not in df.columns or f"{PRIMARY_METHOD}_dice" not in df.columns:
        return
    x = pd.to_numeric(df["pred_to_gt_vol_ratio"], errors="coerce")
    y = pd.to_numeric(df[f"{PRIMARY_METHOD}_dice"], errors="coerce")
    sub = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 10:
        return
    sub = sub[sub["x"] <= sub["x"].quantile(0.99)]
    plt.figure(figsize=(8, 5.5))
    plt.scatter(sub["x"], sub["y"], alpha=0.55, s=18)
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted / ground-truth tumor-volume ratio")
    plt.ylabel("v2 + LCC Dice")
    plt.title("Volume inflation/deflation vs external Dice")
    plt.ylim(0, 1.0)
    fig_save(FIGURES_DIR / "fig06_predicted_to_gt_volume_ratio_vs_dice.png")


def plot_components_vs_lcc_gain(df: pd.DataFrame) -> None:
    comp_candidates = ["v2_raw_num_components", "v2_raw_components", "num_components", "raw_num_components"]
    comp_col = next((c for c in comp_candidates if c in df.columns), None)
    gain_col = "delta_v2_lcc_minus_v2_raw_dice"
    if comp_col is None or gain_col not in df.columns:
        return
    x = pd.to_numeric(df[comp_col], errors="coerce")
    y = pd.to_numeric(df[gain_col], errors="coerce")
    sub = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 10:
        return
    sub = sub[sub["x"] <= sub["x"].quantile(0.99)]
    plt.figure(figsize=(8, 5.5))
    plt.scatter(sub["x"], sub["y"], alpha=0.55, s=18)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Number of connected components before LCC")
    plt.ylabel("Dice gain from LCC")
    plt.title("Connected components vs LCC improvement")
    fig_save(FIGURES_DIR / "fig07_components_vs_lcc_gain.png")


def plot_top_correlations(corr: pd.DataFrame, target: str) -> None:
    if corr is None or len(corr) == 0:
        return
    sub = corr[corr["target"] == target].sort_values("abs_rho", ascending=False).head(TOP_N_CORR).copy()
    if len(sub) == 0:
        return
    sub = sub.sort_values("spearman_rho")
    plt.figure(figsize=(10, 7))
    plt.barh(sub["feature"], sub["spearman_rho"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Spearman rho")
    plt.title(f"Top exploratory correlations with {target}")
    fig_save(FIGURES_DIR / f"fig08_top_correlations_{target}.png")


def plot_dice_by_cohort_phase(df: pd.DataFrame) -> None:
    if "cohort" not in df.columns or "overall" not in df.columns or f"{PRIMARY_METHOD}_dice" not in df.columns:
        return
    tab = df.groupby(["cohort", "overall"], dropna=False)[f"{PRIMARY_METHOD}_dice"].mean().reset_index()
    if len(tab) == 0:
        return
    cohorts = [c for c in ["DUKE", "ISPY1", "NACT"] if c in tab["cohort"].astype(str).unique()]
    phases = [p for p in ["good", "acceptable", "poor", "no_times"] if p in tab["overall"].astype(str).unique()]
    if not cohorts or not phases:
        return
    x = np.arange(len(cohorts))
    width = 0.8 / len(phases)
    plt.figure(figsize=(10, 5.5))
    for j, ph in enumerate(phases):
        vals = []
        for c in cohorts:
            tmp = tab[(tab["cohort"].astype(str) == c) & (tab["overall"].astype(str) == ph)]
            vals.append(float(tmp[f"{PRIMARY_METHOD}_dice"].iloc[0]) if len(tmp) else np.nan)
        plt.bar(x + (j - (len(phases)-1)/2) * width, vals, width=width, label=ph)
    plt.xticks(x, cohorts)
    plt.ylabel("v2 + LCC Dice")
    plt.title("External Dice by cohort and phase-quality label")
    plt.ylim(0, 0.8)
    plt.legend(fontsize=8)
    fig_save(FIGURES_DIR / "fig09_dice_by_cohort_and_phase_quality.png")


def write_takeaways(df: pd.DataFrame, methods: list[str], corr: pd.DataFrame) -> None:
    lines = []
    lines.append("08 EXTERNAL FAILURE ATTRIBUTION — QUICK TAKEAWAYS")
    lines.append("=" * 72)
    lines.append(f"Cases analyzed: {len(df)}")
    lines.append("")

    lines.append("Pipeline performance, mean Dice:")
    for m in methods:
        col = f"{m}_dice"
        if col in df.columns:
            lines.append(f"  {m}: {pd.to_numeric(df[col], errors='coerce').mean():.4f}")
    lines.append("")

    if f"{PRIMARY_METHOD}_precision" in df.columns and f"{PRIMARY_METHOD}_recall" in df.columns:
        lines.append(f"Primary pipeline ({PRIMARY_METHOD}) precision/recall:")
        lines.append(f"  precision: {pd.to_numeric(df[f'{PRIMARY_METHOD}_precision'], errors='coerce').mean():.4f}")
        lines.append(f"  recall   : {pd.to_numeric(df[f'{PRIMARY_METHOD}_recall'], errors='coerce').mean():.4f}")
        lines.append("")

    if "primary_failure_type" in df.columns:
        lines.append("Failure-type counts:")
        for k, v in df["primary_failure_type"].value_counts().items():
            lines.append(f"  {k}: {int(v)}")
        lines.append("")

    if "pred_to_gt_vol_ratio" in df.columns:
        r = pd.to_numeric(df["pred_to_gt_vol_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(r):
            lines.append("Predicted/GT volume ratio, primary pipeline:")
            lines.append(f"  median: {r.median():.3f}")
            lines.append(f"  mean  : {r.mean():.3f}")
            lines.append(f"  q25/q75: {r.quantile(0.25):.3f} / {r.quantile(0.75):.3f}")
            lines.append("")

    if corr is not None and len(corr) > 0:
        target = f"{PRIMARY_METHOD}_dice"
        sub = corr[corr["target"] == target].sort_values("abs_rho", ascending=False).head(10)
        if len(sub):
            lines.append(f"Strongest exploratory correlations with {target}:")
            for _, row in sub.iterrows():
                p = row.get("p_value", np.nan)
                q = row.get("q_value_bh", np.nan)
                lines.append(
                    f"  {row['feature']}: rho={row['spearman_rho']:.3f}, "
                    f"p={p:.4g}, q(BH)={q:.4g}, n={int(row['n'])}"
                )
            lines.append("")

    lines.append("Suggested manuscript framing:")
    lines.append("  Use this analysis to explain external failure modes, especially oversegmentation, tumor-size")
    lines.append("  sensitivity, component-level false positives, and cohort/domain effects. Do not present")
    lines.append("  these post-hoc analyses as a tuned final method unless validated on a separate held-out split.")
    lines.append("")
    lines.append("Important caution:")
    lines.append("  Correlations are exploratory. If many factors are tested, report effect sizes and use BH/FDR")
    lines.append("  correction or clearly mark the analysis as hypothesis-generating.")

    TAKEAWAYS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("08 finish-from-saved-CSV")
    print("=========================")
    print(f"Script folder: {SCRIPT_DIR}")
    print(f"Case table   : {CASE_TABLE_PATH}")

    df = read_case_table(CASE_TABLE_PATH)
    print(f"Cases loaded : {len(df)}")
    df = add_derived_columns(df)

    # Save enhanced table so the derived columns are preserved.
    enhanced_path = METRICS_DIR / "failure_attribution_case_table_enhanced.csv"
    df.to_csv(enhanced_path, index=False)
    print(f"Enhanced case table saved: {enhanced_path}")

    methods = available_method_prefixes(df)
    print(f"Detected methods: {methods}")

    print("Writing summary tables...")
    save_group_summaries(df, methods)

    print("Computing exploratory correlations...")
    targets = [
        f"{PRIMARY_METHOD}_dice",
        f"{PRIMARY_METHOD}_precision",
        f"{PRIMARY_METHOD}_recall",
        "delta_v2_lcc_minus_v2_raw_dice",
    ]
    corr = correlation_table(df, [t for t in targets if t in df.columns])
    corr_path = METRICS_DIR / "correlations_failure_factors_vs_external_metrics.csv"
    corr.to_csv(corr_path, index=False)
    print(f"  saved {corr_path.name}")

    print("Creating figures...")
    plot_external_pipeline_dice(df, methods)
    plot_dice_by_cohort(df, methods)
    plot_precision_recall(df, methods)
    plot_dice_by_volume_quintile(df)
    plot_failure_counts(df)
    plot_volume_ratio_vs_dice(df)
    plot_components_vs_lcc_gain(df)
    plot_top_correlations(corr, f"{PRIMARY_METHOD}_dice")
    plot_top_correlations(corr, f"{PRIMARY_METHOD}_precision")
    plot_top_correlations(corr, f"{PRIMARY_METHOD}_recall")
    plot_top_correlations(corr, "delta_v2_lcc_minus_v2_raw_dice")
    plot_dice_by_cohort_phase(df)

    write_takeaways(df, methods, corr)

    print("\nDONE — finished 08 from saved CSV, no MRI recomputation.")
    print(f"Metrics : {METRICS_DIR}")
    print(f"Figures : {FIGURES_DIR}")
    print(f"Takeaways: {TAKEAWAYS_PATH}")

    if f"{PRIMARY_METHOD}_dice" in df.columns:
        print("\nQuick numbers:")
        for m in methods:
            col = f"{m}_dice"
            if col in df.columns:
                print(f"  {col}: {pd.to_numeric(df[col], errors='coerce').mean():.4f}")
        if f"{PRIMARY_METHOD}_precision" in df.columns:
            print(f"  {PRIMARY_METHOD}_precision: {pd.to_numeric(df[f'{PRIMARY_METHOD}_precision'], errors='coerce').mean():.4f}")
        if f"{PRIMARY_METHOD}_recall" in df.columns:
            print(f"  {PRIMARY_METHOD}_recall: {pd.to_numeric(df[f'{PRIMARY_METHOD}_recall'], errors='coerce').mean():.4f}")


if __name__ == "__main__":
    main()
