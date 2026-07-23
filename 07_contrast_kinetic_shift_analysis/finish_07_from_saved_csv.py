# finish_07_from_saved_csv.py
# Folder: 07_contrast_kinetic_shift_analysis
#
# Purpose:
#   Finish the 07 contrast-kinetic analysis AFTER the long case-processing step
#   already completed. This script DOES NOT read MRI images and DOES NOT recompute
#   case-level kinetic features. It only reads the saved CSV files and regenerates:
#     - summary CSV tables
#     - correlation CSV table
#     - publication-ready PNG figures
#     - article_takeaways.txt
#
# Run from:
#   C:\Users\olegk\Desktop\MRI Project\07_contrast_kinetic_shift_analysis
#
# Command:
#   python finish_07_from_saved_csv.py
#
# Edit only CONFIG if your project path differs.

from __future__ import annotations

from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception as e:
    raise ImportError("This script needs scipy. Install with: pip install scipy") from e

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError("This script needs matplotlib. Install with: pip install matplotlib") from e


# =============================================================================
# CONFIG — EDIT ONLY HERE IF NEEDED
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")
OUT_DIR = PROJECT_ROOT / "07_contrast_kinetic_shift_analysis"
METRICS_DIR = OUT_DIR / "metrics"
FIGURES_DIR = OUT_DIR / "figures"
PREVIEW_DIR = OUT_DIR / "previews"

CASE_CSV = METRICS_DIR / "contrast_kinetic_case_features_and_metrics.csv"
CURVE_CSV = METRICS_DIR / "contrast_kinetic_curves_long_format.csv"
ERROR_CSV = METRICS_DIR / "contrast_kinetic_errors.csv"

TARGET_EARLY_S = 90
TARGET_LATE_S = 420


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    for d in [OUT_DIR, METRICS_DIR, FIGURES_DIR, PREVIEW_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def read_case_csv() -> pd.DataFrame:
    if not CASE_CSV.exists():
        # Make the error helpful by listing nearby CSV files.
        nearby = sorted(METRICS_DIR.glob("*.csv")) if METRICS_DIR.exists() else []
        msg = [
            f"Could not find case-level CSV: {CASE_CSV}",
            "The long script must have saved this before it crashed.",
        ]
        if nearby:
            msg.append("CSV files currently found in metrics folder:")
            msg.extend([f"  - {p.name}" for p in nearby])
        raise FileNotFoundError("\n".join(msg))
    return pd.read_csv(CASE_CSV)


def read_curve_csv() -> pd.DataFrame:
    if CURVE_CSV.exists():
        return pd.read_csv(CURVE_CSV)
    return pd.DataFrame()


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def summarize(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    # This is the fixed version. For overall summary we group the copied dataframe,
    # not the original df, so the temporary "group" column exists.
    if not group_cols:
        work = df.copy()
        work["group"] = "ALL"
        group_cols = ["group"]
    else:
        work = df.copy()

    rows = []
    for key, grp in work.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row["n"] = int(len(grp))
        for col in metric_cols:
            if col in grp.columns:
                vals = pd.to_numeric(grp[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
                row[f"{col}_std"] = float(vals.std()) if vals.notna().sum() > 1 else np.nan
                row[f"{col}_median"] = float(vals.median()) if vals.notna().any() else np.nan
                row[f"{col}_q25"] = float(vals.quantile(0.25)) if vals.notna().any() else np.nan
                row[f"{col}_q75"] = float(vals.quantile(0.75)) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_box_by_cohort(df: pd.DataFrame) -> None:
    needed = {"cohort", "first_post_s", "last_post_s"}
    if not needed.issubset(df.columns):
        return
    cohorts = [c for c in ["DUKE", "ISPY1", "NACT"] if c in set(df["cohort"].dropna().astype(str))]
    if not cohorts:
        return
    data_first = [numeric(df.loc[df["cohort"] == c], "first_post_s").dropna().values for c in cohorts]
    data_last = [numeric(df.loc[df["cohort"] == c], "last_post_s").dropna().values for c in cohorts]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions_first = np.arange(len(cohorts)) * 2.0
    positions_last = positions_first + 0.7
    ax.boxplot(data_first, positions=positions_first, widths=0.55)
    ax.boxplot(data_last, positions=positions_last, widths=0.55)
    ax.axhline(TARGET_EARLY_S, linestyle="--", linewidth=1, label="target early 90s")
    ax.axhline(TARGET_LATE_S, linestyle=":", linewidth=1, label="target late 420s")
    ax.set_xticks(positions_first + 0.35)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Acquisition time (s)")
    ax.set_title("External DCE acquisition timing by cohort")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_acquisition_timing_by_cohort.png", dpi=300)
    plt.close(fig)


def plot_mean_curves(curve_df: pd.DataFrame) -> None:
    if curve_df.empty or not {"cohort", "time_s", "tumor_rel_enh"}.issubset(curve_df.columns):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for cohort, grp in curve_df.groupby("cohort"):
        tmp = grp.dropna(subset=["time_s", "tumor_rel_enh"]).copy()
        if tmp.empty:
            continue
        tmp["time_s"] = pd.to_numeric(tmp["time_s"], errors="coerce")
        tmp["tumor_rel_enh"] = pd.to_numeric(tmp["tumor_rel_enh"], errors="coerce")
        tmp = tmp.dropna(subset=["time_s", "tumor_rel_enh"])
        if tmp.empty:
            continue
        tmp["time_bin"] = (tmp["time_s"] / 60).round() * 60
        agg = tmp.groupby("time_bin")["tumor_rel_enh"].mean().reset_index()
        ax.plot(agg["time_bin"], agg["tumor_rel_enh"], marker="o", label=str(cohort))
    ax.axvline(TARGET_EARLY_S, linestyle="--", linewidth=1)
    ax.axvline(TARGET_LATE_S, linestyle=":", linewidth=1)
    ax.set_xlabel("Acquisition time (s)")
    ax.set_ylabel("Mean tumor relative enhancement")
    ax.set_title("Observed tumor enhancement curves by external cohort")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_observed_enhancement_curves_by_cohort.png", dpi=300)
    plt.close(fig)


def plot_bar_metric(df: pd.DataFrame, group_col: str, metric: str, filename: str, title: str, ylabel: str) -> None:
    if group_col not in df.columns or metric not in df.columns:
        return
    sub = df.copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return
    preferred = ["good", "acceptable", "poor", "no_times", "unknown"]
    order = [x for x in preferred if x in set(sub[group_col].astype(str))]
    if not order:
        order = sorted(sub[group_col].dropna().astype(str).unique())
    means = [sub.loc[sub[group_col].astype(str) == g, metric].mean() for g in order]
    errs = [sub.loc[sub[group_col].astype(str) == g, metric].std() for g in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(order, means, yerr=errs, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x: str, y: str, filename: str, title: str, xlabel: str, ylabel: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    sub = df.copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna(subset=[x, y])
    sub = sub[np.isfinite(sub[x]) & np.isfinite(sub[y])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    if "cohort" in sub.columns:
        cohorts = sorted(sub["cohort"].dropna().astype(str).unique())
        for c in cohorts:
            g = sub[sub["cohort"].astype(str) == c]
            ax.scatter(g[x], g[y], s=18, alpha=0.65, label=c)
        if len(cohorts) > 1:
            ax.legend(fontsize=8)
    else:
        ax.scatter(sub[x], sub[y], s=18, alpha=0.65)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def write_takeaways(df: pd.DataFrame, corr_df: pd.DataFrame, post_df: pd.DataFrame) -> None:
    lines = []
    lines.append("CONTRAST-KINETIC SHIFT ANALYSIS — QUICK TAKEAWAYS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Cases analyzed: {len(df)}")
    if "cohort" in df.columns:
        lines.append("Cohort counts:")
        for c, n in df["cohort"].value_counts().items():
            lines.append(f"  {c}: {n}")
    lines.append("")
    if "first_post_s" in df.columns and "cohort" in df.columns:
        lines.append("First post-contrast timing by cohort:")
        for c, grp in df.groupby("cohort"):
            vals = pd.to_numeric(grp["first_post_s"], errors="coerce").dropna()
            if len(vals):
                lines.append(f"  {c}: mean={vals.mean():.1f}s, median={vals.median():.1f}s, n={len(vals)}")
    lines.append("")
    if "v2_lcc_dice" in df.columns:
        lines.append(f"Overall v2+LCC Dice: {pd.to_numeric(df['v2_lcc_dice'], errors='coerce').mean():.4f}")
        if "overall" in df.columns:
            lines.append("v2+LCC Dice by phase quality:")
            for q, grp in df.groupby("overall"):
                vals = pd.to_numeric(grp["v2_lcc_dice"], errors="coerce").dropna()
                if len(vals):
                    lines.append(f"  {q}: mean={vals.mean():.4f}, n={len(vals)}")
    lines.append("")
    if not post_df.empty:
        lines.append("Postprocessing comparison, overall mean Dice:")
        for col in ["v2_raw_dice", "v2_lcc_dice", "v2_kinetic_cc_dice"]:
            if col in df.columns:
                lines.append(f"  {col}: {pd.to_numeric(df[col], errors='coerce').mean():.4f}")
    lines.append("")
    if not corr_df.empty and {"metric", "spearman_rho"}.issubset(corr_df.columns):
        lines.append("Strongest absolute Spearman correlations with v2+LCC Dice:")
        tmp = corr_df[corr_df["metric"] == "v2_lcc_dice"].copy()
        if not tmp.empty:
            tmp["abs_rho"] = tmp["spearman_rho"].abs()
            tmp = tmp.sort_values("abs_rho", ascending=False).head(10)
            for _, r in tmp.iterrows():
                lines.append(f"  {r['feature']}: rho={r['spearman_rho']:.3f}, p={r['p_value']:.4g}, n={int(r['n'])}")
    lines.append("")
    lines.append("Suggested manuscript framing:")
    lines.append("  Use these results as contrast-kinetic/failure-attribution evidence, not as proof")
    lines.append("  that timing alone explains the external Dice drop. If no single feature has a")
    lines.append("  strong effect size, write that degradation appears multifactorial.")
    lines.append("")
    lines.append("Important caution:")
    lines.append("  Kinetic-component postprocessing is exploratory and should not be described as")
    lines.append("  a tuned final method unless it is validated on a separate held-out split.")

    (OUT_DIR / "article_takeaways.txt").write_text("\n".join(lines), encoding="utf-8")


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    corr_features = [
        "first_post_s", "last_post_s", "acquisition_time_span_s", "early_dev_s", "late_dev_s",
        "gt_max_enh", "gt_auc_enh_norm", "gt_observed_uptake_rate", "gt_late_slope", "gt_washout_index",
        "ring_max_enh", "ring_auc_enh_norm", "fg_max_enh", "fg_auc_enh_norm",
        "tumor_to_ring_max_enh_diff", "tumor_to_ring_auc_norm_diff", "tumor_to_fg_max_enh_diff",
    ]
    corr_metrics = ["v2_raw_dice", "v2_lcc_dice", "v2_lcc_precision", "v2_lcc_recall", "v2_kinetic_cc_dice"]
    rows = []
    for feat in corr_features:
        if feat not in df.columns:
            continue
        for met in corr_metrics:
            if met not in df.columns:
                continue
            sub = df[[feat, met]].apply(pd.to_numeric, errors="coerce").dropna()
            sub = sub[np.isfinite(sub[feat]) & np.isfinite(sub[met])]
            if len(sub) >= 10:
                rho, p = spearmanr(sub[feat], sub[met])
                rows.append({
                    "feature": feat,
                    "metric": met,
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n": int(len(sub)),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_rho"] = out["spearman_rho"].abs()
        out = out.sort_values(["metric", "abs_rho"], ascending=[True, False])
    return out


def main() -> None:
    ensure_dirs()
    print("Loading saved 07 CSV files...")
    df = read_case_csv()
    curve_df = read_curve_csv()
    print(f"Case table: {CASE_CSV}")
    print(f"Cases loaded: {len(df)}")
    if not curve_df.empty:
        print(f"Curve rows loaded: {len(curve_df)}")
    else:
        print("Curve CSV not found or empty; curve figure will be skipped.")

    metric_cols = [
        "first_post_s", "last_post_s", "early_dev_s", "late_dev_s",
        "gt_max_enh", "gt_auc_enh_norm", "gt_observed_uptake_rate", "gt_late_slope", "gt_washout_index",
        "tumor_to_ring_max_enh_diff", "tumor_to_ring_auc_norm_diff", "tumor_to_fg_max_enh_diff",
        "v2_raw_dice", "v2_raw_precision", "v2_raw_recall",
        "v2_lcc_dice", "v2_lcc_precision", "v2_lcc_recall",
        "v2_kinetic_cc_dice", "v2_kinetic_cc_precision", "v2_kinetic_cc_recall",
        "delta_v2_lcc_minus_raw_dice", "delta_kineticcc_minus_lcc_dice",
    ]

    print("Writing summary tables...")
    for name, groups in {
        "overall": [],
        "by_cohort": ["cohort"],
        "by_phase_quality": ["overall"],
        "by_cohort_and_phase_quality": ["cohort", "overall"],
    }.items():
        missing_groups = [g for g in groups if g not in df.columns]
        if missing_groups:
            print(f"  skipping summary_{name}.csv because missing columns: {missing_groups}")
            continue
        summary = summarize(df, groups, metric_cols)
        path = METRICS_DIR / f"summary_{name}.csv"
        summary.to_csv(path, index=False)
        print(f"  saved {path.name}")

    print("Computing correlations...")
    corr_df = compute_correlations(df)
    corr_path = METRICS_DIR / "correlations_kinetic_timing_vs_external_metrics.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  saved {corr_path.name}")

    print("Writing postprocessing comparison...")
    post_cols = [c for c in [
        "v2_raw_dice", "v2_lcc_dice", "v2_kinetic_cc_dice",
        "v2_raw_precision", "v2_lcc_precision", "v2_kinetic_cc_precision",
        "v2_raw_recall", "v2_lcc_recall", "v2_kinetic_cc_recall",
    ] if c in df.columns]
    post_df = summarize(df, [], post_cols) if post_cols else pd.DataFrame()
    post_path = METRICS_DIR / "postprocessing_comparison_overall.csv"
    post_df.to_csv(post_path, index=False)
    print(f"  saved {post_path.name}")

    print("Creating figures...")
    plot_box_by_cohort(df)
    plot_mean_curves(curve_df)
    plot_bar_metric(df, "overall", "v2_lcc_dice", "fig03_v2_lcc_dice_by_phase_quality.png", "v2+LCC Dice by phase quality", "Dice")
    plot_bar_metric(df, "cohort", "v2_lcc_dice", "fig04_v2_lcc_dice_by_cohort.png", "v2+LCC Dice by cohort", "Dice")
    plot_scatter(df, "first_post_s", "v2_lcc_dice", "fig05_first_post_time_vs_v2_lcc_dice.png", "First post-contrast time vs external Dice", "First post-contrast time (s)", "v2+LCC Dice")
    plot_scatter(df, "tumor_to_ring_max_enh_diff", "v2_lcc_dice", "fig06_tumor_to_ring_enhancement_vs_dice.png", "Tumor-to-ring enhancement contrast vs external Dice", "Tumor-to-ring max enhancement difference", "v2+LCC Dice")
    plot_scatter(df, "gt_observed_uptake_rate", "v2_lcc_dice", "fig07_observed_uptake_rate_vs_dice.png", "Observed uptake rate vs external Dice", "Observed uptake rate", "v2+LCC Dice")

    # Postprocessing comparison figure.
    labels = []
    means = []
    for col, label in [("v2_raw_dice", "v2 raw"), ("v2_lcc_dice", "v2 + LCC"), ("v2_kinetic_cc_dice", "v2 + kinetic component")]:
        if col in df.columns:
            labels.append(label)
            means.append(pd.to_numeric(df[col], errors="coerce").mean())
    if means:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(labels, means)
        ax.set_ylabel("Dice")
        ax.set_title("No-retraining external postprocessing comparison")
        ax.set_ylim(0, max(0.75, float(np.nanmax(means)) + 0.05))
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig08_postprocessing_comparison.png", dpi=300)
        plt.close(fig)

    write_takeaways(df, corr_df, post_df)

    print()
    print("DONE — finished from saved CSV, no MRI recomputation.")
    print(f"Summary tables: {METRICS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Takeaways: {OUT_DIR / 'article_takeaways.txt'}")

    if "v2_lcc_dice" in df.columns:
        print()
        print("Quick numbers:")
        for col in ["v2_raw_dice", "v2_lcc_dice", "v2_kinetic_cc_dice"]:
            if col in df.columns:
                print(f"  {col}: {pd.to_numeric(df[col], errors='coerce').mean():.4f}")


if __name__ == "__main__":
    main()
