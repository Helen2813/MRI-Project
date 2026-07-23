# run_10_final_sanity_checks.py
# Final sanity checks for external failure-attribution analysis.
#
# Purpose:
# 1) Check whether predicted/GT volume-ratio findings are partly a mathematical consequence
#    of the Dice formula by computing the theoretical maximum Dice allowed by volume ratio.
# 2) Check collinearity among external-failure factors so the paper does not overclaim
#    multiple independent causes when metrics are correlated.
#
# Run from this folder with:
#   python run_10_final_sanity_checks.py

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

# =============================================================================
# CONFIG — EDIT ONLY HERE IF YOUR FOLDERS ARE DIFFERENT
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")

# This script reads the saved output from folder 08. It does NOT recompute MRI masks.
INPUT_08_CASE_TABLE = PROJECT_ROOT / "08_external_failure_attribution_analysis" / "metrics" / "failure_attribution_case_table_enhanced.csv"
INPUT_08_CASE_TABLE_FALLBACK = PROJECT_ROOT / "08_external_failure_attribution_analysis" / "metrics" / "failure_attribution_case_table.csv"

OUTPUT_ROOT = PROJECT_ROOT / "10_final_external_sanity_checks"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURES_DIR = OUTPUT_ROOT / "figures"

# Main method used in the paper-safe no-retraining external pipeline.
PRIMARY_METHOD = "v2_lcc"
COMPARISON_METHODS = ["v2_raw", "v2_lcc", "v1_raw", "v1_lcc"]

# For volume-ratio sanity checks.
NEAR_BOUND_UTILIZATION = 0.80
HIGH_BOUND_LOW_DICE_BOUND = 0.75
HIGH_BOUND_LOW_DICE_DICE = 0.50
LOW_BOUND_THRESHOLD = 0.55

# =============================================================================
# UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_case_table() -> pd.DataFrame:
    if INPUT_08_CASE_TABLE.exists():
        path = INPUT_08_CASE_TABLE
    elif INPUT_08_CASE_TABLE_FALLBACK.exists():
        path = INPUT_08_CASE_TABLE_FALLBACK
    else:
        raise FileNotFoundError(
            "Could not find failure attribution case table. Expected one of:\n"
            f"  {INPUT_08_CASE_TABLE}\n"
            f"  {INPUT_08_CASE_TABLE_FALLBACK}\n"
            "Run folder 08 first."
        )
    print(f"Loading case table: {path}")
    df = pd.read_csv(path)
    print(f"Cases loaded: {len(df)}")
    return df


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def method_cols(method: str) -> Dict[str, str]:
    return {
        "dice": f"{method}_dice",
        "precision": f"{method}_precision",
        "recall": f"{method}_recall",
        "pred_vol": f"{method}_pred_vol_cm3",
        "ratio": f"{method}_pred_to_gt_vol_ratio",
        "n_components": f"{method}_n_components",
        "largest_fraction": f"{method}_largest_component_fraction",
        "second_fraction": f"{method}_second_largest_component_fraction",
    }


def add_ratio_if_missing(df: pd.DataFrame, method: str) -> pd.DataFrame:
    cols = method_cols(method)
    if cols["ratio"] not in df.columns and cols["pred_vol"] in df.columns and "gt_vol_cm3" in df.columns:
        pred = safe_float_series(df[cols["pred_vol"]])
        gt = safe_float_series(df["gt_vol_cm3"])
        df[cols["ratio"]] = pred / gt.replace(0, np.nan)
    return df


def theoretical_max_dice_from_ratio(r: np.ndarray) -> np.ndarray:
    """Maximum possible Dice if one mask were perfectly contained in the other.

    Let r = |P| / |G|. If r >= 1, the best possible overlap is |G|,
    so max Dice = 2|G|/(|P|+|G|) = 2/(r+1).
    If r < 1, the best possible overlap is |P|,
    so max Dice = 2|P|/(|P|+|G|) = 2r/(r+1).
    Unified: 2 * min(r, 1) / (r + 1).
    """
    r = np.asarray(r, dtype=float)
    out = np.full_like(r, np.nan, dtype=float)
    ok = np.isfinite(r) & (r > 0)
    out[ok] = 2.0 * np.minimum(r[ok], 1.0) / (r[ok] + 1.0)
    return out


def bh_fdr(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ordered = ranked * n / (np.arange(n) + 1)
    q_ordered = np.minimum.accumulate(q_ordered[::-1])[::-1]
    q_ordered = np.minimum(q_ordered, 1.0)
    q_valid = np.empty_like(q_ordered)
    q_valid[order] = q_ordered
    q[valid] = q_valid
    return q


def bootstrap_ci_mean(x: pd.Series, n_boot: int = 1000, seed: int = 17) -> Tuple[float, float]:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# =============================================================================
# VOLUME-RATIO / DICE-UPPER-BOUND SANITY CHECK
# =============================================================================

def add_volume_bound_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    cols = method_cols(method)
    df = add_ratio_if_missing(df, method)

    dice_col = cols["dice"]
    ratio_col = cols["ratio"]
    bound_col = f"{method}_theoretical_max_dice_by_volume_ratio"
    util_col = f"{method}_dice_utilization_of_volume_bound"
    gap_col = f"{method}_dice_gap_to_volume_bound"
    bound_failure_col = f"{method}_volume_bound_failure_signature"

    if ratio_col not in df.columns or dice_col not in df.columns:
        return df

    ratio = safe_float_series(df[ratio_col]).to_numpy(dtype=float)
    dice = safe_float_series(df[dice_col]).to_numpy(dtype=float)
    bound = theoretical_max_dice_from_ratio(ratio)

    df[bound_col] = bound
    df[util_col] = dice / np.where(bound > 0, bound, np.nan)
    df[gap_col] = bound - dice

    # Diagnostic categories. These are not intended as causal labels.
    labels = []
    for r, d, b, u in zip(ratio, dice, bound, df[util_col].to_numpy(dtype=float)):
        if not np.isfinite(r) or not np.isfinite(d) or not np.isfinite(b):
            labels.append("unknown")
        elif b < LOW_BOUND_THRESHOLD:
            labels.append("volume_ratio_limits_dice")
        elif b >= HIGH_BOUND_LOW_DICE_BOUND and d < HIGH_BOUND_LOW_DICE_DICE:
            labels.append("spatial_or_boundary_mismatch_beyond_volume")
        elif np.isfinite(u) and u >= NEAR_BOUND_UTILIZATION:
            labels.append("near_volume_bound")
        else:
            labels.append("mixed_volume_and_spatial_error")
    df[bound_failure_col] = labels
    return df


def volume_bound_summary(df: pd.DataFrame, method: str) -> pd.DataFrame:
    cols = method_cols(method)
    dice_col = cols["dice"]
    ratio_col = cols["ratio"]
    bound_col = f"{method}_theoretical_max_dice_by_volume_ratio"
    util_col = f"{method}_dice_utilization_of_volume_bound"
    gap_col = f"{method}_dice_gap_to_volume_bound"
    group_col = f"{method}_volume_bound_failure_signature"

    rows = []
    if group_col not in df.columns:
        return pd.DataFrame()

    for name, grp in df.groupby(group_col, dropna=False):
        row = {"group": name, "n": len(grp)}
        for c in [dice_col, ratio_col, bound_col, util_col, gap_col, "gt_vol_cm3"]:
            if c in grp.columns:
                s = safe_float_series(grp[c])
                row[f"{c}_mean"] = float(s.mean())
                row[f"{c}_median"] = float(s.median())
                row[f"{c}_std"] = float(s.std())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


def plot_volume_ratio_bound(df: pd.DataFrame, method: str) -> None:
    cols = method_cols(method)
    dice_col = cols["dice"]
    ratio_col = cols["ratio"]
    bound_col = f"{method}_theoretical_max_dice_by_volume_ratio"
    if dice_col not in df.columns or ratio_col not in df.columns or bound_col not in df.columns:
        return

    x = safe_float_series(df[ratio_col])
    y = safe_float_series(df[dice_col])
    cohort = df["cohort"] if "cohort" in df.columns else pd.Series(["ALL"] * len(df))
    plot_df = pd.DataFrame({"ratio": x, "dice": y, "cohort": cohort}).dropna()
    plot_df = plot_df[plot_df["ratio"] > 0]
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for c, grp in plot_df.groupby("cohort"):
        ax.scatter(grp["ratio"], grp["dice"], s=18, alpha=0.55, label=str(c))
    xr = np.logspace(-2, 2, 500)
    ax.plot(xr, theoretical_max_dice_from_ratio(xr), linewidth=2.5, linestyle="--", label="Theoretical max Dice from volume ratio")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted / GT volume ratio (log scale)")
    ax.set_ylabel("Observed Dice")
    ax.set_title(f"{method}: observed Dice vs volume-ratio upper bound")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig01_{method}_dice_vs_volume_ratio_upper_bound.png", dpi=300)
    plt.close(fig)


def plot_bound_gap_by_group(df: pd.DataFrame, method: str) -> None:
    group_col = f"{method}_volume_bound_failure_signature"
    gap_col = f"{method}_dice_gap_to_volume_bound"
    if group_col not in df.columns or gap_col not in df.columns:
        return
    plot_df = df[[group_col, gap_col]].copy()
    plot_df[gap_col] = safe_float_series(plot_df[gap_col])
    order = plot_df.groupby(group_col)[gap_col].median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [plot_df.loc[plot_df[group_col] == g, gap_col].dropna().values for g in order]
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("Theoretical max Dice - observed Dice")
    ax.set_title(f"{method}: Dice deficit beyond volume-ratio limit")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig02_{method}_dice_gap_to_volume_bound_by_group.png", dpi=300)
    plt.close(fig)

# =============================================================================
# COLLINEARITY / PREDICTOR CORRELATION CHECKS
# =============================================================================

def choose_predictor_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "gt_vol_cm3",
        "log_gt_vol_cm3",
        "first_post_s",
        "last_post_s",
        "early_dev_s",
        "late_dev_s",
        "n_phases",
        "v2_raw_pred_vol_cm3",
        "v2_lcc_pred_vol_cm3",
        "v2_raw_pred_to_gt_vol_ratio",
        "v2_lcc_pred_to_gt_vol_ratio",
        "v2_raw_n_components",
        "v2_raw_largest_component_fraction",
        "v2_raw_second_largest_component_fraction",
        "v1_raw_pred_to_gt_vol_ratio",
        "v1_raw_n_components",
        "v1_raw_largest_component_fraction",
    ]

    # Add selected kinetic columns if 07 was merged into 08.
    keyword_patterns = [
        "tumor_to_ring", "tumor_to_fg", "gt_max_enh", "gt_auc", "gt_washout",
        "gt_late_slope", "ring_max_enh", "ring_auc", "fg_max_enh", "fg_auc",
    ]
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in keyword_patterns):
            candidates.append(col)

    out = []
    for c in candidates:
        if c in df.columns and c not in out:
            s = safe_float_series(df[c])
            if s.notna().sum() >= 30 and s.nunique(dropna=True) >= 3:
                out.append(c)
    return out


def spearman_corr_p(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    a = safe_float_series(x)
    b = safe_float_series(y)
    sub = pd.DataFrame({"a": a, "b": b}).dropna()
    n = len(sub)
    if n < 3 or sub["a"].nunique() < 2 or sub["b"].nunique() < 2:
        return np.nan, np.nan, n
    if spearmanr is None:
        rho = sub["a"].corr(sub["b"], method="spearman")
        return float(rho), np.nan, n
    rho, p = spearmanr(sub["a"], sub["b"])
    return float(rho), float(p), n


def predictor_target_correlations(df: pd.DataFrame, predictors: List[str], targets: List[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        if target not in df.columns:
            continue
        for f in predictors:
            if f == target or f not in df.columns:
                continue
            rho, p, n = spearman_corr_p(df[f], df[target])
            rows.append({"target": target, "feature": f, "spearman_rho": rho, "p_value": p, "n": n, "abs_rho": abs(rho) if np.isfinite(rho) else np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value_bh"] = bh_fdr(out["p_value"].values)
        out = out.sort_values(["target", "abs_rho"], ascending=[True, False])
    return out


def predictor_intercorrelations(df: pd.DataFrame, predictors: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, a in enumerate(predictors):
        for b in predictors[i + 1:]:
            rho, p, n = spearman_corr_p(df[a], df[b])
            rows.append({"feature_a": a, "feature_b": b, "spearman_rho": rho, "p_value": p, "n": n, "abs_rho": abs(rho) if np.isfinite(rho) else np.nan})
    pairs = pd.DataFrame(rows)
    if len(pairs):
        pairs["q_value_bh"] = bh_fdr(pairs["p_value"].values)
        pairs = pairs.sort_values("abs_rho", ascending=False)

    # Matrix without p-values for plotting.
    mat = pd.DataFrame(index=predictors, columns=predictors, dtype=float)
    for a in predictors:
        mat.loc[a, a] = 1.0
    for _, r in pairs.iterrows():
        mat.loc[r["feature_a"], r["feature_b"]] = r["spearman_rho"]
        mat.loc[r["feature_b"], r["feature_a"]] = r["spearman_rho"]
    return pairs, mat


def compute_vif(df: pd.DataFrame, predictors: List[str], max_predictors: int = 14) -> pd.DataFrame:
    # VIF on too many sparse/collinear predictors can be unstable, so use a curated subset.
    priority = [
        "gt_vol_cm3",
        "first_post_s",
        "early_dev_s",
        "late_dev_s",
        "v2_raw_pred_to_gt_vol_ratio",
        "v2_raw_n_components",
        "v2_raw_largest_component_fraction",
        "v2_raw_second_largest_component_fraction",
        "v2_lcc_pred_to_gt_vol_ratio",
    ]
    selected = [c for c in priority if c in predictors]
    for c in predictors:
        if c not in selected and len(selected) < max_predictors:
            selected.append(c)
    if len(selected) < 2:
        return pd.DataFrame()

    X = df[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    if len(X) < 30:
        return pd.DataFrame()

    # Standardize.
    Xs = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    Xs = Xs.dropna(axis=1)
    selected = list(Xs.columns)
    rows = []
    for target in selected:
        y = Xs[target].values
        others = [c for c in selected if c != target]
        A = Xs[others].values
        A = np.column_stack([np.ones(len(A)), A])
        try:
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ beta
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            vif = 1.0 / max(1e-12, 1.0 - r2) if np.isfinite(r2) else np.nan
        except Exception:
            r2, vif = np.nan, np.nan
        rows.append({"feature": target, "r2_explained_by_other_predictors": r2, "vif": vif, "n_complete": len(Xs)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def ols_r2(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> Tuple[float, int, int]:
    cols = [y_col] + [c for c in x_cols if c in df.columns]
    work = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 30 or len(cols) < 2:
        return np.nan, len(work), max(0, len(cols) - 1)
    y = work[y_col].values.astype(float)
    X = work[[c for c in cols if c != y_col]]
    X = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    X = X.dropna(axis=1)
    if X.shape[1] == 0:
        return np.nan, len(work), 0
    A = np.column_stack([np.ones(len(X)), X.values])
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    except Exception:
        r2 = np.nan
    return float(r2), len(work), int(X.shape[1])


def model_family_r2_table(df: pd.DataFrame, target: str = "v2_lcc_dice") -> pd.DataFrame:
    families = {
        "tumor_size_only": ["gt_vol_cm3", "log_gt_vol_cm3"],
        "timing_only": ["first_post_s", "last_post_s", "early_dev_s", "late_dev_s", "n_phases"],
        "volume_ratio_only": ["v2_raw_pred_to_gt_vol_ratio", "v2_lcc_pred_to_gt_vol_ratio"],
        "component_structure_only": ["v2_raw_n_components", "v2_raw_largest_component_fraction", "v2_raw_second_largest_component_fraction"],
        "volume_ratio_plus_components": ["v2_raw_pred_to_gt_vol_ratio", "v2_raw_n_components", "v2_raw_largest_component_fraction", "v2_raw_second_largest_component_fraction"],
    }

    # Add cohort as dummy variables if present.
    work = df.copy()
    if "cohort" in work.columns:
        dummies = pd.get_dummies(work["cohort"].astype(str), prefix="cohort", drop_first=True)
        for c in dummies.columns:
            work[c] = dummies[c].astype(float)
        cohort_cols = list(dummies.columns)
        families["cohort_only"] = cohort_cols
        families["cohort_plus_volume_components"] = cohort_cols + families["volume_ratio_plus_components"]

    rows = []
    for name, cols in families.items():
        r2, n, p = ols_r2(work, target, cols)
        rows.append({"predictor_family": name, "ols_r2_exploratory": r2, "n_complete": n, "n_predictors_used": p})
    return pd.DataFrame(rows).sort_values("ols_r2_exploratory", ascending=False)


def plot_correlation_heatmap(mat: pd.DataFrame, filename: str) -> None:
    if mat.empty:
        return
    # Limit to the most relevant manageable set for readability.
    priority = [
        "gt_vol_cm3",
        "first_post_s",
        "early_dev_s",
        "late_dev_s",
        "v2_raw_pred_to_gt_vol_ratio",
        "v2_raw_n_components",
        "v2_raw_largest_component_fraction",
        "v2_raw_second_largest_component_fraction",
        "v2_lcc_pred_to_gt_vol_ratio",
    ]
    cols = [c for c in priority if c in mat.index]
    if len(cols) < 2:
        cols = list(mat.index[:min(12, len(mat.index))])
    sub = mat.loc[cols, cols].astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title("Spearman correlation among failure-factor predictors")
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = sub.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)

# =============================================================================
# TAKEAWAYS
# =============================================================================

def write_takeaways(df: pd.DataFrame, method: str, vol_summary: pd.DataFrame, predictor_corr: pd.DataFrame, inter_pairs: pd.DataFrame, r2_table: pd.DataFrame) -> None:
    cols = method_cols(method)
    dice_col = cols["dice"]
    ratio_col = cols["ratio"]
    bound_col = f"{method}_theoretical_max_dice_by_volume_ratio"
    util_col = f"{method}_dice_utilization_of_volume_bound"
    gap_col = f"{method}_dice_gap_to_volume_bound"

    def mean_col(c: str) -> float:
        if c in df.columns:
            return float(safe_float_series(df[c]).mean())
        return np.nan

    def med_col(c: str) -> float:
        if c in df.columns:
            return float(safe_float_series(df[c]).median())
        return np.nan

    lines = []
    lines.append("10 FINAL EXTERNAL SANITY CHECKS — QUICK TAKEAWAYS")
    lines.append("=" * 72)
    lines.append(f"Cases analyzed: {len(df)}")
    lines.append("")
    lines.append(f"Primary method: {method}")
    lines.append(f"  Mean Dice: {mean_col(dice_col):.4f}")
    lines.append(f"  Median predicted/GT volume ratio: {med_col(ratio_col):.4f}")
    lines.append(f"  Mean predicted/GT volume ratio: {mean_col(ratio_col):.4f}")
    lines.append(f"  Mean theoretical max Dice from volume ratio: {mean_col(bound_col):.4f}")
    lines.append(f"  Mean Dice utilization of that bound: {mean_col(util_col):.4f}")
    lines.append(f"  Mean Dice gap to bound: {mean_col(gap_col):.4f}")
    lines.append("")

    if not vol_summary.empty:
        lines.append("Volume-bound diagnostic groups:")
        for _, r in vol_summary.iterrows():
            group = r.get("group", "unknown")
            n = int(r.get("n", 0))
            d = r.get(f"{dice_col}_mean", np.nan)
            b = r.get(f"{bound_col}_mean", np.nan)
            u = r.get(f"{util_col}_mean", np.nan)
            lines.append(f"  {group}: n={n}, Dice={d:.4f}, max_by_volume={b:.4f}, utilization={u:.4f}")
        lines.append("")

    if not predictor_corr.empty:
        top = predictor_corr[predictor_corr["target"] == dice_col].head(8)
        if len(top):
            lines.append(f"Top exploratory correlations with {dice_col}:")
            for _, r in top.iterrows():
                lines.append(
                    f"  {r['feature']}: rho={r['spearman_rho']:.3f}, "
                    f"p={r['p_value']:.3g}, q={r['q_value_bh']:.3g}, n={int(r['n'])}"
                )
            lines.append("")

    if not inter_pairs.empty:
        high = inter_pairs[inter_pairs["abs_rho"] >= 0.70].head(12)
        lines.append("High inter-feature correlations among candidate failure factors (|rho| >= 0.70):")
        if len(high):
            for _, r in high.iterrows():
                lines.append(
                    f"  {r['feature_a']} vs {r['feature_b']}: "
                    f"rho={r['spearman_rho']:.3f}, q={r['q_value_bh']:.3g}"
                )
        else:
            lines.append("  None above threshold.")
        lines.append("")

    if not r2_table.empty:
        lines.append("Exploratory OLS R^2 by predictor family, for interpretation only:")
        for _, r in r2_table.iterrows():
            val = r.get("ols_r2_exploratory", np.nan)
            lines.append(f"  {r['predictor_family']}: R2={val:.3f}, n={int(r['n_complete'])}, p={int(r['n_predictors_used'])}")
        lines.append("")

    lines.append("Manuscript-safe interpretation:")
    lines.append("  1) Predicted/GT volume ratio should be described as a diagnostic decomposition of Dice failure, not an independent causal factor, because Dice is mathematically constrained by mask volumes.")
    lines.append("  2) Component-count, largest-component fraction, and volume-ratio findings should be grouped as one component/foreground-instability signature if they are collinear.")
    lines.append("  3) Use effect sizes and BH/FDR-corrected exploratory correlations; avoid claiming that any single factor explains the external drop unless effect sizes are large and robust.")
    lines.append("  4) If actual Dice is far below the theoretical volume-ratio upper bound, describe this as spatial or boundary mismatch beyond simple volume imbalance.")
    lines.append("")

    out_path = OUTPUT_ROOT / "article_takeaways_10.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Takeaways saved: {out_path}")

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("10 Final external sanity checks")
    print("=" * 36)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root : {OUTPUT_ROOT}")
    ensure_dirs()

    df = read_case_table()

    # Add theoretical volume-bound columns for relevant methods.
    methods_done = []
    for method in COMPARISON_METHODS:
        if method_cols(method)["dice"] in df.columns:
            df = add_volume_bound_columns(df, method)
            methods_done.append(method)

    primary = PRIMARY_METHOD if PRIMARY_METHOD in methods_done else (methods_done[0] if methods_done else None)
    if primary is None:
        raise RuntimeError("No recognized method columns were found. Expected v2_lcc_dice or similar.")

    enhanced_path = METRICS_DIR / "final_sanity_case_table_with_volume_bounds.csv"
    df.to_csv(enhanced_path, index=False)
    print(f"Enhanced case table saved: {enhanced_path}")

    # Volume-bound summaries.
    all_vol_summaries = []
    for method in methods_done:
        summ = volume_bound_summary(df, method)
        if len(summ):
            summ.insert(0, "method", method)
            all_vol_summaries.append(summ)
            summ.to_csv(METRICS_DIR / f"volume_bound_summary_{method}.csv", index=False)
            plot_volume_ratio_bound(df, method)
            plot_bound_gap_by_group(df, method)
    vol_summary = pd.concat(all_vol_summaries, ignore_index=True) if all_vol_summaries else pd.DataFrame()
    if len(vol_summary):
        vol_summary.to_csv(METRICS_DIR / "volume_bound_summary_all_methods.csv", index=False)

    # Worst cases: high theoretical bound but low actual Dice.
    primary_cols = method_cols(primary)
    bound_col = f"{primary}_theoretical_max_dice_by_volume_ratio"
    gap_col = f"{primary}_dice_gap_to_volume_bound"
    util_col = f"{primary}_dice_utilization_of_volume_bound"
    keep_cols = [c for c in [
        "case_id", "cohort", "overall", "strict_phase_group", "failure_type", "gt_vol_cm3",
        primary_cols["dice"], primary_cols["precision"], primary_cols["recall"], primary_cols["pred_vol"], primary_cols["ratio"],
        bound_col, util_col, gap_col,
        primary_cols["n_components"], primary_cols["largest_fraction"], primary_cols["second_fraction"],
    ] if c in df.columns]
    if gap_col in df.columns:
        df.sort_values(gap_col, ascending=False)[keep_cols].head(60).to_csv(METRICS_DIR / f"top_60_spatial_or_boundary_gap_{primary}.csv", index=False)
    if bound_col in df.columns and primary_cols["dice"] in df.columns:
        candidate = df[(safe_float_series(df[bound_col]) >= HIGH_BOUND_LOW_DICE_BOUND) & (safe_float_series(df[primary_cols["dice"]]) < HIGH_BOUND_LOW_DICE_DICE)]
        candidate[keep_cols].to_csv(METRICS_DIR / f"cases_high_volume_bound_low_actual_dice_{primary}.csv", index=False)

    # Predictor collinearity and correlations.
    predictors = choose_predictor_columns(df)
    targets = [c for c in [f"{primary}_dice", f"{primary}_precision", f"{primary}_recall", f"{primary}_dice_gap_to_volume_bound", f"{primary}_dice_utilization_of_volume_bound"] if c in df.columns]
    pred_corr = predictor_target_correlations(df, predictors, targets)
    pred_corr.to_csv(METRICS_DIR / "predictor_correlations_with_primary_metrics.csv", index=False)

    inter_pairs, inter_mat = predictor_intercorrelations(df, predictors)
    inter_pairs.to_csv(METRICS_DIR / "predictor_intercorrelation_pairs.csv", index=False)
    inter_mat.to_csv(METRICS_DIR / "predictor_intercorrelation_matrix.csv")
    plot_correlation_heatmap(inter_mat, "fig03_predictor_intercorrelation_heatmap.png")

    vif = compute_vif(df, predictors)
    vif.to_csv(METRICS_DIR / "exploratory_vif_collinearity_table.csv", index=False)

    r2_table = model_family_r2_table(df, target=f"{primary}_dice")
    r2_table.to_csv(METRICS_DIR / "exploratory_predictor_family_ols_r2.csv", index=False)

    # Takeaways.
    primary_vol_summary = vol_summary[vol_summary["method"] == primary].copy() if len(vol_summary) else pd.DataFrame()
    write_takeaways(df, primary, primary_vol_summary, pred_corr, inter_pairs, r2_table)

    # Console output — robust, no missing-column crash.
    print("\nDONE — final sanity checks complete.")
    print(f"Metrics : {METRICS_DIR}")
    print(f"Figures : {FIGURES_DIR}")
    print(f"Primary : {primary}")

    dice_col = f"{primary}_dice"
    ratio_col = method_cols(primary)["ratio"]
    bound_col = f"{primary}_theoretical_max_dice_by_volume_ratio"
    util_col = f"{primary}_dice_utilization_of_volume_bound"
    gap_col = f"{primary}_dice_gap_to_volume_bound"
    print("\nQuick numbers:")
    for label, col in [
        ("mean Dice", dice_col),
        ("median predicted/GT volume ratio", ratio_col),
        ("mean theoretical max Dice by volume ratio", bound_col),
        ("mean utilization of volume-bound Dice", util_col),
        ("mean Dice gap to volume bound", gap_col),
    ]:
        if col in df.columns:
            val = safe_float_series(df[col]).median() if "median" in label else safe_float_series(df[col]).mean()
            print(f"  {label}: {val:.4f}")

    if len(primary_vol_summary):
        print("\nVolume-bound diagnostic groups:")
        show_cols = [c for c in ["group", "n", f"{dice_col}_mean", f"{bound_col}_mean", f"{util_col}_mean", f"{gap_col}_mean"] if c in primary_vol_summary.columns]
        print(primary_vol_summary[show_cols].to_string(index=False))

    if not pred_corr.empty:
        top = pred_corr[pred_corr["target"] == dice_col].head(10)
        if len(top):
            print(f"\nTop correlations with {dice_col}:")
            print(top[["feature", "spearman_rho", "p_value", "q_value_bh", "n"]].to_string(index=False))

    if not inter_pairs.empty:
        high = inter_pairs[inter_pairs["abs_rho"] >= 0.70].head(10)
        print("\nHigh predictor intercorrelations |rho| >= 0.70:")
        if len(high):
            print(high[["feature_a", "feature_b", "spearman_rho", "q_value_bh", "n"]].to_string(index=False))
        else:
            print("  None above threshold.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR")
        print("===========")
        print(repr(e))
        print("If outputs were partially saved, send me the last lines and I will make a finish-from-saved-CSV helper.")
        raise
