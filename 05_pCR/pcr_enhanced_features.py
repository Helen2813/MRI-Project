# pcr_enhanced_features.py
# Enhanced pCR prediction using TSV-derived shape and interaction features.
# No MRI files required - all features derived from metadata.
# Run: python pcr_enhanced_features.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TSV_PATH = r"C:\Users\olegk\Desktop\MRI Project\data_main\BreastDCEDL_ISPY2_full_metadata.tsv"
OUT_PATH  = r"C:\Users\olegk\Desktop\MRI Project\data_main\pcr_enhanced_results.csv"

SEED     = 42
N_SPLITS = 5

# ── LOAD ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(TSV_PATH, sep="\t")
df = df.dropna(subset=["pCR"]).copy()
df["pCR"] = df["pCR"].astype(int)

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

# log-transform tumor volume (wide range benefits from log scale)
df["log_tum_vol"] = np.log1p(df["tum_vol"].clip(lower=0))

# bounding box shape features from TSV columns (no MRI loading needed)
df["tumor_z_extent"]   = (df["mask_end"] - df["mask_start"]).clip(lower=0)
df["tumor_row_extent"] = (df["eraw"] - df["sraw"]).clip(lower=0)
df["tumor_col_extent"] = (df["ecol"] - df["scol"]).clip(lower=0)

df["bbox_vol_mm3"] = (
    df["tumor_z_extent"] * df["slice_thick"] *
    df["tumor_row_extent"] * df["xy_spacing"] *
    df["tumor_col_extent"] * df["xy_spacing"]
).clip(lower=1)

# compactness: how much tumor fills its bounding box
df["compactness"]          = (df["tum_vol"] / df["bbox_vol_mm3"]).clip(0, 1)
df["aspect_ratio_col_row"] = (df["tumor_col_extent"] / (df["tumor_row_extent"] + 1e-6)).clip(0, 10)
df["aspect_ratio_z_row"]   = (df["tumor_z_extent"]   / (df["tumor_row_extent"] + 1e-6)).clip(0, 10)
df["log_z_extent"]         = np.log1p(df["tumor_z_extent"])
df["log_row_extent"]       = np.log1p(df["tumor_row_extent"])
df["log_col_extent"]       = np.log1p(df["tumor_col_extent"])

# interaction features: volume effect differs by molecular subtype
df["vol_x_triplneg"]  = df["log_tum_vol"] * df["TripleNeg"]
df["vol_x_her2pos"]   = df["log_tum_vol"] * df["HER2pos"]
df["vol_x_hrher2neg"] = df["log_tum_vol"] * df["HRposHER2neg"]
df["age_x_menopause"] = df["age"] * df["menopause"].fillna(0)

# simplify verbose menopausal_status string into categories
def simplify_menopause(s):
    if pd.isna(s):
        return "unknown"
    s = str(s).lower()
    if "postmenopausal" in s or "(post" in s:
        return "post"
    if "<6" in s or "< 6" in s:
        return "pre_recent"
    if "6-12" in s or "perimenopausal" in s:
        return "peri"
    return "other"

df["menopause_cat"] = df["menopausal_status"].apply(simplify_menopause)
df["race_cat"]      = df["Race"].fillna("Unknown")

# ── FEATURE SETS ──────────────────────────────────────────────────────────────

drug_cols = [
    "Neratinib","Pembrolizumab","Trastuzumab","Ganitumab",
    "Ganetespib","MK-2206","T-DM1","Pertuzumab",
    "Carboplatin","ABT 888","AMG 386"
]

clinical_base = [
    "HR","HER2","age","TripleNeg","HER2pos","HRposHER2neg",
    "race_white","race_black","tum_vol","menopause"
]

clinical_enhanced = clinical_base + [
    "log_tum_vol","menopause_cat","race_cat","age_x_menopause"
]

shape_features = [
    "tumor_z_extent","tumor_row_extent","tumor_col_extent",
    "bbox_vol_mm3","compactness",
    "aspect_ratio_col_row","aspect_ratio_z_row",
    "log_z_extent","log_row_extent","log_col_extent","log_tum_vol"
]

interaction_features = [
    "vol_x_triplneg","vol_x_her2pos","vol_x_hrher2neg","age_x_menopause"
]

def dedup(cols):
    return list(dict.fromkeys(cols))

feature_sets = {
    "A_clinical_only":      dedup(clinical_base),
    "B_clinical_plus_arm":  dedup(clinical_base + ["Arm"]),
    "C_clinical_enhanced":  dedup(clinical_enhanced),
    "D_clinical_enh_arm":   dedup(clinical_enhanced + ["Arm"]),
    "E_clinical_shape":     dedup(clinical_base + shape_features),
    "F_clinical_shape_arm": dedup(clinical_base + shape_features + ["Arm"]),
    "G_enh_shape_arm":      dedup(clinical_enhanced + shape_features + ["Arm"]),
    "H_shape_arm_interact": dedup(clinical_base + shape_features + ["Arm"] + interaction_features),
    "I_full_all":           dedup(clinical_enhanced + shape_features + ["Arm"] + drug_cols + interaction_features),
}

# ── MODELS ────────────────────────────────────────────────────────────────────

models = {
    "logreg_l2": LogisticRegression(
        C=0.3, penalty="l2", solver="lbfgs",
        class_weight="balanced", max_iter=10000, random_state=SEED
    ),
    "logreg_enet": LogisticRegression(
        C=0.3, penalty="elasticnet", solver="saga", l1_ratio=0.3,
        class_weight="balanced", max_iter=10000, random_state=SEED
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=600, max_depth=6, min_samples_leaf=6,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=600, max_depth=6, min_samples_leaf=6,
        class_weight="balanced", n_jobs=-1, random_state=SEED
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, random_state=SEED
    ),
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def make_pipeline(X, cols, model):
    cat = [c for c in cols if X[c].dtype == "object"]
    num = [c for c in cols if X[c].dtype != "object"]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("var",    VarianceThreshold(threshold=1e-8)),
            ("scaler", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imp",    SimpleImputer(strategy="most_frequent")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("model", model)])


def boot_ci(y, p, n=500):
    rng  = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.percentile(vals, [2.5, 50, 97.5]))


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

y    = df["pCR"].values
cv   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rows = []

for fs_name, cols in feature_sets.items():
    missing = [c for c in cols if c not in df.columns]
    if missing:
        continue
    X = df[cols].copy()
    for model_name, model in models.items():
        oof = np.zeros(len(df))
        for tr, va in cv.split(X, y):
            pipe = make_pipeline(X, cols, model)
            pipe.fit(X.iloc[tr], y[tr])
            oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        auc    = roc_auc_score(y, oof)
        pr_auc = average_precision_score(y, oof)
        lo, mid, hi = boot_ci(y, oof)
        rows.append({
            "feature_set": fs_name, "model": model_name,
            "roc_auc": round(auc, 4), "pr_auc": round(pr_auc, 4),
            "ci_low": round(lo, 3),   "ci_high": round(hi, 3),
            "n_features": len(cols),
        })

# ── OUTPUT ────────────────────────────────────────────────────────────────────

results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
results.to_csv(OUT_PATH, index=False)

print("Top 10 results:")
print(results.head(10).to_string(index=False))

print("\nBest AUC per feature set:")
best = results.groupby("feature_set")["roc_auc"].max().sort_values(ascending=False)
for fs, auc in best.items():
    delta = f"  (+{auc-0.7105:+.4f} vs arm baseline)" if fs not in ["A_clinical_only","B_clinical_plus_arm"] else ""
    print(f"  {fs:40s}: {auc:.4f}{delta}")

# feature importance for best model
best_row  = results.iloc[0]
best_cols = feature_sets[best_row["feature_set"]]
X_best    = df[best_cols].copy()

rf_imp = RandomForestClassifier(
    n_estimators=600, max_depth=6, min_samples_leaf=6,
    class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
)
pipe_imp = make_pipeline(X_best, best_cols, rf_imp)
pipe_imp.fit(X_best, y)

try:
    prep      = pipe_imp.named_steps["prep"]
    rf_fitted = pipe_imp.named_steps["model"]
    cat = [c for c in best_cols if X_best[c].dtype == "object"]
    num = [c for c in best_cols if X_best[c].dtype != "object"]
    feat_names = []
    if num:
        var_sel     = prep.named_transformers_["num"].named_steps["var"]
        feat_names += list(np.array(num)[var_sel.get_support()])
    if cat:
        ohe         = prep.named_transformers_["cat"].named_steps["ohe"]
        feat_names += list(ohe.get_feature_names_out(cat))
    imp = pd.Series(rf_fitted.feature_importances_, index=feat_names[:len(rf_fitted.feature_importances_)])
    print(f"\nTop 20 feature importances ({best_row['feature_set']}):")
    print(imp.nlargest(20).round(4).to_string())
except Exception:
    pass

print(f"\nSaved: {OUT_PATH}")