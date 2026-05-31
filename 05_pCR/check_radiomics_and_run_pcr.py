# check_radiomics_and_run_pcr.py
# Inspects extracted radiomics features and runs pCR model
# combining clinical + treatment arm + radiomics.
# Run: python check_radiomics_and_run_pcr.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, average_precision_score

# ── CONFIG ────────────────────────────────────────────────────────────────────

RAD_PATH = r"C:\Users\olegk\Desktop\MRI Project\data_main\radiomics_features_ispy2.csv"
TSV_PATH = r"C:\Users\olegk\Desktop\MRI Project\data_main\BreastDCEDL_ISPY2_full_metadata.tsv"
OUT_PATH = r"C:\Users\olegk\Desktop\MRI Project\data_main\pcr_radiomics_results.csv"

SEED     = 42
N_SPLITS = 5

# ── LOAD & INSPECT ────────────────────────────────────────────────────────────

rad = pd.read_csv(RAD_PATH)
tsv = pd.read_csv(TSV_PATH, sep="\t")

print(f"Radiomics: {rad.shape[0]} cases, {rad.shape[1]-1} features")
print(f"TSV:       {len(tsv)} cases")

# feature category breakdown
rad_cols = [c for c in rad.columns if c != "pid"]
for prefix in ["shape", "pre", "early", "late", "sub"]:
    n = sum(1 for c in rad_cols if c.startswith(prefix))
    print(f"  {prefix:10s}: {n} features")

# ── MERGE ─────────────────────────────────────────────────────────────────────

df = tsv.merge(rad, on="pid", how="inner")
df = df.dropna(subset=["pCR"]).copy()
df["pCR"] = df["pCR"].astype(int)
print(f"\nAfter merge: {len(df)} patients with both radiomics and pCR")
print(f"pCR=1: {df['pCR'].sum()}  pCR=0: {(df['pCR']==0).sum()}")

# ── FEATURE ENGINEERING (same as enhanced model) ──────────────────────────────

df["log_tum_vol"]      = np.log1p(df["tum_vol"].clip(lower=0))
df["vol_x_triplneg"]   = df["log_tum_vol"] * df["TripleNeg"]
df["vol_x_her2pos"]    = df["log_tum_vol"] * df["HER2pos"]
df["vol_x_hrher2neg"]  = df["log_tum_vol"] * df["HRposHER2neg"]
df["age_x_menopause"]  = df["age"] * df["menopause"].fillna(0)

# ── FEATURE SETS ──────────────────────────────────────────────────────────────

clinical = [
    "HR","HER2","age","TripleNeg","HER2pos","HRposHER2neg",
    "race_white","race_black","tum_vol","log_tum_vol","menopause",
    "vol_x_triplneg","vol_x_her2pos","vol_x_hrher2neg","age_x_menopause"
]

drug_cols = [
    "Neratinib","Pembrolizumab","Trastuzumab","Ganitumab",
    "Ganetespib","MK-2206","T-DM1","Pertuzumab",
    "Carboplatin","ABT 888","AMG 386"
]

# only radiomics columns (no pid, no clinical)
rad_feature_cols = [c for c in rad_cols if c != "pid"]

# split radiomics by image type
shape_cols = [c for c in rad_feature_cols if c.startswith("shape")]
pre_cols   = [c for c in rad_feature_cols if c.startswith("pre_")]
early_cols = [c for c in rad_feature_cols if c.startswith("early_")]
late_cols  = [c for c in rad_feature_cols if c.startswith("late_")]
sub_cols   = [c for c in rad_feature_cols if c.startswith("sub_")]

def dedup(cols):
    return list(dict.fromkeys(cols))

feature_sets = {
    # baselines
    "A_clinical_arm":               dedup(clinical + ["Arm"]),
    # radiomics only
    "B_shape_only":                 shape_cols,
    "C_early_texture":              dedup(shape_cols + early_cols),
    "D_sub_texture":                dedup(shape_cols + sub_cols),
    "E_all_radiomics":              dedup(shape_cols + pre_cols + early_cols + late_cols + sub_cols),
    # clinical + radiomics combinations
    "F_clinical_shape":             dedup(clinical + shape_cols),
    "G_clinical_arm_shape":         dedup(clinical + ["Arm"] + shape_cols),
    "H_clinical_arm_early":         dedup(clinical + ["Arm"] + shape_cols + early_cols),
    "I_clinical_arm_sub":           dedup(clinical + ["Arm"] + shape_cols + sub_cols),
    "J_clinical_arm_all_rad":       dedup(clinical + ["Arm"] + rad_feature_cols),
    "K_clinical_arm_drugs_all_rad": dedup(clinical + ["Arm"] + drug_cols + rad_feature_cols),
}

# ── MODELS ────────────────────────────────────────────────────────────────────

models = {
    "logreg_l2": LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs",
        class_weight="balanced", max_iter=10000, random_state=SEED
    ),
    "logreg_enet": LogisticRegression(
        C=0.1, penalty="elasticnet", solver="saga", l1_ratio=0.5,
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
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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

print("\nRunning pCR models...")
for fs_name, cols in feature_sets.items():
    # check all columns exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  SKIP {fs_name}: {len(missing)} missing columns")
        continue

    X = df[cols].copy()
    best_auc = 0.0

    for model_name, model in models.items():
        oof = np.zeros(len(df))
        for tr, va in cv.split(X, y):
            pipe = make_pipeline(X, cols, model)
            pipe.fit(X.iloc[tr], y[tr])
            oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]

        auc    = roc_auc_score(y, oof)
        pr_auc = average_precision_score(y, oof)
        lo, _, hi = boot_ci(y, oof)
        best_auc  = max(best_auc, auc)

        rows.append({
            "feature_set": fs_name, "model": model_name,
            "roc_auc": round(auc, 4), "pr_auc": round(pr_auc, 4),
            "ci_low": round(lo, 3),   "ci_high": round(hi, 3),
            "n_features": len(cols),
        })

    print(f"  {fs_name:40s}: best AUC = {best_auc:.4f}")

# ── OUTPUT ────────────────────────────────────────────────────────────────────

results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
results.to_csv(OUT_PATH, index=False)

print("\nTop 10:")
print(results.head(10).to_string(index=False))

print("\nBest AUC per feature set:")
best = results.groupby("feature_set")["roc_auc"].max().sort_values(ascending=False)
baseline = 0.7127
for fs, auc in best.items():
    delta = f"  ({auc-baseline:+.4f} vs metadata baseline)" if fs != "A_clinical_arm" else ""
    print(f"  {fs:45s}: {auc:.4f}{delta}")

print(f"\nSaved: {OUT_PATH}")