# quick_pcr_arm.py
# Быстрая pCR модель с treatment arm данными
# Запуск: python quick_pcr_arm.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TSV_PATH = r"C:\Users\olegk\Desktop\MRI Project\data_main\BreastDCEDL_ISPY2_full_metadata.tsv"
OUT_PATH  = r"C:\Users\olegk\Desktop\MRI Project\data_main\pcr_arm_results.csv"

SEED     = 42
N_SPLITS = 5

# ── LOAD ──────────────────────────────────────────────────────────────────────

print("Загружаем данные...")
df = pd.read_csv(TSV_PATH, sep="\t")
df = df.dropna(subset=["pCR"]).copy()
df["pCR"] = df["pCR"].astype(int)
print(f"Пациентов: {len(df)}  |  pCR=1: {df['pCR'].sum()}  |  pCR=0: {(df['pCR']==0).sum()}")

# ── FEATURE SETS ──────────────────────────────────────────────────────────────

drug_cols = [
    "Neratinib", "Pembrolizumab", "Trastuzumab", "Ganitumab",
    "Ganetespib", "MK-2206", "T-DM1", "Pertuzumab",
    "Carboplatin", "ABT 888", "AMG 386"
]

clinical = ["HR", "HER2", "age", "TripleNeg", "HER2pos", "HRposHER2neg",
            "race_white", "race_black", "tum_vol"]

feature_sets = {
    "clinical_only":            clinical,
    "arm_only":                 ["Arm"],
    "drugs_only":               drug_cols,
    "clinical_plus_arm":        clinical + ["Arm"],
    "clinical_plus_drugs":      clinical + drug_cols,
    "clinical_arm_drugs":       clinical + ["Arm"] + drug_cols,
    "arm_plus_drugs":           ["Arm"] + drug_cols,
}

# ── MODELS ────────────────────────────────────────────────────────────────────

models = {
    "logreg": LogisticRegression(
        C=0.3, penalty="l2", solver="lbfgs",
        class_weight="balanced", max_iter=10000, random_state=SEED
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=8,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED
    ),
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def make_pipeline(X, cols, model):
    cat = [c for c in cols if X[c].dtype == object]
    num = [c for c in cols if X[c].dtype != object]

    transformers = []
    if num:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num
        ))
    if cat:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat
        ))

    prep = ColumnTransformer(transformers)
    return Pipeline([("prep", prep), ("model", model)])


def boot_ci(y, p, n=500, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, mid, hi = np.percentile(vals, [2.5, 50, 97.5])
    return lo, mid, hi


# ── MAIN ──────────────────────────────────────────────────────────────────────

y   = df["pCR"].values
cv  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
sep = "=" * 65
rows = []

print()
print(sep)
print("  РЕЗУЛЬТАТЫ pCR PREDICTION С TREATMENT ARM")
print(sep)

for fs_name, cols in feature_sets.items():
    # проверить что все колонки есть
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"\n  ПРОПУСК {fs_name}: нет колонок {missing}")
        continue

    X = df[cols].copy()
    print(f"\n  Feature set: {fs_name}  ({len(cols)} признаков)")

    for model_name, model in models.items():
        oof = np.zeros(len(df))

        for tr, va in cv.split(X, y):
            pipe = make_pipeline(X, cols, model)
            pipe.fit(X.iloc[tr], y[tr])
            oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]

        auc    = roc_auc_score(y, oof)
        pr_auc = average_precision_score(y, oof)
        lo, mid, hi = boot_ci(y, oof)

        print(f"    {model_name:20s}: AUC = {auc:.4f}  "
              f"PR-AUC = {pr_auc:.4f}  "
              f"CI [{lo:.3f} – {hi:.3f}]")

        rows.append({
            "feature_set":   fs_name,
            "model":         model_name,
            "roc_auc":       round(auc, 4),
            "pr_auc":        round(pr_auc, 4),
            "ci_low":        round(lo, 3),
            "ci_mid":        round(mid, 3),
            "ci_high":       round(hi, 3),
            "n_features":    len(cols),
        })

# ── SAVE + SUMMARY ────────────────────────────────────────────────────────────

results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
results.to_csv(OUT_PATH, index=False)

print()
print(sep)
print("  ТОП-5 РЕЗУЛЬТАТОВ")
print(sep)
print(results.head(5).to_string(index=False))
print()
print(f"  Все результаты сохранены: {OUT_PATH}")
print(sep)

# Сравнение ключевых feature sets (лучшая модель для каждого)
print()
print("  СВОДКА: лучший AUC по каждому feature set")
print(sep)
best = results.groupby("feature_set")["roc_auc"].max().sort_values(ascending=False)
for fs, auc in best.items():
    print(f"  {fs:35s}: {auc:.4f}")
print(sep)