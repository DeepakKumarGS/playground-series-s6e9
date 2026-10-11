"""
================================================================================
PS S6E9 v4 — Grandmaster Tri-Model Ensemble (LightGBM + XGBoost + CatBoost)
================================================================================
Community Benchmarks & Attribution Credits:
  • Chris Deotte (@cdeotte): Probit generator reverse-engineering, Simpson's Paradox
    analysis (City_Type x Home_Charging_Possible), and multi-seed stabilization.
  • Naji (@najiama): Pure LightGBM architecture, empirical target-mean mapping
    from the original real-world dataset (itzzomkar), and Triple-TE framework.
  • Markus / maiernator: Multi-scale "Smooth Keys" binning (integer & floor scales).
  • Evgen Dvorkin (@evgendvorkin): Digit decomposition feature concept.
  • Finezza (@Finezza, Topic #740824): Integer-scaled integer division fix
    for IEEE 754 precision underflow in negative digit powers.
  • Vladimir Demidov (@yekenot): High-information 2-way categorical pairs.
  • Berat Erol Çelik (@beraterolelk): Physics-based boundary shifts & zero-tie lexsort.
================================================================================
Runs on Kaggle GPU (T4 / P100). Produces submission.csv (100% Zero-Tie Calibrated).
"""

import gc
import glob
import os
import time
import warnings
from pathlib import Path

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


# ─── 1. Data Ingestion & Original Dataset Linkage ────────────────────────────
def find_file(pattern: str) -> Path:
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"Could not locate file pattern: {pattern}")
    return Path(matches[0])


train_path = find_file("/kaggle/input/**/train.csv")
test_path = find_file("/kaggle/input/**/test.csv")
sub_path = find_file("/kaggle/input/**/sample_submission.csv")
OUT_DIR = Path("/kaggle/working")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

# Empirical real-world priors (Credit: Chris Deotte & Naji)
orig_matches = glob.glob("/kaggle/input/**/EV_Adoption*.csv", recursive=True)
if orig_matches:
    orig = pd.read_csv(orig_matches[0])
    if "Buyer_ID" in orig.columns:
        orig = orig.drop(columns=["Buyer_ID"])
    orig = orig.dropna().reset_index(drop=True)
    log(f"✅ Loaded original dataset: {orig_matches[0]} ({len(orig)} rows)")
else:
    log("ℹ️ Original dataset not mounted directly; proceeding with empty proxy.")
    orig = pd.DataFrame()

TARGET = "Will_Buy_EV"
train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0})
if TARGET in orig.columns:
    orig[TARGET] = orig[TARGET].map({"Yes": 1, "No": 0}) if orig[TARGET].dtype == "object" else orig[TARGET]

train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan
combined = pd.concat([train, test], ignore_index=True)

# Drop uninformative noise column
combined.drop(columns=["Number_of_Cars_Owned"], inplace=True, errors="ignore")

cat_cols = combined.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in combined.columns if c not in cat_cols + ["id", "is_train", TARGET]]

# ─── 2. Feature Engineering ──────────────────────────────────────────────────
# A. Fixed Integer-Scaled Digit Powers (Credit: Evgen Dvorkin & Finezza fix)
log(f"🔢 Extracting integer-scaled digit powers for {len(num_cols)} continuous features...")
digit_features = []
for c in num_cols:
    scaled = np.rint(combined[c].fillna(0) * 10000).astype(np.int64)
    for k in range(-4, 4):
        col_name = f"{c}_digit{k}"
        combined[col_name] = (scaled // (10 ** (k + 4)) % 10).astype(np.int8)
        digit_features.append(col_name)

num_cols.extend(digit_features)

# B. Real-World Priors from Original Dataset (Credit: Naji)
if len(orig) > 0 and TARGET in orig.columns:
    orig_global_mean = float(orig[TARGET].mean())
    log("Mapping real-world target priors from original dataset...")
    for col in cat_cols + num_cols:
        if col in orig.columns:
            real_world_stats = orig.groupby(col, observed=False)[TARGET].mean()
            combined[f"{col}_org_mean"] = (
                combined[col].map(real_world_stats).fillna(orig_global_mean).astype(np.float32)
            )

# C. Markus's Smooth Keys (Multi-Scale Binning)
log("🔑 Adding Markus's Smooth Keys (Income/Commute Multi-Scale Bins)...")
combined["income_exact_int"] = np.floor(combined["Annual_Income_USD"]).astype(str)
combined["income100_floor"] = np.floor(combined["Annual_Income_USD"] / 100.0).astype(str)
combined["income1000_floor"] = np.floor(combined["Annual_Income_USD"] / 1000.0).astype(str)
combined["commute_integer"] = np.floor(combined["Daily_Commute_km"]).astype(str)

# D. High-Information 2-Way Combinations (Credit: Vladimir Demidov & Chris Deotte)
log("🔗 Generating Grandmaster 2-Way Interaction Pairs...")
combined["Pair_Concern_Subsidy"] = (
    combined["Environmental_Concern_Level"].astype(str) + "_" + combined["Subsidy_Available"].astype(str)
)
combined["Pair_Income_Anxiety"] = (
    (combined["Annual_Income_USD"] // 10000).astype(str) + "_" + combined["Range_Anxiety_Level"].astype(str)
)
combined["Pair_City_HomeCharge"] = (
    combined["City_Type"].astype(str) + "_" + combined["Home_Charging_Possible"].astype(str)
)

interaction_pairs = ["Pair_Concern_Subsidy", "Pair_Income_Anxiety", "Pair_City_HomeCharge"]

# E. Convert Numerics to String Categories for Frequency & Target Encoding
num_to_cat_cols = []
for col in num_cols:
    cat_name = f"{col}_cat"
    combined[cat_name] = combined[col].fillna("NaN").astype(str)
    num_to_cat_cols.append(cat_name)

# F. Global Frequency Encoding
all_cats = cat_cols + num_to_cat_cols + interaction_pairs + [
    "income_exact_int", "income100_floor", "income1000_floor", "commute_integer"
]
for col in all_cats:
    freq_mapping = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_fe"] = combined[col].map(freq_mapping).astype(np.float32).fillna(0.0)

# G. Deterministic Boundaries & Spikes
combined["is_30k_spike"] = (combined["Annual_Income_USD"] == 30000.0).astype(np.int8)
combined["is_millionaire_cliff"] = (combined["Annual_Income_USD"] >= 170537.0).astype(np.int8)
combined["is_dead_zone"] = (
    (combined["Annual_Income_USD"] >= 31004.0) & (combined["Annual_Income_USD"] <= 41970.0)
).astype(np.int8)
combined["is_5km_commute"] = (combined["Daily_Commute_km"] == 5.0).astype(np.int8)
combined["is_83km_barrier"] = (combined["Daily_Commute_km"] >= 83.0).astype(np.int8)
combined["is_env_hater"] = (combined["Environmental_Concern_Level"] == 1).astype(np.int8)

# Domain Interactions
combined["Total_Charging"] = combined["Charging_Stations_Near_Home"] + combined["Charging_Stations_Near_Work"]
combined["Commute_per_Station"] = (
    combined["Daily_Commute_km"] / (combined["Total_Charging"] + 1.0)
).astype(np.float32)
combined["Charging_Home_x_Possible"] = (
    combined["Charging_Stations_Near_Home"] * (combined["Home_Charging_Possible"] == "Yes").astype(np.float32)
)

train_df = combined[combined["is_train"] == 1].drop(columns=["is_train"])
test_df = combined[combined["is_train"] == 0].drop(columns=["is_train", TARGET])

# H. Feature Pruning (drop constant and perfectly correlated features)
eval_cols = [c for c in train_df.columns if c not in ["id", TARGET] and pd.api.types.is_numeric_dtype(train_df[c])]
corr_matrix = train_df[eval_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] == 1.0)]
to_drop_const = [c for c in train_df.columns if train_df[c].nunique() == 1] + [
    c for c in test_df.columns if test_df[c].nunique() == 1
]
DROP = set(to_drop_corr).union(set(to_drop_const))
DROP = [c for c in DROP if c not in ["id", TARGET]]

if len(DROP) > 0:
    log(f"Dropping {len(DROP)} redundant/constant features")
    train_df.drop(columns=DROP, inplace=True, errors="ignore")
    test_df.drop(columns=DROP, inplace=True, errors="ignore")

FEATURES = [c for c in test_df.columns if c != "id"]
TARGET_ENCODE_COLS = [c for c in all_cats if c not in DROP and c in train_df.columns]
log(f"✅ Total Base Features: {len(FEATURES)} | Columns to Target Encode: {len(TARGET_ENCODE_COLS)}")

# ─── 3. Grandmaster Tri-Model Cross-Validation ───────────────────────────────
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

X = train_df[FEATURES]
y = train_df[TARGET]
X_test = test_df[FEATURES]

oof_lgb = np.zeros(len(train_df), dtype=np.float32)
test_lgb = np.zeros(len(test_df), dtype=np.float32)

oof_xgb = np.zeros(len(train_df), dtype=np.float32)
test_xgb = np.zeros(len(test_df), dtype=np.float32)

oof_cat = np.zeros(len(train_df), dtype=np.float32)
test_cat = np.zeros(len(test_df), dtype=np.float32)

# Tuned Hyperparameters
LGB_PARAMS_1 = {
    "n_estimators": 10000,
    "learning_rate": 0.02,
    "max_depth": 5,
    "num_leaves": 32,
    "min_child_samples": 10,
    "subsample": 0.8127,
    "colsample_bytree": 0.3029,
    "reg_alpha": 0.0709,
    "reg_lambda": 2.033,
    "max_bin": 1024,
    "random_state": 42,
    "feature_pre_filter": False,
    "metric": "auc",
    "n_jobs": -1,
    "verbose": -1,
}

LGB_PARAMS_2 = dict(LGB_PARAMS_1)
LGB_PARAMS_2["random_state"] = 101  # Dual-seed stabilization

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "max_bin": 1024,
    "n_estimators": 4000,
    "early_stopping_rounds": 100,
    "n_jobs": -1,
    "random_state": 42,
}

CAT_PARAMS = {
    "iterations": 3500,
    "learning_rate": 0.04,
    "depth": 6,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "task_type": "GPU",
    "early_stopping_rounds": 100,
    "random_seed": 42,
    "verbose": False,
}

log(f"🚀 Training Tri-Model Stack (LightGBM Dual-Seed + XGBoost + CatBoost) with {FOLDS} Folds...")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    log(f"=== Fold {fold}/{FOLDS} ===")
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx].copy(), y.iloc[valid_idx]
    X_test_fold = X_test.copy()

    # Triple Target Encoding (Auto, Strict 10, Massive 100) (Credit: Naji & Markus)
    te_auto = TargetEncoder(shuffle=True, cv=FOLDS, smooth="auto", random_state=42)
    te_10 = TargetEncoder(shuffle=True, cv=FOLDS, smooth=10.0, random_state=42)
    te_100 = TargetEncoder(shuffle=True, cv=FOLDS, smooth=100.0, random_state=42)

    X_tr_auto = te_auto.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
    X_va_auto = te_auto.transform(X_valid[TARGET_ENCODE_COLS])
    X_te_auto = te_auto.transform(X_test_fold[TARGET_ENCODE_COLS])

    X_tr_10 = te_10.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
    X_va_10 = te_10.transform(X_valid[TARGET_ENCODE_COLS])
    X_te_10 = te_10.transform(X_test_fold[TARGET_ENCODE_COLS])

    X_tr_100 = te_100.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
    X_va_100 = te_100.transform(X_valid[TARGET_ENCODE_COLS])
    X_te_100 = te_100.transform(X_test_fold[TARGET_ENCODE_COLS])

    for i, col in enumerate(TARGET_ENCODE_COLS):
        X_train[f"{col}_TE_auto"] = X_tr_auto[:, i].astype(np.float32)
        X_valid[f"{col}_TE_auto"] = X_va_auto[:, i].astype(np.float32)
        X_test_fold[f"{col}_TE_auto"] = X_te_auto[:, i].astype(np.float32)

        X_train[f"{col}_TE_10"] = X_tr_10[:, i].astype(np.float32)
        X_valid[f"{col}_TE_10"] = X_va_10[:, i].astype(np.float32)
        X_test_fold[f"{col}_TE_10"] = X_te_10[:, i].astype(np.float32)

        X_train[f"{col}_TE_100"] = X_tr_100[:, i].astype(np.float32)
        X_valid[f"{col}_TE_100"] = X_va_100[:, i].astype(np.float32)
        X_test_fold[f"{col}_TE_100"] = X_te_100[:, i].astype(np.float32)

        X_train.drop(columns=[col], inplace=True)
        X_valid.drop(columns=[col], inplace=True)
        X_test_fold.drop(columns=[col], inplace=True)

    # 1. LightGBM (Dual-Seed Average: Seed 42 + Seed 101)
    clf_lgb1 = lgb.LGBMClassifier(**LGB_PARAMS_1)
    clf_lgb1.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
    )
    clf_lgb2 = lgb.LGBMClassifier(**LGB_PARAMS_2)
    clf_lgb2.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
    )
    val_pred_lgb = 0.5 * (clf_lgb1.predict_proba(X_valid)[:, 1] + clf_lgb2.predict_proba(X_valid)[:, 1])
    te_pred_lgb = 0.5 * (clf_lgb1.predict_proba(X_test_fold)[:, 1] + clf_lgb2.predict_proba(X_test_fold)[:, 1])

    oof_lgb[valid_idx] = val_pred_lgb
    test_lgb += te_pred_lgb / FOLDS
    log(f"Fold {fold} Dual-Seed LightGBM AUC: {roc_auc_score(y_valid, val_pred_lgb):.5f}")

    # 2. XGBoost (CUDA Hist)
    clf_xgb = XGBClassifier(**XGB_PARAMS)
    clf_xgb.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    val_pred_xgb = clf_xgb.predict_proba(X_valid)[:, 1]
    te_pred_xgb = clf_xgb.predict_proba(X_test_fold)[:, 1]

    oof_xgb[valid_idx] = val_pred_xgb
    test_xgb += te_pred_xgb / FOLDS
    log(f"Fold {fold} XGBoost AUC:           {roc_auc_score(y_valid, val_pred_xgb):.5f}")

    # 3. CatBoost (Oblivious Trees on GPU)
    clf_cat = CatBoostClassifier(**CAT_PARAMS)
    clf_cat.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        verbose=False,
    )
    val_pred_cat = clf_cat.predict_proba(X_valid)[:, 1]
    te_pred_cat = clf_cat.predict_proba(X_test_fold)[:, 1]

    oof_cat[valid_idx] = val_pred_cat
    test_cat += te_pred_cat / FOLDS
    log(f"Fold {fold} CatBoost AUC:          {roc_auc_score(y_valid, val_pred_cat):.5f}")

    del clf_lgb1, clf_lgb2, clf_xgb, clf_cat, X_train, X_valid, X_test_fold
    gc.collect()

# ─── 4. Meta-Optimization & Optimal Rank Blending ────────────────────────────
auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
auc_cat = roc_auc_score(y, oof_cat)

log("==========================================")
log(f"🏆 LightGBM Dual-Seed OOF AUC: {auc_lgb:.5f}")
log(f"🏆 XGBoost           OOF AUC: {auc_xgb:.5f}")
log(f"🏆 CatBoost GPU      OOF AUC: {auc_cat:.5f}")
log("==========================================")

# Convert to Percentile Ranks
r_oof_lgb = rankdata(oof_lgb) / len(oof_lgb)
r_oof_xgb = rankdata(oof_xgb) / len(oof_xgb)
r_oof_cat = rankdata(oof_cat) / len(oof_cat)

r_te_lgb = rankdata(test_lgb) / len(test_lgb)
r_te_xgb = rankdata(test_xgb) / len(test_xgb)
r_te_cat = rankdata(test_cat) / len(test_cat)

# Optimize 3-Way Weights via Scipy SLSQP (Constrained: w1 + w2 + w3 = 1, wi >= 0)
def loss(weights):
    w1, w2, w3 = weights
    blend = w1 * r_oof_lgb + w2 * r_oof_xgb + w3 * r_oof_cat
    return -roc_auc_score(y, blend)

res = minimize(
    loss,
    x0=[0.45, 0.30, 0.25],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
    method="SLSQP",
)
w_lgb, w_xgb, w_cat = res.x
best_ensemble_auc = -res.fun

log(f"🎯 Optimal Ensemble Weights: {w_lgb:.3f} LGBM + {w_xgb:.3f} XGB + {w_cat:.3f} CatBoost")
log(f"🎯 Tri-Model Ensemble OOF AUC: {best_ensemble_auc:.5f} (Gain over best single: +{best_ensemble_auc - max(auc_lgb, auc_xgb, auc_cat):.5f})")

# Blend Test Predictions
test_blend = w_lgb * r_te_lgb + w_xgb * r_te_xgb + w_cat * r_te_cat

# ─── 5. Physics Boundary Shifts & Zero-Tie Lexsort (Credit: Berat Erol Çelik) ───
log("⚙️ Applying Physics Boundary Calibration & Zero-Tie Lexsort...")
test_incomes = pd.to_numeric(test_df["Annual_Income_USD"], errors="coerce").to_numpy()
test_commute = pd.to_numeric(test_df["Daily_Commute_km"], errors="coerce").to_numpy()
subsidy_no = (test_df["Subsidy_Available"].astype(str) == "No").to_numpy()
env_1 = (pd.to_numeric(test_df["Environmental_Concern_Level"], errors="coerce") == 1.0).to_numpy()
anx_med_high = test_df["Range_Anxiety_Level"].isin(["Medium", "High"]).to_numpy()

order = np.lexsort((test_blend, test_blend))
ranks = np.empty(len(order), dtype=np.int64)
ranks[order] = np.arange(1, len(order) + 1)
calibrated_preds = (ranks - 0.5) / len(ranks)

# Hard Deterministic Shifts (Empirically pure regions)
calibrated_preds[test_incomes >= 170537.0] += 10.0
calibrated_preds[(test_incomes >= 31004.0) & (test_incomes <= 41970.0)] -= 10.0
calibrated_preds[test_commute >= 83.0] -= 5.0
calibrated_preds[(test_incomes == 30000.0) & subsidy_no & (env_1 | anx_med_high)] -= 5.0

# Zero-Tie Lexsort Final Ranking
order_final = np.lexsort((test_blend, calibrated_preds))
ranks_final = np.empty(len(order_final), dtype=np.int64)
ranks_final[order_final] = np.arange(1, len(order_final) + 1)
final_ranks = (ranks_final - 0.5) / len(ranks_final)

# ─── 6. Output & Artifact Preservation ───────────────────────────────────────
np.save(OUT_DIR / "oof_lgb_v4.npy", oof_lgb)
np.save(OUT_DIR / "test_lgb_v4.npy", test_lgb)
np.save(OUT_DIR / "oof_xgb_v4.npy", oof_xgb)
np.save(OUT_DIR / "test_xgb_v4.npy", test_xgb)
np.save(OUT_DIR / "oof_cat_v4.npy", oof_cat)
np.save(OUT_DIR / "test_cat_v4.npy", test_cat)

sub = pd.DataFrame({"id": test["id"], TARGET: final_ranks})
sub.to_csv(OUT_DIR / "submission.csv", index=False)
log(f"💾 Saved submission.csv ({len(sub)} rows, 100% Zero Ties)")

assert len(sub) == len(sample_sub), f"Expected {len(sample_sub)} rows, got {len(sub)}"
assert list(sub.columns) == ["id", TARGET], f"Unexpected columns: {sub.columns}"
assert sub["id"].equals(sample_sub["id"]), "IDs do not match sample submission"
assert sub[TARGET].nunique() == len(sub), "Ties detected in submission"
log("✅ All Grandmaster ensemble sanity checks passed ✓")

