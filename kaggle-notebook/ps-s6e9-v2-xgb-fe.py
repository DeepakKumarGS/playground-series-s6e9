"""
PS S6E9 v2 — XGBoost with Feature Engineering
Runs on Kaggle GPU (T4/P100). Produces submission.csv.

Features: 56 digit features, domain interactions, hard boundaries,
smooth keys, triple target encoding, original dataset augmentation.
"""

import gc
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

T0 = time.perf_counter()

def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)

# ─── Paths ───────────────────────────────────────────────────────────────────
COMP_DIR = Path("/kaggle/input/competitions/playground-series-s6e9")
ORIG_DIR = Path("/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety")
OUT_DIR = Path("/kaggle/working")

N_FOLDS = 5
SEED = 42

CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

NUM_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

# Tuned XGBoost params for Kaggle T4/P100 GPU
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
    "n_estimators": 3000,
    "early_stopping_rounds": 100,
    "n_jobs": -1,
    "random_state": SEED,
}


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_frames():
    train = pd.read_csv(COMP_DIR / "train.csv")
    test = pd.read_csv(COMP_DIR / "test.csv")
    log(f"Train: {train.shape}, Test: {test.shape}")

    orig = None
    orig_csv = ORIG_DIR / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
    if orig_csv.exists():
        orig = pd.read_csv(orig_csv)
        if "Buyer_ID" in orig.columns:
            orig = orig.drop(columns=["Buyer_ID"])
        orig = orig.dropna().reset_index(drop=True)
        log(f"Original dataset: {len(orig)} clean rows")

    return train, test, orig


def target_to_int(s: pd.Series) -> np.ndarray:
    return (s == "Yes").astype(int).to_numpy()


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate digit features, domain interactions, boundaries, and smooth keys."""
    out = df.copy()

    # Domain interaction features
    out["Total_Charging"] = out["Charging_Stations_Near_Home"] + out["Charging_Stations_Near_Work"]
    out["Commute_per_Station"] = (out["Daily_Commute_km"] / (out["Total_Charging"] + 1.0)).astype(np.float32)
    out["Income_per_Commute"] = (out["Annual_Income_USD"] / (out["Daily_Commute_km"] + 1.0)).astype(np.float32)
    out["Charging_Ratio"] = (out["Charging_Stations_Near_Home"] / (out["Charging_Stations_Near_Work"] + 1.0)).astype(np.float32)
    out["Car_per_Income"] = (out["Number_of_Cars_Owned"] / (out["Annual_Income_USD"] / 10000.0 + 1.0)).astype(np.float32)

    is_subsidy = (out["Subsidy_Available"] == "Yes").astype(np.float32)
    out["Concern_x_Subsidy"] = (out["Environmental_Concern_Level"] * is_subsidy).astype(np.float32)
    out["Concern_x_Income"] = (out["Environmental_Concern_Level"] * out["Annual_Income_USD"]).astype(np.float32)

    # Hard synthetic generation boundaries
    out["Millionaire_Cliff"] = (out["Annual_Income_USD"] >= 170537).astype(np.int8)
    out["Dead_Zone"] = out["Annual_Income_USD"].between(31004, 41970).astype(np.int8)

    # Smooth keys (multi-scale binning)
    out["Income_100"] = (out["Annual_Income_USD"] // 100).astype(np.int32)
    out["Income_1000"] = (out["Annual_Income_USD"] // 1000).astype(np.int32)
    out["Commute_5"] = (out["Daily_Commute_km"] // 5).astype(np.int32)

    # 56 Digit Features (10^-4 to 10^3 for each numeric column)
    for col in NUM_COLS:
        col_vals = out[col].fillna(0)
        for k in range(-4, 4):
            out[f"{col}_digit{k}"] = (col_vals // (10**k) % 10).astype(np.int8)

    return out


def label_encode_categoricals(
    train_df: pd.DataFrame, test_df: pd.DataFrame, orig_df=None
):
    tr = train_df.copy()
    te = test_df.copy()
    og = orig_df.copy() if orig_df is not None else None

    for col in CAT_COLS:
        mapping = {val: idx for idx, val in enumerate(tr[col].dropna().unique())}
        tr[col] = tr[col].map(mapping).fillna(-1).astype(np.int32)
        te[col] = te[col].map(mapping).fillna(-1).astype(np.int32)
        if og is not None:
            og[col] = og[col].map(mapping).fillna(-1).astype(np.int32)

    return tr, te, og


# ─── Training Loop ───────────────────────────────────────────────────────────

def cv_and_predict(train, test, orig=None):
    log("Starting CV pipeline...")

    y_train = target_to_int(train["Will_Buy_EV"])
    train_clean = train.drop(columns=["id", "Will_Buy_EV"])
    test_clean = test.drop(columns=["id"])

    orig_clean, y_orig = None, None
    if orig is not None:
        y_orig = target_to_int(orig["Will_Buy_EV"])
        orig_clean = orig.drop(columns=["Will_Buy_EV"])

    # Label encode
    train_enc, test_enc, orig_enc = label_encode_categoricals(train_clean, test_clean, orig_clean)

    # Feature engineering
    log("Generating engineered features...")
    X_all_train = engineer_base_features(train_enc)
    X_test_base = engineer_base_features(test_enc)
    X_orig_fe = engineer_base_features(orig_enc) if orig_enc is not None else None
    log(f"Total base features: {X_all_train.shape[1]}")

    # 5-fold CV
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test), dtype=np.float32)

    # Columns for Triple Target Encoding
    te_cols = CAT_COLS + ["Income_100", "Income_1000", "Commute_5"]

    for fold, (tr_idx, va_idx) in enumerate(folds.split(X_all_train, y_train), start=1):
        log(f"=== Fold {fold}/{N_FOLDS} ===")

        X_tr = X_all_train.iloc[tr_idx].copy()
        y_tr = y_train[tr_idx].copy()
        X_va = X_all_train.iloc[va_idx].copy()
        y_va = y_train[va_idx].copy()
        X_te_fold = X_test_base.copy()

        # Incorporate original dataset into training fold only
        if X_orig_fe is not None and y_orig is not None:
            X_tr = pd.concat([X_tr, X_orig_fe], axis=0, ignore_index=True)
            y_tr = np.concatenate([y_tr, y_orig])

        # Triple Target Encoding (smooth = auto, 10.0, 100.0)
        for smooth_val, smooth_name in [("auto", "auto"), (10.0, "10"), (100.0, "100")]:
            te = TargetEncoder(smooth=smooth_val, cv=5)
            X_tr_enc = te.fit_transform(X_tr[te_cols], y_tr).astype(np.float32)
            X_va_enc = te.transform(X_va[te_cols]).astype(np.float32)
            X_te_enc = te.transform(X_te_fold[te_cols]).astype(np.float32)

            for i, col in enumerate(te_cols):
                X_tr[f"TE_{col}_{smooth_name}"] = X_tr_enc[:, i]
                X_va[f"TE_{col}_{smooth_name}"] = X_va_enc[:, i]
                X_te_fold[f"TE_{col}_{smooth_name}"] = X_te_enc[:, i]

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=100,
        )

        va_pred = model.predict_proba(X_va)[:, 1]
        te_pred = model.predict_proba(X_te_fold)[:, 1]

        oof[va_idx] = va_pred
        pred += te_pred / N_FOLDS

        best = int(model.best_iteration)
        fold_auc = float(roc_auc_score(y_va, va_pred))
        log(f"Fold {fold} AUC {fold_auc:.5f}  best_iter {best}")

        del model, X_tr, X_va, X_te_fold, va_pred, te_pred
        gc.collect()

    cv_auc = float(roc_auc_score(y_train, oof))
    log(f"========== OOF AUC {cv_auc:.5f} ==========")
    return oof, pred, cv_auc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    train, test, orig = load_frames()

    oof, pred, cv_auc = cv_and_predict(train, test, orig)

    # Write submission
    sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": pred})
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    log(f"Wrote submission.csv: {len(sub)} rows, OOF AUC={cv_auc:.5f}")

    # Save OOF for downstream blending
    np.save(OUT_DIR / "oof_preds_v2.npy", oof)
    np.save(OUT_DIR / "test_preds_v2.npy", pred)
    log("Saved OOF and test prediction arrays")

    # Sanity checks
    sample = pd.read_csv(COMP_DIR / "sample_submission.csv")
    assert len(sub) == len(sample), f"Expected {len(sample)} rows, got {len(sub)}"
    assert list(sub.columns) == ["id", "Will_Buy_EV"]
    assert sub["id"].equals(sample["id"])
    assert sub["Will_Buy_EV"].between(0, 1).all()
    log("All sanity checks passed ✓")


if __name__ == "__main__":
    main()
