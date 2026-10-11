"""v2 XGBoost: digit decomposition, domain interactions, boundary flags, triple target encoding, and original dataset integration."""

from __future__ import annotations

import gc
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
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

# Tuned parameters for v2
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "n_estimators": 3000,
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "n_jobs": 1,
    "random_state": SEED,
    "early_stopping_rounds": 100,
}


def list_gpu_ids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi did not list any GPUs") from exc
    gpus: list[int] = []
    for line in out.strip().splitlines():
        idx, name, mem = [p.strip() for p in line.split(",", 2)]
        gid = int(idx)
        gpus.append(gid)
        print(f"GPU {gid}: {name} ({mem})", flush=True)
    if not gpus:
        raise RuntimeError("No NVIDIA GPUs available")
    return gpus


def assert_booster_on_gpu(model: XGBClassifier) -> None:
    cfg = json.loads(model.get_booster().save_config())
    generic = cfg.get("learner", {}).get("generic_param", {})
    train_param = cfg.get("learner", {}).get("gradient_booster", {}).get("gbtree_train_param", {})
    tree_method = train_param.get("tree_method") or generic.get("tree_method")
    predictor = generic.get("predictor")
    if tree_method != "gpu_hist":
        raise RuntimeError(f"expected gpu_hist, got tree_method={tree_method!r} predictor={predictor!r}")
    print(f"booster tree_method={tree_method} predictor={predictor}", flush=True)


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    
    orig_path = DATA / "original" / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
    orig = None
    if orig_path.exists():
        orig = pd.read_csv(orig_path)
        if "Buyer_ID" in orig.columns:
            orig = orig.drop(columns=["Buyer_ID"])
        # Align column dtypes and drop incomplete rows
        orig = orig.dropna().reset_index(drop=True)
        print(f"Loaded original dataset: {len(orig)} clean rows", flush=True)
        
    return train, test, orig


def target_to_int(s: pd.Series) -> np.ndarray:
    return (s == "Yes").astype(int).to_numpy()


def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate digit features, domain interactions, boundaries, and smooth keys."""
    out = df.copy()

    # 1. Domain interaction features
    out["Total_Charging"] = out["Charging_Stations_Near_Home"] + out["Charging_Stations_Near_Work"]
    out["Commute_per_Station"] = (out["Daily_Commute_km"] / (out["Total_Charging"] + 1.0)).astype(np.float32)
    out["Income_per_Commute"] = (out["Annual_Income_USD"] / (out["Daily_Commute_km"] + 1.0)).astype(np.float32)
    out["Charging_Ratio"] = (out["Charging_Stations_Near_Home"] / (out["Charging_Stations_Near_Work"] + 1.0)).astype(np.float32)
    out["Car_per_Income"] = (out["Number_of_Cars_Owned"] / (out["Annual_Income_USD"] / 10000.0 + 1.0)).astype(np.float32)

    is_subsidy = (out["Subsidy_Available"] == "Yes").astype(np.float32)
    out["Concern_x_Subsidy"] = (out["Environmental_Concern_Level"] * is_subsidy).astype(np.float32)
    out["Concern_x_Income"] = (out["Environmental_Concern_Level"] * out["Annual_Income_USD"]).astype(np.float32)

    # 2. Hard synthetic generation boundaries (Millionaire Cliff & Dead Zone)
    out["Millionaire_Cliff"] = (out["Annual_Income_USD"] >= 170537).astype(np.int8)
    out["Dead_Zone"] = out["Annual_Income_USD"].between(31004, 41970).astype(np.int8)

    # 3. Smooth keys (multi-scale binning)
    out["Income_100"] = (out["Annual_Income_USD"] // 100).astype(np.int32)
    out["Income_1000"] = (out["Annual_Income_USD"] // 1000).astype(np.int32)
    out["Commute_5"] = (out["Daily_Commute_km"] // 5).astype(np.int32)

    # 4. 56 Digit Features across the 7 numeric variables (from 10^-4 to 10^3)
    for col in NUM_COLS:
        col_vals = out[col].fillna(0)
        for k in range(-4, 4):
            out[f"{col}_digit{k}"] = (col_vals // (10**k) % 10).astype(np.int8)

    return out


def label_encode_categoricals(
    train_df: pd.DataFrame, test_df: pd.DataFrame, orig_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
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


PREDICT_BATCH_SIZE = 50_000


def _predict_batched(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict in chunks to avoid CUDA launch timeout on low-VRAM GPUs."""
    n = len(X)
    if n <= PREDICT_BATCH_SIZE:
        return model.predict_proba(X)[:, 1]
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, PREDICT_BATCH_SIZE):
        end = min(start + PREDICT_BATCH_SIZE, n)
        out[start:end] = model.predict_proba(X.iloc[start:end])[:, 1]
    return out


def cv_and_predict(
    train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    gpu_ids = list_gpu_ids()
    print(f"Preparing features for train ({len(train)}), test ({len(test)})...", flush=True)

    y_train = target_to_int(train["Will_Buy_EV"])
    train_clean = train.drop(columns=["id", "Will_Buy_EV"])
    test_clean = test.drop(columns=["id"])

    orig_clean = None
    y_orig = None
    if orig is not None:
        y_orig = target_to_int(orig["Will_Buy_EV"])
        orig_clean = orig.drop(columns=["Will_Buy_EV"])

    # Label encode categoricals
    train_encoded, test_encoded, orig_encoded = label_encode_categoricals(train_clean, test_clean, orig_clean)

    # Generate engineered base features
    print("Generating engineered features (interactions, digit features, boundaries)...", flush=True)
    X_all_train = engineer_base_features(train_encoded)
    X_test_base = engineer_base_features(test_encoded)
    X_orig_fe = engineer_base_features(orig_encoded) if orig_encoded is not None else None

    print(f"Total base features: {X_all_train.shape[1]}", flush=True)

    # Set up 5-fold CV
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test), dtype=np.float32)

    # Columns for Triple Target Encoding (categoricals + key interactions)
    te_cols = CAT_COLS + ["Income_100", "Income_1000", "Commute_5"]

    for fold, (tr_idx, va_idx) in enumerate(folds.split(X_all_train, y_train), start=1):
        gpu_id = gpu_ids[(fold - 1) % len(gpu_ids)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        X_tr = X_all_train.iloc[tr_idx].copy()
        y_tr = y_train[tr_idx].copy()
        X_va = X_all_train.iloc[va_idx].copy()
        y_va = y_train[va_idx].copy()
        X_te_fold = X_test_base.copy()

        # Incorporate original dataset strictly into training fold (never in val!)
        if X_orig_fe is not None and y_orig is not None:
            X_tr = pd.concat([X_tr, X_orig_fe], axis=0, ignore_index=True)
            y_tr = np.concatenate([y_tr, y_orig])

        # Apply Triple Target Encoding (smooth = 'auto', 10.0, 100.0)
        for smooth_val, smooth_name in [("auto", "auto"), (10.0, "10"), (100.0, "100")]:
            te = TargetEncoder(smooth=smooth_val, cv=5)
            X_tr_enc = te.fit_transform(X_tr[te_cols], y_tr).astype(np.float32)
            X_va_enc = te.transform(X_va[te_cols]).astype(np.float32)
            X_te_enc = te.transform(X_te_fold[te_cols]).astype(np.float32)

            for i, col in enumerate(te_cols):
                X_tr[f"TE_{col}_{smooth_name}"] = X_tr_enc[:, i]
                X_va[f"TE_{col}_{smooth_name}"] = X_va_enc[:, i]
                X_te_fold[f"TE_{col}_{smooth_name}"] = X_te_enc[:, i]

        params = dict(XGB_PARAMS)
        params["gpu_id"] = 0
        model = XGBClassifier(**params)

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=100,
        )

        if fold == 1:
            assert_booster_on_gpu(model)

        va_pred = _predict_batched(model, X_va)
        te_pred = _predict_batched(model, X_te_fold)

        oof[va_idx] = va_pred
        pred += te_pred / N_FOLDS

        best = int(model.best_iteration)
        fold_auc = float(roc_auc_score(y_va, va_pred))
        print(f"Fold {fold} GPU {gpu_id} AUC {fold_auc:.5f}  best_iter {best}", flush=True)

        # Free GPU memory before next fold
        del model, X_tr, X_va, X_te_fold, va_pred, te_pred
        gc.collect()

    cv_auc = float(roc_auc_score(y_train, oof))
    print(f"=== OOF AUC {cv_auc:.5f} ===", flush=True)
    return oof, pred, cv_auc


def write_submission(test: pd.DataFrame, pred: np.ndarray) -> Path:
    out = ROOT / "submission.csv"
    pd.DataFrame({"id": test["id"], "Will_Buy_EV": pred}).to_csv(out, index=False)
    return out


def main() -> None:
    train, test, orig = load_frames()
    oof, pred, cv_auc = cv_and_predict(train, test, orig)
    path = write_submission(test, pred)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = pd.read_csv(path)
    assert len(sub) == len(sample), f"Expected {len(sample)} rows, got {len(sub)}"
    assert list(sub.columns) == ["id", "Will_Buy_EV"], f"Unexpected columns: {sub.columns}"
    assert sub["id"].equals(sample["id"]), "Submission IDs do not match sample submission"
    assert sub["Will_Buy_EV"].between(0, 1).all(), "Predictions out of [0, 1] range"
    print(f"Successfully wrote {path} (rows={len(sub)}, OOF AUC={cv_auc:.5f})", flush=True)


if __name__ == "__main__":
    main()
