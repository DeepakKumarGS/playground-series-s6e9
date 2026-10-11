"""v1 XGBoost baseline: raw columns, 5-fold CV on every visible NVIDIA GPU."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
FEATURES = NUM_COLS + CAT_COLS

# MX130 is Maxwell SM 50. XGBoost 2.x/3.x wheels drop that arch (or need CUDA 13).
# 1.7.6 gpu_hist still runs on this card. Requires numpy<2.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 1500,
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "enable_categorical": True,
    "n_jobs": 1,
    "random_state": SEED,
    "early_stopping_rounds": 50,
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


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, test


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CAT_COLS:
        out[c] = out[c].astype("category")
    return out


def target_to_int(s: pd.Series) -> np.ndarray:
    return (s == "Yes").astype(int).to_numpy()


def _fit_one_fold(
    fold: int,
    gpu_id: int,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_test: pd.DataFrame,
) -> tuple[int, np.ndarray, np.ndarray, int, float]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    params = dict(XGB_PARAMS)
    params["gpu_id"] = 0  # first (only) device in this process
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    if fold == 1:
        assert_booster_on_gpu(model)
    va_pred = model.predict_proba(X_va)[:, 1]
    te_pred = model.predict_proba(X_test)[:, 1]
    best = int(model.best_iteration)
    auc = float(roc_auc_score(y_va, va_pred))
    print(f"fold {fold} GPU {gpu_id} AUC {auc:.5f}  best_iter {best}", flush=True)
    return fold, va_pred, te_pred, best, auc


def cv_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    gpu_ids = list_gpu_ids()
    y = target_to_int(train["Will_Buy_EV"])
    X = encode(train[FEATURES])
    X_test = encode(test[FEATURES])

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(folds.split(X, y))
    oof = np.zeros(len(train), dtype=float)
    pred = np.zeros(len(test), dtype=float)

    jobs = []
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        gpu_id = gpu_ids[(fold - 1) % len(gpu_ids)]
        jobs.append(
            (
                fold,
                gpu_id,
                tr_idx,
                va_idx,
            )
        )

    n_workers = min(len(gpu_ids), N_FOLDS)
    print(f"training {N_FOLDS} folds on {len(gpu_ids)} GPU(s), parallel workers={n_workers}", flush=True)

    if n_workers == 1:
        results = []
        for fold, gpu_id, tr_idx, va_idx in jobs:
            results.append(
                _fit_one_fold(
                    fold,
                    gpu_id,
                    X.iloc[tr_idx],
                    y[tr_idx],
                    X.iloc[va_idx],
                    y[va_idx],
                    X_test,
                )
            )
    else:
        # One worker per GPU so two boosters never share a card.
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = []
            for fold, gpu_id, tr_idx, va_idx in jobs:
                futs.append(
                    pool.submit(
                        _fit_one_fold,
                        fold,
                        gpu_id,
                        X.iloc[tr_idx],
                        y[tr_idx],
                        X.iloc[va_idx],
                        y[va_idx],
                        X_test,
                    )
                )
            results = [f.result() for f in futs]

    for fold, va_pred, te_pred, _best, _auc in results:
        _, va_idx = splits[fold - 1]
        oof[va_idx] = va_pred
        pred += te_pred / N_FOLDS

    cv_auc = float(roc_auc_score(y, oof))
    print(f"OOF AUC {cv_auc:.5f}", flush=True)
    return oof, pred, cv_auc


def write_submission(test: pd.DataFrame, pred: np.ndarray) -> Path:
    out = ROOT / "submission.csv"
    pd.DataFrame({"id": test["id"], "Will_Buy_EV": pred}).to_csv(out, index=False)
    return out


def main() -> None:
    train, test = load_frames()
    assert set(FEATURES).issubset(test.columns)
    _, pred, _ = cv_and_predict(train, test)
    path = write_submission(test, pred)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = pd.read_csv(path)
    assert len(sub) == len(sample)
    assert list(sub.columns) == ["id", "Will_Buy_EV"]
    assert sub["id"].equals(sample["id"])
    assert sub["Will_Buy_EV"].between(0, 1).all()
    print(f"wrote {path} rows={len(sub)}", flush=True)


if __name__ == "__main__":
    main()
