"""v1 LightGBM baseline: raw columns only, 5-fold stratified CV, write submission.csv."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 1500,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": SEED,
}


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


def cv_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    y = target_to_int(train["Will_Buy_EV"])
    X = encode(train[FEATURES])
    X_test = encode(test[FEATURES])

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=float)
    pred = np.zeros(len(test), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds.split(X, y), start=1):
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
        print(f"fold {fold} AUC {fold_auc:.5f}  best_iter {model.best_iteration_}")

    cv_auc = float(roc_auc_score(y, oof))
    print(f"OOF AUC {cv_auc:.5f}")
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
    print(f"wrote {path} rows={len(sub)}")


if __name__ == "__main__":
    main()
