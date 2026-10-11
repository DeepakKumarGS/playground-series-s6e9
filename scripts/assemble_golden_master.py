"""
Assemble Golden Master Submission
Combines:
  - Stream 1: v3 Dual Ensemble (0.94635 LB verified) [weight: 0.20]
  - Stream 2: v4 Tri-Model Grandmaster Stack (0.94612 OOF) [weight: 0.30]
  - Stream 3: v5 10-Fold Latent-Prior Tri-Model Stack (0.94614 OOF) [weight: 0.50]
Resulting cross-version OOF AUC: 0.946221 (+0.00015 gain over best single stream)
Applies deterministic physics boundary shifts and zero-tie lexsort calibration.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

test = pd.read_csv("data/test.csv")

# 1. Load test predictions from each stream
# v3: 5-fold LGBM (0.61) + XGB (0.39)
te_v3_lgb = np.load("/tmp/v3_output/test_lgb.npy")
te_v3_xgb = np.load("/tmp/v3_output/test_xgb.npy")
r_v3 = 0.61 * rankdata(te_v3_lgb) / len(test) + 0.39 * rankdata(te_v3_xgb) / len(test)

# v4: 5-fold Dual-Seed LGBM (0.45) + XGB (0.30) + CatBoost GPU (0.25)
te_v4_lgb = np.load("kaggle-output/test_lgb_v4.npy")
te_v4_xgb = np.load("kaggle-output/test_xgb_v4.npy")
te_v4_cat = np.load("kaggle-output/test_cat_v4.npy")
r_v4 = 0.45 * rankdata(te_v4_lgb) / len(test) + 0.30 * rankdata(te_v4_xgb) / len(test) + 0.25 * rankdata(te_v4_cat) / len(test)

# v5: 10-fold Latent-Prior LGBM (0.50) + XGB (0.35) + CatBoost GPU (0.15)
te_v5_lgb = np.load("kaggle-output/test_lgb_v5.npy")
te_v5_xgb = np.load("kaggle-output/test_xgb_v5.npy")
te_v5_cat = np.load("kaggle-output/test_cat_v5.npy")
r_v5 = 0.50 * rankdata(te_v5_lgb) / len(test) + 0.35 * rankdata(te_v5_xgb) / len(test) + 0.15 * rankdata(te_v5_cat) / len(test)

# 2. Optimal Golden Master Blend (OOF AUC: 0.946221)
test_blend = 0.20 * rankdata(r_v3) + 0.30 * rankdata(r_v4) + 0.50 * rankdata(r_v5)
test_blend = (rankdata(test_blend) - 0.5) / len(test_blend)

# 3. Physics Boundary Calibration
test_incomes = pd.to_numeric(test["Annual_Income_USD"], errors="coerce").to_numpy()
test_commute = pd.to_numeric(test["Daily_Commute_km"], errors="coerce").to_numpy()
subsidy_no = (test["Subsidy_Available"].astype(str) == "No").to_numpy()
env_1 = (pd.to_numeric(test["Environmental_Concern_Level"], errors="coerce") == 1.0).to_numpy()
anx_med_high = test["Range_Anxiety_Level"].isin(["Medium", "High"]).to_numpy()

order = np.lexsort((test_blend, test_blend))
ranks = np.empty(len(order), dtype=np.int64)
ranks[order] = np.arange(1, len(order) + 1)
calibrated_preds = (ranks - 0.5) / len(ranks)

# Hard Deterministic Shifts
calibrated_preds[test_incomes >= 170537.0] += 10.0
calibrated_preds[(test_incomes >= 31004.0) & (test_incomes <= 41970.0)] -= 10.0
calibrated_preds[test_commute >= 83.0] -= 5.0
calibrated_preds[(test_incomes == 30000.0) & subsidy_no & (env_1 | anx_med_high)] -= 5.0

# 4. Zero-Tie Lexsort Final Ranking
order_final = np.lexsort((test_blend, calibrated_preds))
ranks_final = np.empty(len(order_final), dtype=np.int64)
ranks_final[order_final] = np.arange(1, len(order_final) + 1)
final_ranks = (ranks_final - 0.5) / len(ranks_final)

sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": final_ranks})
sub.to_csv("submission.csv", index=False)

sample_sub = pd.read_csv("data/sample_submission.csv")
assert len(sub) == len(sample_sub)
assert list(sub.columns) == ["id", "Will_Buy_EV"]
assert sub["id"].equals(sample_sub["id"])
assert sub["Will_Buy_EV"].nunique() == len(sub)
print("Golden Master submission.csv successfully generated and verified!")
