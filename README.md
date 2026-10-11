# Predicting Electric Vehicle Purchases — Playground Series S6E9

[Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e9)

## Case study

### Objective

Predict the probability that a customer will buy an electric vehicle (`Will_Buy_EV`). Kaggle scores probability submissions by ROC AUC.

### Constraints

- Validation must preserve the competition's row and label boundaries. Target encodings are fitted within each training fold to avoid target leakage.
- The pipeline can use the optional original EV adoption dataset when it is available locally; it is auxiliary training data, not a replacement for the competition test set.
- The ensemble experiments use GPU versions of LightGBM, XGBoost, and CatBoost. The 10-fold run took about 37.5 minutes in its logged Kaggle execution.
- Downloaded data, submissions, prediction arrays, and training outputs are local artifacts and are excluded from Git.

### Strategy and iterations

| Version | Change | Pooled OOF ROC AUC | Kaggle public score |
|---|---|---:|---:|
| v1 | Raw-column LightGBM and XGBoost baselines; five-fold validation. | No retained score in the tracked run logs. | — |
| v2 | XGBoost feature engineering: digit decomposition, boundary flags, interactions, fold-safe target encodings, and optional original data. | 0.94446 | 0.94432 on 2026-09-20 |
| v3 | Multi-scale feature engineering and LightGBM + XGBoost ensemble. | 0.94607 | 0.94635 on 2026-09-26 |
| v4 | Dual-seed LightGBM, XGBoost, and CatBoost tri-model blend. | 0.94612 | 0.94627 on 2026-09-26 |
| v5 | Ten-fold tri-model stack, latent-prior features, and optimized blend weights (0.50 LightGBM, 0.35 XGBoost, 0.15 CatBoost). | 0.94614 | 0.94630 on 2026-09-27 |

The extra models and folds helped, but gains flattened: v5 improved pooled OOF AUC by only 0.00002 over v4. The 2026-10-11 Kaggle public leaderboard snapshot showed the team's best score as **0.94635**, rank **873 of 3,576 teams**. The leaderboard's best team score differs from the most recent submission's score.

### Model-agent work

The `agents/` directory contains versioned predictive-model pipelines, not autonomous code-editing agents. The scripts carried out the baseline comparisons, feature engineering, cross-validation, ensemble fitting, and submission checks described above. No separate coding-agent transcript or attribution is stored for this project.

### Evaluation and failure cases

The local comparison metric is pooled out-of-fold ROC AUC from stratified five-fold validation (ten folds for v5). The Kaggle public leaderboard is an independent check on generalization. The saved runs show that the larger stack's OOF gain was marginal; adding complexity did not produce a clear improvement over v4. The retained evidence does not include a documented failed Kaggle submission or a per-row error analysis.

The leaderboard snapshot and submission history were read from Kaggle on **2026-10-11**. Public scores above are submission results; OOF scores are local validation results and should not be compared as if they were the same split.

## Repository guide

- `main.py` — local GPU XGBoost pipeline and submission generation.
- `agents/` — saved experiment versions.
- `kaggle-notebook/` — Kaggle-ready scripts and notebook metadata.
- `notebooks/` — competition walkthrough.
- `scripts/eval_local.py` — local evaluation helper.

## Data and reproduction

Download the competition data into the ignored local `data/` directory. The pipeline reads `data/train.csv` and `data/test.csv`; the optional auxiliary dataset belongs at `data/original/EV_Adoption_and_Range_Anxiety_Dataset.csv`. Run `python main.py` in an environment with the script's dependencies and a compatible NVIDIA GPU. It writes a local `submission.csv` and validation artifacts. For the later ensemble experiments, use the corresponding version in `agents/` or its Kaggle-ready script under `kaggle-notebook/`.
