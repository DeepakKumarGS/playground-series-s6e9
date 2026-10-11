# Predicting Electric Vehicle Purchases — Playground Series S6E9

[Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e9)

## Goal

Predict the probability that each test-set customer will buy an electric vehicle (`Will_Buy_EV`). Kaggle evaluates submissions with ROC AUC.

## Repository layout

- `main.py` — current GPU XGBoost pipeline, including feature engineering, cross-validation, and submission generation.
- `agents/` — saved experiment versions.
- `kaggle-notebook/` — Kaggle-ready experiment scripts and metadata.
- `notebooks/` — exploration and competition walkthrough.
- `scripts/eval_local.py` — local evaluation helper.

## Data and running locally

Download the competition data from Kaggle and keep it in the local `data/` directory. The pipeline reads `data/train.csv` and `data/test.csv`; it can also use the optional original dataset at `data/original/EV_Adoption_and_Range_Anxiety_Dataset.csv`.

Run `python main.py` in an environment with the script's Python dependencies and a compatible NVIDIA GPU. It writes a local `submission.csv`; downloaded data and generated CSVs and prediction arrays are excluded from Git.

See the Kaggle competition page for the current rules and submission requirements.
