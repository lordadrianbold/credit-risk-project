"""
Shared evaluation utilities for Weeks 3-5.

Centralizes the train/test split, feature engineering, and metrics logging
so every model script (logistic regression, tree-based, etc.) evaluates on
the exact same split and reports metrics the exact same way — a direct
analogue of backtest.py in the time-series project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from s3_utils import upload_file

RESULTS_PATH = Path("results/model_comparison.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET = "default"

NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "annual_inc", "dti",
    "open_acc", "revol_util", "emp_length_years",
]
CATEGORICAL_FEATURES = ["grade", "home_ownership", "purpose", "term"]


def parse_emp_length(series: pd.Series) -> pd.Series:
    """
    Real Lending Club data stores employment length as text, e.g.
    "10+ years", "< 1 year", "3 years", or "n/a" — not as a ready-made
    number. Converts to a numeric years value:
      "< 1 year"  -> 0
      "1 year"    -> 1
      "3 years"   -> 3
      "10+ years" -> 10
      "n/a" / NaN -> NaN (missing, handled later by imputation)
    This is exactly the kind of real-world messiness that a synthetic
    test dataset can hide — worth remembering when trusting a synthetic
    smoke test as "fully verified."
    """
    def parse_one(val):
        if pd.isna(val) or val == "n/a":
            return np.nan
        val = str(val).strip()
        if val.startswith("< 1"):
            return 0
        if val.startswith("10+"):
            return 10
        # "3 years" / "1 year" -> extract the leading number
        digits = "".join(ch for ch in val if ch.isdigit())
        return float(digits) if digits else np.nan

    return series.apply(parse_one)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few finance-informed engineered features on top of the raw
    columns. Kept intentionally simple and interpretable — in credit risk,
    simple ratio features are often as predictive as anything more elaborate,
    and they're far easier to explain to a non-technical stakeholder (or a
    regulator) than an opaque interaction term.
    """
    df = df.copy()

    # Loan amount relative to income — a classic affordability signal
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)

    # Parse the real text-based emp_length column into a numeric feature.
    # (Falls back gracefully if a pre-parsed emp_length_years column is
    # already present instead — e.g. if this is ever run against synthetic
    # test data shaped like the earlier smoke tests.)
    if "emp_length_years" not in df.columns and "emp_length" in df.columns:
        df["emp_length_years"] = parse_emp_length(df["emp_length"])

    # Interest rate already reflects the lender's own risk assessment at
    # origination, so keeping it as a raw feature is fine (it's not
    # leakage — it's known at approval time, unlike post-outcome fields).

    return df


def prepare_features(df: pd.DataFrame):
    """
    Build the final X, y matrices: engineer features, select columns,
    one-hot encode categoricals. Returns (X, y, feature_names).
    """
    df = engineer_features(df)

    numeric_cols = NUMERIC_FEATURES + ["loan_to_income"]
    X_numeric = df[numeric_cols]
    X_categorical = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True)

    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = df[TARGET]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Stratified split — preserves the class balance (default rate) in both
    train and test sets. With an imbalanced target (~11% default here),
    a plain random split can occasionally produce a test set with a
    meaningfully different default rate than the training set purely by
    chance; stratifying removes that risk.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluate_and_log(model_name: str, y_true, y_pred, y_proba):
    """
    Compute the metrics that actually matter for imbalanced classification
    (accuracy is deliberately NOT included here — with ~89% negative class,
    a model that always predicts "no default" would score 89% accuracy
    while being useless) and append them to the shared comparison table.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n{model_name}:")
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"  ROC-AUC: {roc_auc:.3f}  PR-AUC: {pr_auc:.3f}")
    print(f"  Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    row = pd.DataFrame([{
        "model": model_name, "precision": precision, "recall": recall,
        "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }])

    if RESULTS_PATH.exists():
        existing = pd.read_csv(RESULTS_PATH)
        existing = existing[existing["model"] != model_name]
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    combined.to_csv(RESULTS_PATH, index=False)
    upload_file(RESULTS_PATH, s3_key="results/model_comparison.csv")



    