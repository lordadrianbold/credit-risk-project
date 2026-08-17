"""
Week 4: tree-based models (random forest, XGBoost) with class imbalance
handling.

Compares two approaches to the imbalance problem head-to-head rather than
picking one blindly:
  1. Class weighting (scale_pos_weight / class_weight="balanced") — the
     model penalizes mistakes on the minority class more heavily during
     training, without changing the training data itself.
  2. SMOTE oversampling — synthesizes new minority-class examples so the
     training set itself is balanced.
Neither is "correct" in general; which one works better is an empirical
question this script actually answers for this dataset, rather than
assuming.
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from eval_utils import prepare_features, split_data, evaluate_and_log

DATA_PATH = Path("data/processed/clean_loans.csv")


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Tree-based models handle NaN natively in XGBoost, but sklearn's
    # RandomForestClassifier does not — impute for both, using
    # training-set medians only (same leakage-avoidance rule as Week 3).
    train_medians = X_train.median()
    X_train_filled = X_train.fillna(train_medians)
    X_test_filled = X_test.fillna(train_medians)

    # --- Random Forest, class-weighted ---
    rf_weighted = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
    )
    rf_weighted.fit(X_train_filled, y_train)
    preds = rf_weighted.predict(X_test_filled)
    proba = rf_weighted.predict_proba(X_test_filled)[:, 1]
    evaluate_and_log("random_forest_class_weighted", y_test, preds, proba)

    # --- Random Forest, SMOTE-resampled ---
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_filled, y_train)
    rf_smote = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf_smote.fit(X_train_smote, y_train_smote)
    preds = rf_smote.predict(X_test_filled)
    proba = rf_smote.predict_proba(X_test_filled)[:, 1]
    evaluate_and_log("random_forest_smote", y_test, preds, proba)

    # --- XGBoost, class-weighted (scale_pos_weight) ---
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos  # ratio of majority to minority class
    xgb_weighted = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42,
        eval_metric="logloss",
    )
    xgb_weighted.fit(X_train_filled, y_train)
    preds = xgb_weighted.predict(X_test_filled)
    proba = xgb_weighted.predict_proba(X_test_filled)[:, 1]
    evaluate_and_log("xgboost_class_weighted", y_test, preds, proba)

    # --- XGBoost, SMOTE-resampled ---
    xgb_smote = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42, eval_metric="logloss",
    )
    xgb_smote.fit(X_train_smote, y_train_smote)
    preds = xgb_smote.predict(X_test_filled)
    proba = xgb_smote.predict_proba(X_test_filled)[:, 1]
    evaluate_and_log("xgboost_smote", y_test, preds, proba)

    print("\nAll four tree-based variants logged to results/model_comparison.csv")
    print("Compare recall and PR-AUC specifically — with ~11% default rate,")
    print("PR-AUC is more informative than ROC-AUC for judging real performance.")


if __name__ == "__main__":
    main()




    