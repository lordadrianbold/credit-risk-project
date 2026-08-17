"""
Week 5: model interpretation (SHAP) + deliberate threshold selection.

Two things happen here that are easy to skip but matter a lot in a real
credit risk workflow:
  1. SHAP explains WHY the model flags a given loan as risky — not just
     that it did. This is the difference between "the model said no" and
     being able to explain a denial.
  2. The default 0.5 classification threshold is essentially arbitrary.
     Choosing a threshold deliberately, based on the real cost tradeoff
     between false positives (rejecting a good borrower) and false
     negatives (approving a loan that defaults), is a business decision,
     not a modeling detail — and it's usually skipped by self-taught
     projects, which is exactly what makes doing it properly stand out.
"""

import pandas as pd
import numpy as np
import shap
from pathlib import Path
from xgboost import XGBClassifier
from eval_utils import prepare_features, split_data, TARGET

DATA_PATH = Path("data/processed/clean_loans.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    train_medians = X_train.median()
    X_train_filled = X_train.fillna(train_medians)
    X_test_filled = X_test.fillna(train_medians)

    # Use the class-weighted XGBoost as the model to interpret — pick
    # whichever model actually won in model_trees.py's comparison table
    # once you have real data; this is a reasonable default.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=neg / pos, random_state=42, eval_metric="logloss",
    )
    model.fit(X_train_filled, y_train)

    # --- SHAP: global feature importance ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_filled)

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_test_filled.columns
    ).sort_values(ascending=False)
    print("Top 10 features by mean |SHAP value| (overall importance):")
    print(mean_abs_shap.head(10).to_string())

    # --- SHAP: walk through 2 individual predictions ---
    y_proba = model.predict_proba(X_test_filled)[:, 1]
    high_risk_idx = np.argmax(y_proba)
    low_risk_idx = np.argmin(y_proba)

    for label, idx in [("HIGHEST predicted risk", high_risk_idx), ("LOWEST predicted risk", low_risk_idx)]:
        print(f"\n--- Example: {label} (predicted probability={y_proba[idx]:.3f}) ---")
        row_shap = pd.Series(shap_values[idx], index=X_test_filled.columns).sort_values(key=abs, ascending=False)
        print("Top contributing features (positive = pushed risk up):")
        print(row_shap.head(5).to_string())

    # --- Deliberate threshold selection ---
    # Assume, for this write-up, that a missed default (false negative)
    # costs roughly 5x more than a wrongly-rejected good loan (false
    # positive) — a reasonable starting assumption for unsecured consumer
    # lending, though a real deployment would use the lender's actual loss
    # figures instead of an assumption stated here explicitly.
    FN_COST_RATIO = 5

    # Evaluate real confusion-matrix counts across a threshold grid, rather
    # than relying on precision/recall curve approximations — those can
    # produce a degenerate "optimal" threshold (e.g. flagging almost every
    # loan) because precision and recall alone don't reconstruct the true
    # false-positive RATE without knowing the actual negative-class count.
    threshold_grid = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in threshold_grid:
        preds_at_t = (y_proba >= t).astype(int)
        fn = ((preds_at_t == 0) & (y_test == 1)).sum()
        fp = ((preds_at_t == 1) & (y_test == 0)).sum()
        costs.append(FN_COST_RATIO * fn + fp)

    best_idx = int(np.argmin(costs))
    chosen_threshold = threshold_grid[best_idx]

    def metrics_at_threshold(t):
        preds_at_t = (y_proba >= t).astype(int)
        tp = ((preds_at_t == 1) & (y_test == 1)).sum()
        fp = ((preds_at_t == 1) & (y_test == 0)).sum()
        fn = ((preds_at_t == 0) & (y_test == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall

    def cost_at_threshold(t):
        preds_at_t = (y_proba >= t).astype(int)
        fn = ((preds_at_t == 0) & (y_test == 1)).sum()
        fp = ((preds_at_t == 1) & (y_test == 0)).sum()
        return FN_COST_RATIO * fn + fp

    default_precision, default_recall = metrics_at_threshold(0.5)
    chosen_precision, chosen_recall = metrics_at_threshold(chosen_threshold)

    print(f"\n--- Threshold selection (assuming a missed default costs "
          f"{FN_COST_RATIO}x a wrongly-rejected good loan) ---")
    print(f"Default 0.5 threshold: precision={default_precision:.3f}, recall={default_recall:.3f}, "
          f"total cost={cost_at_threshold(0.5)}")
    print(f"Chosen threshold ({chosen_threshold:.3f}): "
          f"precision={chosen_precision:.3f}, recall={chosen_recall:.3f}, "
          f"total cost={costs[best_idx]}")
    print("Lower threshold catches more true defaults (higher recall) at the "
          "cost of flagging more good loans (lower precision) — the right "
          "tradeoff depends on the stated cost ratio above, not a universal answer. "
          "If this lands at an extreme (near 0 or near 1), that's a signal the "
          "cost ratio itself needs revisiting, not that the threshold search is broken.")


if __name__ == "__main__":
    main()




    