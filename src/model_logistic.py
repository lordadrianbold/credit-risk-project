"""
Week 3: feature engineering + logistic regression baseline.

Logistic regression isn't just a "toy baseline" here — in credit risk,
simple, interpretable models are often what's actually deployed, since
regulators and risk teams typically want to understand exactly why a loan
was flagged, not just trust an opaque score. Coefficients are directly
interpretable (a positive coefficient = higher risk), which a black-box
model can't offer without extra tooling.
"""

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from eval_utils import prepare_features, split_data, evaluate_and_log

DATA_PATH = Path("data/processed/clean_loans.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Logistic regression needs imputation (can't handle NaN) and benefits
    # from scaling (coefficients aren't comparable across features on very
    # different scales otherwise). Fill with training-set medians only —
    # using the full dataset's median would leak test-set information.
    train_medians = X_train.median()
    X_train_filled = X_train.fillna(train_medians)
    X_test_filled = X_test.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_test_scaled = scaler.transform(X_test_filled)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    evaluate_and_log("logistic_regression", y_test, y_pred, y_proba)

    # Coefficients ranked by magnitude — this is the interpretability payoff
    coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
    print("\nTop 10 coefficients by magnitude (positive = higher default risk):")
    print(coefs.head(10).to_string())


if __name__ == "__main__":
    main()




    