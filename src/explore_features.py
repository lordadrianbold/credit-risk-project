import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from s3_utils import upload_dir

DATA_PATH = Path("data/processed/clean_loans.csv")
PLOTS_DIR = Path("results/eda_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate features to inspect — adjust based on what's in your dataset
NUMERIC_FEATURES = ["annual_inc", "dti", "loan_amnt", "int_rate",
                     "revol_util", "open_acc", "emp_length_years"]
CATEGORICAL_FEATURES = ["grade", "purpose", "home_ownership", "term"]

def plot_numeric_vs_default(df: pd.DataFrame, col: str, bins: int = 10):
    """Bucket a numeric feature into bins and plot default rate per bin."""
    df = df.copy()
    df["bucket"] = pd.qcut(df[col], q=bins, duplicates="drop")
    rate_by_bucket = df.groupby("bucket", observed=True)["default"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    rate_by_bucket.plot(kind="bar", ax=ax)
    ax.set_title(f"Default Rate by {col} (binned)")
    ax.set_ylabel("Default rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"default_rate_by_{col}.png")
    plt.close()

def plot_categorical_vs_default(df: pd.DataFrame, col: str):
    rate_by_cat = df.groupby(col, observed=True)["default"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    rate_by_cat.plot(kind="bar", ax=ax)
    ax.set_title(f"Default Rate by {col}")
    ax.set_ylabel("Default rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"default_rate_by_{col}.png")
    plt.close()

def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].notna().sum() > 0:
            plot_numeric_vs_default(df, col)
            print(f"Saved plot for {col}")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            plot_categorical_vs_default(df, col)
            print(f"Saved plot for {col}")

    print(f"\nAll plots saved to {PLOTS_DIR}")

    upload_dir(PLOTS_DIR, s3_prefix="results/eda_plots/")

if __name__ == "__main__":
    main()


