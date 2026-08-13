import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from s3_utils import upload_file

DATA_PATH = Path("data/processed/clean_loans.csv")
PLOTS_DIR = Path("results/eda_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = ["annual_inc", "dti", "loan_amnt", "int_rate",
                     "revol_util", "open_acc"]

def plot_correlation_matrix(df: pd.DataFrame):
    corr = df[NUMERIC_FEATURES + ["default"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    out_path = PLOTS_DIR / "correlation_matrix.png"
    plt.savefig(out_path)
    plt.close()
    print("Saved correlation matrix")
    upload_file(out_path, s3_key="results/eda_plots/correlation_matrix.png")

def missingness_vs_target(df: pd.DataFrame):
    """Check whether missing values themselves are predictive."""
    results = []
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        df["_is_missing"] = df[col].isnull().astype(int)
        if df["_is_missing"].nunique() < 2:
            continue
        default_rate_missing = df.loc[df["_is_missing"] == 1, "default"].mean()
        default_rate_present = df.loc[df["_is_missing"] == 0, "default"].mean()
        results.append({
            "column": col,
            "pct_missing": df["_is_missing"].mean() * 100,
            "default_rate_when_missing": default_rate_missing,
            "default_rate_when_present": default_rate_present,
        })
    result_df = pd.DataFrame(results).sort_values("pct_missing", ascending=False)
    print(result_df.head(15).to_string(index=False))

    out_path = Path("results/missingness_vs_target.csv")
    result_df.to_csv(out_path, index=False)
    upload_file(out_path, s3_key="results/missingness_vs_target.csv")
    return result_df

def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    plot_correlation_matrix(df)
    print("\n--- Missingness vs. default rate ---")
    missingness_vs_target(df)

if __name__ == "__main__":
    main()


