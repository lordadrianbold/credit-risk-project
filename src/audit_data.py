import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/lending_club_loans.csv")

def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)

    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    print(list(df.columns))

    print("\n--- loan_status value counts ---")
    print(df["loan_status"].value_counts())

    print("\n--- Missingness (top 20 worst columns) ---")
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    print(missing_pct.head(20))

    print("\n--- Dtypes ---")
    print(df.dtypes.value_counts())

if __name__ == "__main__":
    main()


