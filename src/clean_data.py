import pandas as pd
from pathlib import Path
from s3_utils import upload_file

RAW_PATH = Path("data/raw/lending_club_loans.csv")
OUT_PATH = Path("data/processed/clean_loans.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Outcomes that count as "default" for this project
DEFAULT_STATUSES = ["Charged Off", "Default"]
# Outcomes that count as "paid" (the negative class)
PAID_STATUSES = ["Fully Paid"]
# Everything else (e.g. "Current", "In Grace Period", "Late") has an unknown
# final outcome and must be dropped — including them would leak future info
# or introduce mislabeled examples.

# Columns known only AFTER a loan's outcome is determined — must be dropped
# to avoid leakage. This list grows as you audit more columns in week 1.
LEAKAGE_COLUMNS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "debt_settlement_flag", "hardship_flag",
    "settlement_status",
]

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["loan_status"].isin(DEFAULT_STATUSES + PAID_STATUSES)].copy()
    df["default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    return df

def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    cols_present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=cols_present)

def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Starting rows: {len(df):,}")

    df = build_target(df)
    print(f"Rows after keeping only resolved loans: {len(df):,}")

    df = drop_leakage(df)
    print(f"Columns after dropping leakage: {df.shape[1]}")

    print("\nClass balance:")
    print(df["default"].value_counts(normalize=True))

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to {OUT_PATH}")

    upload_file(OUT_PATH, s3_key="data/processed/clean_loans.csv")

if __name__ == "__main__":
    main()
