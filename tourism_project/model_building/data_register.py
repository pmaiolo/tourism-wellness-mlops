"""
Data Registration
-----------------
Reads the raw tourism dataset straight from the repository (tourism_project/data),
validates that every expected column is present and prints a short summary.
The dataset is versioned inside the GitHub repo, so no external data store is needed.
Exits with a non-zero code if validation fails, which stops the GitHub Actions pipeline.
"""
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path("tourism_project/data/tourism.csv")

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]
TARGET = "ProdTaken"


def main() -> None:
    if not DATA_PATH.exists():
        sys.exit(f"[ERROR] Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 1) Schema check: all expected columns must be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    # 2) Target check: must be binary 0/1
    if not set(df[TARGET].unique()).issubset({0, 1}):
        sys.exit(f"[ERROR] Target '{TARGET}' is not binary")

    # 3) Summary
    print("=" * 60)
    print("DATASET REGISTERED:", DATA_PATH)
    print("=" * 60)
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print(f"All {len(EXPECTED_COLUMNS)} expected columns present: OK")
    if extra:
        print(f"Extra columns (will be dropped in data prep): {extra}")
    print(f"Missing values (total): {int(df.isna().sum().sum())}")
    print(f"Duplicate CustomerIDs: {int(df['CustomerID'].duplicated().sum())}")
    rate = df[TARGET].mean()
    print(f"Target '{TARGET}' -> purchase rate: {rate:.2%} "
          f"(1: {int(df[TARGET].sum())}, 0: {int((1 - df[TARGET]).sum())})")
    print("\nColumn types:")
    print(df.dtypes.value_counts().to_string())
    print("\nNumeric summary:")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
