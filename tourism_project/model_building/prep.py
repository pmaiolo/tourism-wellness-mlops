"""
Data Preparation
----------------
1. Loads the dataset directly from the repository data folder.
2. Cleans it: drops identifier / index columns, fixes inconsistent category labels,
   removes exact duplicate customer profiles.
3. Splits it into stratified train / test sets and saves them locally as CSV files
   (Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv). The GitHub Actions workflow
   uploads these files as an artifact for the model-training job.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("tourism_project/data/tourism.csv")
TARGET = "ProdTaken"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# --- 1. Remove unnecessary columns -------------------------------------------
# 'Unnamed: 0' is a leftover index and 'CustomerID' is a unique identifier:
# neither carries predictive information and both would leak row identity.
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df = df.drop(columns=drop_cols)
print("Dropped columns:", drop_cols)

# --- 2. Fix inconsistent category labels -------------------------------------
# 'Fe Male' is a typo of 'Female'; 'Free Lancer' is normalised to 'Freelancer'.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})

# --- 3. Handle missing values (defensive: new data drops may contain NaNs) ---
num_cols = df.select_dtypes(include="number").columns.drop(TARGET)
cat_cols = df.select_dtypes(exclude="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# --- 4. Remove exact duplicate rows ------------------------------------------
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows -> {df.shape}")

# --- 5. Split (stratified to preserve the ~19% purchase rate) ---------------
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")
print(f"Purchase rate -> train {ytrain.mean():.2%} | test {ytest.mean():.2%}")
print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
