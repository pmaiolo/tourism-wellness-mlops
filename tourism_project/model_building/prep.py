"""
Data Preparation
----------------
1. Loads the dataset directly from the repository data folder.
2. Cleans it: drops identifier / index columns, fixes inconsistent category labels.
3. Keeps ONLY the information available BEFORE the sales team contacts the customer
   (the business goal is to score leads before contact). Post-contact columns are removed.
4. Removes exact duplicate customer profiles.
5. Splits it into stratified train / test sets and saves them locally as CSV files
   (Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv). The GitHub Actions workflow
   uploads these files as an artifact for the model-training job.

Missing values are NOT imputed here: imputation lives inside the model pipeline
(SimpleImputer in train.py) so it is fitted on the training split only.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("tourism_project/data/tourism.csv")
TARGET = "ProdTaken"
RANDOM_STATE = 42

# Columns that only exist AFTER the customer has been contacted / pitched.
# Using them would leak future information into a "before contact" prediction.
POST_CONTACT_COLS = [
    "DurationOfPitch",         # length of the sales pitch -> known only after the pitch
    "NumberOfFollowups",       # follow-ups happen after the pitch
    "PitchSatisfactionScore",  # customer rates the pitch after it is delivered
    "ProductPitched",          # chosen during the sales interaction; it is also fully
                               # determined by Designation in this data (redundant)
]
# 'TypeofContact' is KEPT: it is the lead source (Self Enquiry vs Company Invited),
# recorded when the lead enters the CRM, i.e. before the sales call.

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# --- 1. Remove unnecessary columns -------------------------------------------
# 'Unnamed: 0' is a leftover index and 'CustomerID' is a unique identifier:
# neither carries predictive information and both would leak row identity.
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df = df.drop(columns=drop_cols)
print("Dropped identifier columns:", drop_cols)

# --- 2. Fix inconsistent category labels -------------------------------------
# 'Fe Male' is a typo of 'Female'; 'Free Lancer' is normalised to 'Freelancer'.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})

# --- 3. Keep only pre-contact information ------------------------------------
df = df.drop(columns=[c for c in POST_CONTACT_COLS if c in df.columns])
print("Dropped post-contact columns:", POST_CONTACT_COLS)

# --- 4. Report missing values (imputation is done inside the model pipeline) --
n_missing = int(df.isna().sum().sum())
print(f"Missing values: {n_missing} (imputed later inside the pipeline, fitted on train only)")

# --- 5. Remove exact duplicate rows ------------------------------------------
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows -> {df.shape}")

# --- 6. Split (stratified to preserve the ~19% purchase rate) ---------------
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Features used ({X.shape[1]}): {list(X.columns)}")
print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")
print(f"Purchase rate -> train {ytrain.mean():.2%} | test {ytest.mean():.2%}")
print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
