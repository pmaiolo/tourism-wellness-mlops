# Wellness Tourism Package – MLOps Pipeline

End-to-end MLOps project for **"Visit with Us"**: a model that predicts whether a customer will buy the new **Wellness Tourism Package** *before* the sales team contacts them, trained and deployed automatically with **GitHub Actions** and served with **Streamlit Community Cloud**.

**Live app:** https://pmaiolo-tourism-wellness.streamlit.app

## Repository structure

```
.github/workflows/pipeline.yml        # CI/CD pipeline (3 jobs)
tourism_project/
├── data/tourism.csv                  # raw dataset (versioned in the repo)
├── model_building/
│   ├── data_register.py              # validates schema and target, prints a summary
│   ├── prep.py                       # cleaning, pre-contact feature selection, train/test split
│   └── train.py                      # XGBoost pipeline + GridSearchCV + MLflow tracking
├── deployment/
│   ├── app.py                        # Streamlit app
│   ├── requirements.txt              # app dependencies
│   ├── best_tourism_model_v1.joblib  # trained model (committed by the pipeline)
│   └── model_meta.json               # threshold, best params, test metrics
└── requirements.txt                  # pipeline dependencies
```

## Pipeline (GitHub Actions)

Runs on every push to `main` (or manually via *Run workflow*):

1. **register-dataset** – checks that all expected columns exist and the target is binary; fails the run otherwise.
2. **data-prep** – removes identifiers, fixes label typos, drops post-contact columns, removes duplicates and creates a stratified 80/20 split.
3. **model-traning** – tunes an XGBoost pipeline (36 combinations × 5-fold CV, F1), logs every run to MLflow, picks the decision threshold on out-of-fold training predictions and commits the model back to `main` (`[skip ci]`). Streamlit redeploys automatically.

## Model

- **Pre-contact features only (14):** Age, Gender, MaritalStatus, Occupation, Designation, MonthlyIncome, CityTier, Passport, OwnCar, NumberOfTrips, NumberOfPersonVisiting, NumberOfChildrenVisiting, PreferredPropertyStar, TypeofContact (lead source).
- **Excluded (known only after contact):** DurationOfPitch, NumberOfFollowups, PitchSatisfactionScore, ProductPitched.
- **Preprocessing inside the pipeline:** SimpleImputer + StandardScaler (numeric), SimpleImputer + OneHotEncoder (categorical), fitted on training data only.
- **Test performance:** accuracy 0.920 · precision 0.821 · recall 0.752 · F1 0.785 · ROC-AUC 0.940 (decision threshold 0.475).

These are results on a held-out historical test set; the expected uplift should be validated with a pilot campaign before full roll-out.

## Run locally

```bash
pip install -r tourism_project/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```

Python 3.11 is required (the same version used by the workflow and the Streamlit app).
