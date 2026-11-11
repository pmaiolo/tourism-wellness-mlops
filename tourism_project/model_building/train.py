"""
Model Training with Experiment Tracking
---------------------------------------
1. Loads the train / test splits produced by the data-prep job (workflow artifact).
2. Builds a preprocessing + XGBoost pipeline and tunes it with GridSearchCV.
3. Logs every tuned parameter combination (nested runs) and the best model's
   parameters, metrics and artifacts to MLflow.
4. Picks a decision threshold that maximises F1 on the cross-validated
   training predictions (never on the test set) and evaluates on the test set.
5. Saves the best model to tourism_project/deployment/ so the workflow can commit it.
"""
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
MODEL_DIR = Path("tourism_project/deployment")
MODEL_PATH = MODEL_DIR / "best_tourism_model_v1.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

# --- MLflow setup -------------------------------------------------------------
# In GitHub Actions an MLflow server is started on localhost:5000 and
# MLFLOW_TRACKING_URI points to it; locally we fall back to a ./mlruns folder.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("tourism-wellness-package")

# --- Load data ----------------------------------------------------------------
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")
print(f"Train {Xtrain.shape} | Test {Xtest.shape}")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# --- Model definition ---------------------------------------------------------
# Class imbalance (~19% buyers): weight the positive class by the neg/pos ratio.
scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=-1,
)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# --- Hyperparameter grid ------------------------------------------------------
param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__subsample": [0.8],
    "xgbclassifier__reg_lambda": [1.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def metrics(y_true, y_pred, y_prob, prefix):
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_prob),
    }


with mlflow.start_run(run_name="xgb-gridsearch") as parent:
    grid = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring="f1",
                        n_jobs=-1, return_train_score=False)
    grid.fit(Xtrain, ytrain)

    # Log EVERY tuned parameter combination as a nested child run
    res = grid.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"candidate-{i:02d}", nested=True):
            mlflow.log_params({k.split("__")[1]: v for k, v in params.items()})
            mlflow.log_metric("cv_mean_f1", res["mean_test_score"][i])
            mlflow.log_metric("cv_std_f1", res["std_test_score"][i])

    best = grid.best_estimator_
    best_params = {k.split("__")[1]: v for k, v in grid.best_params_.items()}
    mlflow.log_params(best_params)
    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
    mlflow.log_metric("best_cv_f1", grid.best_score_)

    # Choose the decision threshold on out-of-fold TRAIN predictions (no test leakage)
    oof_prob = cross_val_predict(best, Xtrain, ytrain, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(ytrain, oof_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    threshold = float(thr[np.argmax(f1s)])
    mlflow.log_param("decision_threshold", round(threshold, 4))

    # Evaluate the refit best model
    train_prob = best.predict_proba(Xtrain)[:, 1]
    test_prob = best.predict_proba(Xtest)[:, 1]
    train_pred = (train_prob >= threshold).astype(int)
    test_pred = (test_prob >= threshold).astype(int)
    all_metrics = {**metrics(ytrain, train_pred, train_prob, "train"),
                   **metrics(ytest, test_pred, test_prob, "test")}
    mlflow.log_metrics(all_metrics)

    print("Best params:", best_params)
    print(f"Best CV F1: {grid.best_score_:.4f} | decision threshold: {threshold:.3f}")
    print("\nTest-set classification report:")
    print(classification_report(ytest, test_pred, digits=3))
    for k, v in all_metrics.items():
        print(f"{k:>18}: {v:.4f}")

    # Feature importance (top 10) for interpretability
    ohe_names = best.named_steps["columntransformer"].get_feature_names_out()
    importances = pd.Series(best.named_steps["xgbclassifier"].feature_importances_,
                            index=[n.split("__", 1)[1] for n in ohe_names])
    top = importances.sort_values(ascending=False).head(10)
    print("\nTop-10 feature importances:\n" + top.round(4).to_string())

    # --- Save best model + metadata for the Streamlit app ---------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    meta = {
        "model_file": MODEL_PATH.name,
        "decision_threshold": threshold,
        "best_params": best_params,
        "test_metrics": {k: round(v, 4) for k, v in all_metrics.items() if k.startswith("test")},
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgb.__version__},
        "mlflow_run_id": parent.info.run_id,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, default=float))
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
    mlflow.log_artifact(str(META_PATH), artifact_path="model")
    print(f"\nSaved model -> {MODEL_PATH}\nSaved metadata -> {META_PATH}")
