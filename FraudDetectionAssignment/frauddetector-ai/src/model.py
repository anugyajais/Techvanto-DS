# src/model.py

# LATER LightGBMClassifier or CatBoostClassifier
# Add logging instead of print()
# Create a ModelManager class later to encapsulate all saving logic
# Save model version with timestamp or hash for reproducibility

import pandas as pd
import joblib
import os
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from preprocessing import load_and_clean_data, split_data

from resampling import apply_smote
from evaluation import evaluate_model_cv
 

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    print(f"\n Training {model_name}...")
    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()
    train_time = end_time - start_time

    y_pred = model.predict(X_test)

    print(f"\n {model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n {model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred)
    return model, f1, train_time, model.get_params(), y_pred


if __name__ == "__main__":
    print(" Loading and preprocessing data...")
    X, y, scaler = load_and_clean_data("../data/creditcard.csv", return_scaler=True)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, y_train = apply_smote(X_train, y_train)  # as classes are highly imbalanced.

    # Models to compare
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_model = None
    best_f1 = 0
    best_model_name = ""
    best_train_time = None
    best_hyperparams = None
    best_report_dict = None

    for name, model in models.items():
        trained_model, f1, train_time, hyperparams, y_pred = train_and_evaluate_model(
            model, name, X_train, X_test, y_train, y_test
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model = trained_model
            best_model_name = name
            best_train_time = train_time
            best_hyperparams = hyperparams
            best_report_dict = classification_report(y_test, y_pred, output_dict=True)

    evaluate_model_cv(best_model, X.values, y.values)

    # Save model
    os.makedirs("models", exist_ok=True)
    best_model_path = f"models/{best_model_name.lower()}.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"\n Best model: {best_model_name} with F1 score = {best_f1:.4f}")
    print(f" Model saved to: {best_model_path}")

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print(" Scaler saved to: models/scaler.pkl")

    # Save classification report
    with open("models/classification_report.json", "w") as f:
        json.dump(best_report_dict, f, indent=4)
    print(" Classification report saved to: models/classification_report.json")

    # Save metadata (train time and hyperparameters)
    metadata = {
        "model_name": best_model_name,
        "f1_score": round(best_f1, 4),
        "train_time_sec": round(best_train_time, 4),
        "hyperparameters": best_hyperparams
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(" Metadata saved to: models/metadata.json")
