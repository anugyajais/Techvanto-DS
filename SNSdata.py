# confusion matrix for the Social Network Ads data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# 1) Load & preprocess
sns_data = pd.read_csv("Datasets/Social_Network_Ads.csv")
sns_data.head()
sns_data.shape
sns_data.columns
sns_data.info()
sns_data.describe()
sns_data.head()
sns_data.drop("User ID", axis=1, inplace=True)
sns_data["Gender"] = LabelEncoder().fit_transform(sns_data["Gender"])  # Male=1, Female=0

# 2) Split X/y & scale features
X = sns_data.drop("Purchased", axis=1)
y = sns_data["Purchased"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

# 3) Train models
log_reg   = LogisticRegression(random_state=42).fit(X_train, y_train)
ridge_reg = Ridge().fit(X_train, y_train)

# 4) Make predictions
y_pred_log   = log_reg.predict(X_test)
y_pred_ridge = np.where(ridge_reg.predict(X_test) >= 0.5, 1, 0)
y_pred_ridge

# 5) Evaluation helper
def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

evaluate("Logistic Regression", y_test, y_pred_log)
evaluate("Ridge (thresholded)",  y_test, y_pred_ridge)

# 6) Interactive predictor
def predict_purchase():
    gender = input("Gender (M/F): ").strip().lower()
    age    = int(input("Age: "))
    sal    = int(input("Estimated Salary: "))
    g_enc  = 1 if gender.startswith("m") else 0

    x_new = np.array([[g_enc, age, sal]], dtype=float)
    x_new[:, 1:] = scaler.transform(x_new[:, 1:])
    pred  = log_reg.predict(x_new)[0]
    prob  = log_reg.predict_proba(x_new)[0, 1]

    label = "Will Purchase" if pred == 1 else "Will Not Purchase"
    print(f"{label} (P={prob:.2f})")

# To run:
# predict_purchase()







