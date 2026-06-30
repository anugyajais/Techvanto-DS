# src/evaluation.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)


def evaluate_model_cv(model, X, y, n_splits=5, global_metrics=True):
    # internally using X and y as numpy array not pandas dataframe
    X = np.array(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1_scores, roc_aucs, pr_aucs = [], [], []

    print(f"\nPerforming {n_splits}-Fold Cross-Validation:\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)

        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

        print(f"Fold {fold}: F1 = {f1:.4f}, ROC AUC = {roc_auc:.4f}, PR AUC = {pr_auc:.4f}")

    print("\nAverage Scores:")
    print(f"F1 Score       : {np.mean(f1_scores):.4f}")
    print(f"ROC AUC        : {np.mean(roc_aucs):.4f}")
    print(f"PR AUC         : {np.mean(pr_aucs):.4f}")

    # Optional global metrics (may fail in highly imbalanced datasets)
    if global_metrics:
        try:
            print("\nGenerating global metrics using cross_val_predict...")
            y_prob_global = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
            y_pred_global = (y_prob_global > 0.5).astype(int)

            print("\nGlobal Confusion Matrix:")
            print(confusion_matrix(y, y_pred_global))

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y, y_prob_global)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label="ROC Curve")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Global ROC Curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plot Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y, y_prob_global)
            plt.figure(figsize=(6, 4))
            plt.plot(recall, precision, label="PR Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Global Precision-Recall Curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except ValueError as e:
            print(f"\n[Warning] Skipping global predictions: {e}")
