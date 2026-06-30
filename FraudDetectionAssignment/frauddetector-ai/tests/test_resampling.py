# tests/test_resampling.py

import pandas as pd
from sklearn.datasets import make_classification
from src.resampling import apply_smote

def test_apply_smote_balances_classes():
    X, y = make_classification(n_samples=500, n_features=5, weights=[0.9, 0.1], random_state=42)
    X_resampled, y_resampled = apply_smote(X, y)

    # Assert class balance
    unique, counts = pd.Series(y_resampled).value_counts().sort_index().values, pd.Series(y_resampled).value_counts().sort_index().index
    assert unique[0] == unique[1], "Classes should be balanced after SMOTE"
