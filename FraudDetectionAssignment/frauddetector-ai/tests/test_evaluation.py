# tests/test_evaluation.py

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from src.evaluation import evaluate_model_cv


def test_evaluate_model_cv_runs_without_error(monkeypatch):
    """
    Tests if evaluate_model_cv runs correctly on a small synthetic dataset,
    and doesn't raise exceptions during fold-wise evaluation.
    """

    # Generate imbalanced binary classification dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42
    )

    model = LogisticRegression(solver='liblinear', random_state=42)

    # Monkeypatch plt.show to prevent actual plotting during test
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    # Run with global metrics off to keep test fast and robust
    try:
        evaluate_model_cv(model, X, y, n_splits=3, global_metrics=False)
    except Exception as e:
        assert False, f"evaluate_model_cv raised an exception: {e}"
