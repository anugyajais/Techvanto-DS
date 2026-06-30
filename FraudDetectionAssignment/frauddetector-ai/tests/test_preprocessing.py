# tests/test_preprocessing.py

import pandas as pd
from src.preprocessing import load_and_clean_data, split_data

def test_load_and_clean_data_shape():
    X, y = load_and_clean_data("data/creditcard.csv")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert not X.isnull().any().any()  # no missing values

def test_split_data_shapes():
    X, y = load_and_clean_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    total_samples = len(y)
    assert len(X_train) + len(X_test) == total_samples
    assert len(y_train) + len(y_test) == total_samples
