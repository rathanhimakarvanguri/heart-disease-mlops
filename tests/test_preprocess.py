import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_data, build_pipeline, get_X_y, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from sklearn.linear_model import LogisticRegression


def _sample_df(n=20):
    rng = np.random.default_rng(0)
    data = {c: rng.integers(0, 5, n).astype(float) for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    data["target"] = rng.integers(0, 2, n)
    df = pd.DataFrame(data)
    df.loc[0, "chol"] = np.nan  # inject missing value
    return df


def test_clean_data_removes_duplicates():
    df = _sample_df()
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(df_dup)
    assert len(cleaned) == len(df)


def test_clean_data_coerces_numeric():
    df = _sample_df()
    df["age"] = df["age"].astype(str)
    cleaned = clean_data(df)
    assert pd.api.types.is_numeric_dtype(cleaned["age"])


def test_pipeline_handles_missing_values():
    df = clean_data(_sample_df())
    X, y = get_X_y(df)
    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(X, y)  # should not raise despite NaN in 'chol'
    preds = pipeline.predict(X)
    assert len(preds) == len(y)


def test_pipeline_predict_proba_shape():
    df = clean_data(_sample_df())
    X, y = get_X_y(df)
    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(y), 2)
