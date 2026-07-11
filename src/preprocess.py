"""Reusable preprocessing pipeline: cleaning, imputation, scaling, encoding.

Kept separate from training so it can be unit-tested and reused
identically at inference time (loaded from the saved pickle/joblib
artifact) — this is what gives us reproducibility.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates, coerce dtypes. Missing values are handled
    inside the pipeline (SimpleImputer) so they survive to inference time
    consistently rather than being dropped here."""
    df = df.drop_duplicates().reset_index(drop=True)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_pipeline(model) -> Pipeline:
    """Wrap a scikit-learn estimator together with its preprocessing
    so `pipeline.fit(X, y)` / `pipeline.predict(X)` is the single
    reproducible artifact we ship."""
    numeric_transform = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_transform = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transform, NUMERIC_FEATURES),
        ("cat", categorical_transform, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def get_X_y(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y
