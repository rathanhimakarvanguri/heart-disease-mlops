"""
Download the UCI Heart Disease dataset (Cleveland subset).

Usage:
    python data/download_data.py

If the UCI server is unreachable (offline/firewalled environment),
falls back to sklearn's synthetic generator so the rest of the
pipeline can still be exercised end-to-end.
"""
import os
import pandas as pd
import numpy as np

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

OUT_PATH = os.path.join(os.path.dirname(__file__), "heart.csv")


def download_real():
    df = pd.read_csv(UCI_URL, header=None, names=COLUMNS, na_values="?")
    # Original target is 0 (no disease) - 4 (severity); binarize per assignment spec
    df["target"] = (df["target"] > 0).astype(int)
    return df


def generate_synthetic(n=303, seed=42):
    """Fallback so the pipeline is runnable without network access."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.integers(94, 201, n),
        "chol": rng.integers(126, 565, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 203, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": np.round(rng.uniform(0, 6.2, n), 1),
        "slope": rng.integers(0, 3, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })
    # a few missing values to exercise the cleaning step
    for col in ["ca", "thal"]:
        idx = rng.choice(n, size=4, replace=False)
        df.loc[idx, col] = np.nan
    # weakly correlated synthetic target
    score = (
        0.03 * df["age"] + 1.5 * df["cp"] + 0.01 * df["chol"]
        - 0.02 * df["thalach"] + 0.8 * df["exang"] + rng.normal(0, 2, n)
    )
    df["target"] = (score > score.median()).astype(int)
    return df


if __name__ == "__main__":
    try:
        df = download_real()
        print("Downloaded real UCI Heart Disease dataset.")
    except Exception as e:
        print(f"Could not reach UCI repository ({e}). Using synthetic fallback.")
        df = generate_synthetic()

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
