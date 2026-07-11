import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SAMPLE_INPUT = {
    "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 246,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 3,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Train a tiny throwaway model so the test doesn't depend on
    # a pre-existing models/model.pkl artifact.
    import joblib
    from preprocess import build_pipeline
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(1)
    from preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = pd.DataFrame({c: rng.integers(0, 5, 40).astype(float) for c in cols})
    y = rng.integers(0, 2, 40)
    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(pipeline, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    import app as app_module
    import importlib
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_predict_returns_prediction_and_confidence(client):
    resp = client.post("/predict", json=SAMPLE_INPUT)
    assert resp.status_code == 200
    body = resp.get_json()
    assert "prediction" in body
    assert "confidence" in body
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_missing_fields(client):
    bad_input = {"age": 55}
    resp = client.post("/predict", json=bad_input)
    assert resp.status_code == 400
