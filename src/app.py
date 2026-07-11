"""
Flask model-serving API.
  GET  /health   -> liveness check
  POST /predict  -> {"prediction": 0|1, "confidence": float}

Run locally:    python src/app.py
Run in Docker:  see Dockerfile
"""
import os
import logging
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics

from preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES

REQUIRED_FIELDS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(
    os.path.dirname(__file__), "..", "models", "model.pkl"))

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
)
logger = logging.getLogger("heart-api")

app = Flask(__name__)
metrics = PrometheusMetrics(app)

_model = None


def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        logger.warning(f"Rejected request, missing fields: {missing}")
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = pd.DataFrame([{f: payload[f] for f in REQUIRED_FIELDS}])
        model = get_model()
        proba = model.predict_proba(X)[0]
        pred = int(proba.argmax())
        confidence = float(proba[pred])
        logger.info(f"Prediction={pred} confidence={confidence:.3f}")
        return jsonify({"prediction": pred, "confidence": round(confidence, 4)}), 200
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": "inference failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
