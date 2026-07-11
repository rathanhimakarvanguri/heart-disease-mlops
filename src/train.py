"""
Train + tune Logistic Regression and Random Forest, tracked in MLflow.
Run: python src/train.py
Then: mlflow ui   (to browse runs at http://localhost:5000)

Personalize: try a third model, change the CV fold count, adjust the
param grids, or add SHAP feature-importance plots as an extra artifact.
"""
import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay,
)

from preprocess import load_data, clean_data, build_pipeline, get_X_y

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_experiment("heart-disease-classification")

MODEL_GRIDS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10]},
    ),
}


def evaluate_and_log(name, best_pipeline, X_test, y_test, X, y):
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring="roc_auc")

    mlflow.log_metrics({
        "accuracy": acc, "precision": prec, "recall": rec, "roc_auc": auc,
        "cv_roc_auc_mean": cv_scores.mean(), "cv_roc_auc_std": cv_scores.std(),
    })

    # ROC curve artifact
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(best_pipeline, X_test, y_test, ax=ax)
    roc_path = f"/tmp/{name}_roc.png"
    fig.savefig(roc_path)
    plt.close(fig)
    mlflow.log_artifact(roc_path)

    # Confusion matrix artifact
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(best_pipeline, X_test, y_test, ax=ax)
    cm_path = f"/tmp/{name}_confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)
    mlflow.log_artifact(cm_path)

    print(f"[{name}] acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} "
          f"auc={auc:.3f} cv_auc={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    return auc


def main():
    df = clean_data(load_data(DATA_PATH))
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_overall = None
    best_auc = -1

    for name, (estimator, param_grid) in MODEL_GRIDS.items():
        with mlflow.start_run(run_name=name):
            pipeline = build_pipeline(estimator)
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
            grid.fit(X_train, y_train)

            mlflow.log_params(grid.best_params_)
            mlflow.log_param("model_type", name)

            auc = evaluate_and_log(name, grid.best_estimator_, X_test, y_test, X, y)
            mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")

            if auc > best_auc:
                best_auc = auc
                best_overall = (name, grid.best_estimator_)

    # Save the best pipeline (preprocessing + model) as the reusable artifact
    name, pipeline = best_overall
    out_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(pipeline, out_path)
    print(f"\nBest model: {name} (AUC={best_auc:.3f}) saved to {out_path}")


if __name__ == "__main__":
    main()
