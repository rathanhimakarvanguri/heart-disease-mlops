# Heart Disease Risk Prediction — MLOps Pipeline

End-to-end pipeline: data → EDA → model training (MLflow-tracked) →
packaging → CI/CD → Docker → Kubernetes → monitoring.

> **Before submitting:** this scaffold is a *starting point*, not a
> final answer — adapt the EDA commentary, try different models/params,
> restructure the API or manifests to your taste. The assignment's
> academic-integrity note flags near-identical submissions, so make
> the architecture decisions and write-up genuinely your own.

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python data/download_data.py          # downloads UCI dataset (falls back to synthetic if offline)
```

## 2. EDA

```bash
python notebooks/eda.py               # writes plots to screenshots/
```

## 3. Train + track experiments

```bash
cd src && python train.py
mlflow ui                              # browse http://localhost:5000
```
Trains Logistic Regression + Random Forest with GridSearchCV, logs
params/metrics/ROC & confusion-matrix plots per run, saves the best
pipeline to `models/model.pkl`.

## 4. Tests

```bash
pytest tests/ -v
```

## 5. Run the API locally

```bash
cd src && python app.py                # http://localhost:5000
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":246,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.2,"slope":1,"ca":0,"thal":3}'
```

## 6. Docker

```bash
docker build -t heart-disease-api:latest .
docker run -p 5000:5000 heart-disease-api:latest
curl http://localhost:5000/health
```

## 7. CI/CD

`.github/workflows/ci-cd.yml` runs on every push/PR: lint (flake8) →
unit tests → dataset generation → model training → Docker build →
container smoke test. Any failed step stops the pipeline with logs
visible in the Actions tab (satisfies "fail on error" requirement).

## 8. Deploy to Kubernetes

```bash
# Minikube (local):
minikube start
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
minikube service heart-disease-api-service
```
For a public cloud (GKE/EKS/AKS): push the image to that provider's
container registry, update the `image:` field in `k8s/deployment.yaml`,
then `kubectl apply -f k8s/`.

## 9. Monitoring

- Every `/predict` call is logged (structured JSON) — see `app.py`'s
  logging config; ship these to `kubectl logs` or a log aggregator.
- For metrics: add `prometheus_flask_exporter` to expose `/metrics`,
  then scrape it with a Prometheus + Grafana stack (or use a hosted
  APM). Document whichever you set up with a dashboard screenshot.

## Architecture

```
 UCI CSV → clean_data() → ColumnTransformer(impute+scale+onehot)
                              │
                    GridSearchCV(LogReg / RandomForest)
                              │
                     MLflow tracking (params/metrics/artifacts)
                              │
                  models/model.pkl  (preprocessing + model, one artifact)
                              │
                     Flask app.py  (/predict, /health)
                              │
                    Docker image ──► GitHub Actions CI/CD
                              │
                 Kubernetes Deployment + Service (LoadBalancer)
```

## Repo layout
```
data/download_data.py     # dataset acquisition
src/preprocess.py         # cleaning + reusable sklearn pipeline
src/train.py              # training + MLflow tracking
src/app.py                # Flask serving API
notebooks/eda.py          # EDA plots
tests/                    # pytest unit tests
Dockerfile
requirements.txt
.github/workflows/ci-cd.yml
k8s/deployment.yaml, k8s/service.yaml
screenshots/              # EDA + deployment screenshots for report
```
