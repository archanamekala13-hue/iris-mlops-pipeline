
import json
import logging
import math
import threading
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "metrics.json"

# Load the model logged in MLflow.

mlflow.set_tracking_uri((BASE_DIR / "mlruns").as_uri())

# Find the latest successful training run.
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Iris Neural Network")

if experiment is None:
    raise RuntimeError(
        "MLflow experiment not found. Run python train.py first."
    )

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

if not runs:
    raise RuntimeError(
        "No successful MLflow run found. Run python train.py first."
    )

run_id = runs[0].info.run_id
model_path = (
    BASE_DIR
    / "mlruns"
    / experiment.experiment_id
    / run_id
    / "artifacts"
    / "iris_model"
)

model = mlflow.sklearn.load_model(str(model_path))

app = FastAPI(
    title="Iris Flower Classifier",
    description="Predict Iris species using a trained neural network tracked in MLflow.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("iris-api")

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
MONITORING_PATH = "/monitoring/metrics"

REQUEST_COUNT = Counter(
    "iris_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "iris_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["path"],
)
PREDICTION_COUNT = Counter(
    "iris_predictions_total",
    "Predictions served, by predicted species",
    ["species"],
)
PREDICTION_ERRORS = Counter(
    "iris_prediction_errors_total",
    "Predictions that raised an error",
)
PREDICTION_LATENCY = Histogram(
    "iris_prediction_duration_seconds",
    "Time spent inside model.predict",
)
FEATURE_VALUE = Histogram(
    "iris_input_feature_value",
    "Distribution of incoming feature values (for drift checks)",
    ["feature"],
    buckets=(0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 10),
)
MODEL_TEST_ACCURACY = Gauge("iris_model_test_accuracy", "Test accuracy from training")
MODEL_TRAIN_ACCURACY = Gauge("iris_model_train_accuracy", "Training accuracy from training")

if METRICS_PATH.exists():
    _training_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    MODEL_TEST_ACCURACY.set(_training_metrics.get("test_accuracy", 0))
    MODEL_TRAIN_ACCURACY.set(_training_metrics.get("train_accuracy", 0))

# Simple data-drift check: compare the live mean of each feature with the
# mean/std of the training data (iris.csv).
_train_data = pd.read_csv(BASE_DIR / "iris.csv")
TRAIN_STATS = {
    f: {"mean": float(_train_data[f].mean()), "std": float(_train_data[f].std())}
    for f in FEATURES
}
_live = {f: {"n": 0, "sum": 0.0} for f in FEATURES}
_live_species = {}
_lock = threading.Lock()
STARTED_AT = time.time()
DRIFT_MIN_SAMPLES = 30
DRIFT_Z_THRESHOLD = 3.0


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        if path != MONITORING_PATH:
            REQUEST_COUNT.labels(request.method, path, str(status)).inc()
            REQUEST_LATENCY.labels(path).observe(time.perf_counter() - start)


class FlowerFeatures(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {
        "message": "Iris Neural Network API is running!",
        "docs": "/docs",
        "model_source": "MLflow",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_source": "MLflow",
    }


@app.get("/metrics")
def metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {"message": "Metrics not found"}


@app.post("/predict")
def predict(flower: FlowerFeatures):
    values = [getattr(flower, f) for f in FEATURES]
    features = pd.DataFrame([values], columns=FEATURES)

    try:
        start = time.perf_counter()
        prediction = str(model.predict(features)[0])
        duration = time.perf_counter() - start
    except Exception:
        PREDICTION_ERRORS.inc()
        logger.exception("prediction_failed")
        raise

    PREDICTION_LATENCY.observe(duration)
    PREDICTION_COUNT.labels(prediction).inc()
    for name, value in zip(FEATURES, values):
        FEATURE_VALUE.labels(name).observe(value)

    with _lock:
        for name, value in zip(FEATURES, values):
            _live[name]["n"] += 1
            _live[name]["sum"] += value
        _live_species[prediction] = _live_species.get(prediction, 0) + 1

    # One structured JSON line per prediction (visible in Render / Docker logs)
    logger.info(json.dumps({
        "event": "prediction",
        "input": dict(zip(FEATURES, values)),
        "predicted_species": prediction,
        "latency_ms": round(duration * 1000, 3),
    }))

    return {"predicted_species": prediction}


@app.get(MONITORING_PATH, include_in_schema=False)
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/summary")
def monitoring_summary():
    """Human-readable monitoring view: traffic, prediction mix and drift check."""
    with _lock:
        n = _live[FEATURES[0]]["n"]
        drift = {}
        for name in FEATURES:
            count = _live[name]["n"]
            train = TRAIN_STATS[name]
            if count == 0:
                drift[name] = {"live_mean": None, "train_mean": round(train["mean"], 3),
                               "z_score": None, "status": "no_data"}
                continue
            live_mean = _live[name]["sum"] / count
            z = (live_mean - train["mean"]) / (train["std"] / math.sqrt(count))
            if count < DRIFT_MIN_SAMPLES:
                status = f"need_{DRIFT_MIN_SAMPLES}_samples"
            else:
                status = "DRIFT" if abs(z) > DRIFT_Z_THRESHOLD else "ok"
            drift[name] = {"live_mean": round(live_mean, 3),
                           "train_mean": round(train["mean"], 3),
                           "z_score": round(z, 2), "status": status}
        species = dict(_live_species)

    return {
        "uptime_seconds": round(time.time() - STARTED_AT),
        "predictions_served": n,
        "predictions_by_species": species,
        "model_test_accuracy": MODEL_TEST_ACCURACY._value.get(),
        "model_train_accuracy": MODEL_TRAIN_ACCURACY._value.get(),
        "drift_check": drift,
    }
