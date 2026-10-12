import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "iris.csv"

# Store MLflow experiment data inside this project folder.
mlflow.set_tracking_uri((BASE_DIR / "mlruns").as_uri())
mlflow.set_experiment("Iris Neural Network")

# 1. Load data
data = pd.read_csv(DATA_PATH)
assert len(data) == 150, f"Expected 150 rows, found {len(data)}"

X = data.drop(columns=["species"])
y = data["species"]

# 2. Keep the original train/test split unchanged
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. Keep the original model configuration unchanged
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(10, 10),
        activation="relu",
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )),
])

# 4. Train, evaluate, and track this run in MLflow
with mlflow.start_run(run_name="iris-mlp-baseline"):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=sorted(y.unique())))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Log model configuration and data-split details
    mlflow.log_params({
        "model": "MLPClassifier",
        "hidden_layer_sizes": "(10, 10)",
        "activation": "relu",
        "solver": "lbfgs",
        "max_iter": 5000,
        "random_state": 42,
        "test_size": 0.2,
        "stratify": True,
        "scaler": "StandardScaler",
        "dataset_rows": len(data),
    })

    # Log evaluation metrics
    mlflow.log_metrics({
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(accuracy),
        "test_precision_weighted": float(precision),
        "test_recall_weighted": float(recall),
        "test_f1_weighted": float(f1),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
    })

    # Preserve the existing model artifact
    joblib.dump(model, BASE_DIR / "model.joblib")

    metrics = {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(accuracy),
        "test_precision_weighted": float(precision),
        "test_recall_weighted": float(recall),
        "test_f1_weighted": float(f1),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "model": "MLPClassifier",
        "hidden_layers": [10, 10],
        "random_state": 42,
    }

    (BASE_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Save the trained pipeline in MLflow as well
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
    )

    # Keep the original 100% accuracy requirement
    if accuracy < 1.0:
        raise SystemExit(
            "Pipeline failed: expected 100% test accuracy, "
            f"got {accuracy * 100:.2f}%"
        )

    print("\nSUCCESS: 100% accuracy target passed.")
    print("Saved model.joblib, metrics.json, and MLflow model.")

    print(f"MLflow run ID: {mlflow.active_run().info.run_id}")