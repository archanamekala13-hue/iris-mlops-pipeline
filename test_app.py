
from pathlib import Path

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_model_artifact_exists():
    assert (Path(__file__).parent / "model.joblib").exists()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_setosa_prediction():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })

    assert response.status_code == 200
    assert response.json()["predicted_species"] == "setosa"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["test_accuracy"] == 1.0
    assert response.json()["train_accuracy"] >= 0.95


def test_versicolor_and_virginica_predictions():
    versicolor = client.post("/predict", json={
        "sepal_length": 6.4, "sepal_width": 3.2,
        "petal_length": 4.5, "petal_width": 1.5,
    })
    virginica = client.post("/predict", json={
        "sepal_length": 6.3, "sepal_width": 3.3,
        "petal_length": 6.0, "petal_width": 2.5,
    })
    assert versicolor.json()["predicted_species"] == "versicolor"
    assert virginica.json()["predicted_species"] == "virginica"


def test_invalid_input_rejected():
    response = client.post("/predict", json={
        "sepal_length": -1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2,
    })
    assert response.status_code == 422


def test_monitoring_metrics_endpoint():
    client.post("/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2,
    })
    response = client.get("/monitoring/metrics")
    assert response.status_code == 200
    body = response.text
    assert "iris_predictions_total" in body
    assert "iris_http_requests_total" in body
    assert "iris_model_test_accuracy" in body


def test_monitoring_summary_endpoint():
    response = client.get("/monitoring/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["predictions_served"] >= 1
    assert set(data["drift_check"]) == {
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    }
