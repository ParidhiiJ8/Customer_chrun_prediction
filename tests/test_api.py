import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


# Load the sample request from the JSON file.
SAMPLE_PATH = (
    Path(__file__).parent / "sample_payload.json"
)

with open(SAMPLE_PATH, encoding="utf-8") as file:
    SAMPLE_PAYLOAD = json.load(file)


def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["status"] == "running"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "ok"

    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():

    response = client.post(
        "/predict",
        json=SAMPLE_PAYLOAD
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["Yes", "No"]

    assert "churn_probability" in result

    assert (
        0.0
        <= result["churn_probability"]
        <= 1.0
    )


def test_missing_required_feature():

    invalid_payload = SAMPLE_PAYLOAD.copy()

    # tenure is a required field.
    invalid_payload.pop("tenure")

    response = client.post(
        "/predict",
        json=invalid_payload
    )

    assert response.status_code == 422