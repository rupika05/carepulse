"""
test_prediction.py — Tests for model inference and API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.prediction_service import prediction_service

client = TestClient(app)

SAMPLE_RED_PATIENT = {
    "patient_id": "TEST_RED_001",
    "age": 45,
    "gender": "M",
    "respiratory_rate": 36,
    "pulse_rate": 130,
    "systolic_bp": 85,
    "spo2": 82,
    "gcs_total": 9,
    "avpu": "Pain",
    "respiratory_effort": "Labored",
    "radial_pulse_present": "No",
    "capillary_refill_sec": 3.8,
    "major_hemorrhage": "Yes",
    "ambulatory": "No",
    "injury_type": "Penetrating",
    "injury_severity": "Critical",
    "mechanism_of_injury": "GSW"
}

SAMPLE_GREEN_PATIENT = {
    "patient_id": "TEST_GREEN_001",
    "age": 28,
    "gender": "F",
    "respiratory_rate": 18,
    "pulse_rate": 72,
    "systolic_bp": 120,
    "spo2": 99,
    "gcs_total": 15,
    "avpu": "Alert",
    "respiratory_effort": "Normal",
    "radial_pulse_present": "Yes",
    "capillary_refill_sec": 1.2,
    "major_hemorrhage": "No",
    "ambulatory": "Yes",
    "injury_type": "None",
    "injury_severity": "Minor",
    "mechanism_of_injury": "Fall"
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_prediction_endpoint_red():
    # Ensure model loaded
    if not prediction_service.is_loaded:
        prediction_service.load()

    response = client.post("/predict", json=SAMPLE_RED_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "TEST_RED_001"
    assert data["triage"] in ["RED", "YELLOW", "GREEN", "BLACK"]
    assert "probabilities" in data
    assert "expected_costs" in data
    # Sum of probabilities ~ 1
    total_prob = sum(data["probabilities"].values())
    assert pytest.approx(total_prob, 0.01) == 1.0


def test_prediction_endpoint_validation_error():
    # Invalid GCS (must be <= 15)
    bad_patient = SAMPLE_RED_PATIENT.copy()
    bad_patient["gcs_total"] = 25

    response = client.post("/predict", json=bad_patient)
    assert response.status_code == 422  # Pydantic validation error


def test_prediction_endpoint_appends_to_csv():
    if not prediction_service.is_loaded:
        prediction_service.load()

    patient_payload = {
        "subject_id": "TEST_CSV_PATIENT_99",
        "gender": "F",
        "heartrate": 110,
        "resprate": 24,
        "sbp": 95,
        "o2sat": 92,
        "chiefcomplaint": "FRACTURE",
    }
    response = client.post("/predict", json=patient_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["saved_to_csv"] is True
    assert data["total_assessed_records"] is not None
    assert data["total_assessed_records"] > 0

    # Verify download endpoint
    dl_response = client.get("/download-assessed-patients")
    assert dl_response.status_code == 200
    assert "TEST_CSV_PATIENT_99" in dl_response.text

    # Verify analytics endpoint
    list_response = client.get("/assessed-patients")
    assert list_response.status_code == 200
    assert list_response.json()["total"] > 0

