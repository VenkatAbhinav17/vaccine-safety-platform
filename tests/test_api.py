"""
Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform
Automated API Verification & Quality Assurance Suite
File: tests/test_api.py

Validates:
1. System Health Check Endpoint (GET /)
2. Safe / Low-Risk Patient Inference (POST /predict)
3. Moderate / High-Risk Patient Reactogenicity Flagging (POST /predict)
4. Strict Pydantic Payload Validation & Error Handling (HTTP 422)
5. Contract Compliance (Keys: risk_probability, risk_level, recommendation)
"""

import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

# Ensure parent directory and backend are in python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import app
from backend.schemas import PatientData, PredictionResponse

client = TestClient(app)


def test_health_check_endpoint():
    """Verify GET / returns 200 OK and valid health telemetry."""
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert data.get("model_loaded") is True, "Trained model.pkl should be loaded upon startup"
    assert "version" in data
    print("[PASS] Health check endpoint (GET /) is healthy and model is loaded.")


def test_predict_low_risk_candidate():
    """
    Verify POST /predict correctly identifies low-risk candidate:
    - Young adult (age 24, BMI 21.8)
    - No allergies, no underlying conditions
    - Normal WBC (5.4 k/uL)
    - Adapted/Booster dosage (0.25 mL)
    """
    payload = {
        "age": 24,
        "bmi": 21.8,
        "has_allergies": 0,
        "prior_covid_infection": 1,
        "underlying_conditions": 0,
        "biomarker_wbc_count": 5.4,
        "proposed_dosage_ml": 0.25
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "risk_probability" in data
    assert "risk_level" in data
    assert "recommendation" in data
    
    # Contract validation with Pydantic
    parsed = PredictionResponse(**data)
    assert 0.0 <= parsed.risk_probability <= 1.0
    assert parsed.risk_level == "Low"
    assert "safe" in parsed.recommendation.lower() or "suitable" in parsed.recommendation.lower()
    print(f"[PASS] Low-risk prediction passed (prob: {parsed.risk_probability}, level: {parsed.risk_level}).")


def test_predict_high_risk_candidate():
    """
    Verify POST /predict flags high reactogenicity risk:
    - Age 56, BMI 33.2
    - Allergies positive, Underlying chronic conditions positive
    - Elevated WBC count (12.6 k/uL)
    - High primary series dosage (0.50 mL)
    """
    payload = {
        "age": 56,
        "bmi": 33.2,
        "has_allergies": 1,
        "prior_covid_infection": 1,
        "underlying_conditions": 1,
        "biomarker_wbc_count": 12.6,
        "proposed_dosage_ml": 0.50
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    parsed = PredictionResponse(**data)
    assert parsed.risk_level in ["High", "Moderate"]
    assert parsed.risk_probability >= 0.60
    assert "0.25" in parsed.recommendation or "caution" in parsed.recommendation.lower()
    print(f"[PASS] High-risk prediction passed (prob: {parsed.risk_probability}, level: {parsed.risk_level}).")


def test_pydantic_validation_age_out_of_bounds():
    """Verify HTTP 422 on invalid age (< 18 or > 100)."""
    invalid_payload = {
        "age": 14,  # Under 18
        "bmi": 22.0,
        "has_allergies": 0,
        "prior_covid_infection": 0,
        "underlying_conditions": 0,
        "biomarker_wbc_count": 6.0,
        "proposed_dosage_ml": 0.30
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422, f"Expected 422 for pediatric age, got {response.status_code}"
    print("[PASS] Pydantic validation rejected invalid age (14) with HTTP 422.")


def test_pydantic_validation_dosage_out_of_bounds():
    """Verify HTTP 422 on invalid dosage (< 0.1 mL or > 1.0 mL)."""
    invalid_payload = {
        "age": 35,
        "bmi": 24.0,
        "has_allergies": 0,
        "prior_covid_infection": 0,
        "underlying_conditions": 0,
        "biomarker_wbc_count": 6.0,
        "proposed_dosage_ml": 2.50  # Out of range
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422, f"Expected 422 for dosage 2.50 mL, got {response.status_code}"
    print("[PASS] Pydantic validation rejected invalid dosage (2.50 mL) with HTTP 422.")


def test_pydantic_validation_missing_fields():
    """Verify HTTP 422 when required clinical features are omitted."""
    incomplete_payload = {
        "age": 40,
        "bmi": 25.0
        # Missing required biomarker, dosage, conditions
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
    print("[PASS] Pydantic validation rejected incomplete payload with HTTP 422.")


def run_all_tests():
    """Self-test runner executing all test cases and summarizing results."""
    print("\n" + "=" * 65)
    print("RUNNING AUTOMATED TEST SUITE: MODERNA mRNA SAFETY PLATFORM")
    print("=" * 65)
    
    tests = [
        ("Health Check Endpoint (GET /)", test_health_check_endpoint),
        ("Low Risk Prediction (POST /predict)", test_predict_low_risk_candidate),
        ("High Risk Reactogenicity Alert (POST /predict)", test_predict_high_risk_candidate),
        ("Schema Validation: Age Bounds (HTTP 422)", test_pydantic_validation_age_out_of_bounds),
        ("Schema Validation: Dosage Bounds (HTTP 422)", test_pydantic_validation_dosage_out_of_bounds),
        ("Schema Validation: Missing Fields (HTTP 422)", test_pydantic_validation_missing_fields),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n--> Running test: {name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1
            
    print("\n" + "=" * 65)
    print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED (TOTAL: {len(tests)})")
    print("=" * 65 + "\n")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
