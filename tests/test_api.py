"""
Tests for the FastAPI credit risk API
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import json

from src.api.main import app
from src.api.pydantic_models import CustomerFeatures


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_customer():
    """Sample customer data for testing"""
    return {
        "recency_days": 45.0,
        "transaction_frequency": 12.0,
        "total_monetary_value": 1500.50
    }


class TestHealthCheck:
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data


class TestModelInfo:
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "performance" in data
        assert "basel_ii_compliant" in data


class TestSinglePrediction:
    def test_valid_prediction(self, client, sample_customer):
        """Test valid single prediction"""
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200
        data = response.json()
        assert "is_high_risk" in data
        assert "risk_score" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert 0 <= data["risk_score"] <= 1
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert 0 <= data["confidence"] <= 1
    
    def test_invalid_recency_days(self, client, sample_customer):
        """Test with invalid recency days"""
        invalid_data = sample_customer.copy()
        invalid_data["recency_days"] = 400  # Out of range
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_missing_field(self, client):
        """Test with missing required field"""
        invalid_data = {
            "recency_days": 45.0,
            "transaction_frequency": 12.0
            # Missing total_monetary_value
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422


class TestBatchPrediction:
    def test_valid_batch_prediction(self, client):
        """Test valid batch prediction"""
        batch_data = {
            "customers": [
                {
                    "recency_days": 45.0,
                    "transaction_frequency": 12.0,
                    "total_monetary_value": 1500.50
                },
                {
                    "recency_days": 30.0,
                    "transaction_frequency": 20.0,
                    "total_monetary_value": 2500.00
                }
            ]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "processing_time_ms" in data
        assert "total_customers" in data
        assert len(data["predictions"]) == 2
    
    def test_empty_batch(self, client):
        """Test with empty batch"""
        batch_data = {"customers": []}
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 422  # Validation error
    
    def test_large_batch(self, client):
        """Test with batch exceeding limit"""
        customers = []
        for i in range(1001):  # Exceeds 1000 limit
            customers.append({
                "recency_days": 30.0,
                "transaction_frequency": 10.0,
                "total_monetary_value": 1000.0
            })
        
        batch_data = {"customers": customers}
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 422  # Validation error


class TestFeaturesDescription:
    def test_features_description(self, client):
        """Test features description endpoint"""
        response = client.get("/features/description")
        assert response.status_code == 200
        data = response.json()
        assert "recency_days" in data
        assert "transaction_frequency" in data
        assert "total_monetary_value" in data