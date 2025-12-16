"""
Tests for the credit risk model
"""

import pytest
import pickle
import numpy as np
import pandas as pd
import json
import os

from sklearn.metrics import roc_auc_score


class TestModelFiles:
    def test_model_files_exist(self):
        """Test that all required model files exist"""
        assert os.path.exists("models/best_model/model.pkl")
        assert os.path.exists("models/best_model/preprocessor.pkl")
        assert os.path.exists("models/best_model/metadata.json")
    
    def test_load_model(self):
        """Test that model can be loaded"""
        with open("models/best_model/model.pkl", "rb") as f:
            model = pickle.load(f)
        assert model is not None
        
        # Test that model has predict method
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
    
    def test_load_preprocessor(self):
        """Test that preprocessor can be loaded"""
        with open("models/best_model/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        assert preprocessor is not None
    
    def test_metadata(self):
        """Test metadata structure"""
        with open("models/best_model/metadata.json", "r") as f:
            metadata = json.load(f)
        
        assert "model_name" in metadata
        assert "training_date" in metadata
        assert "performance" in metadata
        assert "features" in metadata
        assert "basel_ii_compliance" in metadata


class TestModelPerformance:
    def test_basel_compliance(self):
        """Test that model meets Basel II requirements"""
        with open("models/best_model/metadata.json", "r") as f:
            metadata = json.load(f)
        
        performance = metadata.get("performance", {})
        basel_compliance = metadata.get("basel_ii_compliance", {})
        
        # Check ROC-AUC >= 0.7
        roc_auc = performance.get("roc_auc", 0)
        assert roc_auc >= 0.7, f"ROC-AUC {roc_auc} < 0.7"
        
        # Check false negative rate <= 20%
        fnr = performance.get("false_negative_rate", 1.0)
        assert fnr <= 0.2, f"False negative rate {fnr} > 0.2"
        
        # Check overall compliance
        assert basel_compliance.get("overall", False), "Model not Basel II compliant"
    
    def test_model_predictions(self):
        """Test model predictions on sample data"""
        # Load model and preprocessor
        with open("models/best_model/model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("models/best_model/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        
        # Test sample data
        test_samples = np.array([
            [30, 20, 2000],   # Low risk (low recency, high frequency, high monetary)
            [120, 5, 500],    # High risk (high recency, low frequency, low monetary)
            [60, 10, 1000],   # Medium risk
        ])
        
        # Scale features
        test_scaled = preprocessor.transform(test_samples)
        
        # Make predictions
        predictions = model.predict(test_scaled)
        probabilities = model.predict_proba(test_scaled)
        
        # Check predictions shape
        assert predictions.shape[0] == 3
        assert probabilities.shape == (3, 2)  # 3 samples, 2 classes
        
        # Check probabilities sum to 1
        for probs in probabilities:
            assert np.isclose(probs.sum(), 1.0, atol=1e-10)
        
        # Second sample should be high risk
        assert predictions[1] == 1, "High-risk sample should be predicted as high risk"
        
        # First sample should be low risk
        assert predictions[0] == 0, "Low-risk sample should be predicted as low risk"