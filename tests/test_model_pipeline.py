"""
Unit Tests for Bati Bank Credit Risk Model
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def test_data_loading():
    """Test that data loads correctly with expected columns"""
    try:
        df = pd.read_csv('data/processed/customer_rfm_with_target.csv')
        assert 'is_high_risk' in df.columns, "Target column missing"
        assert len(df) > 1000, "Insufficient data"
        assert df['is_high_risk'].isin([0, 1]).all(), "Invalid target values"
        print("[PASS] Data loading test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Data loading test failed: {e}")
        return False

def test_feature_engineering():
    """Test that feature engineering produces expected features"""
    # This would test your feature engineering functions
    pass

def test_model_training():
    """Test that model can be trained and makes predictions"""
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y), "Prediction length mismatch"
    assert predictions.shape == y.shape, "Prediction shape mismatch"
    print("[PASS] Model training test passed")
    return True

def test_production_model():
    """Test that production model files exist"""
    import os
    import pickle

    required_files = [
        '../../models/best_model/model.pkl',
        '../../models/best_model/preprocessor.pkl',
        '../../models/best_model/metadata.json'
    ]

    all_exist = all(os.path.exists(f) for f in required_files)
    assert all_exist, f"Missing production files. Found: {[f for f in required_files if os.path.exists(f)]}"
    print("[PASS] Production model files exist")

    # Test model can be loaded
    with open('../../models/best_model/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Test preprocessor can be loaded
    with open('../../models/best_model/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    print("[PASS] Model and preprocessor can be loaded")
    return True

def test_basel_compliance():
    """Test that model meets Basel II compliance requirements"""
    import json

    with open('../../models/best_model/metadata.json', 'r') as f:
        metadata = json.load(f)

    # Basel II requirements
    roc_auc_met = metadata['basel_ii_compliance']['roc_auc_met']
    fnr_met = metadata['basel_ii_compliance']['fnr_met']

    assert roc_auc_met, f"ROC-AUC {metadata['performance']['roc_auc']:.3f} < 0.7"
    assert fnr_met, f"FNR {metadata['performance']['false_negative_rate']:.3f} > 0.2"

    print(f"[PASS] Basel II compliance met: ROC-AUC={metadata['performance']['roc_auc']:.3f}, FNR={metadata['performance']['false_negative_rate']:.3f}")
    return True

if __name__ == "__main__":
    results = []
    print("Running Bati Bank Credit Risk Model Tests...")
    print("-" * 50)

    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Training", test_model_training()))
    results.append(("Production Model", test_production_model()))
    results.append(("Basel Compliance", test_basel_compliance()))

    print("-" * 50)
    print("Test Results Summary:")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(result for _, result in results)
    if all_passed:
        print("[SUCCESS] All tests passed!")
    else:
        print("[FAILURE] Some tests failed")
        raise AssertionError("One or more tests failed")
