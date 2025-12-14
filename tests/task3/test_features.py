"""
Unit tests for Task 3 - CLEAN VERSION (No warnings)
"""

import pandas as pd
import numpy as np
import sys
import os
import pytest

# ============================================================================
# FIX IMPORT PATHS
# ============================================================================

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import from your actual files
from src.features.aggregate import AggregationEngine
from src.features.temporal_features import TemporalFeatureExtractor

# ============================================================================
# TEST CLASS
# ============================================================================

class TestTask3Features:
    """Test Task 3 requirements - Clean version"""
    
    def setup_method(self):
        """Create test data"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'TransactionId': range(1, 101),
            'CustomerId': np.repeat(['C001', 'C002', 'C003', 'C004'], [25, 25, 25, 25]),
            'Value': np.random.uniform(10, 1000, 100),
            'TransactionStartTime': pd.date_range('2024-01-01', periods=100, freq='h'),  # Use 'h' not 'H'
            'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Food'], 100)
        })
    
    def test_01_task3_requirement1_aggregates(self):
        """Test Task 3 Requirement 1: Create 4 aggregate features"""
        aggregates = AggregationEngine.create_required_aggregations(self.test_data)
        
        # Check all 4 required features exist
        required_features = [
            'total_transaction_amount',
            'avg_transaction_amount', 
            'transaction_count',
            'std_transaction_amount'
        ]
        
        for feature in required_features:
            assert feature in aggregates.columns, f"Missing: {feature}"
        
        # Check calculations
        assert len(aggregates) == 4  # 4 customers
        assert not aggregates.isnull().any().any()  # No NaN values
    
    def test_02_task3_requirement2_temporal(self):
        """Test Task 3 Requirement 2: Extract 4 temporal features"""
        temporal = TemporalFeatureExtractor.extract_basic_temporal_features(self.test_data)
        
        # Check all 4 required features exist
        required_features = [
            'transaction_hour',
            'transaction_day',
            'transaction_month',
            'transaction_year'
        ]
        
        for feature in required_features:
            assert feature in temporal.columns, f"Missing: {feature}"
        
        # Check valid ranges
        assert temporal['transaction_hour'].between(0, 23).all()
        assert temporal['transaction_day'].between(1, 31).all()
        assert temporal['transaction_month'].between(1, 12).all()
    
    def test_03_complete_task3(self):
        """Complete Task 3 verification"""
        # Test aggregates
        aggregates = AggregationEngine.create_required_aggregations(self.test_data)
        
        # Test temporal
        temporal = TemporalFeatureExtractor.extract_basic_temporal_features(self.test_data)
        
        # Verify all Task 3 requirements
        task3_requirements = {
            "Aggregate Features": [
                'total_transaction_amount',
                'avg_transaction_amount',
                'transaction_count',
                'std_transaction_amount'
            ],
            "Temporal Features": [
                'transaction_hour',
                'transaction_day',
                'transaction_month',
                'transaction_year'
            ]
        }
        
        for category, features in task3_requirements.items():
            if category == "Aggregate Features":
                data = aggregates
            else:
                data = temporal
            
            for feature in features:
                assert feature in data.columns, f"Missing {category}: {feature}"

# ============================================================================
# SIMPLE TEST FUNCTIONS
# ============================================================================

def test_aggregate_feature_creation():
    """Test aggregate feature creation"""
    test = TestTask3Features()
    test.setup_method()
    
    aggregates = AggregationEngine.create_required_aggregations(test.test_data)
    
    required = ['total_transaction_amount', 'avg_transaction_amount', 
                'transaction_count', 'std_transaction_amount']
    
    for feature in required:
        assert feature in aggregates.columns

def test_temporal_feature_extraction():
    """Test temporal feature extraction"""
    test = TestTask3Features()
    test.setup_method()
    
    temporal = TemporalFeatureExtractor.extract_basic_temporal_features(test.test_data)
    
    required = ['transaction_hour', 'transaction_day', 
                'transaction_month', 'transaction_year']
    
    for feature in required:
        assert feature in temporal.columns

def test_task3_completion():
    """Test complete Task 3"""
    test = TestTask3Features()
    test.setup_method()
    
    # Test both requirements
    test.test_01_task3_requirement1_aggregates()
    test.test_02_task3_requirement2_temporal()