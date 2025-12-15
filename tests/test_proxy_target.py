"""
Unit tests for proxy target creation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.proxy_target import RFMTargetCreator

@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    data = {
        'CustomerId': np.random.choice([f'CUST_{i}' for i in range(1, 101)], size=1000),
        'TransactionId': [f'TXN_{i}' for i in range(1000)],
        'Amount': np.random.exponential(1000, 1000),
        'TransactionStartTime': np.random.choice(dates, 1000)
    }
    
    return pd.DataFrame(data)

def test_rfm_calculation(sample_transaction_data):
    """Test RFM metrics calculation"""
    creator = RFMTargetCreator()
    rfm_df = creator.calculate_rfm(sample_transaction_data)
    
    # Check columns
    expected_columns = ['recency', 'frequency', 'monetary']
    assert all(col in rfm_df.columns for col in expected_columns)
    
    # Check all customers present
    unique_customers = sample_transaction_data['CustomerId'].unique()
    assert len(rfm_df) == len(unique_customers)
    
    # Check recency is non-negative
    assert rfm_df['recency'].min() >= 0
    
    # Check frequency is positive
    assert rfm_df['frequency'].min() > 0

def test_cluster_creation():
    """Test K-Means clustering"""
    # Create synthetic RFM data
    np.random.seed(42)
    n_customers = 100
    
    # Create distinct groups
    rfm_data = pd.DataFrame({
        'recency': np.concatenate([
            np.random.uniform(0, 30, n_customers//3),  # Recent, active
            np.random.uniform(30, 180, n_customers//3), # Medium
            np.random.uniform(180, 365, n_customers//3) # Inactive
        ]),
        'frequency': np.concatenate([
            np.random.uniform(10, 50, n_customers//3),  # Frequent
            np.random.uniform(5, 15, n_customers//3),   # Medium
            np.random.uniform(1, 5, n_customers//3)     # Infrequent
        ]),
        'monetary': np.concatenate([
            np.random.uniform(1000, 10000, n_customers//3),  # High value
            np.random.uniform(500, 2000, n_customers//3),    # Medium value
            np.random.uniform(50, 500, n_customers//3)       # Low value
        ])
    })
    
    creator = RFMTargetCreator(random_state=42)
    rfm_df, high_risk_cluster, profiles = creator.create_clusters(rfm_data)
    
    # Check cluster assignment
    assert 'cluster' in rfm_df.columns
    assert len(rfm_df['cluster'].unique()) == 3
    
    # Check all clusters have customers
    cluster_sizes = rfm_df['cluster'].value_counts()
    assert all(size > 0 for size in cluster_sizes.values)
    
    # Check high-risk cluster is identified
    assert high_risk_cluster in [0, 1, 2]

def test_target_variable_creation():
    """Test binary target variable creation"""
    # Create sample RFM with clusters
    rfm_df = pd.DataFrame({
        'recency': [10, 100, 200, 15, 250],
        'frequency': [20, 5, 2, 15, 1],
        'monetary': [5000, 1000, 500, 3000, 200],
        'cluster': [0, 1, 2, 0, 2]  # Assume cluster 2 is high-risk
    }, index=['C1', 'C2', 'C3', 'C4', 'C5'])
    
    creator = RFMTargetCreator()
    target = creator.create_target_variable(rfm_df, high_risk_cluster=2)
    
    # Check target is binary
    assert set(target.unique()).issubset({0, 1})
    
    # Check correct customers marked as high-risk
    assert target['C3'] == 1
    assert target['C5'] == 1
    assert target['C1'] == 0

def test_validation_logic():
    """Test proxy validation"""
    creator = RFMTargetCreator()
    
    # Create test data with clear patterns
    rfm_df = pd.DataFrame({
        'recency': [10, 20, 300, 350],  # Last two are inactive
        'frequency': [20, 15, 2, 1],    # Last two have low frequency
        'monetary': [5000, 4000, 500, 200],  # Last two have low monetary
        'cluster': [0, 0, 1, 1]
    })
    
    target = pd.Series([0, 0, 1, 1], index=rfm_df.index)
    
    # Create original transaction data with fraud indicator
    original_data = pd.DataFrame({
        'CustomerId': ['C1', 'C2', 'C3', 'C4'],
        'FraudResult': [0, 0, 1, 0]  # C3 had fraud
    })
    
    metrics = creator.validate_proxy(rfm_df, target, original_data)
    
    # Check validation metrics exist
    assert 'fraud_rate_high_risk' in metrics
    assert 'makes_business_sense' in metrics
    
    # High-risk should have higher recency ratio
    assert metrics.get('recency_ratio', 0) > 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])