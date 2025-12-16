"""
Helper module to calculate all 17 features from basic RFM data
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def calculate_all_features(customer_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate all 17 features from basic customer data
    
    Args:
        customer_data: Dictionary containing at least:
            - recency_days
            - transaction_frequency  
            - total_monetary_value
    
    Returns:
        Dictionary with all 17 features
    """
    # Extract core RFM features
    recency = customer_data.get("recency_days", 0)
    frequency = customer_data.get("transaction_frequency", 0)
    monetary = customer_data.get("total_monetary_value", 0)
    
    # Initialize features dict
    features = {
        "Unnamed: 0": customer_data.get("Unnamed_0", 0.0),
        "recency_days": recency,
        "transaction_frequency": frequency,
        "total_monetary_value": monetary,
    }
    
    # Calculate derived features
    # 1. Average transaction value
    features["avg_transaction_value"] = monetary / frequency if frequency > 0 else 0.0
    
    # 2. Standard deviation (simplified - in reality this would need historical data)
    features["std_transaction_value"] = customer_data.get("std_transaction_value", 0.0)
    
    # 3. Cluster (default)
    features["cluster"] = customer_data.get("cluster", 0)
    
    # 4. RFM Scores
    # Recency score (lower days = better)
    if recency <= 30:
        features["recency_score"] = 5
    elif recency <= 60:
        features["recency_score"] = 4
    elif recency <= 90:
        features["recency_score"] = 3
    elif recency <= 180:
        features["recency_score"] = 2
    else:
        features["recency_score"] = 1
    
    # Frequency score (higher frequency = better)
    if frequency >= 20:
        features["frequency_score"] = 5
    elif frequency >= 10:
        features["frequency_score"] = 4
    elif frequency >= 5:
        features["frequency_score"] = 3
    elif frequency >= 2:
        features["frequency_score"] = 2
    else:
        features["frequency_score"] = 1
    
    # Monetary score (higher value = better)
    if monetary >= 10000:
        features["monetary_score"] = 5
    elif monetary >= 5000:
        features["monetary_score"] = 4
    elif monetary >= 1000:
        features["monetary_score"] = 3
    elif monetary >= 500:
        features["monetary_score"] = 2
    else:
        features["monetary_score"] = 1
    
    # 5. Customer value (simplified projection)
    features["customer_value"] = customer_data.get("customer_value", 
                                                   monetary * (frequency / 12) * 24)  # 2-year projection
    
    # 6. Engagement index (combination of RFM scores)
    features["engagement_index"] = (
        features["recency_score"] * 0.4 +
        features["frequency_score"] * 0.3 +
        features["monetary_score"] * 0.3
    ) / 5.0  # Normalize to 0-1
    
    # 7. Value concentration (if only 1 transaction, concentration is 100%)
    features["value_concentration"] = customer_data.get("value_concentration", 1.0 if frequency <= 1 else 0.5)
    
    # 8. Value per transaction (same as avg_transaction_value)
    features["value_per_transaction"] = features["avg_transaction_value"]
    
    # 9. Transaction size variability
    features["transaction_size_variability"] = customer_data.get("transaction_size_variability", 0.1)
    
    # 10. Estimated tenure (simplified: assume 1 month per 5 transactions)
    features["estimated_tenure_months"] = customer_data.get("estimated_tenure_months", 
                                                            max(1, frequency / 5))
    
    # 11. Monthly activity (transactions per month)
    features["monthly_activity"] = customer_data.get("monthly_activity", 
                                                     frequency / max(1, features["estimated_tenure_months"]))
    
    return features