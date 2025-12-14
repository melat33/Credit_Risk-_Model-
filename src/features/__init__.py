"""
Credit Risk Feature Engineering Package
Bank-Grade Feature Engineering for Basel II/III Compliance
"""

__version__ = "1.0.0"
__author__ = "Bati Bank Analytics Team"

# Import from your ACTUAL filenames
from .aggregate import AggregationEngine  # Note: aggregate.py (not aggregations.py)
from .temporal_features import TemporalFeatureExtractor  # Note: temporal_features.py

__all__ = [
    'AggregationEngine',
    'TemporalFeatureExtractor'
]

print("✅ src.features package loaded successfully with actual module names")