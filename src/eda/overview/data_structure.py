"""
Data structure analysis module - Fixed version
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class DataStructureAnalyzer:
    """Analyzes data structure and dimensions"""
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze data structure"""
        result = {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2)
        }
        
        # Convert dtypes to string for JSON serialization
        dtype_counts = {}
        for dtype, count in df.dtypes.value_counts().to_dict().items():
            dtype_counts[str(dtype)] = int(count)
        
        result['dtypes'] = dtype_counts
        return result
    
    def check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows"""
        duplicates = df.duplicated().sum()
        return {
            'duplicate_rows': int(duplicates),
            'duplicate_percentage': float((duplicates / len(df)) * 100)
        }
    
    def get_data_types(self, df: pd.DataFrame) -> Dict:
        """Get data type distribution"""
        return {
            'numerical': int(df.select_dtypes(include=[np.number]).shape[1]),
            'categorical': int(df.select_dtypes(include=['object']).shape[1]),
            'datetime': int(df.select_dtypes(include=['datetime']).shape[1]),
            'boolean': int(df.select_dtypes(include=['bool']).shape[1])
        }