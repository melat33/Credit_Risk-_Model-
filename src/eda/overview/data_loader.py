"""
Data loading module with optimization for large datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

class DataLoader:
    """Handles efficient data loading and type optimization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV file with optimization
        """
        try:
            # Check file existence
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            self.logger.info(f"Loading data from {filepath}")
            
            # Read CSV with optimized parameters
            df = pd.read_csv(
                filepath,
                parse_dates=self._infer_date_columns(filepath),
                infer_datetime_format=True,
                low_memory=False,
                nrows=self.config['data'].get('sample_size')  # For sampling if needed
            )
            
            self.logger.info(f"Raw data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def _infer_date_columns(self, filepath: str) -> List[str]:
        """
        Infer which columns might be dates
        """
        # Read first row to check column names
        try:
            first_row = pd.read_csv(filepath, nrows=1)
            date_like_cols = []
            
            for col in first_row.columns:
                col_lower = col.lower()
                if any(date_term in col_lower for date_term in 
                      ['time', 'date', 'timestamp', 'start', 'end', 'created', 'updated']):
                    date_like_cols.append(col)
            
            return date_like_cols
        except:
            return []
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for memory efficiency
        """
        self.logger.info("Optimizing data types...")
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Downcast numerical columns
        df = self._downcast_numericals(df)
        
        # Convert categorical columns
        df = self._optimize_categoricals(df)
        
        # Convert datetime columns
        df = self._optimize_datetimes(df)
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = ((original_memory - optimized_memory) / original_memory) * 100
        
        self.logger.info(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB ({reduction:.1f}% reduction)")
        
        return df
    
    def _downcast_numericals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast numerical columns to smallest possible type"""
        # Integer columns
        int_cols = df.select_dtypes(include=['int64', 'int32']).columns
        for col in int_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df[col] = pd.to_numeric(df[col], downcast='unsigned')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:  # Signed integers
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Float columns
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        return df
    
    def _optimize_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert object columns to category where appropriate"""
        object_cols = df.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            # Convert to category if low cardinality
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    def _optimize_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize datetime columns"""
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        
        for col in datetime_cols:
            # Convert to datetime[ns] if not already
            df[col] = pd.to_datetime(df[col])
            
            # Extract useful features
            if 'TransactionStartTime' in col:
                df[f'{col}_hour'] = df[col].dt.hour.astype('uint8')
                df[f'{col}_day'] = df[col].dt.day.astype('uint8')
                df[f'{col}_month'] = df[col].dt.month.astype('uint8')
                df[f'{col}_year'] = df[col].dt.year.astype('uint16')
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek.astype('uint8')
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Perform basic data validation
        Returns dictionary of validation results
        """
        validation = {}
        
        # Check for required columns in credit risk context
        required_columns = ['TransactionId', 'CustomerId', 'Amount']
        for col in required_columns:
            validation[f'has_{col}'] = col in df.columns
        
        # Check for negative amounts (should be credits)
        if 'Amount' in df.columns:
            negative_amounts = (df['Amount'] < 0).sum()
            validation['has_negative_amounts'] = negative_amounts > 0
            validation['negative_amount_percentage'] = (negative_amounts / len(df)) * 100
        
        # Check for duplicate transactions
        if 'TransactionId' in df.columns:
            duplicates = df['TransactionId'].duplicated().sum()
            validation['has_duplicate_transactions'] = duplicates > 0
        
        return validation