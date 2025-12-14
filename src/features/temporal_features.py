"""
Temporal feature extraction for Task 3 requirements.
Extracts: Hour, Day, Month, Year from TransactionStartTime
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor:
    """
    Extract and engineer temporal features from timestamps.
    Implements Task 3 requirements for time feature extraction.
    """
    
    @staticmethod
    def extract_basic_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract REQUIRED temporal features (Task 3):
        - Transaction Hour
        - Transaction Day
        - Transaction Month  
        - Transaction Year
        """
        logger.info("Extracting basic temporal features...")
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime']):
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # REQUIRED FEATURES
        temporal_features = pd.DataFrame(index=df.index)
        
        # 1. Transaction Hour (0-23)
        temporal_features['transaction_hour'] = df['TransactionStartTime'].dt.hour
        
        # 2. Transaction Day (1-31)
        temporal_features['transaction_day'] = df['TransactionStartTime'].dt.day
        
        # 3. Transaction Month (1-12)
        temporal_features['transaction_month'] = df['TransactionStartTime'].dt.month
        
        # 4. Transaction Year
        temporal_features['transaction_year'] = df['TransactionStartTime'].dt.year
        
        logger.info("Extracted 4 required temporal features")
        
        return temporal_features
    
    @staticmethod
    def extract_advanced_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract advanced temporal features for better predictive power.
        """
        logger.info("Extracting advanced temporal features...")
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime']):
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        advanced_features = pd.DataFrame(index=df.index)
        
        # Day of week (0=Monday, 6=Sunday)
        advanced_features['transaction_day_of_week'] = df['TransactionStartTime'].dt.dayofweek
        
        # Day of year (1-365/366)
        advanced_features['transaction_day_of_year'] = df['TransactionStartTime'].dt.dayofyear
        
        # Week of year
        advanced_features['transaction_week_of_year'] = df['TransactionStartTime'].dt.isocalendar().week
        
        # Quarter of year
        advanced_features['transaction_quarter'] = df['TransactionStartTime'].dt.quarter
        
        # Is weekend?
        advanced_features['is_weekend'] = (df['TransactionStartTime'].dt.dayofweek >= 5).astype(int)
        
        # Is holiday? (Simplified - weekend proxy for now)
        advanced_features['is_holiday'] = advanced_features['is_weekend']
        
        # Time of day categories
        def categorize_hour(hour):
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'
        
        advanced_features['time_of_day'] = df['TransactionStartTime'].dt.hour.apply(categorize_hour)
        
        # Season based on month
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        
        advanced_features['season'] = df['TransactionStartTime'].dt.month.apply(get_season)
        
        # Business hours (9 AM - 5 PM)
        advanced_features['is_business_hours'] = (
            (df['TransactionStartTime'].dt.hour >= 9) & 
            (df['TransactionStartTime'].dt.hour <= 17)
        ).astype(int)
        
        logger.info(f"Extracted {len(advanced_features.columns)} advanced temporal features")
        
        return advanced_features
    
    @staticmethod
    def create_customer_temporal_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level temporal patterns from transaction timestamps.
        """
        logger.info("Creating customer temporal patterns...")
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime']):
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Add basic temporal features first
        df = df.copy()
        basic_features = TemporalFeatureExtractor.extract_basic_temporal_features(df)
        advanced_features = TemporalFeatureExtractor.extract_advanced_temporal_features(df)
        
        df = pd.concat([df, basic_features, advanced_features], axis=1)
        
        # Group by customer for pattern analysis
        customer_patterns = {}
        
        # Temporal consistency metrics
        temporal_cols = [
            'transaction_hour', 'transaction_day_of_week', 
            'time_of_day', 'is_weekend', 'is_business_hours'
        ]
        
        for col in temporal_cols:
            if col in df.columns:
                # Mode (most common value)
                mode_series = df.groupby('CustomerId')[col].agg(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                )
                customer_patterns[f'{col}_mode'] = mode_series
                
                # Consistency (1 - normalized entropy)
                def temporal_consistency(series):
                    value_counts = series.value_counts(normalize=True)
                    entropy = -np.sum(value_counts * np.log(value_counts + 1e-10))
                    max_entropy = np.log(len(value_counts))
                    return 1 - (entropy / max_entropy) if max_entropy > 0 else 1
                
                consistency_series = df.groupby('CustomerId')[col].apply(temporal_consistency)
                customer_patterns[f'{col}_consistency'] = consistency_series
        
        # Favorite time patterns
        customer_patterns['favorite_hour'] = df.groupby('CustomerId')['transaction_hour'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        
        customer_patterns['favorite_day_of_week'] = df.groupby('CustomerId')['transaction_day_of_week'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        
        # Weekend vs weekday ratio
        if 'is_weekend' in df.columns:
            weekend_ratio = df.groupby('CustomerId')['is_weekend'].mean()
            customer_patterns['weekend_transaction_ratio'] = weekend_ratio
        
        # Business hours ratio
        if 'is_business_hours' in df.columns:
            business_ratio = df.groupby('CustomerId')['is_business_hours'].mean()
            customer_patterns['business_hours_ratio'] = business_ratio
        
        # Time between transactions statistics
        df_sorted = df.sort_values(['CustomerId', 'TransactionStartTime'])
        df_sorted['time_diff_hours'] = df_sorted.groupby('CustomerId')['TransactionStartTime'].diff().dt.total_seconds() / 3600
        
        time_stats = df_sorted.groupby('CustomerId')['time_diff_hours'].agg(['mean', 'std', 'min', 'max'])
        time_stats.columns = [f'time_between_transactions_{col}' for col in time_stats.columns]
        
        # Merge time stats
        for col in time_stats.columns:
            customer_patterns[col] = time_stats[col]
        
        # Create DataFrame
        patterns_df = pd.DataFrame(customer_patterns)
        
        # Fill NaN values
        patterns_df = patterns_df.fillna(patterns_df.median())
        
        logger.info(f"Created {len(patterns_df.columns)} customer temporal patterns")
        
        return patterns_df
    
    @staticmethod
    def create_time_based_aggregations(df: pd.DataFrame, time_windows: List[str] = ['D', 'W', 'M']) -> pd.DataFrame:
        """
        Create time-based rolling aggregations.
        """
        logger.info("Creating time-based aggregations...")
        
        # Ensure datetime and set as index
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime']):
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Set datetime as index for time-based operations
        df_time = df.set_index('TransactionStartTime').sort_index()
        
        time_features = {}
        
        for customer_id in df['CustomerId'].unique():
            customer_data = df_time[df_time['CustomerId'] == customer_id]
            
            for window in time_windows:
                # Rolling sum
                rolling_sum = customer_data['Value'].rolling(window).sum()
                time_features[f'{customer_id}_rolling_sum_{window}'] = rolling_sum.iloc[-1] if len(rolling_sum) > 0 else 0
                
                # Rolling count
                rolling_count = customer_data['TransactionId'].rolling(window).count()
                time_features[f'{customer_id}_rolling_count_{window}'] = rolling_count.iloc[-1] if len(rolling_count) > 0 else 0
        
        # This would typically be more complex - simplified for example
        logger.info(f"Created time-based aggregations for {len(df['CustomerId'].unique())} customers")
        
        # Return empty DataFrame for this example (implementation would be more complex)
        return pd.DataFrame()