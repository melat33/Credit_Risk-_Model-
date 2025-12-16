"""
Specialized aggregation functions for Task 3 requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable
import logging

logger = logging.getLogger(__name__)


class AggregationEngine:
    """
    Engine for creating customer-level aggregations.
    Focuses on Task 3 required features.
    """
    
    @staticmethod
    def create_required_aggregations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ALL Task 3 required aggregate features.
        
        Returns:
            DataFrame with CustomerId as index and required features
        """
        logger.info("Creating Task 3 required aggregations...")
        
        # Group by customer
        customer_groups = df.groupby('CustomerId')
        
        # REQUIRED FEATURES (Task 3 Instructions)
        aggregations = {
            # 1. Total Transaction Amount
            'total_transaction_amount': ('Value', 'sum'),
            
            # 2. Average Transaction Amount
            'avg_transaction_amount': ('Value', 'mean'),
            
            # 3. Transaction Count
            'transaction_count': ('TransactionId', 'count'),
            
            # 4. Standard Deviation of Transaction Amounts
            'std_transaction_amount': ('Value', 'std'),
        }
        
        # Execute aggregations
        results = {}
        for feature_name, (column, func) in aggregations.items():
            if func == 'count':
                results[feature_name] = customer_groups[column].count()
            else:
                results[feature_name] = getattr(customer_groups[column], func)()
        
        # Create DataFrame
        result_df = pd.DataFrame(results)
        
        # Fill NaN for std (when only 1 transaction)
        result_df['std_transaction_amount'] = result_df['std_transaction_amount'].fillna(0)
        
        logger.info(f"Created {len(result_df.columns)} required aggregations for {len(result_df)} customers")
        
        return result_df
    
    @staticmethod
    def create_extended_aggregations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create extended aggregations for better model performance.
        """
        logger.info("Creating extended aggregations...")
        
        customer_groups = df.groupby('CustomerId')
        
        extended_features = {
            # Central tendency
            'median_transaction_amount': ('Value', 'median'),
            'mode_transaction_amount': ('Value', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            
            # Range and spread
            'min_transaction_amount': ('Value', 'min'),
            'max_transaction_amount': ('Value', 'max'),
            'range_transaction_amount': ('Value', lambda x: x.max() - x.min()),
            'iqr_transaction_amount': ('Value', lambda x: x.quantile(0.75) - x.quantile(0.25)),
            
            # Shape
            'skew_transaction_amount': ('Value', 'skew'),
            'kurtosis_transaction_amount': ('Value', lambda x: x.kurtosis()),
            
            # Percentiles
            'p25_transaction_amount': ('Value', lambda x: x.quantile(0.25)),
            'p75_transaction_amount': ('Value', lambda x: x.quantile(0.75)),
            'p90_transaction_amount': ('Value', lambda x: x.quantile(0.90)),
            
            # Zero and negative values
            'zero_transaction_count': ('Value', lambda x: (x == 0).sum()),
            'negative_transaction_count': ('Value', lambda x: (x < 0).sum()),
            'positive_transaction_count': ('Value', lambda x: (x > 0).sum()),
        }
        
        # Execute extended aggregations
        results = {}
        for feature_name, (column, func) in extended_features.items():
            try:
                if callable(func):
                    results[feature_name] = customer_groups[column].apply(func)
                else:
                    results[feature_name] = getattr(customer_groups[column], func)()
            except Exception as e:
                logger.warning(f"Failed to create {feature_name}: {e}")
                results[feature_name] = np.nan
        
        # Create DataFrame
        extended_df = pd.DataFrame(results)
        
        # Fill NaN values
        for col in extended_df.columns:
            if extended_df[col].isnull().any():
                if 'count' in col:
                    extended_df[col] = extended_df[col].fillna(0)
                else:
                    extended_df[col] = extended_df[col].fillna(extended_df[col].median())
        
        logger.info(f"Created {len(extended_df.columns)} extended aggregations")
        
        return extended_df
    
    @staticmethod
    def create_time_window_aggregations(df: pd.DataFrame, windows_days: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """
        Create aggregations for different time windows.
        """
        logger.info("Creating time window aggregations...")
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime']):
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        latest_date = df['TransactionStartTime'].max()
        
        window_features = {}
        
        for window in windows_days:
            # Calculate cutoff date
            cutoff_date = latest_date - pd.Timedelta(days=window)
            
            # Filter recent transactions
            recent_df = df[df['TransactionStartTime'] > cutoff_date]
            
            # Group recent transactions by customer
            recent_groups = recent_df.groupby('CustomerId')
            
            # Create window-specific features
            window_features[f'total_amount_{window}d'] = recent_groups['Value'].sum()
            window_features[f'transaction_count_{window}d'] = recent_groups['TransactionId'].count()
            window_features[f'avg_amount_{window}d'] = recent_groups['Value'].mean()
            
            # Fill NaN for customers with no recent transactions
            all_customers = df['CustomerId'].unique()
            for feature in [f'total_amount_{window}d', f'transaction_count_{window}d', f'avg_amount_{window}d']:
                window_features[feature] = window_features[feature].reindex(all_customers).fillna(0)
        
        # Create DataFrame
        window_df = pd.DataFrame(window_features)
        
        # Calculate growth rates
        for window in [30, 90]:
            if f'total_amount_{window}d' in window_df.columns and 'total_transaction_amount' in df.columns:
                window_df[f'growth_rate_{window}d'] = (
                    window_df[f'total_amount_{window}d'] / 
                    df.groupby('CustomerId')['Value'].sum()
                ).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logger.info(f"Created time window aggregations for windows: {windows_days}")
        
        return window_df
    
    @staticmethod
    def create_feature_interactions(aggregated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between aggregates.
        """
        logger.info("Creating feature interactions...")
        
        interactions = pd.DataFrame(index=aggregated_df.index)
        
        # Ratio features
        if 'total_transaction_amount' in aggregated_df.columns and 'transaction_count' in aggregated_df.columns:
            interactions['avg_amount_per_transaction'] = (
                aggregated_df['total_transaction_amount'] / 
                aggregated_df['transaction_count']
            ).replace([np.inf, -np.inf], 0)
        
        # Variability ratios
        if 'std_transaction_amount' in aggregated_df.columns and 'avg_transaction_amount' in aggregated_df.columns:
            interactions['cv_transaction_amount'] = (
                aggregated_df['std_transaction_amount'] / 
                aggregated_df['avg_transaction_amount']
            ).replace([np.inf, -np.inf], 0)
        
        # Range to average ratio
        if all(col in aggregated_df.columns for col in ['max_transaction_amount', 'min_transaction_amount', 'avg_transaction_amount']):
            interactions['range_to_avg_ratio'] = (
                (aggregated_df['max_transaction_amount'] - aggregated_df['min_transaction_amount']) / 
                aggregated_df['avg_transaction_amount']
            ).replace([np.inf, -np.inf], 0)
        
        # Log features for skewed distributions
        for col in ['total_transaction_amount', 'avg_transaction_amount', 'max_transaction_amount']:
            if col in aggregated_df.columns:
                interactions[f'log_{col}'] = np.log1p(np.abs(aggregated_df[col])) * np.sign(aggregated_df[col])
        
        # Fill NaN values
        interactions = interactions.fillna(0)
        
        logger.info(f"Created {len(interactions.columns)} interaction features")
        
        return interactions