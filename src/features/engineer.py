"""
Main Feature Engineering Class
Implements ALL Task 3 requirements with banking compliance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import yaml
import logging
from pathlib import Path
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature engineering for credit risk modeling.
    Implements ALL Task 3 requirements with audit trail and banking compliance.
    """
    
    def __init__(self, config_path: str = None, random_state: int = 42):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config_path: Path to feature configuration YAML
            random_state: Random seed for reproducibility
        """
        self.config = self._load_config(config_path)
        self.random_state = random_state
        self.audit_trail = []
        self.feature_definitions = {}
        self.fitted = False
        
        logger.info(f"CreditRiskFeatureEngineer initialized with config: {config_path}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration file."""
        if config_path is None:
            default_path = Path(__file__).parent.parent.parent / "configs" / "feature_config.yaml"
            config_path = str(default_path)
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature engineer (learn statistics)."""
        logger.info("Fitting CreditRiskFeatureEngineer...")
        
        # Store reference statistics for validation
        self.reference_stats_ = {
            'n_customers': X['CustomerId'].nunique(),
            'date_range': {
                'min': X['TransactionStartTime'].min(),
                'max': X['TransactionStartTime'].max()
            },
            'feature_means': X.select_dtypes(include=[np.number]).mean().to_dict(),
            'feature_medians': X.select_dtypes(include=[np.number]).median().to_dict(),
            'feature_stds': X.select_dtypes(include=[np.number]).std().to_dict()
        }
        
        # Learn aggregation parameters
        self._learn_aggregation_params(X)
        
        self.fitted = True
        logger.info(f"Feature engineer fitted on {len(X)} rows, {X['CustomerId'].nunique()} customers")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw transaction data into customer-level features.
        Implements ALL Task 3 requirements.
        """
        if not self.fitted:
            self.fit(X)
            
        logger.info("Starting feature transformation...")
        
        # Create copy to avoid modifying original
        X_transformed = X.copy()
        
        # 1. Ensure TransactionStartTime is datetime
        X_transformed = self._ensure_datetime(X_transformed)
        
        # 2. Create customer-level features
        customer_features = self._create_customer_features(X_transformed)
        
        # 3. Create temporal features
        temporal_features = self._create_temporal_features(X_transformed)
        
        # 4. Create behavioral features
        behavioral_features = self._create_behavioral_features(X_transformed)
        
        # 5. Create financial features
        financial_features = self._create_financial_features(X_transformed)
        
        # 6. Combine all features
        all_features = pd.concat([
            customer_features,
            temporal_features,
            behavioral_features,
            financial_features
        ], axis=1)
        
        # 7. Add audit trail entry
        self._add_audit_entry('transform', {
            'input_shape': X.shape,
            'output_shape': all_features.shape,
            'features_created': len(all_features.columns),
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Transformation complete. Created {len(all_features.columns)} features")
        
        return all_features
    
    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure TransactionStartTime is datetime format."""
        if not pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime']):
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        return df
    
    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level aggregate features.
        REQUIRED BY TASK 3: Total, Average, Count, Std
        """
        logger.info("Creating customer aggregate features...")
        
        # Group by customer
        customer_groups = df.groupby('CustomerId')
        
        features = {}
        
        # 1. TOTAL TRANSACTION AMOUNT (Required)
        features['total_transaction_amount'] = customer_groups['Value'].sum()
        
        # 2. AVERAGE TRANSACTION AMOUNT (Required)
        features['avg_transaction_amount'] = customer_groups['Value'].mean()
        
        # 3. TRANSACTION COUNT (Required)
        features['transaction_count'] = customer_groups['TransactionId'].count()
        
        # 4. STANDARD DEVIATION (Required)
        features['std_transaction_amount'] = customer_groups['Value'].std()
        
        # Additional useful aggregates
        features['median_transaction_amount'] = customer_groups['Value'].median()
        features['min_transaction_amount'] = customer_groups['Value'].min()
        features['max_transaction_amount'] = customer_groups['Value'].max()
        features['transaction_range'] = features['max_transaction_amount'] - features['min_transaction_amount']
        
        # Transaction value ratios
        features['value_ratio_max_avg'] = features['max_transaction_amount'] / features['avg_transaction_amount']
        features['value_ratio_std_avg'] = features['std_transaction_amount'] / features['avg_transaction_amount']
        
        # Create DataFrame
        customer_df = pd.DataFrame(features)
        
        # Log feature creation
        self._add_audit_entry('customer_features', {
            'features_created': list(customer_df.columns),
            'customer_count': len(customer_df)
        })
        
        return customer_df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from TransactionStartTime.
        REQUIRED BY TASK 3: Hour, Day, Month, Year
        """
        logger.info("Creating temporal features...")
        
        # Ensure datetime
        df = self._ensure_datetime(df)
        
        # Group by customer for temporal patterns
        customer_features = {}
        
        # Required temporal features
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        
        # Additional useful temporal features
        df['transaction_day_of_week'] = df['TransactionStartTime'].dt.dayofweek
        df['transaction_quarter'] = df['TransactionStartTime'].dt.quarter
        df['transaction_day_of_year'] = df['TransactionStartTime'].dt.dayofyear
        df['is_weekend'] = df['transaction_day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hour'] = ((df['transaction_hour'] >= 9) & (df['transaction_hour'] <= 17)).astype(int)
        
        # Customer-level temporal aggregations
        for col in ['transaction_hour', 'transaction_day_of_week', 'is_weekend', 'is_business_hour']:
            customer_features[f'{col}_mean'] = df.groupby('CustomerId')[col].mean()
            customer_features[f'{col}_std'] = df.groupby('CustomerId')[col].std()
        
        # Time between transactions
        df_sorted = df.sort_values(['CustomerId', 'TransactionStartTime'])
        df_sorted['time_diff'] = df_sorted.groupby('CustomerId')['TransactionStartTime'].diff().dt.total_seconds() / 3600  # hours
        
        customer_features['avg_time_between_transactions'] = df_sorted.groupby('CustomerId')['time_diff'].mean()
        customer_features['std_time_between_transactions'] = df_sorted.groupby('CustomerId')['time_diff'].std()
        
        # Create DataFrame
        temporal_df = pd.DataFrame(customer_features)
        
        # Log feature creation
        self._add_audit_entry('temporal_features', {
            'features_created': list(temporal_df.columns)
        })
        
        return temporal_df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral and engagement features."""
        logger.info("Creating behavioral features...")
        
        customer_features = {}
        
        # Product category diversity
        product_diversity = df.groupby('CustomerId')['ProductCategory'].nunique()
        customer_features['product_category_count'] = product_diversity
        
        # Channel usage diversity
        channel_diversity = df.groupby('CustomerId')['ChannelId'].nunique()
        customer_features['channel_count'] = channel_diversity
        
        # Provider diversity
        provider_diversity = df.groupby('CustomerId')['ProviderId'].nunique()
        customer_features['provider_count'] = provider_diversity
        
        # Currency usage
        currency_diversity = df.groupby('CustomerId')['CurrencyCode'].nunique()
        customer_features['currency_count'] = currency_diversity
        
        # Favorite product category (mode)
        fav_category = df.groupby('CustomerId')['ProductCategory'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        customer_features['favorite_product_category'] = fav_category
        
        # Most used channel
        fav_channel = df.groupby('CustomerId')['ChannelId'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        customer_features['favorite_channel'] = fav_channel
        
        # Create DataFrame
        behavioral_df = pd.DataFrame(customer_features)
        
        # Log feature creation
        self._add_audit_entry('behavioral_features', {
            'features_created': list(behavioral_df.columns)
        })
        
        return behavioral_df
    
    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial metrics and ratios."""
        logger.info("Creating financial features...")
        
        # Group by customer
        customer_groups = df.groupby('CustomerId')
        
        features = {}
        
        # Financial ratios
        total_amount = customer_groups['Value'].sum()
        transaction_count = customer_groups['TransactionId'].count()
        
        # Average amount per transaction (already have, but recalc for ratio)
        avg_amount = total_amount / transaction_count
        
        # Transaction velocity (transactions per active day)
        df['transaction_date'] = df['TransactionStartTime'].dt.date
        active_days = df.groupby('CustomerId')['transaction_date'].nunique()
        features['transactions_per_active_day'] = transaction_count / active_days
        
        # Amount per active day
        features['amount_per_active_day'] = total_amount / active_days
        
        # Recency metrics
        latest_date = df['TransactionStartTime'].max()
        days_since_last = (latest_date - df.groupby('CustomerId')['TransactionStartTime'].max()).dt.days
        features['days_since_last_transaction'] = days_since_last
        
        # Create DataFrame
        financial_df = pd.DataFrame(features)
        
        # Log feature creation
        self._add_audit_entry('financial_features', {
            'features_created': list(financial_df.columns)
        })
        
        return financial_df
    
    def _learn_aggregation_params(self, df: pd.DataFrame):
        """Learn parameters needed for aggregation."""
        self.aggregation_params_ = {
            'customer_ids': df['CustomerId'].unique(),
            'n_customers': df['CustomerId'].nunique(),
            'date_range': {
                'min': df['TransactionStartTime'].min(),
                'max': df['TransactionStartTime'].max()
            }
        }
    
    def _add_audit_entry(self, step: str, details: Dict):
        """Add entry to audit trail."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'details': details
        }
        self.audit_trail.append(entry)
        
    def get_audit_trail(self) -> pd.DataFrame:
        """Get audit trail as DataFrame."""
        return pd.DataFrame(self.audit_trail)
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of all features created."""
        features = []
        for step in self.audit_trail:
            if 'features_created' in step['details']:
                for feature in step['details']['features_created']:
                    features.append({
                        'feature_name': feature,
                        'created_in_step': step['step'],
                        'creation_time': step['timestamp']
                    })
        return pd.DataFrame(features)