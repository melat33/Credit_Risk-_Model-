"""
Complete sklearn Pipeline builder for Task 3.
Implements all required transformations in a single reproducible pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

from .engineer import CreditRiskFeatureEngineer
from .aggregations import AggregationEngine
from .temporal import TemporalFeatureExtractor
from .encoders import SmartCategoricalEncoder
from .transformers import SmartImputer, FeatureScaler, OutlierHandler
from .woe_iv import WoETransformer

logger = logging.getLogger(__name__)


def create_feature_pipeline(config_path: str = None) -> Pipeline:
    """
    Create complete feature engineering pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        sklearn Pipeline with all transformations
    """
    logger.info("Creating feature engineering pipeline...")
    
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "feature_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define pipeline steps
    pipeline_steps = []
    
    # Step 1: Feature Engineering
    if config.get('feature_engineering', {}).get('enabled', True):
        pipeline_steps.append(('feature_engineer', CreditRiskFeatureEngineer(config_path=config_path)))
    
    # Step 2: Categorical Encoding
    encoding_config = config.get('categorical_encoding', {})
    if encoding_config.get('enabled', True):
        encoder = SmartCategoricalEncoder(
            max_onehot_categories=encoding_config.get('max_categories', 15),
            min_frequency_for_rare=encoding_config.get('rare_threshold', 0.01)
        )
        pipeline_steps.append(('categorical_encoder', encoder))
    
    # Step 3: Missing Value Imputation
    imputation_config = config.get('missing_values', {})
    if imputation_config.get('enabled', True):
        imputer = SmartImputer(
            strategy=imputation_config.get('strategy', 'simple'),
            numerical_strategy=imputation_config.get('numerical_strategy', 'median'),
            categorical_strategy=imputation_config.get('categorical_strategy', 'most_frequent'),
            add_missing_indicator=imputation_config.get('add_missing_indicators', True)
        )
        pipeline_steps.append(('imputer', imputer))
    
    # Step 4: Outlier Handling
    outlier_config = config.get('outlier_handling', {})
    if outlier_config.get('enabled', True):
        outlier_handler = OutlierHandler(
            method=outlier_config.get('method', 'cap'),
            lower_percentile=outlier_config.get('lower_percentile', 1),
            upper_percentile=outlier_config.get('upper_percentile', 99),
            add_outlier_flags=outlier_config.get('add_outlier_flags', True)
        )
        pipeline_steps.append(('outlier_handler', outlier_handler))
    
    # Step 5: Feature Scaling
    scaling_config = config.get('feature_scaling', {})
    if scaling_config.get('enabled', True):
        scaler = FeatureScaler(
            strategy=scaling_config.get('method', 'standard'),
            with_mean=scaling_config.get('with_mean', True),
            with_std=scaling_config.get('with_std', True),
            feature_range=tuple(scaling_config.get('feature_range', [0, 1]))
        )
        pipeline_steps.append(('scaler', scaler))
    
    # Step 6: WoE Transformation
    woe_config = config.get('woe_transformation', {})
    if woe_config.get('enabled', True):
        woe_transformer = WoETransformer(
            target_column=woe_config.get('target_column', 'is_high_risk'),
            min_bin_size=woe_config.get('min_bin_size', 50),
            max_bins=woe_config.get('max_bins', 10),
            iv_threshold=woe_config.get('iv_threshold', 0.02),
            monotonic_binning=woe_config.get('monotonic_binning', True)
        )
        pipeline_steps.append(('woe_transformer', woe_transformer))
    
    # Create pipeline
    pipeline = Pipeline(steps=pipeline_steps)
    
    logger.info(f"Created pipeline with {len(pipeline_steps)} steps")
    
    return pipeline


def create_simple_pipeline() -> Pipeline:
    """
    Create simplified pipeline with only Task 3 required steps.
    """
    logger.info("Creating simplified pipeline for Task 3 requirements...")
    
    # Define custom transformers for Task 3 requirements
    def create_aggregates(df):
        """Create Task 3 required aggregate features."""
        aggregates = AggregationEngine.create_required_aggregations(df)
        return aggregates
    
    def extract_temporal(df):
        """Extract Task 3 required temporal features."""
        temporal = TemporalFeatureExtractor.extract_basic_temporal_features(df)
        return temporal
    
    # Create pipeline
    pipeline = Pipeline([
        ('aggregates', FunctionTransformer(create_aggregates, validate=False)),
        ('temporal', FunctionTransformer(extract_temporal, validate=False)),
        ('imputer', SmartImputer(strategy='median')),
        ('encoder', SmartCategoricalEncoder()),
        ('scaler', FeatureScaler(strategy='standard'))
    ])
    
    return pipeline


def save_pipeline(pipeline: Pipeline, filepath: str):
    """
    Save pipeline to file.
    
    Args:
        pipeline: Trained pipeline
        filepath: Path to save pipeline
    """
    import joblib
    
    joblib.dump(pipeline, filepath)
    logger.info(f"Pipeline saved to {filepath}")


def load_pipeline(filepath: str) -> Pipeline:
    """
    Load pipeline from file.
    
    Args:
        filepath: Path to pipeline file
        
    Returns:
        Loaded pipeline
    """
    import joblib
    
    pipeline = joblib.load(filepath)
    logger.info(f"Pipeline loaded from {filepath}")
    
    return pipeline