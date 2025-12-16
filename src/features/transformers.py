"""
Missing value imputation and feature scaling implementations.
Implements Task 3 requirements for missing value handling and normalization.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging

logger = logging.getLogger(__name__)


class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Smart missing value imputer with multiple strategies.
    Implements Task 3 requirements for missing value handling.
    """
    
    def __init__(self, 
                 strategy: str = 'median',
                 numerical_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent',
                 add_missing_indicator: bool = True,
                 knn_neighbors: int = 5):
        """
        Initialize smart imputer.
        
        Args:
            strategy: Overall strategy ('simple', 'knn', 'iterative')
            numerical_strategy: For numerical columns ('mean', 'median', 'constant')
            categorical_strategy: For categorical columns ('most_frequent', 'constant')
            add_missing_indicator: Whether to add missing indicators
            knn_neighbors: Number of neighbors for KNN imputation
        """
        self.strategy = strategy
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.add_missing_indicator = add_missing_indicator
        self.knn_neighbors = knn_neighbors
        
        # Store imputers and column info
        self.imputers_ = {}
        self.column_stats_ = {}
        self.missing_indicators_ = []
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit imputer to data.
        
        Args:
            X: DataFrame to fit on
            y: Target variable (optional)
        """
        logger.info(f"Fitting smart imputer with strategy: {self.strategy}")
        
        # Store column information
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check for missing values
        missing_counts = X.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0].index.tolist()
        
        logger.info(f"Columns with missing values: {missing_columns}")
        logger.info(f"Missing value counts:\n{missing_counts[missing_counts > 0]}")
        
        if self.strategy == 'simple':
            # Fit separate imputers for numerical and categorical
            if self.numerical_columns_:
                num_imputer = SimpleImputer(strategy=self.numerical_strategy)
                num_imputer.fit(X[self.numerical_columns_])
                self.imputers_['numerical'] = num_imputer
                
                # Store imputation values
                self.column_stats_['numerical_imputation'] = dict(
                    zip(self.numerical_columns_, num_imputer.statistics_)
                )
            
            if self.categorical_columns_:
                cat_imputer = SimpleImputer(strategy=self.categorical_strategy)
                cat_imputer.fit(X[self.categorical_columns_])
                self.imputers_['categorical'] = cat_imputer
                
                self.column_stats_['categorical_imputation'] = dict(
                    zip(self.categorical_columns_, cat_imputer.statistics_)
                )
                
        elif self.strategy == 'knn':
            # KNN imputer (only for numerical)
            knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            knn_imputer.fit(X[self.numerical_columns_])
            self.imputers_['knn'] = knn_imputer
            
        elif self.strategy == 'iterative':
            # Iterative imputer
            iterative_imputer = IterativeImputer(
                max_iter=10,
                random_state=42
            )
            iterative_imputer.fit(X[self.numerical_columns_])
            self.imputers_['iterative'] = iterative_imputer
        
        self.fitted = True
        logger.info(f"Fitted imputer on {len(self.numerical_columns_)} numerical and "
                   f"{len(self.categorical_columns_)} categorical columns")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by imputing missing values.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Imputed DataFrame
        """
        if not self.fitted:
            raise ValueError("Imputer must be fitted before transformation")
        
        logger.info("Transforming data with imputation...")
        
        X_transformed = X.copy()
        
        # Create missing indicators if requested
        if self.add_missing_indicator:
            X_transformed = self._add_missing_indicators(X_transformed)
        
        # Apply imputation based on strategy
        if self.strategy == 'simple':
            # Impute numerical columns
            if 'numerical' in self.imputers_ and self.numerical_columns_:
                num_data = self.imputers_['numerical'].transform(X_transformed[self.numerical_columns_])
                X_transformed[self.numerical_columns_] = num_data
            
            # Impute categorical columns
            if 'categorical' in self.imputers_ and self.categorical_columns_:
                cat_data = self.imputers_['categorical'].transform(X_transformed[self.categorical_columns_])
                X_transformed[self.categorical_columns_] = cat_data
                
        elif self.strategy == 'knn':
            if 'knn' in self.imputers_ and self.numerical_columns_:
                num_data = self.imputers_['knn'].transform(X_transformed[self.numerical_columns_])
                X_transformed[self.numerical_columns_] = num_data
                
        elif self.strategy == 'iterative':
            if 'iterative' in self.imputers_ and self.numerical_columns_:
                num_data = self.imputers_['iterative'].transform(X_transformed[self.numerical_columns_])
                X_transformed[self.numerical_columns_] = num_data
        
        # Verify no missing values remain
        remaining_missing = X_transformed.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"{remaining_missing} missing values remain after imputation")
            # Fill remaining with column means
            X_transformed = X_transformed.fillna(X_transformed.mean())
        
        logger.info(f"Imputation complete. Remaining missing values: {X_transformed.isnull().sum().sum()}")
        
        return X_transformed
    
    def _add_missing_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary indicators for missing values.
        
        Args:
            X: DataFrame to add indicators to
            
        Returns:
            DataFrame with missing indicators
        """
        missing_cols = X.columns[X.isnull().any()].tolist()
        
        for col in missing_cols:
            indicator_name = f'{col}_missing'
            X[indicator_name] = X[col].isnull().astype(int)
            self.missing_indicators_.append(indicator_name)
        
        logger.info(f"Added {len(self.missing_indicators_)} missing indicators")
        
        return X
    
    def get_imputation_summary(self) -> pd.DataFrame:
        """
        Get summary of imputation operations.
        
        Returns:
            DataFrame with imputation details
        """
        if not self.fitted:
            raise ValueError("Imputer must be fitted first")
        
        summary = []
        
        if 'numerical_imputation' in self.column_stats_:
            for col, value in self.column_stats_['numerical_imputation'].items():
                summary.append({
                    'column': col,
                    'type': 'numerical',
                    'imputation_strategy': self.numerical_strategy,
                    'imputation_value': value
                })
        
        if 'categorical_imputation' in self.column_stats_:
            for col, value in self.column_stats_['categorical_imputation'].items():
                summary.append({
                    'column': col,
                    'type': 'categorical',
                    'imputation_strategy': self.categorical_strategy,
                    'imputation_value': value
                })
        
        return pd.DataFrame(summary)


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Feature scaler with multiple scaling strategies.
    Implements Task 3 requirements for normalization/standardization.
    """
    
    def __init__(self, 
                 strategy: str = 'standard',
                 columns: Optional[List[str]] = None,
                 with_mean: bool = True,
                 with_std: bool = True,
                 feature_range: tuple = (0, 1)):
        """
        Initialize feature scaler.
        
        Args:
            strategy: Scaling strategy ('standard', 'minmax', 'robust', 'maxabs')
            columns: Columns to scale. If None, scales all numerical columns.
            with_mean: Whether to center data (for StandardScaler)
            with_std: Whether to scale to unit variance (for StandardScaler)
            feature_range: Desired range for MinMaxScaler
        """
        self.strategy = strategy
        self.columns = columns
        self.with_mean = with_mean
        self.with_std = with_std
        self.feature_range = feature_range
        
        # Store scaler and parameters
        self.scaler_ = None
        self.scaled_columns_ = []
        self.scaling_params_ = {}
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit scaler to data.
        
        Args:
            X: DataFrame to fit on
            y: Target variable (optional)
        """
        logger.info(f"Fitting feature scaler with strategy: {self.strategy}")
        
        # Identify columns to scale
        if self.columns is None:
            self.scaled_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.scaled_columns_ = [col for col in self.columns if col in X.columns]
        
        logger.info(f"Scaling {len(self.scaled_columns_)} columns: {self.scaled_columns_[:5]}...")
        
        # Select and fit scaler
        if self.strategy == 'standard':
            self.scaler_ = StandardScaler(
                with_mean=self.with_mean,
                with_std=self.with_std
            )
            
        elif self.strategy == 'minmax':
            self.scaler_ = MinMaxScaler(
                feature_range=self.feature_range
            )
            
        elif self.strategy == 'robust':
            self.scaler_ = RobustScaler(
                with_centering=self.with_mean,
                with_scaling=self.with_std
            )
            
        elif self.strategy == 'maxabs':
            from sklearn.preprocessing import MaxAbsScaler
            self.scaler_ = MaxAbsScaler()
        
        # Fit scaler
        self.scaler_.fit(X[self.scaled_columns_])
        
        # Store scaling parameters for audit trail
        if hasattr(self.scaler_, 'scale_'):
            self.scaling_params_['scale'] = dict(
                zip(self.scaled_columns_, self.scaler_.scale_)
            )
        
        if hasattr(self.scaler_, 'mean_'):
            self.scaling_params_['mean'] = dict(
                zip(self.scaled_columns_, self.scaler_.mean_)
            )
        
        if hasattr(self.scaler_, 'min_'):
            self.scaling_params_['min'] = dict(
                zip(self.scaled_columns_, self.scaler_.min_)
            )
        
        if hasattr(self.scaler_, 'data_min_'):
            self.scaling_params_['data_min'] = dict(
                zip(self.scaled_columns_, self.scaler_.data_min_)
            )
        
        if hasattr(self.scaler_, 'data_max_'):
            self.scaling_params_['data_max'] = dict(
                zip(self.scaled_columns_, self.scaler_.data_max_)
            )
        
        self.fitted = True
        logger.info("Scaler fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by scaling features.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Scaled DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transformation")
        
        logger.info("Transforming data with scaling...")
        
        X_transformed = X.copy()
        
        # Scale specified columns
        if self.scaled_columns_:
            scaled_data = self.scaler_.transform(X_transformed[self.scaled_columns_])
            X_transformed[self.scaled_columns_] = scaled_data
        
        logger.info(f"Scaling complete for {len(self.scaled_columns_)} columns")
        
        return X_transformed
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data.
        
        Args:
            X: Scaled DataFrame
            
        Returns:
            Original scale DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transformation")
        
        X_inverse = X.copy()
        
        if self.scaled_columns_:
            inverse_data = self.scaler_.inverse_transform(X_inverse[self.scaled_columns_])
            X_inverse[self.scaled_columns_] = inverse_data
        
        return X_inverse
    
    def get_scaling_summary(self) -> pd.DataFrame:
        """
        Get summary of scaling transformations.
        
        Returns:
            DataFrame with scaling details
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted first")
        
        summary = []
        for col in self.scaled_columns_:
            col_summary = {'column': col, 'scaling_strategy': self.strategy}
            
            if 'mean' in self.scaling_params_ and col in self.scaling_params_['mean']:
                col_summary['original_mean'] = self.scaling_params_['mean'][col]
            
            if 'scale' in self.scaling_params_ and col in self.scaling_params_['scale']:
                col_summary['scale'] = self.scaling_params_['scale'][col]
            
            if 'min' in self.scaling_params_ and col in self.scaling_params_['min']:
                col_summary['transformed_min'] = self.scaling_params_['min'][col]
            
            if 'data_min' in self.scaling_params_ and col in self.scaling_params_['data_min']:
                col_summary['original_min'] = self.scaling_params_['data_min'][col]
            
            if 'data_max' in self.scaling_params_ and col in self.scaling_params_['data_max']:
                col_summary['original_max'] = self.scaling_params_['data_max'][col]
            
            summary.append(col_summary)
        
        return pd.DataFrame(summary)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handle outliers in numerical features.
    """
    
    def __init__(self, 
                 method: str = 'cap',
                 lower_percentile: float = 1,
                 upper_percentile: float = 99,
                 add_outlier_flags: bool = True):
        """
        Initialize outlier handler.
        
        Args:
            method: Outlier handling method ('cap', 'remove', 'ignore')
            lower_percentile: Lower percentile for capping
            upper_percentile: Upper percentile for capping
            add_outlier_flags: Whether to add outlier indicator columns
        """
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.add_outlier_flags = add_outlier_flags
        
        # Store outlier bounds
        self.outlier_bounds_ = {}
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit outlier handler to data.
        
        Args:
            X: DataFrame to fit on
            y: Target variable (optional)
        """
        logger.info(f"Fitting outlier handler with method: {self.method}")
        
        # Identify numerical columns
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate bounds for each column
        for col in self.numerical_columns_:
            lower_bound = np.percentile(X[col].dropna(), self.lower_percentile)
            upper_bound = np.percentile(X[col].dropna(), self.upper_percentile)
            
            self.outlier_bounds_[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'iqr': upper_bound - lower_bound
            }
        
        self.fitted = True
        logger.info(f"Calculated outlier bounds for {len(self.numerical_columns_)} columns")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by handling outliers.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            DataFrame with outliers handled
        """
        if not self.fitted:
            raise ValueError("Outlier handler must be fitted before transformation")
        
        logger.info("Transforming data with outlier handling...")
        
        X_transformed = X.copy()
        
        for col in self.numerical_columns_:
            if col in X.columns:
                bounds = self.outlier_bounds_[col]
                
                if self.method == 'cap':
                    # Cap outliers at percentiles
                    X_transformed[col] = X_transformed[col].clip(
                        lower=bounds['lower'],
                        upper=bounds['upper']
                    )
                    
                elif self.method == 'remove':
                    # Remove outliers (set to NaN, to be imputed later)
                    is_outlier = (X_transformed[col] < bounds['lower']) | (X_transformed[col] > bounds['upper'])
                    X_transformed.loc[is_outlier, col] = np.nan
                
                # Add outlier flags if requested
                if self.add_outlier_flags:
                    is_outlier = (X[col] < bounds['lower']) | (X[col] > bounds['upper'])
                    X_transformed[f'{col}_outlier'] = is_outlier.astype(int)
        
        logger.info(f"Outlier handling complete using {self.method} method")
        
        return X_transformed
    
    def get_outlier_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of outliers in data.
        
        Args:
            X: DataFrame to analyze
            
        Returns:
            DataFrame with outlier statistics
        """
        if not self.fitted:
            raise ValueError("Outlier handler must be fitted first")
        
        summary = []
        for col in self.numerical_columns_:
            if col in X.columns:
                bounds = self.outlier_bounds_[col]
                
                is_outlier = (X[col] < bounds['lower']) | (X[col] > bounds['upper'])
                n_outliers = is_outlier.sum()
                pct_outliers = (n_outliers / len(X)) * 100
                
                summary.append({
                    'column': col,
                    'lower_bound': bounds['lower'],
                    'upper_bound': bounds['upper'],
                    'n_outliers': n_outliers,
                    'pct_outliers': pct_outliers,
                    'min_value': X[col].min(),
                    'max_value': X[col].max()
                })
        
        return pd.DataFrame(summary)