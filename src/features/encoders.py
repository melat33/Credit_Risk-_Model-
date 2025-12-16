"""
Categorical encoding implementations for Task 3.
Supports: One-Hot Encoding, Label Encoding, and Frequency Encoding
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
import logging

logger = logging.getLogger(__name__)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Flexible categorical encoder supporting multiple strategies.
    Implements Task 3 requirements for categorical variable encoding.
    """
    
    def __init__(self, 
                 encoding_strategy: str = 'onehot',
                 columns: Optional[List[str]] = None,
                 handle_unknown: str = 'error',
                 min_frequency: float = 0.01,
                 max_categories: int = 20):
        """
        Initialize categorical encoder.
        
        Args:
            encoding_strategy: 'onehot', 'label', 'frequency', 'binary'
            columns: List of columns to encode. If None, encodes all object/category columns.
            handle_unknown: How to handle unknown categories ('error', 'ignore', 'encode')
            min_frequency: Minimum frequency for category to be kept (for frequency encoding)
            max_categories: Maximum number of categories for one-hot encoding
        """
        self.encoding_strategy = encoding_strategy
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        
        # Store encoders for each column
        self.encoders_ = {}
        self.column_categories_ = {}
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit encoder to data.
        
        Args:
            X: DataFrame containing categorical columns
            y: Target variable (optional, used for some encoding strategies)
        """
        logger.info(f"Fitting categorical encoder with strategy: {self.encoding_strategy}")
        
        # Identify columns to encode
        if self.columns is None:
            self.columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.columns_ = [col for col in self.columns if col in X.columns]
        
        logger.info(f"Encoding columns: {self.columns_}")
        
        # Fit encoder for each column
        for col in self.columns_:
            # Get unique categories
            categories = X[col].dropna().unique()
            self.column_categories_[col] = categories
            
            if self.encoding_strategy == 'label':
                encoder = LabelEncoder()
                encoder.fit(categories)
                self.encoders_[col] = encoder
                
            elif self.encoding_strategy == 'onehot':
                # For one-hot, we'll handle it differently in transform
                pass
                
            elif self.encoding_strategy == 'frequency':
                # Calculate frequencies
                frequencies = X[col].value_counts(normalize=True)
                self.encoders_[col] = frequencies.to_dict()
                
            elif self.encoding_strategy == 'binary':
                # Binary encoding using category_encoders
                encoder = ce.BinaryEncoder(cols=[col], handle_unknown='value')
                encoder.fit(X[[col]])
                self.encoders_[col] = encoder
        
        self.fitted = True
        logger.info(f"Fitted encoder on {len(self.columns_)} columns")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transformation")
        
        logger.info(f"Transforming categorical columns with {self.encoding_strategy} encoding")
        
        X_transformed = X.copy()
        
        if self.encoding_strategy == 'onehot':
            # One-hot encode all columns at once
            X_transformed = self._onehot_encode(X_transformed)
            
        else:
            # Encode each column individually
            for col in self.columns_:
                if col in X_transformed.columns:
                    if self.encoding_strategy == 'label':
                        X_transformed[col] = self._label_encode_column(X_transformed, col)
                        
                    elif self.encoding_strategy == 'frequency':
                        X_transformed[col] = self._frequency_encode_column(X_transformed, col)
                        
                    elif self.encoding_strategy == 'binary':
                        X_transformed = self._binary_encode_column(X_transformed, col)
        
        logger.info(f"Transformed {len(self.columns_)} categorical columns")
        
        return X_transformed
    
    def _onehot_encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform one-hot encoding."""
        # Use pandas get_dummies for simplicity
        encoded_df = pd.get_dummies(
            X, 
            columns=self.columns_,
            prefix=self.columns_,
            drop_first=False,
            dtype=int
        )
        
        return encoded_df
    
    def _label_encode_column(self, X: pd.DataFrame, col: str) -> pd.Series:
        """Label encode a single column."""
        encoder = self.encoders_[col]
        
        # Handle unseen categories
        def encode_value(val):
            if pd.isna(val):
                return -1
            elif val in encoder.classes_:
                return encoder.transform([val])[0]
            else:
                # Handle unknown based on strategy
                if self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category {val} in column {col}")
                elif self.handle_unknown == 'encode':
                    return len(encoder.classes_)  # Encode as new category
                else:  # 'ignore'
                    return -1
        
        return X[col].apply(encode_value)
    
    def _frequency_encode_column(self, X: pd.DataFrame, col: str) -> pd.Series:
        """Frequency encode a single column."""
        freq_dict = self.encoders_[col]
        
        def encode_value(val):
            if pd.isna(val):
                return 0
            elif val in freq_dict:
                return freq_dict[val]
            else:
                # Handle unknown
                if self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category {val} in column {col}")
                else:
                    return 0  # Treat as rare category
        
        return X[col].apply(encode_value)
    
    def _binary_encode_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        """Binary encode a single column."""
        encoder = self.encoders_[col]
        encoded = encoder.transform(X[[col]])
        
        # Drop original column and add encoded ones
        X = X.drop(columns=[col])
        X = pd.concat([X, encoded], axis=1)
        
        return X
    
    def get_feature_names(self, input_features: List[str] = None) -> List[str]:
        """
        Get names of transformed features.
        
        Args:
            input_features: Original feature names
            
        Returns:
            List of transformed feature names
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        
        if self.encoding_strategy == 'onehot':
            # For one-hot, we can't know feature names without seeing data
            # In practice, you'd store the column names from fit
            return []
        
        # For other strategies, column names remain the same
        return self.columns_
    
    def get_encoding_summary(self) -> pd.DataFrame:
        """
        Get summary of encoding transformations.
        
        Returns:
            DataFrame with encoding details
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        
        summary = []
        for col in self.columns_:
            if self.encoding_strategy == 'label':
                encoder = self.encoders_[col]
                n_categories = len(encoder.classes_)
                categories = list(encoder.classes_)
                
            elif self.encoding_strategy == 'frequency':
                freq_dict = self.encoders_[col]
                n_categories = len(freq_dict)
                categories = list(freq_dict.keys())
                
            else:
                n_categories = len(self.column_categories_[col])
                categories = list(self.column_categories_[col])
            
            summary.append({
                'column': col,
                'encoding_strategy': self.encoding_strategy,
                'n_categories': n_categories,
                'categories': str(categories[:10]) + ('...' if len(categories) > 10 else '')
            })
        
        return pd.DataFrame(summary)


class SmartCategoricalEncoder(CategoricalEncoder):
    """
    Smart encoder that automatically chooses encoding strategy based on data.
    """
    
    def __init__(self, 
                 max_onehot_categories: int = 15,
                 min_frequency_for_rare: float = 0.01,
                 handle_unknown: str = 'encode'):
        """
        Initialize smart encoder.
        
        Args:
            max_onehot_categories: Maximum categories for one-hot encoding
            min_frequency_for_rare: Minimum frequency to not be considered rare
            handle_unknown: How to handle unknown categories
        """
        self.max_onehot_categories = max_onehot_categories
        self.min_frequency_for_rare = min_frequency_for_rare
        self.handle_unknown = handle_unknown
        
        # Store encoding decisions
        self.encoding_decisions_ = {}
        self.encoders_ = {}
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit smart encoder, choosing best strategy for each column.
        """
        logger.info("Fitting smart categorical encoder...")
        
        # Identify categorical columns
        self.columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.columns_:
            n_unique = X[col].nunique()
            frequencies = X[col].value_counts(normalize=True)
            
            # Decision logic
            if n_unique <= 2:
                # Binary column - use label encoding
                strategy = 'label'
                
            elif n_unique <= self.max_onehot_categories:
                # Few categories - use one-hot
                strategy = 'onehot'
                
            else:
                # Many categories - use frequency encoding
                strategy = 'frequency'
                
                # Check for rare categories
                rare_categories = frequencies[frequencies < self.min_frequency_for_rare].index
                if len(rare_categories) > 0:
                    logger.info(f"Column {col} has {len(rare_categories)} rare categories")
            
            self.encoding_decisions_[col] = {
                'strategy': strategy,
                'n_unique': n_unique,
                'most_common': frequencies.index[0] if len(frequencies) > 0 else None,
                'most_common_freq': frequencies.iloc[0] if len(frequencies) > 0 else None
            }
            
            # Fit appropriate encoder
            if strategy == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[col].dropna().unique())
                self.encoders_[col] = encoder
                
            elif strategy == 'frequency':
                self.encoders_[col] = frequencies.to_dict()
        
        self.fitted = True
        
        # Create summary
        summary_df = pd.DataFrame.from_dict(self.encoding_decisions_, orient='index')
        logger.info(f"Encoding decisions:\n{summary_df[['strategy', 'n_unique']].to_string()}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform using chosen encoding strategies.
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transformation")
        
        X_transformed = X.copy()
        
        for col, decision in self.encoding_decisions_.items():
            if col in X.columns:
                strategy = decision['strategy']
                
                if strategy == 'label':
                    encoder = self.encoders_[col]
                    X_transformed[col] = X[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )
                    
                elif strategy == 'onehot':
                    # One-hot encode using pandas
                    dummies = pd.get_dummies(X[col], prefix=col, dtype=int)
                    X_transformed = pd.concat([X_transformed, dummies], axis=1)
                    X_transformed = X_transformed.drop(columns=[col])
                    
                elif strategy == 'frequency':
                    freq_dict = self.encoders_[col]
                    X_transformed[col] = X[col].map(freq_dict).fillna(0)
        
        logger.info(f"Smart encoding complete. Original columns: {len(self.columns_)}, "
                   f"Transformed shape: {X_transformed.shape}")
        
        return X_transformed