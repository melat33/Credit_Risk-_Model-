"""
Weight of Evidence (WoE) and Information Value (IV) implementation.
Implements Task 3 requirements for WoE transformation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import logging
from scipy import stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) transformer for categorical/binomial features.
    Banking-standard implementation for credit risk modeling.
    """
    
    def __init__(self, 
                 target_column: str = 'is_high_risk',
                 min_bin_size: int = 50,
                 max_bins: int = 10,
                 iv_threshold: float = 0.02,
                 monotonic_binning: bool = True,
                 handle_unseen: str = 'return_nan'):
        """
        Initialize WoE transformer.
        
        Args:
            target_column: Name of binary target column
            min_bin_size: Minimum number of observations per bin
            max_bins: Maximum number of bins
            iv_threshold: Minimum IV value to keep feature
            monotonic_binning: Whether to enforce monotonic WoE
            handle_unseen: How to handle unseen categories ('return_nan', 'woe_mean', 'most_frequent')
        """
        self.target_column = target_column
        self.min_bin_size = min_bin_size
        self.max_bins = max_bins
        self.iv_threshold = iv_threshold
        self.monotonic_binning = monotonic_binning
        self.handle_unseen = handle_unseen
        
        # Store WoE mappings and IV values
        self.woe_mappings_ = {}
        self.iv_values_ = {}
        self.bin_edges_ = {}
        self.selected_features_ = []
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit WoE transformer to data.
        
        Args:
            X: Feature DataFrame
            y: Target series. If None, uses self.target_column from X
        """
        logger.info("Fitting WoE transformer...")
        
        # Get target variable
        if y is None:
            if self.target_column in X.columns:
                y = X[self.target_column]
                X = X.drop(columns=[self.target_column])
            else:
                raise ValueError(f"Target column '{self.target_column}' not found in X")
        
        # Identify features to transform
        self.feature_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Calculating WoE/IV for {len(self.feature_columns_)} features...")
        
        # Calculate WoE and IV for each feature
        for feature in self.feature_columns_:
            try:
                woe_mapping, iv, bin_edges = self._calculate_woe_iv(
                    X[feature], 
                    y,
                    feature_name=feature
                )
                
                self.woe_mappings_[feature] = woe_mapping
                self.iv_values_[feature] = iv
                self.bin_edges_[feature] = bin_edges
                
                # Select features based on IV threshold
                if iv >= self.iv_threshold:
                    self.selected_features_.append(feature)
                    logger.info(f"  {feature}: IV = {iv:.4f} (SELECTED)")
                else:
                    logger.info(f"  {feature}: IV = {iv:.4f} (DISCARDED)")
                    
            except Exception as e:
                logger.warning(f"Failed to calculate WoE/IV for {feature}: {e}")
        
        self.fitted = True
        
        # Create IV summary
        self.iv_summary_ = pd.DataFrame({
            'feature': list(self.iv_values_.keys()),
            'iv_value': list(self.iv_values_.values()),
            'selected': [feat in self.selected_features_ for feat in self.iv_values_.keys()]
        }).sort_values('iv_value', ascending=False)
        
        logger.info(f"WoE fitting complete. Selected {len(self.selected_features_)} features "
                   f"(IV >= {self.iv_threshold})")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using WoE encoding.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            WoE-transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("WoE transformer must be fitted before transformation")
        
        logger.info("Transforming features with WoE encoding...")
        
        X_transformed = X.copy()
        
        # Apply WoE transformation to selected features
        for feature in self.selected_features_:
            if feature in X.columns:
                X_transformed[feature] = self._apply_woe_mapping(
                    X_transformed[feature], 
                    self.woe_mappings_[feature],
                    self.bin_edges_[feature]
                )
        
        # Drop non-selected features
        features_to_drop = [f for f in self.feature_columns_ if f not in self.selected_features_]
        X_transformed = X_transformed.drop(columns=features_to_drop)
        
        logger.info(f"WoE transformation complete. Kept {len(self.selected_features_)} features, "
                   f"dropped {len(features_to_drop)} features")
        
        return X_transformed
    
    def _calculate_woe_iv(self, x: pd.Series, y: pd.Series, feature_name: str) -> Tuple[Dict, float, List]:
        """
        Calculate WoE and IV for a single feature.
        
        Args:
            x: Feature values
            y: Binary target
            feature_name: Name of feature
            
        Returns:
            Tuple of (woe_mapping, iv_value, bin_edges)
        """
        # Remove NaN values
        mask = ~x.isna()
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) == 0:
            return {}, 0.0, []
        
        # Create bins for continuous features
        if x_clean.nunique() > self.max_bins:
            # Use quantile-based binning
            bins = pd.qcut(
                x_clean, 
                q=self.max_bins,
                duplicates='drop',
                retbins=True
            )
            binned = bins[0]
            bin_edges = bins[1].tolist()
        else:
            # Use unique values for categorical/discrete features
            binned = x_clean
            bin_edges = sorted(x_clean.unique())
        
        # Calculate WoE for each bin
        woe_mapping = {}
        iv_total = 0
        
        for bin_label in binned.cat.categories if hasattr(binned, 'cat') else binned.unique():
            if pd.isna(bin_label):
                continue
                
            # Get observations in this bin
            if hasattr(binned, 'cat'):
                mask_bin = (binned == bin_label)
            else:
                mask_bin = (binned == bin_label)
            
            y_bin = y_clean[mask_bin]
            
            # Calculate good/bad counts
            good_count = (y_bin == 0).sum()
            bad_count = (y_bin == 1).sum()
            
            # Avoid division by zero
            good_dist = good_count / (y_clean == 0).sum()
            bad_dist = bad_count / (y_clean == 1).sum()
            
            # Add small epsilon to avoid zero
            good_dist = max(good_dist, 1e-10)
            bad_dist = max(bad_dist, 1e-10)
            
            # Calculate WoE
            woe = np.log(bad_dist / good_dist)
            
            # Calculate IV contribution
            iv_contribution = (bad_dist - good_dist) * woe
            
            # Store mapping
            if isinstance(bin_label, pd.Interval):
                woe_mapping[bin_label] = woe
            else:
                woe_mapping[bin_label] = woe
            
            iv_total += iv_contribution
        
        # Check monotonicity if requested
        if self.monotonic_binning and len(woe_mapping) > 2:
            woe_values = list(woe_mapping.values())
            is_monotonic = self._check_monotonic(woe_values)
            
            if not is_monotonic:
                logger.warning(f"WoE for {feature_name} is not monotonic. IV: {iv_total:.4f}")
        
        return woe_mapping, iv_total, bin_edges
    
    def _apply_woe_mapping(self, x: pd.Series, woe_mapping: Dict, bin_edges: List) -> pd.Series:
        """
        Apply WoE mapping to feature values.
        
        Args:
            x: Feature values
            woe_mapping: WoE mapping dictionary
            bin_edges: Bin edges for continuous features
            
        Returns:
            WoE-encoded series
        """
        result = pd.Series(index=x.index, dtype=float)
        
        for idx, value in x.items():
            if pd.isna(value):
                result[idx] = np.nan
                continue
            
            # Find appropriate bin for continuous features
            if len(bin_edges) > 1 and not isinstance(list(woe_mapping.keys())[0], (str, int)):
                # Continuous feature with interval bins
                for interval, woe in woe_mapping.items():
                    if interval.left <= value <= interval.right:
                        result[idx] = woe
                        break
                else:
                    # Value outside all bins
                    result[idx] = self._handle_unseen_value(woe_mapping)
            else:
                # Categorical feature
                if value in woe_mapping:
                    result[idx] = woe_mapping[value]
                else:
                    # Unseen category
                    result[idx] = self._handle_unseen_value(woe_mapping)
        
        return result
    
    def _handle_unseen_value(self, woe_mapping: Dict) -> float:
        """
        Handle unseen values based on strategy.
        
        Args:
            woe_mapping: WoE mapping dictionary
            
        Returns:
            WoE value for unseen category
        """
        if self.handle_unseen == 'return_nan':
            return np.nan
        elif self.handle_unseen == 'woe_mean':
            return np.mean(list(woe_mapping.values()))
        elif self.handle_unseen == 'most_frequent':
            # Return WoE of most frequent bin (approximation)
            return list(woe_mapping.values())[0]
        else:
            return np.nan
    
    def _check_monotonic(self, values: List[float]) -> bool:
        """
        Check if WoE values are monotonic.
        
        Args:
            values: List of WoE values
            
        Returns:
            True if monotonic, False otherwise
        """
        if len(values) <= 2:
            return True
        
        # Check monotonic increasing
        increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        
        # Check monotonic decreasing
        decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
        
        return increasing or decreasing
    
    def get_iv_summary(self) -> pd.DataFrame:
        """
        Get Information Value summary.
        
        Returns:
            DataFrame with IV values and strength categories
        """
        if not self.fitted:
            raise ValueError("WoE transformer must be fitted first")
        
        def iv_strength(iv_value):
            if iv_value < 0.02:
                return "Useless"
            elif iv_value < 0.1:
                return "Weak"
            elif iv_value < 0.3:
                return "Medium"
            elif iv_value < 0.5:
                return "Strong"
            else:
                return "Suspicious"
        
        summary = self.iv_summary_.copy()
        summary['strength'] = summary['iv_value'].apply(iv_strength)
        
        return summary
    
    def plot_woe_bins(self, feature_name: str, figsize=(12, 8)):
        """
        Plot WoE bins for a feature.
        
        Args:
            feature_name: Name of feature to plot
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if feature_name not in self.woe_mappings_:
            raise ValueError(f"Feature '{feature_name}' not found in WoE mappings")
        
        woe_mapping = self.woe_mappings_[feature_name]
        iv_value = self.iv_values_[feature_name]
        
        # Prepare data for plotting
        if isinstance(list(woe_mapping.keys())[0], pd.Interval):
            # Continuous feature with interval bins
            bin_labels = [f"{interval.left:.2f}-{interval.right:.2f}" 
                         for interval in woe_mapping.keys()]
        else:
            # Categorical feature
            bin_labels = [str(key) for key in woe_mapping.keys()]
        
        woe_values = list(woe_mapping.values())
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: WoE values
        bars1 = ax1.bar(range(len(woe_values)), woe_values, color='steelblue')
        ax1.set_xlabel('Bin')
        ax1.set_ylabel('Weight of Evidence (WoE)')
        ax1.set_title(f'{feature_name} - WoE Bins (IV = {iv_value:.4f})')
        ax1.set_xticks(range(len(bin_labels)))
        ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Plot 2: Monotonicity check
        ax2.plot(range(len(woe_values)), woe_values, marker='o', color='darkred', linewidth=2)
        ax2.set_xlabel('Bin')
        ax2.set_ylabel('WoE')
        ax2.set_title('WoE Monotonicity Check')
        ax2.set_xticks(range(len(bin_labels)))
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Check and annotate monotonicity
        is_monotonic = self._check_monotonic(woe_values)
        monotonic_text = "Monotonic ✓" if is_monotonic else "Not Monotonic ✗"
        ax2.text(0.02, 0.98, monotonic_text, 
                transform=ax2.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='green' if is_monotonic else 'red', alpha=0.3))
        
        plt.tight_layout()
        
        return fig


def calculate_iv(df: pd.DataFrame, feature: str, target: str) -> float:
    """
    Calculate Information Value for a single feature.
    
    Args:
        df: DataFrame containing feature and target
        feature: Feature column name
        target: Target column name
        
    Returns:
        Information Value
    """
    # Group data
    temp_df = df[[feature, target]].copy()
    temp_df = temp_df.dropna()
    
    # Calculate good/bad counts
    good = temp_df[temp_df[target] == 0].groupby(feature).size()
    bad = temp_df[temp_df[target] == 1].groupby(feature).size()
    
    # Create DataFrame with all categories
    iv_df = pd.DataFrame({'good': good, 'bad': bad}).fillna(0.5)  # Add 0.5 for smoothing
    
    # Calculate distributions
    good_total = iv_df['good'].sum()
    bad_total = iv_df['bad'].sum()
    
    iv_df['good_dist'] = iv_df['good'] / good_total
    iv_df['bad_dist'] = iv_df['bad'] / bad_total
    
    # Calculate WoE and IV
    iv_df['woe'] = np.log(iv_df['bad_dist'] / iv_df['good_dist'])
    iv_df['iv_contribution'] = (iv_df['bad_dist'] - iv_df['good_dist']) * iv_df['woe']
    
    # Total IV
    iv_total = iv_df['iv_contribution'].sum()
    
    return iv_total


def calculate_iv_for_all_features(df: pd.DataFrame, target: str, 
                                 features: List[str] = None) -> pd.DataFrame:
    """
    Calculate Information Value for all features.
    
    Args:
        df: DataFrame with features and target
        target: Target column name
        features: List of features to calculate IV for. If None, uses all numeric features.
        
    Returns:
        DataFrame with IV values for all features
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f != target]
    
    iv_results = []
    
    for feature in features:
        try:
            iv = calculate_iv(df, feature, target)
            iv_results.append({
                'feature': feature,
                'iv': iv
            })
        except Exception as e:
            print(f"Error calculating IV for {feature}: {e}")
            iv_results.append({
                'feature': feature,
                'iv': np.nan
            })
    
    iv_df = pd.DataFrame(iv_results).sort_values('iv', ascending=False)
    
    # Add strength categories
    def categorize_iv(iv_value):
        if pd.isna(iv_value):
            return 'Error'
        elif iv_value < 0.02:
            return 'Useless'
        elif iv_value < 0.1:
            return 'Weak'
        elif iv_value < 0.3:
            return 'Medium'
        elif iv_value < 0.5:
            return 'Strong'
        else:
            return 'Suspicious'
    
    iv_df['strength'] = iv_df['iv'].apply(categorize_iv)
    
    return iv_df