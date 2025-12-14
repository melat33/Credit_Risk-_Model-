"""
Main EDA orchestrator - Clean working version
"""
import pandas as pd
import numpy as np
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import minimal components
from .overview.data_structure import DataStructureAnalyzer

class CreditRiskEDA:
    """Clean EDA orchestrator"""
    
    def __init__(self, config_path: str = "config/eda_config.yaml"):
        self.config = self._load_config(config_path)
        self.df = None
        self.logger = self._setup_logger()
        
        # Initialize minimal components
        self.data_structure = DataStructureAnalyzer()
        
        self.results = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {
                'data': {
                    'input_path': 'data/raw/data.csv',
                    'output_path': 'data/processed/'
                },
                'analysis': {
                    'correlation_threshold': 0.7,
                    'outlier_threshold': 3.0,
                    'missing_threshold': 0.3
                }
            }
    
    def _setup_logger(self):
        """Setup basic logging"""
        logger = logging.getLogger("CreditRiskEDA")
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        return logger
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load and optimize dataset"""
        if data_path is None:
            data_path = self.config['data']['input_path']
        
        self.logger.info(f"Loading data from {data_path}")
        
        # Read CSV - parse TransactionStartTime as datetime
        self.df = pd.read_csv(
            data_path,
            parse_dates=['TransactionStartTime'],
            infer_datetime_format=True,
            low_memory=False
        )
        
        # Optimize data types
        self.df = self._optimize_data_types(self.df)
        
        self.logger.info(f"Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        self.logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for Xente dataset"""
        # Convert ID columns to string
        id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
        for col in id_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Convert categorical columns to string
        cat_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 
                   'ProductCategory', 'ChannelId', 'PricingStrategy']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Downcast numerical columns
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            if df['Amount'].dtype == 'float64':
                df['Amount'] = df['Amount'].astype('float32')
        
        if 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            if df['Value'].dtype == 'float64':
                df['Value'] = df['Value'].astype('float32')
        
        # FraudResult to uint8
        if 'FraudResult' in df.columns:
            df['FraudResult'] = pd.to_numeric(df['FraudResult'], errors='coerce').fillna(0).astype('uint8')
        
        return df
    
    def run_complete_eda(self) -> Dict:
        """Run comprehensive EDA"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPLETE EDA PIPELINE")
        self.logger.info("=" * 80)
        
        if self.df is None:
            self.load_data()
        
        try:
            # Run all analyses
            self.results['data_overview'] = self._run_data_overview()
            self.results['statistics'] = self._run_descriptive_statistics()
            self.results['missing_values'] = self._run_missing_values_analysis()
            self.results['distributions'] = self._run_distribution_analysis()
            self.results['correlations'] = self._run_correlation_analysis()
            self.results['outliers'] = self._run_outlier_detection()
            self.results['temporal'] = self._run_temporal_analysis()
            self.results['fraud_analysis'] = self._run_fraud_analysis()
            
            # Save results
            self._save_results()
            self._generate_key_insights()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"EDA pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def _run_data_overview(self) -> Dict:
        """Run data overview analysis"""
        overview = self.data_structure.analyze(self.df)
        overview['duplicates'] = self.data_structure.check_duplicates(self.df)
        overview['data_types'] = self.data_structure.get_data_types(self.df)
        
        # Add column information
        overview['columns'] = {}
        for col in self.df.columns:
            overview['columns'][col] = {
                'dtype': str(self.df[col].dtype),
                'non_null': int(self.df[col].count()),
                'null': int(self.df[col].isnull().sum()),
                'null_percentage': float((self.df[col].isnull().sum() / len(self.df)) * 100),
                'unique': int(self.df[col].nunique())
            }
        
        return overview
    
    def _run_descriptive_statistics(self) -> Dict:
        """Run descriptive statistics"""
        stats = {}
        
        # Identify numerical columns
        numerical_cols = []
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numerical_cols.append(col)
        
        if len(numerical_cols) > 0:
            stats['numerical'] = {}
            for col in numerical_cols:
                data = self.df[col].dropna()
                if len(data) > 0:
                    stats['numerical'][col] = {
                        'count': int(data.count()),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        '25%': float(data.quantile(0.25)),
                        '50%': float(data.median()),
                        '75%': float(data.quantile(0.75)),
                        'max': float(data.max()),
                        'skewness': float(data.skew()),
                        'kurtosis': float(data.kurtosis())
                    }
        
        # Categorical statistics
        categorical_cols = []
        for col in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]) and col != 'TransactionStartTime':
                categorical_cols.append(col)
        
        stats['categorical'] = {}
        for col in categorical_cols:
            if self.df[col].dtype == 'object':
                value_counts = self.df[col].value_counts()
                if len(value_counts) > 0:
                    stats['categorical'][col] = {
                        'unique_values': int(value_counts.shape[0]),
                        'top_5': value_counts.head(5).to_dict(),
                        'mode': str(value_counts.index[0]),
                        'mode_percentage': float((value_counts.iloc[0] / len(self.df)) * 100)
                    }
        
        return stats
    
    def _run_missing_values_analysis(self) -> Dict:
        """Run missing values analysis"""
        missing = {}
        
        # Overall statistics
        missing['overall'] = {
            'total_missing_cells': int(self.df.isnull().sum().sum()),
            'total_missing_percentage': float((self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100),
            'columns_with_missing': int((self.df.isnull().sum() > 0).sum()),
            'complete_columns': int((self.df.isnull().sum() == 0).sum())
        }
        
        # By column
        missing['by_column'] = {}
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing['by_column'][col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float((missing_count / len(self.df)) * 100)
                }
        
        return missing
    
    def _run_distribution_analysis(self) -> Dict:
        """Run distribution analysis"""
        distributions = {}
        
        # Numerical distributions
        numerical_cols = []
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numerical_cols.append(col)
        
        distributions['numerical'] = {}
        
        for col in numerical_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                distributions['numerical'][col] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q1': float(data.quantile(0.25)),
                    'q3': float(data.quantile(0.75)),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'is_normal': abs(data.skew()) < 0.5 and abs(data.kurtosis()) < 1
                }
        
        # Categorical distributions (limit to first 5 for performance)
        categorical_cols = []
        for col in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]) and col != 'TransactionStartTime':
                categorical_cols.append(col)
        
        distributions['categorical'] = {}
        
        for col in categorical_cols[:5]:  # Limit to first 5
            if self.df[col].dtype == 'object':
                value_counts = self.df[col].value_counts()
                if len(value_counts) > 0:
                    distributions['categorical'][col] = {
                        'unique_values': int(value_counts.shape[0]),
                        'entropy': float(self._calculate_entropy(value_counts)),
                        'top_categories': value_counts.head(10).to_dict()
                    }
        
        return distributions
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution"""
        probabilities = value_counts / value_counts.sum()
        return float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))
    
    def _run_correlation_analysis(self) -> Dict:
        """Run correlation analysis"""
        correlations = {}
        
        # Get numerical columns
        numerical_cols = []
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numerical_cols.append(col)
        
        if len(numerical_cols) > 1:
            # Pearson correlation
            corr_matrix = self.df[numerical_cols].corr(method='pearson')
            correlations['pearson'] = corr_matrix.to_dict()
            
            # Strong correlations
            threshold = self.config['analysis']['correlation_threshold']
            strong_corrs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        strong_corrs.append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': float(corr_val),
                            'strength': 'Very Strong' if abs(corr_val) > 0.8 else 'Strong'
                        })
            
            correlations['strong'] = strong_corrs
        
        return correlations
    
    def _run_outlier_detection(self) -> Dict:
        """Run outlier detection"""
        outliers = {}
        
        # Get numerical columns
        numerical_cols = []
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numerical_cols.append(col)
        
        for col in numerical_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                outliers[col] = {
                    'iqr_method': {
                        'outliers_count': int(len(iqr_outliers)),
                        'outliers_percentage': float((len(iqr_outliers) / len(data)) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
                }
        
        return outliers
    
    def _run_temporal_analysis(self) -> Dict:
        """Run temporal analysis"""
        temporal = {}
        
        if 'TransactionStartTime' in self.df.columns:
            time_col = 'TransactionStartTime'
            
            # Basic temporal stats
            temporal['range'] = {
                'min': str(self.df[time_col].min()),
                'max': str(self.df[time_col].max()),
                'duration_days': float((self.df[time_col].max() - self.df[time_col].min()).days)
            }
            
            # Add temporal features
            self.df[f'{time_col}_hour'] = self.df[time_col].dt.hour
            self.df[f'{time_col}_day'] = self.df[time_col].dt.day
            self.df[f'{time_col}_month'] = self.df[time_col].dt.month
            self.df[f'{time_col}_year'] = self.df[time_col].dt.year
            self.df[f'{time_col}_dayofweek'] = self.df[time_col].dt.dayofweek
        
        return temporal
    
    def _run_fraud_analysis(self) -> Dict:
        """Run fraud analysis"""
        fraud = {}
        
        if 'FraudResult' in self.df.columns:
            # Overall statistics
            fraud['overall'] = {
                'total_transactions': int(len(self.df)),
                'fraud_transactions': int(self.df['FraudResult'].sum()),
                'fraud_percentage': float((self.df['FraudResult'].sum() / len(self.df)) * 100),
                'non_fraud_transactions': int(len(self.df) - self.df['FraudResult'].sum())
            }
            
            # Fraud by amount if available
            if 'Amount' in self.df.columns:
                fraud_mean = self.df[self.df['FraudResult'] == 1]['Amount'].mean()
                non_fraud_mean = self.df[self.df['FraudResult'] == 0]['Amount'].mean()
                
                fraud['amount_analysis'] = {
                    'fraud_mean': float(fraud_mean) if not pd.isna(fraud_mean) else 0,
                    'non_fraud_mean': float(non_fraud_mean) if not pd.isna(non_fraud_mean) else 0,
                    'ratio': float(fraud_mean / non_fraud_mean) if non_fraud_mean != 0 else 0
                }
        
        return fraud
    
    def _save_results(self):
        """Save all results to files"""
        self.logger.info("Saving results...")
        
        # Create output directory
        output_path = Path(self.config['data']['output_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save cleaned data
            self.df.to_csv(output_path / 'cleaned_data.csv', index=False)
            
            # Save results as JSON
            results_serializable = self._make_serializable(self.results)
            with open(output_path / 'eda_results.json', 'w') as f:
                json.dump(results_serializable, f, indent=4, default=str)
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return str(obj)
    
    def _generate_key_insights(self):
        """Generate key insights CSV"""
        insights = []
        
        # Data overview insights
        if 'data_overview' in self.results:
            overview = self.results['data_overview']
            insights.append({
                'category': 'Data Overview',
                'insight': f"Dataset contains {overview['rows']:,} transactions",
                'business_implication': 'Sufficient data volume for modeling',
                'action': 'Proceed with feature engineering'
            })
        
        # Missing values insights
        if 'missing_values' in self.results:
            missing = self.results['missing_values']['overall']
            if missing['total_missing_percentage'] > 0:
                insights.append({
                    'category': 'Data Quality',
                    'insight': f"{missing['total_missing_percentage']:.1f}% missing values",
                    'business_implication': 'Need imputation strategy',
                    'action': 'Implement appropriate imputation methods'
                })
        
        # Save insights
        output_path = Path(self.config['data']['output_path'])
        insights_df = pd.DataFrame(insights)
        insights_df.to_csv(output_path / 'key_insights.csv', index=False)
        
        self.logger.info(f"Generated {len(insights)} key insights")
    
    def get_clean_data(self) -> pd.DataFrame:
        """Get cleaned data for next tasks"""
        if self.df is None:
            self.load_data()
        
        df_clean = self.df.copy()
        
        # Basic cleaning
        missing_threshold = self.config['analysis']['missing_threshold']
        
        # Remove columns with too many missing values
        missing_percentages = df_clean.isnull().mean()
        cols_to_drop = missing_percentages[missing_percentages > missing_threshold].index.tolist()
        
        if cols_to_drop:
            self.logger.info(f"Dropping columns with >{missing_threshold*100}% missing values: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # Fill remaining missing values
        numerical_cols = []
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                numerical_cols.append(col)
        
        categorical_cols = []
        for col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]) and col != 'TransactionStartTime':
                categorical_cols.append(col)
        
        # Numerical: fill with median
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Categorical: fill with mode
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
        
        self.logger.info(f"Clean data shape: {df_clean.shape}")
        return df_clean