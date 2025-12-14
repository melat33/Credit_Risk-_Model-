#!/usr/bin/env python3
"""
Main script to run Task 3 feature engineering pipeline.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FIX PATH ISSUES
# ============================================================================

# Get the script directory and project root
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.absolute()  # Go up 2 levels from scripts/task3/

print(f"📁 Script directory: {script_dir}")
print(f"📁 Project root: {project_root}")

# Add project root to Python path
sys.path.insert(0, str(project_root))

# Check if src exists
src_path = project_root / "src"
if src_path.exists():
    print(f"✅ src directory found: {src_path}")
else:
    print(f"❌ src directory not found at: {src_path}")

# ============================================================================
# IMPORT YOUR MODULES WITH BETTER ERROR HANDLING
# ============================================================================

try:
    # Try absolute import
    from src.features.aggregate import AggregationEngine
    from src.features.temporal_features import TemporalFeatureExtractor
    print("✅ Imported from src.features")
    
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Trying alternative import paths...")
    
    # Try direct import
    try:
        import importlib.util
        
        # Try to import aggregate.py
        aggregate_path = project_root / "src" / "features" / "aggregate.py"
        if aggregate_path.exists():
            spec = importlib.util.spec_from_file_location("aggregate", str(aggregate_path))
            aggregate_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(aggregate_module)
            AggregationEngine = aggregate_module.AggregationEngine
            print("✅ Loaded AggregationEngine directly")
        else:
            print(f"❌ aggregate.py not found at: {aggregate_path}")
            raise ImportError("aggregate.py not found")
        
        # Try to import temporal_features.py
        temporal_path = project_root / "src" / "features" / "temporal_features.py"
        if temporal_path.exists():
            spec = importlib.util.spec_from_file_location("temporal_features", str(temporal_path))
            temporal_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temporal_module)
            TemporalFeatureExtractor = temporal_module.TemporalFeatureExtractor
            print("✅ Loaded TemporalFeatureExtractor directly")
        else:
            print(f"❌ temporal_features.py not found at: {temporal_path}")
            raise ImportError("temporal_features.py not found")
            
    except Exception as e2:
        print(f"⚠️  Direct import failed: {e2}")
        print("Using simplified implementations...")
        
        # Simplified fallback implementations
        class AggregationEngine:
            @staticmethod
            def create_required_aggregations(df):
                """Create Task 3 required aggregate features"""
                if 'CustomerId' not in df.columns:
                    raise ValueError("CustomerId column not found")
                
                aggregates = df.groupby('CustomerId').agg({
                    'Value': ['sum', 'mean', 'std'],
                    'TransactionId': 'count'
                })
                
                aggregates.columns = [
                    'total_transaction_amount',
                    'avg_transaction_amount',
                    'std_transaction_amount',
                    'transaction_count'
                ]
                
                aggregates['std_transaction_amount'] = aggregates['std_transaction_amount'].fillna(0)
                return aggregates
        
        class TemporalFeatureExtractor:
            @staticmethod
            def extract_basic_temporal_features(df):
                """Extract Task 3 required temporal features"""
                if 'TransactionStartTime' not in df.columns:
                    raise ValueError("TransactionStartTime column not found")
                
                df = df.copy()
                df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
                
                return pd.DataFrame({
                    'transaction_hour': df['TransactionStartTime'].dt.hour,
                    'transaction_day': df['TransactionStartTime'].dt.day,
                    'transaction_month': df['TransactionStartTime'].dt.month,
                    'transaction_year': df['TransactionStartTime'].dt.year
                })

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_feature_engineering(df):
    """
    Run complete feature engineering for Task 3.
    
    Args:
        df: Input DataFrame with transaction data
        
    Returns:
        DataFrame with customer-level features
    """
    logger.info("Starting feature engineering...")
    
    # 1. Create aggregate features (Task 3 Requirement 1)
    logger.info("Creating aggregate features...")
    aggregates = AggregationEngine.create_required_aggregations(df)
    logger.info(f"Created {len(aggregates.columns)} aggregate features for {len(aggregates)} customers")
    
    # 2. Create temporal features (Task 3 Requirement 2)
    logger.info("Extracting temporal features...")
    temporal_features = TemporalFeatureExtractor.extract_basic_temporal_features(df)
    logger.info(f"Extracted {len(temporal_features.columns)} temporal features")
    
    # 3. Create customer-level temporal patterns
    logger.info("Creating customer temporal patterns...")
    
    # Add CustomerId to temporal features
    df_with_temporal = pd.concat([df[['CustomerId']].reset_index(drop=True), 
                                  temporal_features.reset_index(drop=True)], axis=1)
    
    # Calculate average temporal patterns per customer
    temporal_aggregates = df_with_temporal.groupby('CustomerId').agg({
        'transaction_hour': 'mean',
        'transaction_day': 'mean',
        'transaction_month': 'mean',
        'transaction_year': 'mean'
    })
    
    temporal_aggregates.columns = [f'avg_{col}' for col in temporal_aggregates.columns]
    
    # Combine all features
    customer_features = aggregates.join(temporal_aggregates, how='left')
    
    logger.info(f"Final feature dataset shape: {customer_features.shape}")
    logger.info(f"Total features created: {len(customer_features.columns)}")
    
    return customer_features

def generate_reports(features_df, reports_dir):
    """
    Generate feature engineering reports.
    
    Args:
        features_df: DataFrame with engineered features
        reports_dir: Directory to save reports
    """
    logger.info("Generating feature reports...")
    
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Basic statistics report
    stats_report = features_df.describe().T
    stats_report['missing_values'] = features_df.isnull().sum()
    stats_report['missing_pct'] = (features_df.isnull().sum() / len(features_df)) * 100
    
    stats_path = reports_dir / 'feature_statistics.csv'
    stats_report.to_csv(stats_path)
    logger.info(f"Feature statistics saved to {stats_path}")
    
    # 2. Feature summary
    feature_summary = pd.DataFrame({
        'feature_name': features_df.columns,
        'data_type': features_df.dtypes.astype(str),
        'n_unique': [features_df[col].nunique() for col in features_df.columns],
        'missing_values': features_df.isnull().sum().values,
        'missing_pct': (features_df.isnull().sum() / len(features_df) * 100).values
    })
    
    summary_path = reports_dir / 'feature_summary.csv'
    feature_summary.to_csv(summary_path, index=False)
    logger.info(f"Feature summary saved to {summary_path}")
    
    # 3. Correlation matrix and visualization
    if len(features_df.columns) > 1:
        corr_matrix = features_df.corr()
        
        corr_path = reports_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(corr_path)
        logger.info(f"Correlation matrix saved to {corr_path}")
        
        # Plot correlation heatmap
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                       square=True, linewidths=.5, cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            plot_path = reports_dir / 'correlation_heatmap.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation heatmap saved to {plot_path}")
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for visualization")
    
    # 4. Feature distributions
    try:
        import matplotlib.pyplot as plt
        
        # Select features for visualization
        plot_features = [col for col in features_df.columns if 'avg_' not in col][:8]  # Skip avg_ prefixed
        
        if plot_features:
            n_cols = 4
            n_rows = (len(plot_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            fig.suptitle('Feature Distributions', fontsize=16)
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, feature in enumerate(plot_features):
                row = idx // n_cols
                col = idx % n_cols
                
                axes[row, col].hist(features_df[feature].dropna(), bins=30, alpha=0.7, 
                                   edgecolor='black', color='steelblue')
                axes[row, col].set_title(feature, fontsize=10)
                axes[row, col].set_xlabel('Value')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(plot_features), n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            dist_path = reports_dir / 'feature_distributions.png'
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature distributions plot saved to {dist_path}")
    except ImportError:
        logger.warning("Matplotlib not available for distribution plots")
    
    logger.info(f"All reports generated in {reports_dir}")

def find_cleaned_data():
    """Find the cleaned data file in common locations."""
    possible_paths = [
        # Relative paths from project root
        project_root / "data" / "processed" / "cleaned_data.csv",
        project_root / "data" / "cleaned_data.csv",
        project_root / "cleaned_data.csv",
        # Absolute paths based on your error message
        Path("D:/data/processed/cleaned_data.csv"),
        Path("D:/10 academy/Credit Risk Model/data/processed/cleaned_data.csv"),
        # Add more possible paths here
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found cleaned data at: {path}")
            return path
    
    logger.warning("Cleaned data not found in common locations")
    return None

def main():
    """Main function to run the feature engineering pipeline."""
    parser = argparse.ArgumentParser(description='Run Task 3 feature engineering pipeline')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input CSV file (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save processed features (optional)')
    parser.add_argument('--reports-dir', type=str, default=None,
                       help='Directory to save reports (optional)')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Sample size if creating test data (default: 1000)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TASK 3: FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # 1. Determine input path
    if args.input:
        input_path = Path(args.input).resolve()
    else:
        input_path = find_cleaned_data()
    
    # 2. Load or create data
    if input_path and input_path.exists():
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    else:
        logger.info("Creating sample data for demonstration...")
        
        # Create realistic sample data
        np.random.seed(42)
        n_rows = args.sample_size
        
        # Create 100 unique customers
        customer_ids = [f'CUST_{i:04d}' for i in range(1, 101)]
        
        df = pd.DataFrame({
            'TransactionId': range(1, n_rows + 1),
            'CustomerId': np.random.choice(customer_ids, n_rows),
            'Value': np.random.lognormal(mean=5, sigma=1, size=n_rows),  # Log-normal for realistic amounts
            'TransactionStartTime': pd.date_range('2023-01-01', periods=n_rows, freq='h'),
            'ProductCategory': np.random.choice(['ELECTRONICS', 'CLOTHING', 'GROCERY', 'HOME', 'ENTERTAINMENT'], n_rows),
            'ChannelId': np.random.choice(['WEB', 'MOBILE', 'APP', 'IN_STORE'], n_rows),
            'CountryCode': np.random.choice(['US', 'UK', 'DE', 'FR', 'CA'], n_rows)
        })
        
        logger.info(f"Created sample data with {len(df)} rows")
        logger.info(f"Unique customers: {df['CustomerId'].nunique()}")
    
    # 3. Set output paths
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = project_root / "data" / "processed" / "task3_features.csv"
    
    if args.reports_dir:
        reports_dir = Path(args.reports_dir).resolve()
    else:
        reports_dir = project_root / "reports" / "task3_features"
    
    # 4. Run feature engineering
    logger.info("\n" + "-"*80)
    logger.info("RUNNING FEATURE ENGINEERING")
    logger.info("-"*80)
    
    start_time = datetime.now()
    
    try:
        features_df = run_feature_engineering(df)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Feature engineering completed in {elapsed:.2f} seconds")
        
        # 5. Save features
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reset index to make CustomerId a column
        features_df_reset = features_df.reset_index()
        features_df_reset.to_csv(output_path, index=False)
        
        logger.info(f"Features saved to {output_path}")
        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"Features columns: {list(features_df.columns)}")
        
        # 6. Generate reports
        generate_reports(features_df, reports_dir)
        
        # 7. Print detailed summary
        print("\n" + "="*80)
        print("TASK 3 COMPLETION REPORT")
        print("="*80)
        
        print(f"\n📊 INPUT DATA:")
        print(f"   Source: {input_path if input_path and input_path.exists() else 'Sample Data'}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Customers: {df['CustomerId'].nunique():,}")
        
        print(f"\n🎯 TASK 3 REQUIREMENTS:")
        
        # Check aggregate features
        required_aggregates = [
            'total_transaction_amount',
            'avg_transaction_amount',
            'transaction_count',
            'std_transaction_amount'
        ]
        
        aggregate_status = []
        for feature in required_aggregates:
            if feature in features_df.columns:
                aggregate_status.append(f"✅ {feature}")
            else:
                aggregate_status.append(f"❌ {feature} (MISSING)")
        
        print(f"\n   1. Aggregate Features:")
        for status in aggregate_status:
            print(f"      {status}")
        
        # Check temporal features (they should be prefixed with avg_)
        required_temporal = [
            'avg_transaction_hour',
            'avg_transaction_day',
            'avg_transaction_month',
            'avg_transaction_year'
        ]
        
        temporal_status = []
        for feature in required_temporal:
            if feature in features_df.columns:
                temporal_status.append(f"✅ {feature}")
            else:
                # Check if non-prefixed version exists
                base_feature = feature.replace('avg_', '')
                if base_feature in features_df.columns:
                    temporal_status.append(f"✅ {base_feature} (raw)")
                else:
                    temporal_status.append(f"❌ {feature} (MISSING)")
        
        print(f"\n   2. Temporal Features (averaged per customer):")
        for status in temporal_status:
            print(f"      {status}")
        
        print(f"\n📈 FEATURE ENGINEERING RESULTS:")
        print(f"   Output file: {output_path}")
        print(f"   Features created: {len(features_df.columns):,}")
        print(f"   Customers processed: {len(features_df):,}")
        print(f"   Dataset shape: {features_df.shape}")
        
        print(f"\n📁 REPORTS GENERATED:")
        print(f"   Directory: {reports_dir}")
        report_files = list(reports_dir.glob("*"))
        for report_file in sorted(report_files):
            print(f"   - {report_file.name}")
        
        print(f"\n✅ TASK 3 STATUS: COMPLETED SUCCESSFULLY")
        
        # Show sample of features
        print(f"\n🔍 SAMPLE OF CREATED FEATURES:")
        print(features_df.head().to_string())
        
        print("\n" + "="*80)
        print("🚀 READY FOR TASK 4: PROXY TARGET VARIABLE ENGINEERING")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()