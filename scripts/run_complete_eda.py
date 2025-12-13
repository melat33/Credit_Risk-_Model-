#!/usr/bin/env python3
"""
Main script to run complete EDA pipeline for credit risk modeling
"""
import sys
import os
from pathlib import Path
import argparse
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from eda.data_explorer import CreditRiskEDA


def main():
    """Run complete EDA pipeline"""
    parser = argparse.ArgumentParser(description='Run complete EDA for credit risk modeling')
    parser.add_argument('--config', type=str, default='config/eda_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data file (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size (for testing)')
    parser.add_argument('--clean-only', action='store_true',
                       help='Only clean data, skip full EDA')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CREDIT RISK MODELING - COMPLETE EDA PIPELINE")
    print("="*80)
    
    try:
        # Initialize EDA engine
        print("\n1. 📊 Initializing EDA Engine...")
        eda_engine = CreditRiskEDA(args.config)
        
        # Override config if arguments provided
        if args.data:
            eda_engine.config['data']['input_path'] = args.data
        
        if args.output:
            eda_engine.config['data']['output_path'] = args.output
        
        if args.sample:
            eda_engine.config['data']['sample_size'] = args.sample
        
        # Load data
        print("\n2. 📂 Loading Data...")
        df = eda_engine.load_data()
        
        if args.clean_only:
            print("\n3. 🧹 Cleaning Data Only...")
            clean_data = eda_engine.get_clean_data()
            print(f"   Cleaned data shape: {clean_data.shape}")
            print(f"   Saved to: {eda_engine.config['data']['output_path']}")
            return clean_data
        
        # Run complete EDA
        print("\n3. 🔍 Running Complete EDA Analysis...")
        results = eda_engine.run_complete_eda()
        
        # Display summary
        print_summary(results)
        
        # Get clean data for next tasks
        print("\n4. 🧹 Preparing Clean Data for Next Tasks...")
        clean_data = eda_engine.get_clean_data()
        print(f"   Clean data shape: {clean_data.shape}")
        
        print("\n" + "="*80)
        print("✅ EDA PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"\n❌ EDA Pipeline Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_summary(results: dict):
    """Print summary of EDA results - FIXED VERSION"""
    print("\n📋 EDA RESULTS SUMMARY:")
    print("-" * 40)
    
    if 'data_overview' in results:
        # Check structure of data_overview
        if isinstance(results['data_overview'], dict):
            # Try different possible keys
            if 'rows' in results['data_overview']:
                # Direct structure
                overview = results['data_overview']
                print(f"📊 Data Overview:")
                print(f"   • Rows: {overview.get('rows', 'N/A'):,}")
                print(f"   • Columns: {overview.get('columns', 'N/A')}")
                print(f"   • Memory: {overview.get('memory_mb', 'N/A'):.1f} MB")
            elif 'structure' in results['data_overview']:
                # Nested structure
                structure = results['data_overview']['structure']
                print(f"📊 Data Overview:")
                print(f"   • Rows: {structure.get('rows', 'N/A'):,}")
                print(f"   • Columns: {structure.get('columns', 'N/A')}")
                print(f"   • Memory: {structure.get('memory_mb', 'N/A'):.1f} MB")
    
    # Missing values with safe access
    if 'missing_values' in results and 'overall' in results['missing_values']:
        missing = results['missing_values']['overall']
        print(f"\n⚠️  Missing Values:")
        print(f"   • Total missing: {missing.get('total_missing_cells', 'N/A'):,}")
        print(f"   • Missing %: {missing.get('total_missing_percentage', 'N/A'):.1f}%")
    
    # Correlations
    if 'correlations' in results and 'strong' in results['correlations']:
        strong_corrs = results['correlations']['strong']
        if isinstance(strong_corrs, list):
            print(f"\n🔗 Correlations:")
            print(f"   • Strong correlations: {len(strong_corrs)} pairs")
    
    # Outliers
    if 'outliers' in results:
        if 'summary' in results['outliers']:
            outlier_summary = results['outliers']['summary']
            high_outliers = sum(1 for stats in outlier_summary.values() 
                              if isinstance(stats, dict) and stats.get('percentage_iqr', 0) > 5)
            print(f"\n📈 Outliers:")
            print(f"   • Features with >5% outliers: {high_outliers}")
    
    # Fraud analysis
    if 'fraud_analysis' in results and 'overall' in results['fraud_analysis']:
        fraud = results['fraud_analysis']['overall']
        print(f"\n🚨 Fraud Analysis:")
        print(f"   • Fraud rate: {fraud.get('fraud_percentage', 'N/A'):.2f}%")
        print(f"   • Fraud transactions: {fraud.get('fraud_transactions', 'N/A'):,}")
    
    print("\n📁 Outputs saved to:")
    print("   • data/processed/ - Cleaned data and results")
    print("   • reports/task2_eda/ - Reports and visualizations")
    print("   • notebooks/ - Complete analysis notebooks")


if __name__ == "__main__":
    results = main()