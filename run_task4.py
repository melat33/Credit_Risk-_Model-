#!/usr/bin/env python3
"""
Script to run Task 4 pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from proxy_target import RFMTargetCreator

def main():
    print("🚀 Running Task 4: Proxy Target Variable Engineering")
    
    # Load your data
    try:
        # Adjust path as needed
        df = pd.read_csv('data/processed/cleaned_data.csv')
        print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print("Error: Processed data not found. Run data processing first.")
        return
    
    # Initialize and run pipeline
    creator = RFMTargetCreator(random_state=42)
    
    save_paths = {
        'rfm_path': 'data/processed/customer_rfm.csv',
        'target_path': 'data/processed/target_variable.csv',
        'combined_path': 'data/processed/labeled_dataset.csv'
    }
    
    # Run pipeline
    target, rfm, metrics = creator.run_pipeline(df, save_paths)
    
    print("\n" + "="*50)
    print("✅ TASK 4 COMPLETE")
    print("="*50)
    print(f"Created target variable for {len(target)} customers")
    print(f"High-risk customers: {(target == 1).sum()} ({(target == 1).mean()*100:.1f}%)")
    print(f"Low-risk customers: {(target == 0).sum()} ({(target == 0).mean()*100:.1f}%)")
    
    if metrics.get('makes_business_sense', False):
        print("🎯 Validation: Proxy makes business sense!")
    else:
        print("⚠️ Warning: Proxy may not align with business logic")
    
    print("\nOutput files saved:")
    for key, path in save_paths.items():
        print(f"  • {key}: {path}")

if __name__ == "__main__":
    main()