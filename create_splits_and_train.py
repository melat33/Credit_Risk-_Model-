#!/usr/bin/env python
"""
Create train/validation/test splits from customer_rfm_with_target.csv
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def create_data_splits(input_path='data/processed/customer_rfm_with_target.csv'):
    """Create train/validation/test splits from single file"""
    
    print(f"📂 Loading data from: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data file not found: {input_path}")
    
    # Load data
    rfm = pd.read_csv(input_path)
    print(f"✅ Data loaded: {len(rfm):,} customers")
    
    # Check required columns
    required_cols = ['recency_days', 'transaction_frequency', 'total_monetary_value', 'is_high_risk']
    missing_cols = [col for col in required_cols if col not in rfm.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Split data: 70% train, 15% validation, 15% test
    features = ['recency_days', 'transaction_frequency', 'total_monetary_value']
    X = rfm[features]
    y = rfm['is_high_risk']
    
    print(f"📊 Target distribution:")
    print(f"   • High-risk customers: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"   • Low-risk customers: {(len(y)-y.sum()):,} ({100-y.mean()*100:.1f}%)")
    
    # First split: 70% train, 30% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    
    # Second split: 50% validation, 50% test from temp (15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    # Create DataFrames
    train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    val_data = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    
    # Save to files
    data_dir = 'data/processed'
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, 'train_data.csv')
    val_path = os.path.join(data_dir, 'validation_set.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    full_path = os.path.join(data_dir, 'customer_rfm_with_target_full.csv')
    
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    rfm.to_csv(full_path, index=False)  # Save original as full dataset
    
    print(f"\n✅ Data splits created:")
    print(f"   • Train: {train_path} ({len(train_data):,} records, {len(train_data)/len(rfm)*100:.1f}%)")
    print(f"   • Validation: {val_path} ({len(val_data):,} records, {len(val_data)/len(rfm)*100:.1f}%)")
    print(f"   • Test: {test_path} ({len(test_data):,} records, {len(test_data)/len(rfm)*100:.1f}%)")
    print(f"   • Full dataset: {full_path} ({len(rfm):,} records)")
    
    # Create data dictionary
    data_dict = {
        "dataset_name": "Bati Bank Credit Risk Data",
        "created_date": datetime.now().isoformat(),
        "data_files": {
            "train_data.csv": {
                "description": "Training dataset (70% of total)",
                "records": int(len(train_data)),
                "features": list(train_data.columns),
                "target_distribution": {
                    "low_risk": int(len(train_data) - train_data['is_high_risk'].sum()),
                    "high_risk": int(train_data['is_high_risk'].sum()),
                    "high_risk_rate": float(train_data['is_high_risk'].mean() * 100)
                }
            },
            "validation_set.csv": {
                "description": "Validation dataset (15% of total)",
                "records": int(len(val_data)),
                "features": list(val_data.columns),
                "target_distribution": {
                    "low_risk": int(len(val_data) - val_data['is_high_risk'].sum()),
                    "high_risk": int(val_data['is_high_risk'].sum()),
                    "high_risk_rate": float(val_data['is_high_risk'].mean() * 100)
                }
            },
            "test_data.csv": {
                "description": "Test dataset (15% of total)",
                "records": int(len(test_data)),
                "features": list(test_data.columns),
                "target_distribution": {
                    "low_risk": int(len(test_data) - test_data['is_high_risk'].sum()),
                    "high_risk": int(test_data['is_high_risk'].sum()),
                    "high_risk_rate": float(test_data['is_high_risk'].mean() * 100)
                }
            }
        }
    }
    
    data_dict_path = os.path.join(data_dir, 'data_dictionary.json')
    with open(data_dict_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    print(f"📋 Data dictionary saved: {data_dict_path}")
    
    return rfm, train_data, val_data, test_data

if __name__ == "__main__":
    print("=" * 60)
    print("CREATING TRAIN/VALIDATION/TEST DATA SPLITS")
    print("=" * 60)
    
    try:
        rfm, train_data, val_data, test_data = create_data_splits()
        
        print("\n" + "=" * 60)
        print("✅ DATA SPLITS CREATED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📊 Summary:")
        print(f"   Total customers: {len(rfm):,}")
        print(f"   Train: {len(train_data):,} ({len(train_data)/len(rfm)*100:.1f}%)")
        print(f"   Validation: {len(val_data):,} ({len(val_data)/len(rfm)*100:.1f}%)")
        print(f"   Test: {len(test_data):,} ({len(test_data)/len(rfm)*100:.1f}%)")
        print(f"   High-risk rate: {rfm['is_high_risk'].mean()*100:.1f}%")
        print("\n📁 Files created in data/processed/:")
        print("   • train_data.csv")
        print("   • validation_set.csv")
        print("   • test_data.csv")
        print("   • customer_rfm_with_target_full.csv")
        print("   • data_dictionary.json")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")