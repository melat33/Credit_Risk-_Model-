#!/usr/bin/env python
"""
Train model from preprocessed RFM data
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import os
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

RANDOM_SEED = 42
MLFLOW_EXPERIMENT_NAME = "bati_bank_credit_risk_rfm"

def load_rfm_data(data_path):
    """Load preprocessed RFM data"""
    print(f"Loading RFM data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Required columns
    required_cols = ['recency_days', 'transaction_frequency', 'total_monetary_value', 'is_high_risk']
    
    # Check for target with different possible names
    target_aliases = ['is_high_risk', 'target', 'risk_flag', 'default_flag']
    for target_col in target_aliases:
        if target_col in df.columns:
            if target_col != 'is_high_risk':
                df = df.rename(columns={target_col: 'is_high_risk'})
            break
    else:
        raise ValueError(f"No target column found. Expected one of: {target_aliases}")
    
    # Check other columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✅ Data loaded: {len(df)} customers")
    print(f"📊 Risk distribution: {df['is_high_risk'].sum()} high-risk ({df['is_high_risk'].mean()*100:.1f}%)")
    
    return df

def main(data_path, output_dir='models/production'):
    """Main training pipeline for RFM data"""
    print("=" * 60)
    print("Bati Bank Credit Risk Model - RFM Training")
    print("=" * 60)
    
    # Set up MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f'rfm_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
        try:
            # 1. Load RFM data
            rfm_data = load_rfm_data(data_path)
            
            # 2. Prepare features and target
            features = ['recency_days', 'transaction_frequency', 'total_monetary_value']
            X = rfm_data[features]
            y = rfm_data['is_high_risk']
            
            # Log class distribution
            mlflow.log_param("n_customers", len(rfm_data))
            mlflow.log_param("high_risk_pct", float(y.mean() * 100))
            mlflow.log_param("features", str(features))
            
            # 3. Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.125, random_state=RANDOM_SEED, stratify=y_train
            )
            
            print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # 4. Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # 5. Train model
            print("Training Random Forest model...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # 6. Evaluate
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            metrics = {
                'roc_auc': roc_auc_score(y_test, y_prob),
                'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'false_negative_rate': 1 - recall_score(y_test, y_pred)
            }
            
            # 7. Basel compliance check
            basel_compliance = {
                'roc_auc_met': metrics['roc_auc'] >= 0.7,
                'fnr_met': metrics['false_negative_rate'] <= 0.2,
                'overall': (metrics['roc_auc'] >= 0.7) and (metrics['false_negative_rate'] <= 0.2)
            }
            
            # 8. Save model
            os.makedirs(output_dir, exist_ok=True)
            
            model_path = os.path.join(output_dir, 'rfm_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save metadata
            metadata = {
                'model_type': 'RandomForest',
                'training_date': datetime.now().isoformat(),
                'features': features,
                'metrics': {k: float(v) for k, v in metrics.items()},
                'basel_compliance': basel_compliance,
                'data_info': {
                    'total_samples': len(rfm_data),
                    'high_risk_samples': int(y.sum()),
                    'high_risk_percentage': float(y.mean() * 100)
                }
            }
            
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # 9. Log to MLflow
            mlflow.log_params(model.get_params())
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.log_metric('basel_compliant', int(basel_compliance['overall']))
            mlflow.sklearn.log_model(model, "model")
            
            # 10. Print results
            print("\n" + "=" * 60)
            print("TRAINING RESULTS")
            print("=" * 60)
            print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"F1-Score: {metrics['f1']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"False Negative Rate: {metrics['false_negative_rate']:.3f}")
            print(f"Basel II Compliant: {'✅ YES' if basel_compliance['overall'] else '❌ NO'}")
            print(f"Model saved to: {output_dir}")
            print("=" * 60)
            
            return {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metadata_path': metadata_path,
                'metrics': metrics,
                'basel_compliance': basel_compliance
            }
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model from preprocessed RFM data')
    parser.add_argument('--data_path', type=str, default='data/processed/customer_rfm_with_target.csv',
                       help='Path to RFM data CSV file')
    parser.add_argument('--output_dir', type=str, default='models/rfm_production',
                       help='Directory to save trained model')
    
    args = parser.parse_args()
    
    results = main(args.data_path, args.output_dir)
    print("\n✅ Training completed successfully!")