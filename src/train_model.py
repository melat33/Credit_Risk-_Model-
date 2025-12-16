#!/usr/bin/env python
"""
Bati Bank Credit Risk Model - Training Script (Task 6 Ready)
Trains model and registers it in MLflow for deployment
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
    """Load RFM data for training"""
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Check for target column
    if 'is_high_risk' not in df.columns:
        # Try alternative names
        for col in ['target', 'risk_flag', 'default_flag', 'label']:
            if col in df.columns:
                df = df.rename(columns={col: 'is_high_risk'})
                print(f"Renamed column '{col}' to 'is_high_risk'")
                break
    
    # Check required columns
    required_cols = ['recency_days', 'transaction_frequency', 'total_monetary_value', 'is_high_risk']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Data loaded: {len(df):,} customers")
    print(f"High-risk customers: {df['is_high_risk'].sum():,} ({df['is_high_risk'].mean()*100:.1f}%)")
    
    return df

def train_and_register_model(data_path, output_dir='models/best_model'):
    """Train model and register in MLflow (Task 6 requirement)"""
    
    print("=" * 60)
    print("Bati Bank Credit Risk Model Training")
    print("=" * 60)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Required for Task 6
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
        try:
            # 1. Load data
            df = load_rfm_data(data_path)
            
            # 2. Prepare features
            features = ['recency_days', 'transaction_frequency', 'total_monetary_value']
            X = df[features]
            y = df['is_high_risk']
            
            # 3. Split data (70% train, 15% validation, 15% test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
            )
            
            print(f"Data splits:")
            print(f"  Train: {len(X_train):,} samples")
            print(f"  Validation: {len(X_val):,} samples")
            print(f"  Test: {len(X_test):,} samples")
            
            # 4. Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
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
                'roc_auc': float(roc_auc_score(y_test, y_prob)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'false_negative_rate': float(1 - recall_score(y_test, y_pred))
            }
            
            # 7. Check Basel II compliance
            basel_compliance = {
                'roc_auc_met': bool(metrics['roc_auc'] >= 0.7),
                'fnr_met': bool(metrics['false_negative_rate'] <= 0.2),
                'overall': bool((metrics['roc_auc'] >= 0.7) and (metrics['false_negative_rate'] <= 0.2))
            }
            
            # 8. Save model files
            os.makedirs(output_dir, exist_ok=True)
            
            model_path = os.path.join(output_dir, 'model.pkl')
            scaler_path = os.path.join(output_dir, 'preprocessor.pkl')
            metadata_path = os.path.join(output_dir, 'metadata.json')
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # 9. Save metadata
            metadata = {
                'model_name': 'RandomForestClassifier',
                'training_date': datetime.now().isoformat(),
                'performance': metrics,
                'basel_ii_compliance': basel_compliance,
                'features': features,
                'random_seed': int(RANDOM_SEED)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # 10. LOG TO MLFLOW REGISTRY (TASK 6 REQUIREMENT)
            print("Registering model in MLflow...")
            
            # Log parameters
            mlflow.log_params(model.get_params())
            mlflow.log_param("features", str(features))
            mlflow.log_param("data_path", data_path)
            
            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
            
            mlflow.log_metric("basel_compliant", int(basel_compliance['overall']))
            
            # Log model to registry
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="bati_bank_credit_risk_model"
            )
            
            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metadata_path)
            
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE - MODEL REGISTERED")
            print("=" * 60)
            print(f"Performance:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  FNR: {metrics['false_negative_rate']:.3f}")
            print(f"Basel II: {'COMPLIANT' if basel_compliance['overall'] else 'NON-COMPLIANT'}")
            print(f"Files saved to: {output_dir}")
            print(f"MLflow model: bati_bank_credit_risk_model")
            print("=" * 60)
            
            return {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metadata_path': metadata_path,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train credit risk model with MLflow registration')
    parser.add_argument('--data_path', type=str, 
                       default='data/processed/customer_rfm_with_target.csv',
                       help='Path to RFM data CSV file')
    parser.add_argument('--output_dir', type=str, 
                       default='models/best_model',
                       help='Directory to save trained model')
    
    args = parser.parse_args()
    
    print("Starting Bati Bank Credit Risk Model Training...")
    results = train_and_register_model(args.data_path, args.output_dir)
    print("\nTraining completed successfully!")