#!/usr/bin/env python
"""
<<<<<<< HEAD
Bati Bank Credit Risk Model - Training Script (Task 6 Ready)
Trains model and registers it in MLflow for deployment
=======
Bati Bank Credit Risk Model - Production Training Script
Automated script for retraining the model with new data

Usage:
    python train_model.py --data_path path/to/data.csv

Features:
    - Automated data preprocessing
    - Model training with hyperparameter tuning
    - Performance validation
    - Basel II compliance checking
    - MLflow experiment tracking
    - Production model deployment

Supports both:
    1. Raw transaction data (CustomerId, TransactionStartTime, TransactionId, Amount)
    2. Preprocessed RFM data (recency_days, transaction_frequency, total_monetary_value, is_high_risk)
>>>>>>> 39507b391f2d40556edbe467aa24f486731defbc
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
<<<<<<< HEAD
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
=======
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MLFLOW_EXPERIMENT_NAME = "bati_bank_credit_risk_production"

def load_and_preprocess_data(data_path):
    """Load and preprocess data (handles both raw and processed data)"""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Check if this is preprocessed RFM data
    rfm_columns = ['recency_days', 'transaction_frequency', 'total_monetary_value']
    has_rfm_features = all(col in df.columns for col in rfm_columns)
    
    if has_rfm_features:
        print("✅ Detected preprocessed RFM data")
        
        # Check for target column
        target_aliases = ['is_high_risk', 'target', 'risk_flag', 'default_flag']
        target_col = None
        
        for alias in target_aliases:
            if alias in df.columns:
                target_col = alias
                if alias != 'is_high_risk':
                    df = df.rename(columns={alias: 'is_high_risk'})
                print(f"✅ Target column found: {alias}")
                break
        
        if target_col is None:
            raise ValueError(f"No target column found. Expected one of: {target_aliases}")
        
        # Ensure we have all RFM columns
        missing_rfm = [col for col in rfm_columns if col not in df.columns]
        if missing_rfm:
            raise ValueError(f"Missing RFM columns: {missing_rfm}")
        
        # Add CustomerId if missing
        if 'CustomerId' not in df.columns:
            df['CustomerId'] = range(1, len(df) + 1)
        
        # Add calculated features if missing
        if 'avg_transaction_value' not in df.columns:
            df['avg_transaction_value'] = df['total_monetary_value'] / df['transaction_frequency'].replace(0, 1)
            df['avg_transaction_value'] = df['avg_transaction_value'].fillna(0)
        
        if 'std_transaction_value' not in df.columns:
            df['std_transaction_value'] = 0
        
        print(f"📊 Loaded {len(df)} RFM records")
        print(f"🎯 Target distribution: {df['is_high_risk'].sum()} high-risk ({df['is_high_risk'].mean()*100:.1f}%)")
        
        return df
    
    else:
        # Original code for raw transaction data
        print("📊 Detected raw transaction data - calculating RFM features...")
        required_cols = ['CustomerId', 'TransactionStartTime', 'TransactionId', 'Amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert date columns
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Calculate RFM features
        snapshot_date = df['TransactionStartTime'].max()

        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': ['sum', 'mean', 'std']
        }).reset_index()

        rfm.columns = ['CustomerId', 'recency_days', 'transaction_frequency', 
                      'total_monetary_value', 'avg_transaction_value', 'std_transaction_value']

        rfm['total_monetary_value'] = rfm['total_monetary_value'].abs()
        rfm['std_transaction_value'] = rfm['std_transaction_value'].fillna(0)

        print(f"RFM features calculated for {len(rfm)} customers")
        return rfm

def create_target_variable(rfm_df, high_risk_threshold=0.1):
    """Create target variable based on RFM metrics (only if not already present)"""
    
    # Check if target already exists
    if 'is_high_risk' in rfm_df.columns:
        print(f"✅ Using existing target variable")
        print(f"   High-risk customers: {rfm_df['is_high_risk'].sum()} ({rfm_df['is_high_risk'].mean()*100:.1f}%)")
        return rfm_df
    
    print("🔄 Creating target variable from RFM features...")
    
    # Create risk score (higher = more risky)
    features = ['recency_days', 'transaction_frequency', 'total_monetary_value']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[features])

    # Risk formula: high recency + low frequency + low monetary = high risk
    risk_scores = (
        rfm_scaled[:, 0] * 0.5 +    # recency (positive weight)
        rfm_scaled[:, 1] * -0.3 +   # frequency (negative weight)
        rfm_scaled[:, 2] * -0.2     # monetary (negative weight)
    )

    # Create binary target (top X% as high risk)
    threshold = np.percentile(risk_scores, 100 * (1 - high_risk_threshold))
    rfm_df['is_high_risk'] = (risk_scores >= threshold).astype(int)

    print(f"✅ Target created: {rfm_df['is_high_risk'].sum()} high-risk customers "
          f"({rfm_df['is_high_risk'].mean()*100:.1f}%)")

    return rfm_df

def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, model_name, model_params):
    """Train and evaluate a single model"""
    if model_name == 'RandomForest':
        model = RandomForestClassifier(**model_params, random_state=RANDOM_SEED, n_jobs=-1)
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(**model_params, random_state=RANDOM_SEED)
    elif model_name == 'XGBoost':
        model = XGBClassifier(**model_params, random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        'val_roc_auc': roc_auc_score(y_val, y_prob_val),
        'test_roc_auc': roc_auc_score(y_test, y_prob_test),
        'val_f1': f1_score(y_val, y_pred_val),
        'test_f1': f1_score(y_test, y_pred_test),
        'val_precision': precision_score(y_val, y_pred_val),
        'test_precision': precision_score(y_test, y_pred_test),
        'val_recall': recall_score(y_val, y_pred_val),
        'test_recall': recall_score(y_test, y_pred_test),
        'false_negative_rate_val': 1 - recall_score(y_val, y_pred_val),
        'false_negative_rate_test': 1 - recall_score(y_test, y_pred_test),
        'model': model_name
    }

    return model, metrics

def check_basel_compliance(metrics):
    """Check if model meets Basel II requirements"""
    roc_auc_met = metrics['test_roc_auc'] >= 0.7
    fnr_met = metrics['false_negative_rate_test'] <= 0.2
    overall = roc_auc_met and fnr_met

    return {
        'roc_auc_met': bool(roc_auc_met),
        'fnr_met': bool(fnr_met),
        'overall': bool(overall)
    }

def save_production_model(model, preprocessor, metrics, basel_compliance, features, output_dir='models/production'):
    """Save production model and metadata"""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    # Save metadata
    metadata = {
        'model_name': type(model).__name__,
        'training_date': datetime.now().isoformat(),
        'performance': {
            'roc_auc': float(metrics['test_roc_auc']),
            'f1_score': float(metrics['test_f1']),
            'precision': float(metrics['test_precision']),
            'recall': float(metrics['test_recall']),
            'false_negative_rate': float(metrics['false_negative_rate_test'])
        },
        'basel_ii_compliance': basel_compliance,
        'features': [str(f) for f in features],
        'random_seed': int(RANDOM_SEED)
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Model saved to {output_dir}")
    return model_path, preprocessor_path, metadata_path

def main(data_path, output_dir='models/production'):
    """Main training pipeline"""
    print("=" * 60)
    print("Bati Bank Credit Risk Model - Production Training")
    print("=" * 60)

    # Set up MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f'production_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
        # 1. Load and preprocess data
        rfm_data = load_and_preprocess_data(data_path)

        # 2. Create target variable (if not already present)
        rfm_data = create_target_variable(rfm_data, high_risk_threshold=0.1)

        # 3. Prepare features
        features = ['recency_days', 'transaction_frequency', 'total_monetary_value']
        X = rfm_data[features]
        y = rfm_data['is_high_risk']

        # Log class distribution
        mlflow.log_param("total_samples", len(rfm_data))
        mlflow.log_param("high_risk_pct", float(y.mean() * 100))
        mlflow.log_param("features", str(features))

        # 4. Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), 
            random_state=RANDOM_SEED, stratify=y_temp
        )

        print(f"\n📊 Data split:")
        print(f"   • Train: {len(X_train)} samples")
        print(f"   • Validation: {len(X_val)} samples")
        print(f"   • Test: {len(X_test)} samples")
        
        # Log split sizes
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        # 5. Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 6. Define models to try
        models_to_try = {
            'RandomForest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'class_weight': 'balanced'
            },
            'XGBoost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'scale_pos_weight': len(y_train[y_train==0])/len(y_train[y_train==1])
            },
            'LogisticRegression': {
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced'
            }
        }

        # 7. Train and evaluate models
        all_metrics = []
        best_model = None
        best_metrics = None
        best_score = 0
        best_model_name = None

        for model_name, params in models_to_try.items():
            print(f"\n{'='*40}")
            print(f"Training {model_name}...")
            print(f"{'='*40}")
            
            mlflow.log_param(f"{model_name}_params", params)

            model, metrics = train_and_evaluate_model(
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test, model_name, params
            )

            all_metrics.append(metrics)

            # Log metrics to MLflow
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{model_name}_{key}", value)

            # Print model performance
            print(f"✅ {model_name} Performance:")
            print(f"   • Test ROC-AUC: {metrics['test_roc_auc']:.3f}")
            print(f"   • Test F1-Score: {metrics['test_f1']:.3f}")
            print(f"   • Test Recall: {metrics['test_recall']:.3f}")

            # Check if this is the best model
            if metrics['test_roc_auc'] > best_score:
                best_score = metrics['test_roc_auc']
                best_model = model
                best_metrics = metrics
                best_model_name = model_name

        # 8. Check Basel II compliance
        basel_compliance = check_basel_compliance(best_metrics)

        # 9. Save best model
        model_path, preprocessor_path, metadata_path = save_production_model(
            best_model, scaler, best_metrics, basel_compliance, features, output_dir
        )

        # 10. Log best model to MLflow
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_roc_auc", best_metrics['test_roc_auc'])
        mlflow.log_metric("best_f1", best_metrics['test_f1'])
        mlflow.log_metric("basel_compliant", basel_compliance['overall'])

        mlflow.sklearn.log_model(best_model, "best_model")

        # 11. Print summary
        print("\n" + "=" * 60)
        print("🎯 TRAINING COMPLETE - SUMMARY")
        print("=" * 60)
        print(f"📊 Data Summary:")
        print(f"   • Total customers: {len(rfm_data):,}")
        print(f"   • High-risk customers: {y.sum():,} ({y.mean()*100:.1f}%)")
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"\n📈 Performance Metrics:")
        print(f"   • ROC-AUC: {best_metrics['test_roc_auc']:.3f}")
        print(f"   • F1-Score: {best_metrics['test_f1']:.3f}")
        print(f"   • Precision: {best_metrics['test_precision']:.3f}")
        print(f"   • Recall: {best_metrics['test_recall']:.3f}")
        print(f"   • False Negative Rate: {best_metrics['false_negative_rate_test']:.3f}")
        print(f"\n⚠️  Basel II Compliance:")
        print(f"   • ROC-AUC >= 0.7: {'✅ YES' if basel_compliance['roc_auc_met'] else '❌ NO'}")
        print(f"   • FNR <= 0.2: {'✅ YES' if basel_compliance['fnr_met'] else '❌ NO'}")
        print(f"   • Overall Compliant: {'✅ YES' if basel_compliance['overall'] else '❌ NO'}")
        print(f"\n💾 Model saved to: {output_dir}")
        print(f"📝 MLflow tracking enabled")
        print("=" * 60)

        return {
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'metadata_path': metadata_path,
            'metrics': best_metrics,
            'basel_compliance': basel_compliance
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train credit risk model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data CSV file (raw transaction or processed RFM data)')
    parser.add_argument('--output_dir', type=str, default='models/production',
                       help='Directory to save trained model')
    
    # Add example usage to help
    parser.epilog = """
Examples:
  # Train with raw transaction data:
  python train_model.py --data_path data/raw/transactions.csv
  
  # Train with preprocessed RFM data:
  python train_model.py --data_path data/processed/customer_rfm_with_target.csv
  
  # Train with custom output directory:
  python train_model.py --data_path data.csv --output_dir models/my_model
    """

    args = parser.parse_args()

    # Run training
    try:
        print("\n🚀 Starting training pipeline...")
        results = main(args.data_path, args.output_dir)
        print("\n✅ Training completed successfully!")
        
        # Show where files were saved
        print("\n📁 Generated files:")
        print(f"   • Model: {results['model_path']}")
        print(f"   • Preprocessor: {results['preprocessor_path']}")
        print(f"   • Metadata: {results['metadata_path']}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if your CSV file exists")
        print("   2. Verify your data has required columns:")
        print("      - For raw data: CustomerId, TransactionStartTime, TransactionId, Amount")
        print("      - For RFM data: recency_days, transaction_frequency, total_monetary_value, is_high_risk")
        print("   3. Check file permissions")
        raise
>>>>>>> 39507b391f2d40556edbe467aa24f486731defbc
