#!/usr/bin/env python
"""
Bati Bank Credit Risk Model - Production Training Script
Automated script for retraining the model with new data

Usage:
    python train_model.py --data_path path/to/new_data.csv

Features:
    - Automated data preprocessing
    - Model training with hyperparameter tuning
    - Performance validation
    - Basel II compliance checking
    - MLflow experiment tracking
    - Production model deployment
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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MLFLOW_EXPERIMENT_NAME = "bati_bank_credit_risk_production"

def load_and_preprocess_data(data_path):
    """Load and preprocess transaction data"""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Check required columns
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
    """Create target variable based on RFM metrics"""
    # Create risk score (higher = more risky)
    from sklearn.preprocessing import StandardScaler

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

    print(f"Target created: {rfm_df['is_high_risk'].sum()} high-risk customers "
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

    print(f"Model saved to {output_dir}")
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

        # 2. Create target variable
        rfm_data = create_target_variable(rfm_data, high_risk_threshold=0.1)

        # 3. Prepare features
        features = ['recency_days', 'transaction_frequency', 'total_monetary_value']
        X = rfm_data[features]
        y = rfm_data['is_high_risk']

        # 4. Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), 
            random_state=RANDOM_SEED, stratify=y_temp
        )

        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

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
                'min_samples_split': 10
            },
            'XGBoost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'scale_pos_weight': len(y_train[y_train==0])/len(y_train[y_train==1])
            },
            'LogisticRegression': {
                'C': 1.0,
                'max_iter': 1000
            }
        }

        # 7. Train and evaluate models
        all_metrics = []
        best_model = None
        best_metrics = None
        best_score = 0

        for model_name, params in models_to_try.items():
            print(f"Training {model_name}...")
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
        print("TRAINING COMPLETE - SUMMARY")
        print("=" * 60)
        print(f"Best Model: {best_model_name}")
        print(f"ROC-AUC: {best_metrics['test_roc_auc']:.3f}")
        print(f"F1-Score: {best_metrics['test_f1']:.3f}")
        print(f"Recall: {best_metrics['test_recall']:.3f}")
        print(f"False Negative Rate: {best_metrics['false_negative_rate_test']:.3f}")
        print(f"Basel II Compliant: {'YES' if basel_compliance['overall'] else 'NO'}")
        print(f"Model saved to: {output_dir}")
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
                       help='Path to transaction data CSV file')
    parser.add_argument('--output_dir', type=str, default='models/production',
                       help='Directory to save trained model')

    args = parser.parse_args()

    # Run training
    try:
        results = main(args.data_path, args.output_dir)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise