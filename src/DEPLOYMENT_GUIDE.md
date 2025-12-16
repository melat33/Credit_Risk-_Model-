# Bati Bank Credit Risk Model - Deployment Guide

## Quick Start
```bash
# Install dependencies
pip install -r src/requirements.txt

# Train model
python src/train_model.py --data_path data/processed/customer_rfm_with_target.csv
```

## Model Files
- `models/production/model.pkl` - Trained model
- `models/production/preprocessor.pkl` - Feature scaler
- `models/production/metadata.json` - Performance metrics

## Basel II Compliance
- ROC-AUC: Must be ≥ 0.7
- False Negative Rate: Must be ≤ 20%

Last updated: 2025-12-16

## API Deployment Example
```python
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
with open('models/production/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/production/preprocessor.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.post("/predict")
def predict(recency_days: float, transaction_frequency: float, total_monetary_value: float):
    features = np.array([[recency_days, transaction_frequency, total_monetary_value]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "is_high_risk": bool(prediction),
        "risk_score": float(probability),
        "risk_level": "HIGH" if prediction == 1 else "LOW"
    }
```
