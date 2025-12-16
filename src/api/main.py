"""
Bati Bank Credit Risk Model API
FastAPI application for credit risk prediction
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import json
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import Pydantic models correctly
try:
    from src.api.pydantic_models import (
        CustomerFeatures,
        BatchPredictionRequest,
        PredictionResponse,
        BatchPredictionResponse,
        ModelInfo,
        HealthCheck
    )
except ImportError:
    # Create minimal models if import fails
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict, Any
    
    class CustomerFeatures(BaseModel):
        customer_id: str
        recency_days: float
        transaction_frequency: float
        total_monetary_value: float
    
    class BatchPredictionRequest(BaseModel):
        customers: List[CustomerFeatures]
    
    class PredictionResponse(BaseModel):
        customer_id: str
        is_high_risk: bool
        risk_score: float
        risk_level: str
        confidence: float
    
    class BatchPredictionResponse(BaseModel):
        predictions: List[PredictionResponse]
        processing_time_ms: float
        total_customers: int
    
    class ModelInfo(BaseModel):
        model_name: str
        version: str
        training_date: datetime
        performance: Dict[str, Any]
        basel_ii_compliant: bool
    
    class HealthCheck(BaseModel):
        status: str
        timestamp: datetime
        model_loaded: bool
        version: str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== INITIALIZE FASTAPI APP ==========
app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="API for predicting credit risk of BNPL customers",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
MODEL = None
SCALER = None
METADATA = None
EXPECTED_FEATURES = None

def load_model():
    """Load the trained model and preprocessor"""
    global MODEL, SCALER, METADATA, EXPECTED_FEATURES
    
    try:
        model_path = "models/best_model/model.pkl"
        scaler_path = "models/best_model/preprocessor.pkl"
        metadata_path = "models/best_model/metadata.json"
        
        # Check if files exist
        for path in [model_path, scaler_path, metadata_path]:
            if not os.path.exists(path):
                logger.error(f"File not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")
        
        # Load files
        with open(model_path, 'rb') as f:
            MODEL = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            SCALER = pickle.load(f)
        
        with open(metadata_path, 'r') as f:
            METADATA = json.load(f)
        
        EXPECTED_FEATURES = METADATA.get("features", [])
        
        logger.info(f"✅ Model loaded: {METADATA.get('model_name', 'Unknown')}")
        logger.info(f"📊 Features: {len(EXPECTED_FEATURES)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    logger.info("🚀 Starting Bati Bank Credit Risk API...")
    load_model()

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "message": "Bati Bank Credit Risk API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=MODEL is not None,
        version="1.0.0"
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if METADATA is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=METADATA.get("model_name", "Unknown"),
        version="1.0",
        training_date=datetime.fromisoformat(METADATA.get("training_date", "2024-01-01")),
        performance=METADATA.get("performance", {}),
        basel_ii_compliant=METADATA.get("basel_ii_compliance", {}).get("overall", False)
    )

def prepare_features_for_prediction(customer: CustomerFeatures) -> np.ndarray:
    """Prepare 17 features for prediction"""
    features = []
    
    # Add all 17 features in correct order
    expected_order = [
        "Unnamed: 0", "recency_days", "transaction_frequency", "total_monetary_value",
        "avg_transaction_value", "std_transaction_value", "cluster", "recency_score",
        "frequency_score", "monetary_score", "customer_value", "engagement_index",
        "value_concentration", "value_per_transaction", "transaction_size_variability",
        "estimated_tenure_months", "monthly_activity"
    ]
    
    # Convert to dict
    data = customer.dict()
    
    for feature in expected_order:
        if feature == "Unnamed: 0":
            features.append(data.get("Unnamed_0", 0.0))
        elif feature == "avg_transaction_value":
            # Calculate if not provided
            if data.get("avg_transaction_value") is not None:
                features.append(data["avg_transaction_value"])
            elif data["transaction_frequency"] > 0:
                features.append(data["total_monetary_value"] / data["transaction_frequency"])
            else:
                features.append(0.0)
        elif feature == "value_per_transaction":
            # Same as avg_transaction_value
            if data.get("value_per_transaction") is not None:
                features.append(data["value_per_transaction"])
            elif data["transaction_frequency"] > 0:
                features.append(data["total_monetary_value"] / data["transaction_frequency"])
            else:
                features.append(0.0)
        else:
            # Map field names
            field_map = {
                "recency_days": "recency_days",
                "transaction_frequency": "transaction_frequency",
                "total_monetary_value": "total_monetary_value",
                "std_transaction_value": "std_transaction_value",
                "cluster": "cluster",
                "recency_score": "recency_score",
                "frequency_score": "frequency_score",
                "monetary_score": "monetary_score",
                "customer_value": "customer_value",
                "engagement_index": "engagement_index",
                "value_concentration": "value_concentration",
                "transaction_size_variability": "transaction_size_variability",
                "estimated_tenure_months": "estimated_tenure_months",
                "monthly_activity": "monthly_activity"
            }
            
            api_field = field_map.get(feature)
            if api_field and api_field in data and data[api_field] is not None:
                features.append(data[api_field])
            else:
                features.append(0.0)  # Default value
    
    return np.array([features])

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerFeatures):
    """Predict for a single customer"""
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = prepare_features_for_prediction(customer)
        
        # Scale and predict
        features_scaled = SCALER.transform(features)
        prediction = MODEL.predict(features_scaled)[0]
        probability = MODEL.predict_proba(features_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        confidence = max(probability, 1 - probability)
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            is_high_risk=bool(prediction),
            risk_score=float(probability),
            risk_level=risk_level,
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction"""
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        predictions_list = []
        
        for customer in request.customers:
            # Prepare features for each customer
            features = prepare_features_for_prediction(customer)
            
            # Scale and predict
            features_scaled = SCALER.transform(features)
            prediction = MODEL.predict(features_scaled)[0]
            probability = MODEL.predict_proba(features_scaled)[0][1]
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "LOW"
            elif probability < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            confidence = max(probability, 1 - probability)
            
            predictions_list.append(
                PredictionResponse(
                    customer_id=customer.customer_id,
                    is_high_risk=bool(prediction),
                    risk_score=float(probability),
                    risk_level=risk_level,
                    confidence=float(confidence)
                )
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            processing_time_ms=processing_time_ms,
            total_customers=len(predictions_list)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Run the app if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)