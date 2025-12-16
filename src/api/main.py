"""
Bati Bank Credit Risk Model API
FastAPI application for credit risk prediction
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import List, Optional
import os
import json
import logging

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .pydantic_models import (
    CustomerFeatures,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfo,
    HealthCheck
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="API for predicting credit risk of BNPL customers",
    version="1.0.0",
    contact={
        "name": "Bati Bank Risk Modeling Team",
        "email": "risk-modeling@batibank.com",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://www.batibank.com/terms",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
MODEL = None
PREPROCESSOR = None
METADATA = None


def load_model():
    """Load the trained model and preprocessor"""
    global MODEL, PREPROCESSOR, METADATA
    
    try:
        model_path = "models/best_model/model.pkl"
        preprocessor_path = "models/best_model/preprocessor.pkl"
        metadata_path = "models/best_model/metadata.json"
        
        # Check if files exist
        for path in [model_path, preprocessor_path, metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        with open(model_path, 'rb') as f:
            MODEL = pickle.load(f)
        
        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            PREPROCESSOR = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            METADATA = json.load(f)
        
        logger.info("Model loaded successfully")
        logger.info(f"Model: {METADATA.get('model_name', 'Unknown')}")
        logger.info(f"ROC-AUC: {METADATA.get('performance', {}).get('roc_auc', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    logger.info("Starting Bati Bank Credit Risk API...")
    load_model()


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirects to docs"""
    return {"message": "Bati Bank Credit Risk API - See /docs for API documentation"}


@app.get("/health", response_model=HealthCheck, tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=MODEL is not None,
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information and metadata"""
    if METADATA is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfo(
        model_name=METADATA.get("model_name", "Unknown"),
        version="1.0",
        training_date=datetime.fromisoformat(METADATA.get("training_date", "2024-01-01")),
        performance=METADATA.get("performance", {}),
        basel_ii_compliant=METADATA.get("basel_ii_compliance", {}).get("overall", False)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(features: CustomerFeatures):
    """
    Predict credit risk for a single customer
    
    - **recency_days**: Days since last transaction
    - **transaction_frequency**: Number of transactions  
    - **total_monetary_value**: Total transaction amount in USD
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert features to numpy array
        input_features = np.array([[
            features.recency_days,
            features.transaction_frequency,
            features.total_monetary_value
        ]])
        
        # Scale features
        features_scaled = PREPROCESSOR.transform(input_features)
        
        # Make prediction
        prediction = MODEL.predict(features_scaled)[0]
        probability = MODEL.predict_proba(features_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Calculate confidence
        confidence = max(probability, 1 - probability)
        
        return PredictionResponse(
            is_high_risk=bool(prediction),
            risk_score=float(probability),
            risk_level=risk_level,
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict credit risk for multiple customers in batch
    
    - Accepts up to 1000 customers per request
    - Returns predictions for all customers
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    
    try:
        # Convert features to DataFrame
        features_list = []
        for i, customer in enumerate(request.customers):
            features_list.append([
                customer.recency_days,
                customer.transaction_frequency,
                customer.total_monetary_value
            ])
        
        input_features = np.array(features_list)
        
        # Scale features
        features_scaled = PREPROCESSOR.transform(input_features)
        
        # Make predictions
        predictions = MODEL.predict(features_scaled)
        probabilities = MODEL.predict_proba(features_scaled)[:, 1]
        
        # Prepare response
        prediction_responses = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Determine risk level
            if prob < 0.3:
                risk_level = "LOW"
            elif prob < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            # Calculate confidence
            confidence = max(prob, 1 - prob)
            
            prediction_responses.append(
                PredictionResponse(
                    customer_id=f"customer_{i}",
                    is_high_risk=bool(pred),
                    risk_score=float(prob),
                    risk_level=risk_level,
                    confidence=float(confidence)
                )
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            processing_time_ms=processing_time_ms,
            total_customers=len(prediction_responses)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/features/description", tags=["Model"])
async def get_features_description():
    """Get description of model features"""
    return {
        "recency_days": {
            "description": "Days since last transaction",
            "interpretation": "Higher values indicate higher risk (customers who haven't transacted recently are more likely to default)",
            "range": "1-365 days",
            "importance": "High"
        },
        "transaction_frequency": {
            "description": "Number of transactions in the period",
            "interpretation": "Lower values indicate higher risk (customers with fewer transactions are more likely to default)",
            "range": "1+ transactions",
            "importance": "High"
        },
        "total_monetary_value": {
            "description": "Total transaction amount in USD",
            "interpretation": "Lower values indicate higher risk (customers with lower transaction amounts are more likely to default)",
            "range": "1+ USD",
            "importance": "Medium"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# For testing purposes
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )