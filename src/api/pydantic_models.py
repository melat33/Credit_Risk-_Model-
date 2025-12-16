from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

class CustomerFeatures(BaseModel):
    """Data model for customer features (17 features total)"""
    customer_id: str = Field(..., description="Unique customer identifier")
    
    # Core RFM features (required)
    recency_days: float = Field(..., description="Days since last transaction")
    transaction_frequency: float = Field(..., description="Number of transactions")
    total_monetary_value: float = Field(..., description="Total transaction amount")
    
    # Optional features (will be calculated if not provided)
    Unnamed_0: Optional[float] = Field(0.0, description="Index column")
    avg_transaction_value: Optional[float] = Field(None, description="Average transaction amount")
    std_transaction_value: Optional[float] = Field(None, description="Standard deviation of transaction amounts")
    cluster: Optional[int] = Field(0, description="Customer cluster/segment")
    recency_score: Optional[float] = Field(None, description="Recency score (1-5)")
    frequency_score: Optional[float] = Field(None, description="Frequency score (1-5)")
    monetary_score: Optional[float] = Field(None, description="Monetary score (1-5)")
    customer_value: Optional[float] = Field(None, description="Customer lifetime value")
    engagement_index: Optional[float] = Field(None, description="Engagement index (0-1)")
    value_concentration: Optional[float] = Field(None, description="Value concentration ratio")
    value_per_transaction: Optional[float] = Field(None, description="Value per transaction")
    transaction_size_variability: Optional[float] = Field(None, description="Transaction size variability")
    estimated_tenure_months: Optional[float] = Field(None, description="Estimated tenure in months")
    monthly_activity: Optional[float] = Field(None, description="Monthly activity rate")
    
    # Pydantic V2 style - NO WARNINGS
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST001",
                "recency_days": 45.0,
                "transaction_frequency": 12.0,
                "total_monetary_value": 1500.50
            }
        }
    )

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    customers: List[CustomerFeatures] = Field(..., description="List of customers to predict")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customers": [
                    {
                        "customer_id": "CUST001",
                        "recency_days": 45.0,
                        "transaction_frequency": 12.0,
                        "total_monetary_value": 1500.50
                    }
                ]
            }
        }
    )

class PredictionResponse(BaseModel):
    """Response model for a single prediction"""
    customer_id: str = Field(..., description="Customer identifier")
    is_high_risk: bool = Field(..., description="True if customer is high risk")
    risk_score: float = Field(..., description="Risk probability (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST001",
                "is_high_risk": False,
                "risk_score": 0.15,
                "risk_level": "LOW",
                "confidence": 0.85
            }
        }
    )

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    total_customers: int = Field(..., description="Total number of customers processed")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "customer_id": "CUST001",
                        "is_high_risk": False,
                        "risk_score": 0.15,
                        "risk_level": "LOW",
                        "confidence": 0.85
                    }
                ],
                "processing_time_ms": 125.5,
                "total_customers": 1
            }
        }
    )

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    training_date: datetime = Field(..., description="When the model was trained")
    performance: Dict[str, Any] = Field(..., description="Model performance metrics")
    basel_ii_compliant: bool = Field(..., description="Whether model meets Basel II requirements")
    features_used: Optional[List[str]] = Field(None, description="Features used by the model")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "Logistic Regression",
                "version": "2.0",
                "training_date": "2025-12-16T10:18:49.924506",
                "performance": {
                    "roc_auc": 1.0,
                    "f1_score": 1.0
                },
                "basel_ii_compliant": True,
                "features_used": ["recency_days", "transaction_frequency", "total_monetary_value"]
            }
        }
    )

class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    features_supported: Optional[int] = Field(None, description="Number of features supported")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2025-12-16T14:28:45.272000",
                "model_loaded": True,
                "version": "2.0.0",
                "features_supported": 17
            }
        }
    )