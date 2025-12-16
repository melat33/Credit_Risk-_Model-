from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class CustomerFeatures(BaseModel):
    """Features for a single customer prediction"""
    recency_days: float = Field(
        ...,
        gt=0,
        le=365,
        description="Days since last transaction (0-365 days)",
        example=45.0
    )
    transaction_frequency: float = Field(
        ...,
        gt=0,
        description="Number of transactions",
        example=12.0
    )
    total_monetary_value: float = Field(
        ...,
        gt=0,
        description="Total transaction amount in USD",
        example=1500.50
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    customers: List[CustomerFeatures] = Field(
        ...,
        description="List of customer features",
        min_items=1,
        max_items=1000
    )


class PredictionResponse(BaseModel):
    """Single prediction response"""
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer identifier"
    )
    is_high_risk: bool = Field(
        ...,
        description="True if customer is high risk"
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being high risk (0-1)"
    )
    risk_level: str = Field(
        ...,
        description="Risk category: LOW, MEDIUM, or HIGH"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken to process batch (milliseconds)"
    )
    total_customers: int = Field(
        ...,
        description="Total number of customers processed"
    )


class ModelInfo(BaseModel):
    """Model metadata response"""
    model_name: str = Field(
        ...,
        description="Name of the model"
    )
    version: str = Field(
        ..., 
        description="Model version"
    )
    training_date: datetime = Field(
        ...,
        description="When the model was trained"
    )
    performance: dict = Field(
        ...,
        description="Model performance metrics"
    )
    basel_ii_compliant: bool = Field(
        ...,
        description="Whether model meets Basel II requirements"
    )


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(
        ...,
        description="Service status"
    )
    timestamp: datetime = Field(
        ...,
        description="Current timestamp"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded and ready"
    )
    version: str = Field(
        ...,
        description="API version"
    )