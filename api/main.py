"""
ChurnGuard API - FastAPI application for churn prediction.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import predict, is_model_loaded

app = FastAPI(
    title="ChurnGuard API",
    description="Customer churn prediction service",
    version="0.1.0"
)


class CustomerFeatures(BaseModel):
    """Input features for churn prediction."""
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20
            }
        }


class PredictionResponse(BaseModel):
    """Churn prediction output."""
    churn_probability: float
    churn_prediction: bool
    confidence: float
    risk_level: str


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "churnguard"}


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": is_model_loaded(),
        "version": "0.1.0"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    """
    Predict customer churn probability.
    
    Returns churn probability, binary prediction, confidence score, and risk level.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training script first."
        )
    
    try:
        result = predict(customer.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
