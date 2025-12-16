"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel
from typing import Optional


class CustomerFeatures(BaseModel):
    """Input features for churn prediction."""
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: str  # Month-to-month, One year, Two year
    payment_method: str
    # Add more features as we develop the model


class PredictionResponse(BaseModel):
    """Churn prediction output."""
    customer_id: Optional[str] = None
    churn_probability: float
    churn_prediction: bool  # True = likely to churn
    confidence: float
