"""
Prediction logic for churn model.
Loads trained model and makes predictions on new customer data.
"""
import joblib
import pandas as pd
import numpy as np
import os

# Paths
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "churn_model.joblib")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "encoders.joblib")
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.joblib")

# Global model cache (loaded once)
_model = None
_encoders = None
_feature_names = None


def load_artifacts():
    """Load model and encoders into memory."""
    global _model, _encoders, _feature_names
    
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _encoders = joblib.load(ENCODERS_PATH)
        _feature_names = joblib.load(FEATURE_NAMES_PATH)
    
    return _model, _encoders, _feature_names


def is_model_loaded() -> bool:
    """Check if model artifacts exist."""
    return all([
        os.path.exists(MODEL_PATH),
        os.path.exists(ENCODERS_PATH),
        os.path.exists(FEATURE_NAMES_PATH)
    ])


def predict(customer_data: dict) -> dict:
    """
    Make a churn prediction for a single customer.
    
    Args:
        customer_data: Dictionary with customer features
        
    Returns:
        Dictionary with prediction results
    """
    model, encoders, feature_names = load_artifacts()
    
    # Create DataFrame with single row
    df = pd.DataFrame([customer_data])
    
    # Encode categorical columns
    for col, encoder in encoders.items():
        if col in df.columns:
            # Handle unseen categories by using the most frequent
            try:
                df[col] = encoder.transform(df[col])
            except ValueError:
                # Unseen category - use first class as fallback
                df[col] = 0
    
    # Ensure columns are in correct order
    df = df[feature_names]
    
    # Make prediction
    churn_prob = model.predict_proba(df)[0, 1]
    churn_pred = churn_prob >= 0.5
    
    return {
        "churn_probability": round(float(churn_prob), 4),
        "churn_prediction": bool(churn_pred),
        "confidence": round(float(max(churn_prob, 1 - churn_prob)), 4),
        "risk_level": _get_risk_level(churn_prob)
    }


def _get_risk_level(prob: float) -> str:
    """Categorize churn probability into risk levels."""
    if prob < 0.3:
        return "low"
    elif prob < 0.6:
        return "medium"
    else:
        return "high"


# Test the prediction
if __name__ == "__main__":
    # Sample customer (high risk profile)
    test_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 140.70
    }
    
    print("Testing prediction...")
    result = predict(test_customer)
    print(f"Result: {result}")
