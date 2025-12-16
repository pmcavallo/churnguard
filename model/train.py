"""
ChurnGuard Model Training Script
Trains an XGBoost classifier on Telco customer churn data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import os

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")


def load_and_clean_data(path):
    """Load data and fix known issues."""
    df = pd.read_csv(path)
    
    # Fix TotalCharges (empty strings -> 0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    return df


def prepare_features(df):
    """
    Prepare features for training.
    Returns: X (features), y (target), encoders (for inference)
    """
    # Drop customerID - not a feature
    df = df.drop('customerID', axis=1)
    
    # Separate target
    y = (df['Churn'] == 'Yes').astype(int)
    X = df.drop('Churn', axis=1)
    
    # Encode categorical columns
    encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    return X, y, encoders


def train_model(X_train, y_train):
    """Train XGBoost classifier."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    return metrics


def main():
    print("=" * 50)
    print("ChurnGuard Model Training")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"   Loaded {len(df)} rows")
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y, encoders = prepare_features(df)
    print(f"   Features: {X.shape[1]}")
    print(f"   Churn rate: {y.mean():.1%}")
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    print("\n4. Training XGBoost model...")
    model = train_model(X_train, y_train)
    print("   Done!")
    
    # Evaluate
    print("\n5. Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Save model and encoders
    print("\n6. Saving artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Encoders saved to: {ENCODERS_PATH}")
    
    # Save feature names for inference
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    joblib.dump(X.columns.tolist(), feature_names_path)
    print(f"   Feature names saved to: {feature_names_path}")
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
