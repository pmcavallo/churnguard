"""
ChurnGuard Model Training Script
Trains an XGBoost classifier on Telco customer churn data.
Now with MLflow experiment tracking.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import mlflow
import mlflow.xgboost
import os

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")

# MLflow experiment name
EXPERIMENT_NAME = "churnguard-churn-prediction"


def load_and_clean_data(path):
    """Load data and fix known issues."""
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df


def prepare_features(df):
    """Prepare features for training."""
    df = df.drop('customerID', axis=1)
    y = (df['Churn'] == 'Yes').astype(int)
    X = df.drop('Churn', axis=1)
    
    encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    return X, y, encoders


def train_model(X_train, y_train, params):
    """Train XGBoost classifier with given parameters."""
    model = xgb.XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
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
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }


def main():
    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print("=" * 50)
    print("ChurnGuard Model Training (with MLflow)")
    print("=" * 50)
    
    # Define hyperparameters (easy to change for experiments)
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'test_size': 0.2
    }
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Load data
        print("\n1. Loading data...")
        df = load_and_clean_data(DATA_PATH)
        mlflow.log_param("dataset_rows", len(df))
        print(f"   Loaded {len(df)} rows")
        
        # Prepare features
        print("\n2. Preparing features...")
        X, y, encoders = prepare_features(df)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("churn_rate", round(y.mean(), 4))
        print(f"   Features: {X.shape[1]}")
        print(f"   Churn rate: {y.mean():.1%}")
        
        # Split data
        print("\n3. Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params['test_size'], random_state=42, stratify=y
        )
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train model
        print("\n4. Training XGBoost model...")
        model = train_model(X_train, y_train, params)
        print("   Done!")
        
        # Evaluate
        print("\n5. Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        for name, value in metrics.items():
            print(f"   {name}: {value:.4f}")
        
        # Save artifacts locally
        print("\n6. Saving artifacts...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
        
        feature_names_path = os.path.join(MODEL_DIR, "feature_names.joblib")
        joblib.dump(X.columns.tolist(), feature_names_path)
        
        # Log model to MLflow
        mlflow.xgboost.log_model(model, "model")
        
        # Log artifacts to MLflow
        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(ENCODERS_PATH)
        mlflow.log_artifact(feature_names_path)
        
        print(f"   Model saved to: {MODEL_PATH}")
        print(f"   Artifacts logged to MLflow")
        
        # Print MLflow run info
        run_id = mlflow.active_run().info.run_id
        print(f"\n   MLflow Run ID: {run_id}")
        
    print("\n" + "=" * 50)
    print("Training complete! View results with: mlflow ui")
    print("=" * 50)


if __name__ == "__main__":
    main()
