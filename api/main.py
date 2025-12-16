"""
ChurnGuard API - FastAPI application for churn prediction.
"""
from fastapi import FastAPI

app = FastAPI(
    title="ChurnGuard API",
    description="Customer churn prediction service",
    version="0.1.0"
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "churnguard"}


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": False,  # Will be True once we add model
        "version": "0.1.0"
    }
