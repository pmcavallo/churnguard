# ChurnGuard

An end-to-end MLOps pipeline for predicting customer churn, demonstrating production-ready machine learning deployment.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazon-aws&logoColor=white)

## Overview

ChurnGuard predicts whether a telecom customer will churn based on their account characteristics, service usage, and billing information. The project demonstrates the complete ML lifecycle from model training to cloud deployment.

**Live Demo:** `http://54.158.47.223:8000/docs` *(EC2 instance may be stopped to save costs)*

![API Documentation](Screenshots/UI.png)

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost |
| API Framework | FastAPI |
| Containerization | Docker, docker-compose |
| Experiment Tracking | MLflow |
| Container Registry | AWS ECR |
| Cloud Deployment | AWS EC2 |

## Model Performance

Trained on the [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset (7,043 customers, 19 features).

| Metric | Score |
|--------|-------|
| ROC AUC | 0.840 |
| Accuracy | 79.9% |
| Precision | 65.3% |
| Recall | 51.9% |
| F1 Score | 57.8% |

## Experiment Tracking

MLflow tracks all training runs, enabling comparison of different hyperparameter configurations.

![MLflow Experiments](Screenshots/MLflow_exoeriment.png)

The parallel coordinates plot visualizes how hyperparameters affect model performance. In this comparison, the simpler model (depth=5, 100 trees) outperformed the more complex configuration, demonstrating classic overfitting behavior.

![MLflow Comparison](Screenshots/MLflow.png)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check with model status |
| `/predict` | POST | Churn prediction |

### Sample Prediction

**Request:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
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
  "TotalCharges": 70.35
}
```

**Response:**
```json
{
  "churn_probability": 0.752,
  "churn_prediction": true,
  "confidence": 0.752,
  "risk_level": "high"
}
```

![Prediction Response](Screenshots/prediction.png)

## Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop

### Local Development

1. **Clone and setup:**
```bash
git clone https://github.com/pmcavallo/churnguard.git
cd churnguard
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **Download data and train model:**
```bash
python data/download_data.py
python -m model.train
```

3. **Run API:**
```bash
uvicorn api.main:app --reload
```

4. **Access:** http://localhost:8000/docs

### Docker

```bash
# Build and run
docker build -t churnguard:latest .
docker run -p 8000:8000 churnguard:latest

# Or use docker-compose (includes MLflow)
docker-compose up --build
```

### MLflow UI

```bash
mlflow ui
# Access: http://localhost:5000
```

## Cloud Deployment

The application is deployed on AWS using the following architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                            │
│                                                             │
│   ┌─────────┐         ┌─────────────────────────────────┐   │
│   │   ECR   │         │         EC2 Instance            │   │
│   │ (image  │◄────────│  ┌───────────────────────────┐  │   │
│   │ storage)│  pull   │  │    Docker Container       │  │   │
│   └─────────┘         │  │  ┌─────────────────────┐  │  │   │
│                       │  │  │  FastAPI + XGBoost  │  │  │   │
│                       │  │  └─────────────────────┘  │  │   │
│                       │  └───────────────────────────┘  │   │
│                       │              │                   │   │
│                       │         port 8000                │   │
│                       └──────────────┼───────────────────┘   │
│                                      │                       │
└──────────────────────────────────────┼───────────────────────┘
                                       ▼
                                   Internet
```

![EC2 Instance](Screenshots/ec2.png)

### Deployment Steps

1. Push Docker image to ECR
2. Launch EC2 instance with IAM role for ECR access
3. Configure security group to allow port 8000
4. User data script automatically pulls and runs container

## Project Structure

```
churnguard/
├── api/
│   ├── main.py          # FastAPI application
│   └── schemas.py       # Pydantic models
├── model/
│   ├── train.py         # Training script with MLflow
│   └── predict.py       # Inference logic
├── data/
│   └── download_data.py # Dataset downloader
├── notebooks/
│   └── 01_exploration.ipynb
├── Screenshots/         # Documentation images
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Key Features

- **Real-time predictions** via REST API
- **Experiment tracking** with MLflow for reproducibility
- **Containerized** for consistent deployment across environments
- **Multi-service orchestration** with docker-compose
- **Cloud-deployed** on AWS EC2

## Author

**Paulo Cavallo**  
[LinkedIn](https://www.linkedin.com/in/pmcavallo/) | [GitHub](https://github.com/pmcavallo)
