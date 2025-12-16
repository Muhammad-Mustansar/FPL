# FPL Manager Agent - End-to-End ML Engineering System

A production-ready, AI-driven Fantasy Premier League (FPL) Manager Agent built with modern ML Engineering practices. This system predicts player performance using machine learning and optimizes squad selection using constraint satisfaction problem (CSP) solving.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [ML Pipeline](#ml-pipeline)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a complete ML Engineering system for Fantasy Premier League management, featuring:

- **Regression Models**: Predict Expected Points (xP) for players using Linear and Ridge Regression
- **Clustering**: Segment players into categories (Premiums, Budget Gems, Avoid) using K-Means
- **CSP Optimization**: Select optimal 15-player squads maximizing predicted points while satisfying constraints
- **FastAPI Service**: Production-ready REST API for predictions and optimizations
- **Prefect Orchestration**: Automated ML pipeline with retries and error handling
- **ML Validation**: DeepChecks integration for data integrity and drift detection
- **CI/CD**: Complete GitHub Actions workflow for testing and deployment

## ✨ Features

### Machine Learning Tasks

1. **Regression (Supervised Learning)**
   - Linear Regression (baseline)
   - Ridge Regression (with L2 regularization)
   - Metrics: MSE, RMSE, R², MAE
   - Cross-validation support

2. **Clustering (Unsupervised Learning)**
   - K-Means clustering
   - Automatic optimal cluster detection
   - Cluster labeling: Premiums, Budget Gems, Avoid, Overpriced, Mid-range
   - Metrics: Silhouette Score, Davies-Bouldin, Calinski-Harabasz

3. **Optimization (CSP)**
   - Greedy search algorithm
   - Hill climbing with backtracking
   - Constraint satisfaction:
     - 15 players total
     - £100m budget
     - Position constraints (2 GK, 5 DEF, 5 MID, 3 FWD)
     - Max 3 players per club

### ML Engineering Features

- **FastAPI Service**: RESTful API with Pydantic validation
- **Prefect Workflows**: Orchestrated ML pipeline with retries
- **DeepChecks Integration**: Automated ML validation
- **Docker Containerization**: Production-ready deployment
- **CI/CD Pipeline**: Automated testing and deployment
- **Model Versioning**: Tracked experiments and model artifacts

## 🏗️ Architecture

```
┌─────────────────┐
│   FastAPI API   │
│  (REST Endpoints)│
└────────┬────────┘
         │
    ┌────┴────┐
    │        │
┌───▼───┐ ┌──▼──────┐
│ ML    │ │ CSP     │
│ Models│ │Optimizer│
└───┬───┘ └─────────┘
    │
┌───▼──────────────┐
│ Prefect Pipeline │
│ (Orchestration)  │
└──────────────────┘
```

## 📁 Project Structure

```
project/
├── app/                      # FastAPI application
│   ├── __init__.py
│   ├── main.py              # FastAPI app and endpoints
│   ├── schemas.py           # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       └── ml_service.py    # ML service layer
│
├── ml/                       # Machine learning modules
│   ├── __init__.py
│   ├── data_ingestion.py    # Data loading and validation
│   ├── feature_engineering.py  # Feature creation and selection
│   ├── train_regression.py  # Regression model training
│   ├── train_clustering.py  # Clustering model training
│   └── csp_optimizer.py     # Squad optimization
│
├── prefect/                  # Workflow orchestration
│   ├── __init__.py
│   └── flow.py              # Prefect flows and tasks
│
├── tests/                    # Test suite
│   ├── test_data.py         # Data ingestion tests
│   ├── test_model.py        # ML model tests
│   └── test_ml_validation.py # DeepChecks validation
│
├── .github/workflows/        # CI/CD pipelines
│   └── ci.yml               # GitHub Actions workflow
│
├── data/                     # Data directory (generated)
├── models/                   # Saved models (generated)
│
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Multi-service orchestration
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- pip
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML_PROJECT
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p data models logs
   ```

## 💻 Usage

### 1. Train Models

Train all ML models (regression and clustering):

```bash
python -c "
from ml.data_ingestion import DataIngestion
from ml.train_regression import RegressionTrainer
from ml.train_clustering import ClusteringTrainer

# Generate mock data
ingestion = DataIngestion()
df = ingestion.generate_mock_data(n_players=500)
df = ingestion.clean_data(df)

# Train regression models
trainer = RegressionTrainer()
trainer.train_all_models(df, target_column='points')
trainer.save_all_models()

# Train clustering model
cluster_trainer = ClusteringTrainer()
results = cluster_trainer.train_all_models(df, n_clusters=3)
cluster_trainer.save_model(results['model'], results['metrics'], results['cluster_names'])
"
```

### 2. Run Prefect Pipeline

Execute the complete ML pipeline:

```bash
python -c "
from prefect.flow import ml_pipeline_flow
result = ml_pipeline_flow(
    data_dir='data',
    model_dir='models',
    n_players=500,
    n_clusters=3
)
print(result)
"
```

### 3. Start FastAPI Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at: `http://localhost:8000`

Interactive API docs: `http://localhost:8000/docs`

## 📚 API Documentation

### Endpoints

#### Health Check
```http
GET /health
```

#### Predict Player Points
```http
POST /predict-player-points
Content-Type: application/json

{
  "player_id": 1,
  "model_name": "ridge"
}
```

#### Batch Predictions
```http
POST /predict-batch-players
Content-Type: application/json

{
  "player_ids": [1, 2, 3, 4, 5],
  "model_name": "ridge"
}
```

#### Cluster Player
```http
POST /cluster-player
Content-Type: application/json

{
  "player_id": 1
}
```

#### Optimize Squad
```http
POST /optimize-squad
Content-Type: application/json

{
  "budget": 100.0,
  "constraints": {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3
  },
  "max_per_club": 3,
  "use_hill_climbing": true,
  "model_name": "ridge"
}
```

### Example API Calls

```bash
# Predict points for a player
curl -X POST "http://localhost:8000/predict-player-points" \
  -H "Content-Type: application/json" \
  -d '{"player_id": 1, "model_name": "ridge"}'

# Optimize squad
curl -X POST "http://localhost:8000/optimize-squad" \
  -H "Content-Type: application/json" \
  -d '{
    "budget": 100.0,
    "use_hill_climbing": true
  }'
```

## 🔄 ML Pipeline

The Prefect workflow orchestrates:

1. **Data Ingestion**: Load or generate player data
2. **Data Validation**: Check data quality and completeness
3. **Data Cleaning**: Handle missing values and duplicates
4. **Feature Engineering**: Create derived features
5. **Model Training**: Train regression and clustering models
6. **Model Evaluation**: Compare model performance
7. **Model Saving**: Version and save trained models
8. **Notifications**: Send status updates (placeholder)

### Run Pipeline

```python
from prefect.flow import ml_pipeline_flow

result = ml_pipeline_flow(
    data_dir="data",
    model_dir="models",
    n_players=500,
    n_clusters=3,
    send_notifications=True
)
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suites

```bash
# Unit tests
pytest tests/test_data.py tests/test_model.py -v

# ML validation tests
pytest tests/test_ml_validation.py -v

# With coverage
pytest tests/ --cov=ml --cov=app --cov-report=html
```

### DeepChecks Validation

The ML validation tests use DeepChecks to verify:

- **Data Integrity**: Missing values, duplicates, data types
- **Distribution Drift**: Train/test distribution differences
- **Model Performance**: Error analysis and performance regression

## 🔧 CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yml`) includes:

1. **Linting**: Black, Flake8, MyPy
2. **Unit Tests**: pytest with coverage
3. **ML Tests**: DeepChecks validation
4. **Model Training**: Train models in CI
5. **Docker Build**: Build containerized image
6. **Deployment**: Deploy to production (placeholder)

### Manual Trigger

```bash
# Trigger workflow manually via GitHub Actions UI
# Or use workflow_dispatch event
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t fpl-manager-agent:latest .
```

### Run with Docker Compose

```bash
docker-compose up -d
```

This starts:
- **FastAPI service** on port 8000
- **Prefect worker** for pipeline execution

### Access Services

- FastAPI: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### Stop Services

```bash
docker-compose down
```

## 📊 Model Performance

### Regression Models

- **Linear Regression**: Baseline model
- **Ridge Regression**: Regularized model (typically better)
- **Metrics**: R², RMSE, MSE, MAE

### Clustering

- **K-Means**: 3 clusters (configurable)
- **Silhouette Score**: Measures cluster quality
- **Cluster Labels**: Premiums, Budget Gems, Avoid

### CSP Optimization

- **Greedy Search**: Fast initial solution
- **Hill Climbing**: Improves solution with swaps
- **Constraints**: Budget, positions, club limits

## 🔍 Monitoring & Logging

- **Loguru**: Structured logging throughout
- **Prefect UI**: Workflow monitoring (if Prefect server enabled)
- **FastAPI Logs**: Request/response logging
- **Health Checks**: `/health` endpoint for monitoring

## 🛠️ Development

### Code Style

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

### Running Linters

```bash
black .
flake8 .
mypy .
```

## 📝 Notes

- **Mock Data**: The system generates synthetic FPL data for testing. Replace with real FPL API data in production.
- **Model Persistence**: Trained models are saved to `models/` directory
- **Data Storage**: Player data stored in `data/` directory
- **Notifications**: Slack/Discord integration placeholders included

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run CI checks locally
6. Submit a pull request

## 📄 License

This project is for educational purposes (AI321L ML Engineering Lab).

## 👥 Authors

- ML Engineering Team
- AI321L Course

## 🙏 Acknowledgments

- Fantasy Premier League for the domain inspiration
- Prefect for workflow orchestration
- DeepChecks for ML validation
- FastAPI for the web framework

---

**Status**: Production-ready ✅

**Last Updated**: 2024

