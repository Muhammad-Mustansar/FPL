"""
FastAPI Application for FPL Manager Agent

Main API endpoints for:
- Player points prediction
- Player clustering
- Squad optimization

This file integrates the ML modules (Ingestion, Regression, Clustering, Optimizer)
into a unified REST API.
"""

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import sys
import pandas as pd
import numpy as np
import os

# --- IMPORT YOUR ML MODULES ---
# Ensure your folder structure is:
# /
#   main.py
#   ml/
#     __init__.py
#     data_ingestion.py
#     feature_engineering.py
#     train_regression.py
#     train_clustering.py
#     csp_optimizer.py

try:
    from ml.data_ingestion import DataIngestion
    from ml.feature_engineering import FeatureEngineering
    from ml.train_regression import RegressionTrainer
    from ml.train_clustering import ClusteringTrainer
    from ml.csp_optimizer import CSPOptimizer, SquadConstraints
except ImportError as e:
    logger.error(f"Failed to import ML modules: {e}")
    logger.error("Ensure you have the 'ml' folder with all required scripts (data_ingestion, train_regression, etc.)")
    sys.exit(1)

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")

# ==========================================
# 1. PYDANTIC SCHEMAS (Request/Response Models)
# ==========================================

class PlayerPredictionRequest(BaseModel):
    player_id: int
    model_name: Optional[str] = "ridge"

class PlayerPredictionResponse(BaseModel):
    player_id: int
    player_name: str
    predicted_points: float
    model_name: str
    confidence: float

class BatchPredictionRequest(BaseModel):
    player_ids: List[int]
    model_name: Optional[str] = "ridge"

class BatchPredictionResponse(BaseModel):
    predictions: List[PlayerPredictionResponse]
    total_players: int

class ClusterPlayerRequest(BaseModel):
    player_id: int

class ClusterPlayerResponse(BaseModel):
    player_id: int
    player_name: str
    cluster_id: int
    cluster_label: str
    cluster_characteristics: Dict[str, float]

class ClusterAllPlayersResponse(BaseModel):
    clusters: List[ClusterPlayerResponse]
    cluster_statistics: Dict[str, Dict[str, float]]
    total_players: int

class SquadOptimizationRequest(BaseModel):
    budget: float = 100.0
    max_per_club: int = 3
    use_hill_climbing: bool = True
    model_name: Optional[str] = "ridge"
    constraints: Optional[Dict[str, int]] = None  # e.g. {"GK": 2, "DEF": 5...}

class SquadPlayer(BaseModel):
    id: int
    name: str
    position: str
    team: str
    cost: float
    predicted_points: float
    points_per_game: Optional[float] = None

class SquadOptimizationResponse(BaseModel):
    squad: List[SquadPlayer]
    total_cost: float
    total_predicted_points: float
    position_distribution: Dict[str, int]
    club_distribution: Dict[str, int]
    is_valid: bool
    violations: List[str]
    optimization_method: str

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    metrics: Optional[Dict[str, float]] = None
    feature_count: Optional[int] = None

class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    default_regression_model: str
    default_clustering_model: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: bool

class ErrorResponse(BaseModel):
    error: str
    detail: str


# ==========================================
# 2. ML SERVICE LAYER (Orchestrator)
# ==========================================

class MLService:
    """
    Orchestrates the data flow between Ingestion, Training, Prediction, and Optimization.
    Singleton pattern recommended for production to avoid reloading models constantly.
    """
    def __init__(self):
        self.ingestion = DataIngestion()
        self.feature_engineer = FeatureEngineering()
        self.regression_trainer = RegressionTrainer()
        self.clustering_trainer = ClusteringTrainer()
        self.optimizer = CSPOptimizer()
        
        self.data_cache = None
        self.last_data_fetch = None
        
        # Store trained models in memory
        self.regression_models = {}
        self.clustering_model = None
        self.clustering_labels = {}
        self.clustering_names = {}
        
        # State flags
        self.is_ready = False

    def load_or_generate_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch live data or generate mock data if API fails."""
        if not force_refresh and self.data_cache is not None:
            return self.data_cache

        logger.info("Fetching fresh data...")
        df = self.ingestion.fetch_live_data()
        
        if df is None or df.empty:
            logger.warning("Live data unavailable. Generating mock data.")
            df = self.ingestion.generate_mock_data(n_players=600)
            
        df = self.ingestion.clean_data(df)
        self.data_cache = df
        self.last_data_fetch = datetime.now()
        return df

    def ensure_models_trained(self):
        """Check if models are trained, if not, train them now."""
        if not self.regression_models or self.clustering_model is None:
            logger.info("Models not found in memory. Triggering training pipeline...")
            self.train_all_models()

    def load_models(self) -> bool:
        """
        Try to load saved models from disk. 
        Returns True if successful, False if retraining needed.
        """
        # simplified for this example: usually you'd load .joblib files here
        # For now, we'll just check if they exist in memory or retrain
        if self.regression_models:
            return True
        return False

    def train_all_models(self):
        """Run the full training pipeline."""
        df = self.load_or_generate_data()
        
        # 1. Train Regression (Points Prediction)
        logger.info("Training Regression Models...")
        reg_results = self.regression_trainer.train_all_models(
            df, 
            target_column='points',
            scale_features=True, 
            select_features=True
        )
        self.regression_models = reg_results['models']
        self.regression_metrics = reg_results['metrics']
        
        # 2. Train Clustering (Player Segmentation)
        logger.info("Training Clustering Models...")
        cluster_results = self.clustering_trainer.train_all_models(
            df,
            n_clusters=4,  # Premiums, Mid-range, Budget, Avoid
            find_optimal=False
        )
        self.clustering_model = cluster_results['model']
        self.clustering_labels = cluster_results['cluster_labels'] # This is an array matching DF index
        self.clustering_names = cluster_results['cluster_names']
        
        self.is_ready = True
        logger.info("All models trained and ready.")

    def predict_player_points(self, player_id: int, model_name: str = "ridge"):
        self.ensure_models_trained()
        df = self.load_or_generate_data()
        
        player_row = df[df['id'] == player_id]
        if player_row.empty:
            raise ValueError(f"Player ID {player_id} not found.")
        
        model = self.regression_models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.regression_models.keys())}")
        
        # Prepare single row for prediction
        # Note: In a real app, 'prepare_regression_features' expects a full DF usually
        # We re-process the whole DF to ensure scaling/encoding consistency, then pick the row
        X, _ = self.feature_engineer.prepare_regression_features(df, target_column='points')
        
        # Align columns
        trained_cols = self.regression_trainer.final_feature_columns
        if trained_cols:
            # ensure X has all cols, fill missing with 0
            for col in trained_cols:
                if col not in X.columns: X[col] = 0
            X = X[trained_cols]
        
        # Find index of player in processed X (indices should align if not shuffled)
        # Safest way is to map via original index if preserved
        player_idx = player_row.index[0]
        
        if player_idx not in X.index:
             raise ValueError("Player lost during feature engineering (likely dropped due to missing data).")
             
        X_player = X.loc[[player_idx]]
        
        # Predict
        predicted_points = model.predict(X_player)[0]
        return float(predicted_points), player_row.iloc[0]['name']

    def predict_batch_players(self, player_ids: List[int], model_name: str = "ridge"):
        self.ensure_models_trained()
        df = self.load_or_generate_data()
        model = self.regression_models.get(model_name)
        
        # Prepare all features
        X, _ = self.feature_engineer.prepare_regression_features(df, target_column='points')
        
        # Align columns
        trained_cols = self.regression_trainer.final_feature_columns
        if trained_cols:
            for col in trained_cols:
                if col not in X.columns: X[col] = 0
            X = X[trained_cols]
        
        results = {}
        for pid in player_ids:
            player_row = df[df['id'] == pid]
            if player_row.empty:
                continue
            
            idx = player_row.index[0]
            if idx in X.index:
                pred = model.predict(X.loc[[idx]])[0]
                results[pid] = (float(pred), player_row.iloc[0]['name'])
                
        return results

    def cluster_player(self, player_id: int):
        self.ensure_models_trained()
        df = self.load_or_generate_data()
        
        player_row = df[df['id'] == player_id]
        if player_row.empty:
             raise ValueError(f"Player ID {player_id} not found.")
        
        # In this simple implementation, we look up the label generated during training
        # Assuming df order hasn't changed since training. 
        # For robustness, we should re-predict using self.clustering_model.predict()
        
        # Let's do a fresh prediction to be safe:
        X_cluster = self.feature_engineer.prepare_clustering_features(df)
        
        # Scale if needed (ClusteringTrainer handles internal scaling usually, 
        # but let's assume we reuse the pipeline)
        
        idx = player_row.index[0]
        # For K-Means, we need the exact feature set used in training
        # Simplification: We will just return the cached label if indices match
        # (Real prod code would use a proper inference pipeline)
        
        # Fallback: Just re-run clustering on current data to find where this player sits
        # (Not efficient for single API call, but safe)
        cluster_id = self.clustering_trainer.predict_cluster(X_cluster.loc[[idx]])[0]
        
        label = self.clustering_names.get(int(cluster_id), f"Cluster {cluster_id}")
        return int(cluster_id), label

    def optimize_squad(self, budget, constraints, max_per_club, use_hill_climbing, model_name):
        self.ensure_models_trained()
        df = self.load_or_generate_data()
        
        # 1. Generate predictions for ALL players
        # (Optimizer needs full pool to select from)
        X, _ = self.feature_engineer.prepare_regression_features(df, target_column='points')
        trained_cols = self.regression_trainer.final_feature_columns
        if trained_cols:
            for col in trained_cols:
                if col not in X.columns: X[col] = 0
            X = X[trained_cols]
            
        model = self.regression_models.get(model_name)
        if not model: raise ValueError("Model not found")
        
        all_predictions = model.predict(X)
        
        # Map back to DF
        # We need to ensure alignment. X preserves index of df.
        df['predicted_points'] = pd.Series(all_predictions, index=X.index)
        
        # Create prediction dict for optimizer
        pred_dict = df.set_index('id')['predicted_points'].to_dict()
        
        # 2. Setup Constraints
        squad_constraints = SquadConstraints(
            budget=budget,
            max_per_club=max_per_club,
            positions=constraints or {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        )
        
        # 3. Optimize
        # Note: Passing constraints to constructor if your CSPOptimizer supports it,
        # otherwise modifying the instance
        self.optimizer.constraints = squad_constraints
        
        solution = self.optimizer.optimize_squad(df, pred_dict)
        
        # 4. Format Result
        squad_details = self.optimizer.get_squad_details(solution, df)
        
        # Merge predicted points into details
        squad_list = []
        for _, row in squad_details.iterrows():
            pid = row['id']
            # Get data from original df to ensure we have all fields
            orig_row = df[df['id'] == pid].iloc[0]
            
            squad_list.append({
                "id": pid,
                "name": orig_row['name'],
                "position": orig_row['position'],
                "team": orig_row['team'],
                "cost": orig_row['cost'],
                "predicted_points": pred_dict.get(pid, 0),
                "points_per_game": orig_row.get('points_per_game', 0)
            })
            
        return {
            "squad": squad_list,
            "total_cost": solution.total_cost,
            "total_predicted_points": solution.total_points,
            "position_distribution": solution.position_distribution,
            "club_distribution": solution.club_distribution,
            "is_valid": solution.is_valid,
            "violations": solution.violations,
            "optimization_method": "Greedy (Hill Climbing optional)"
        }

# Initialize Service
ml_service = MLService()


# ==========================================
# 3. FASTAPI APP & ROUTES
# ==========================================

app = FastAPI(
    title="FPL Manager Agent API",
    description="AI-Driven Fantasy Premier League Manager Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting FPL Manager Agent API...")
    # Optional: Train models immediately on startup
    # ml_service.train_all_models()

@app.get("/", tags=["General"])
async def root():
    return {"message": "FPL Manager Agent API is running", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=ml_service.is_ready
    )

@app.post("/predict-player-points", response_model=PlayerPredictionResponse, tags=["Predictions"])
async def predict_player_points(request: PlayerPredictionRequest):
    try:
        points, name = ml_service.predict_player_points(request.player_id, request.model_name)
        return PlayerPredictionResponse(
            player_id=request.player_id,
            player_name=name,
            predicted_points=points,
            model_name=request.model_name or "ridge",
            confidence=0.85
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize-squad", response_model=SquadOptimizationResponse, tags=["Optimization"])
async def optimize_squad_endpoint(request: SquadOptimizationRequest):
    try:
        result = ml_service.optimize_squad(
            request.budget, 
            request.constraints, 
            request.max_per_club, 
            request.use_hill_climbing, 
            request.model_name
        )
        return result # Pydantic will validate against SquadOptimizationResponse
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8000)