"""
FastAPI Application for FPL Manager Agent
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
import joblib  # Added for saving/loading models

# --- IMPORT YOUR ML MODULES ---
try:
    from ml.data_ingestion import DataIngestion
    from ml.feature_engineering import FeatureEngineering
    from ml.train_regression import RegressionTrainer
    from ml.train_clustering import ClusteringTrainer
    from ml.csp_optimizer import CSPOptimizer, SquadConstraints
except ImportError as e:
    logger.error(f"Failed to import ML modules: {e}")
    sys.exit(1)

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")

# ==========================================
# 1. PYDANTIC SCHEMAS
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

class SquadOptimizationRequest(BaseModel):
    budget: float = 100.0
    max_per_club: int = 3
    use_hill_climbing: bool = True
    model_name: Optional[str] = "ridge"
    constraints: Optional[Dict[str, int]] = None 

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

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: bool

# ==========================================
# 2. ML SERVICE LAYER
# ==========================================

class MLService:
    def __init__(self):
        self.ingestion = DataIngestion()
        self.feature_engineer = FeatureEngineering()
        self.regression_trainer = RegressionTrainer()
        self.clustering_trainer = ClusteringTrainer()
        self.optimizer = CSPOptimizer()
        
        self.data_cache = None
        self.regression_models = {}
        self.clustering_model = None
        self.clustering_names = {}
        self.is_ready = False
        
        # Ensure models directory exists
        if not os.path.exists("models"):
            os.makedirs("models")

    def load_or_generate_data(self, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.data_cache is not None:
            return self.data_cache
        logger.info("Fetching fresh data...")
        df = self.ingestion.fetch_live_data()
        if df is None or df.empty:
            df = self.ingestion.generate_mock_data(n_players=600)
        df = self.ingestion.clean_data(df)
        self.data_cache = df
        return df

    def train_all_models(self):
        """Run the full training pipeline and set is_ready to True."""
        df = self.load_or_generate_data()
        
        logger.info("Training Regression Models...")
        reg_results = self.regression_trainer.train_all_models(df, target_column='points')
        self.regression_models = reg_results['models']
        
        logger.info("Training Clustering Models...")
        cluster_results = self.clustering_trainer.train_all_models(df, n_clusters=4)
        self.clustering_model = cluster_results['model']
        self.clustering_names = cluster_results['cluster_names']
        
        # Save to disk
        joblib.dump(self.regression_models, "models/regression_models.joblib")
        joblib.dump(self.clustering_model, "models/clustering_model.joblib")
        
        self.is_ready = True
        logger.info("✅ All models trained, saved, and ready.")

    def predict_player_points(self, player_id: int, model_name: str = "ridge"):
        df = self.load_or_generate_data()
        player_row = df[df['id'] == player_id]
        if player_row.empty:
            raise ValueError(f"Player ID {player_id} not found.")
        
        model = self.regression_models.get(model_name)
        X, _ = self.feature_engineer.prepare_regression_features(df, target_column='points')
        
        trained_cols = self.regression_trainer.final_feature_columns
        if trained_cols:
            for col in trained_cols:
                if col not in X.columns: X[col] = 0
            X = X[trained_cols]
        
        player_idx = player_row.index[0]
        predicted_points = model.predict(X.loc[[player_idx]])[0]
        return float(predicted_points), player_row.iloc[0]['name']

    def optimize_squad(self, budget, constraints, max_per_club, use_hill_climbing, model_name):
        df = self.load_or_generate_data()
        X, _ = self.feature_engineer.prepare_regression_features(df, target_column='points')
        
        trained_cols = self.regression_trainer.final_feature_columns
        if trained_cols:
            for col in trained_cols:
                if col not in X.columns: X[col] = 0
            X = X[trained_cols]
            
        model = self.regression_models.get(model_name)
        all_predictions = model.predict(X)
        df['predicted_points'] = pd.Series(all_predictions, index=X.index)
        pred_dict = df.set_index('id')['predicted_points'].to_dict()
        
        squad_constraints = SquadConstraints(
            budget=budget,
            max_per_club=max_per_club,
            positions=constraints or {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        )
        
        self.optimizer.constraints = squad_constraints
        solution = self.optimizer.optimize_squad(df, pred_dict)
        squad_details = self.optimizer.get_squad_details(solution, df)
        
        squad_list = []
        for _, row in squad_details.iterrows():
            pid = row['id']
            orig_row = df[df['id'] == pid].iloc[0]
            squad_list.append({
                "id": pid, "name": orig_row['name'], "position": orig_row['position'],
                "team": orig_row['team'], "cost": orig_row['cost'],
                "predicted_points": pred_dict.get(pid, 0)
            })
            
        return {
            "squad": squad_list, "total_cost": solution.total_cost,
            "total_predicted_points": solution.total_points,
            "position_distribution": solution.position_distribution,
            "club_distribution": solution.club_distribution,
            "is_valid": solution.is_valid, "violations": solution.violations,
            "optimization_method": "Greedy"
        }

ml_service = MLService()

# ==========================================
# 3. FASTAPI APP & ROUTES
# ==========================================

app = FastAPI(title="FPL Manager Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting FPL Manager Agent API...")
    # This automatically trains your models when you start the backend
    try:
        ml_service.train_all_models()
    except Exception as e:
        logger.error(f"Failed to train models on startup: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=ml_service.is_ready
    )

@app.post("/predict-player-points", response_model=PlayerPredictionResponse)
async def predict_endpoint(request: PlayerPredictionRequest):
    try:
        points, name = ml_service.predict_player_points(request.player_id, request.model_name)
        return PlayerPredictionResponse(
            player_id=request.player_id, player_name=name,
            predicted_points=points, model_name=request.model_name or "ridge", confidence=0.85
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize-squad", response_model=SquadOptimizationResponse)
async def optimize_endpoint(request: SquadOptimizationRequest):
    try:
        return ml_service.optimize_squad(
            request.budget, request.constraints, request.max_per_club, 
            request.use_hill_climbing, request.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))