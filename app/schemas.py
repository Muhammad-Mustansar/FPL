"""
Pydantic Schemas for FastAPI Request/Response Models

Defines data validation schemas for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Player-related schemas
class PlayerBase(BaseModel):
    """Base player schema."""
    id: int
    name: str
    position: str
    team: str
    cost: float = Field(..., ge=4.0, le=15.0, description="Player cost in millions")
    points: Optional[float] = None
    goals: Optional[int] = None
    assists: Optional[int] = None
    clean_sheets: Optional[int] = None
    form: Optional[float] = None
    selected_by_percent: Optional[float] = None
    minutes: Optional[int] = None


class PlayerPredictionRequest(BaseModel):
    """Request schema for player points prediction."""
    player_id: int = Field(..., description="Player ID to predict")
    model_name: Optional[str] = Field(default="ridge", description="Model to use: 'linear' or 'ridge'")


class PlayerPredictionResponse(BaseModel):
    """Response schema for player points prediction."""
    player_id: int
    player_name: str
    predicted_points: float = Field(..., description="Predicted Expected Points (xP)")
    model_name: str
    confidence: Optional[float] = Field(None, description="Model confidence score (R²)")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch player predictions."""
    player_ids: List[int] = Field(..., description="List of player IDs to predict")
    model_name: Optional[str] = Field(default="ridge", description="Model to use: 'linear' or 'ridge'")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PlayerPredictionResponse]
    total_players: int


# Clustering schemas
class ClusterPlayerRequest(BaseModel):
    """Request schema for clustering a single player."""
    player_id: int = Field(..., description="Player ID to cluster")


class ClusterPlayerResponse(BaseModel):
    """Response schema for player clustering."""
    player_id: int
    player_name: str
    cluster_id: int
    cluster_label: str = Field(..., description="Cluster label: Premiums, Budget Gems, Avoid, etc.")
    cluster_characteristics: Optional[Dict[str, Any]] = None


class ClusterAllPlayersResponse(BaseModel):
    """Response schema for clustering all players."""
    clusters: List[ClusterPlayerResponse]
    cluster_statistics: Dict[str, Any] = Field(..., description="Statistics for each cluster")
    total_players: int


# Squad optimization schemas
class SquadOptimizationRequest(BaseModel):
    """Request schema for squad optimization."""
    budget: Optional[float] = Field(default=100.0, ge=80.0, le=100.0, description="Budget in millions")
    constraints: Optional[Dict[str, int]] = Field(
        default=None,
        description="Position constraints: {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}"
    )
    max_per_club: Optional[int] = Field(default=3, ge=1, le=5, description="Maximum players per club")
    use_hill_climbing: Optional[bool] = Field(default=True, description="Use hill climbing optimization")
    model_name: Optional[str] = Field(default="ridge", description="Model to use for predictions")


class SquadPlayer(BaseModel):
    """Schema for a player in the optimized squad."""
    id: int
    name: str
    position: str
    team: str
    cost: float
    predicted_points: float
    points_per_game: Optional[float] = None


class SquadOptimizationResponse(BaseModel):
    """Response schema for squad optimization."""
    squad: List[SquadPlayer] = Field(..., description="Selected players")
    total_cost: float = Field(..., description="Total squad cost in millions")
    total_predicted_points: float = Field(..., description="Total predicted points")
    position_distribution: Dict[str, int] = Field(..., description="Players per position")
    club_distribution: Dict[str, int] = Field(..., description="Players per club")
    is_valid: bool = Field(..., description="Whether squad satisfies all constraints")
    violations: List[str] = Field(default_factory=list, description="Constraint violations if any")
    optimization_method: str = Field(..., description="Method used: greedy, hill_climbing, etc.")


# Model information schemas
class ModelInfo(BaseModel):
    """Schema for model information."""
    model_name: str
    model_type: str
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    trained_at: Optional[str] = None
    feature_count: Optional[int] = None


class ModelListResponse(BaseModel):
    """Response schema for listing available models."""
    models: List[ModelInfo]
    default_regression_model: str
    default_clustering_model: str


# Health check schema
class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    timestamp: str
    version: str = "1.0.0"
    models_loaded: bool


# Authentication schemas
# Error schemas
class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

