"""
ML Service Layer for FastAPI

Handles model loading, predictions, clustering, and squad optimization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import joblib
import json

from ml.data_ingestion import DataIngestion
from ml.feature_engineering import FeatureEngineering
from ml.train_regression import RegressionTrainer
from ml.train_clustering import ClusteringTrainer
from ml.csp_optimizer import CSPOptimizer, SquadConstraints


class MLService:
    """
    Service class for ML operations.
    Handles model loading, predictions, and optimizations.
    """
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        """
        Initialize ML service.
        
        Args:
            model_dir: Directory containing saved models
            data_dir: Directory containing data files
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        self.regression_models = {}
        self.clustering_model = None
        self.feature_engineer = None
        self.player_data = None
        
        logger.info("ML Service initialized")
    
    def load_models(self) -> bool:
        """
        Load all trained models from disk.
        
        Returns:
            True if models loaded successfully
        """
        try:
            # Try to load saved models
            model_files = list(self.model_dir.glob("*.pkl"))
            
            if not model_files:
                logger.warning("No saved models found. Models will be trained on first request.")
                return False
            
            # Load feature engineer
            fe_path = self.model_dir / "feature_engineer.pkl"
            if fe_path.exists():
                self.feature_engineer = joblib.load(fe_path)
                logger.info("Loaded feature engineer")
            
            # Load regression models
            for model_file in model_files:
                if "linear" in model_file.stem.lower():
                    self.regression_models['linear'] = joblib.load(model_file)
                    logger.info(f"Loaded linear regression model: {model_file}")
                elif "ridge" in model_file.stem.lower():
                    self.regression_models['ridge'] = joblib.load(model_file)
                    logger.info(f"Loaded ridge regression model: {model_file}")
            
            # Load clustering model
            cluster_files = [f for f in model_files if "kmeans" in f.stem.lower()]
            if cluster_files:
                self.clustering_model = joblib.load(cluster_files[0])
                logger.info(f"Loaded clustering model: {cluster_files[0]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def load_or_generate_data(self) -> pd.DataFrame:
        """
        Load player data from file or generate mock data.
        
        Returns:
            DataFrame with player data
        """
        if self.player_data is not None:
            return self.player_data
        
        ingestion = DataIngestion(str(self.data_dir))

        # Prefer the provided 2023/24 dataset if available
        data_file_new = self.data_dir / "fpl_2023_24_data.csv"
        data_file_legacy = self.data_dir / "fpl_players.csv"

        if data_file_new.exists():
            raw_df = ingestion.load_from_csv(str(data_file_new))
            normalized_df = ingestion.normalize_fpl_2023_data(raw_df)
            self.player_data = ingestion.clean_data(normalized_df)
            # Cache a normalized copy as fpl_players.csv for downstream reuse
            ingestion.save_data(self.player_data, "fpl_players.csv")
            logger.info(f"Loaded {len(self.player_data)} players from {data_file_new}")
        elif data_file_legacy.exists():
            self.player_data = ingestion.load_from_csv(str(data_file_legacy))
            self.player_data = ingestion.clean_data(self.player_data)
            logger.info(f"Loaded {len(self.player_data)} players from {data_file_legacy}")
        else:
            # Generate mock data
            self.player_data = ingestion.generate_mock_data(n_players=500)
            self.player_data = ingestion.clean_data(self.player_data)
            ingestion.save_data(self.player_data, "fpl_players.csv")
            logger.info(f"Generated {len(self.player_data)} mock players")
        
        return self.player_data
    
    def ensure_models_trained(self):
        """Ensure models are trained if not already loaded."""
        if not self.regression_models or not self.feature_engineer:
            logger.info("Training models on first request...")
            df = self.load_or_generate_data()
            
            # Train regression models
            trainer = RegressionTrainer(str(self.model_dir))
            results = trainer.train_all_models(
                df,
                target_column='points',
                test_size=0.2,
                scale_features=True,
                select_features=True,
                n_features=15
            )
            
            self.regression_models = results['models']
            self.feature_engineer = results['feature_engineer']
            
            # Save models
            trainer.save_all_models()
            logger.info("Models trained and saved")
        
        if not self.clustering_model:
            logger.info("Training clustering model...")
            df = self.load_or_generate_data()
            
            # Train clustering model
            cluster_trainer = ClusteringTrainer(str(self.model_dir))
            cluster_results = cluster_trainer.train_all_models(
                df,
                n_clusters=3,
                find_optimal=False,
                scale_features=True
            )
            
            self.clustering_model = cluster_results['model']
            cluster_trainer.save_model(
                cluster_results['model'],
                cluster_results['metrics'],
                cluster_results['cluster_names']
            )
            logger.info("Clustering model trained and saved")
    
    def predict_player_points(
        self,
        player_id: int,
        model_name: str = "ridge"
    ) -> Tuple[float, str]:
        """
        Predict Expected Points (xP) for a player.
        
        Args:
            player_id: Player ID
            model_name: Model to use ('linear' or 'ridge')
            
        Returns:
            Tuple of (predicted_points, player_name)
        """
        self.ensure_models_trained()
        
        df = self.load_or_generate_data()
        
        if player_id not in df['id'].values:
            raise ValueError(f"Player ID {player_id} not found")
        
        # Get player data
        player_row = df[df['id'] == player_id].iloc[0]
        player_name = player_row['name']
        
        # Prepare features
        X, _ = self.feature_engineer.prepare_regression_features(df, target_column='points')
        player_features = X[df['id'] == player_id]
        
        # Scale and select features
        player_features_scaled = self.feature_engineer.scale_features(player_features, fit=False)
        player_features_selected = self.feature_engineer.select_features(
            player_features_scaled,
            pd.Series([0]),  # Dummy target
            k=15,
            fit=False
        )
        
        # Predict
        if model_name not in self.regression_models:
            model_name = "ridge"  # Default
        
        model = self.regression_models[model_name]
        prediction = model.predict(player_features_selected)[0]
        
        return float(prediction), player_name
    
    def predict_batch_players(
        self,
        player_ids: List[int],
        model_name: str = "ridge"
    ) -> Dict[int, Tuple[float, str]]:
        """
        Predict Expected Points for multiple players.
        
        Args:
            player_ids: List of player IDs
            model_name: Model to use
            
        Returns:
            Dictionary mapping player_id to (predicted_points, player_name)
        """
        self.ensure_models_trained()
        
        df = self.load_or_generate_data()
        
        # Filter to requested players
        valid_ids = df[df['id'].isin(player_ids)]['id'].tolist()
        if len(valid_ids) != len(player_ids):
            missing = set(player_ids) - set(valid_ids)
            logger.warning(f"Missing player IDs: {missing}")
        
        # Prepare features for all players
        X, _ = self.feature_engineer.prepare_regression_features(df, target_column='points')
        player_features = X[df['id'].isin(valid_ids)]
        
        # Scale and select
        player_features_scaled = self.feature_engineer.scale_features(player_features, fit=False)
        player_features_selected = self.feature_engineer.select_features(
            player_features_scaled,
            pd.Series([0] * len(player_features)),
            k=15,
            fit=False
        )
        
        # Predict
        if model_name not in self.regression_models:
            model_name = "ridge"
        
        model = self.regression_models[model_name]
        predictions = model.predict(player_features_selected)
        
        # Map to player IDs
        player_df = df[df['id'].isin(valid_ids)][['id', 'name']]
        results = {}
        for idx, (_, row) in enumerate(player_df.iterrows()):
            results[row['id']] = (float(predictions[idx]), row['name'])
        
        return results
    
    def cluster_player(
        self,
        player_id: int
    ) -> Tuple[int, str]:
        """
        Get cluster assignment for a player.
        
        Args:
            player_id: Player ID
            
        Returns:
            Tuple of (cluster_id, cluster_label)
        """
        self.ensure_models_trained()
        
        df = self.load_or_generate_data()
        
        if player_id not in df['id'].values:
            raise ValueError(f"Player ID {player_id} not found")
        
        # Prepare clustering features
        X = self.feature_engineer.prepare_clustering_features(df)
        player_features = X[df['id'] == player_id]
        
        # Scale if model was trained with scaling
        if hasattr(self.clustering_model, 'scaler'):
            player_features = pd.DataFrame(
                self.clustering_model.scaler.transform(player_features),
                columns=player_features.columns,
                index=player_features.index
            )
        
        # Predict cluster
        cluster_id = self.clustering_model.predict(player_features)[0]
        
        # Get cluster label (would need to load from metadata)
        # For now, use simple heuristics
        player_row = df[df['id'] == player_id].iloc[0]
        cost = player_row['cost']
        points = player_row.get('points', 0)
        
        if cost >= 10 and points >= 100:
            label = "Premiums"
        elif cost <= 7 and points >= 100:
            label = "Budget Gems"
        elif points < 50:
            label = "Avoid"
        else:
            label = "Mid-range"
        
        return int(cluster_id), label
    
    def optimize_squad(
        self,
        budget: float = 100.0,
        constraints: Optional[Dict[str, int]] = None,
        max_per_club: int = 3,
        use_hill_climbing: bool = True,
        model_name: str = "ridge"
    ) -> Dict[str, Any]:
        """
        Optimize squad selection using CSP solver.
        
        Args:
            budget: Budget in millions
            constraints: Position constraints
            max_per_club: Maximum players per club
            use_hill_climbing: Use hill climbing optimization
            model_name: Model to use for predictions
            
        Returns:
            Dictionary with squad solution details
        """
        self.ensure_models_trained()
        
        df = self.load_or_generate_data()
        
        # Get predictions for all players
        # Ensure keys/values are standard Python types (not NumPy scalars) to avoid
        # issues when these dicts are later serialized or used as Pydantic fields.
        predictions_dict = {}
        batch_results = self.predict_batch_players(df['id'].tolist(), model_name=model_name)
        for player_id, (pred_points, _) in batch_results.items():
            predictions_dict[int(player_id)] = float(pred_points)
        
        # Set up constraints
        default_positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        allowed_positions = set(default_positions.keys())

        # Normalize/validate incoming constraints from API (Swagger can send placeholder keys
        # like additionalProp1/additionalProp2/etc.). If we don't see any valid position
        # keys, fall back to defaults. Otherwise, merge user-provided values with defaults.
        if not constraints or not (set(constraints.keys()) & allowed_positions):
            constraints = default_positions
        else:
            normalized_constraints = default_positions.copy()
            for key, value in constraints.items():
                if key in allowed_positions:
                    normalized_constraints[key] = int(value)
            constraints = normalized_constraints

        squad_constraints = SquadConstraints(
            total_players=15,
            budget=budget,
            positions=constraints,
            max_per_club=max_per_club
        )
        
        # Optimize
        optimizer = CSPOptimizer(squad_constraints)
        solution = optimizer.optimize_squad(df, predictions_dict, use_hill_climbing)
        
        # Get squad details
        squad_df = optimizer.get_squad_details(solution, df)
        
        # Add predicted points to squad
        squad_df['predicted_points'] = squad_df['id'].map(predictions_dict)
        
        return {
            'squad': squad_df.to_dict('records'),
            'total_cost': solution.total_cost,
            'total_predicted_points': solution.total_points,
            'position_distribution': solution.position_distribution,
            'club_distribution': solution.club_distribution,
            'is_valid': solution.is_valid,
            'violations': solution.violations,
            'optimization_method': 'hill_climbing' if use_hill_climbing else 'greedy'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'regression_models': list(self.regression_models.keys()),
            'clustering_model_loaded': self.clustering_model is not None,
            'feature_engineer_loaded': self.feature_engineer is not None
        }
        
        return info


# Global ML service instance
ml_service = MLService()

