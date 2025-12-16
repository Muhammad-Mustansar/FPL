"""
Unit Tests for ML Models

Tests regression, clustering, and CSP optimization models.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_ingestion import DataIngestion
from ml.feature_engineering import FeatureEngineering
from ml.train_regression import RegressionTrainer
from ml.train_clustering import ClusteringTrainer
from ml.csp_optimizer import CSPOptimizer, SquadConstraints


class TestRegressionModels:
    """Test cases for regression models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for regression testing."""
        ingestion = DataIngestion()
        df = ingestion.generate_mock_data(n_players=200)
        return ingestion.clean_data(df)
    
    @pytest.fixture
    def trainer(self, tmp_path):
        """Create RegressionTrainer instance."""
        return RegressionTrainer(model_dir=str(tmp_path))
    
    def test_linear_regression_training(self, trainer, sample_data):
        """Test Linear Regression model training."""
        X, y = trainer.feature_engineer.prepare_regression_features(sample_data, target_column='points')
        X_scaled = trainer.feature_engineer.scale_features(X, fit=True)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model, metrics = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert 'train_r2' in metrics
        assert 'val_r2' in metrics
        assert metrics['train_r2'] > 0
        assert metrics['val_r2'] > 0
    
    def test_ridge_regression_training(self, trainer, sample_data):
        """Test Ridge Regression model training."""
        X, y = trainer.feature_engineer.prepare_regression_features(sample_data, target_column='points')
        X_scaled = trainer.feature_engineer.scale_features(X, fit=True)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model, metrics = trainer.train_ridge_regression(X_train, y_train, X_test, y_test, alpha=1.0)
        
        assert model is not None
        assert 'alpha' in metrics
        assert metrics['alpha'] == 1.0
        assert metrics['train_r2'] > 0
    
    def test_train_all_models(self, trainer, sample_data):
        """Test training all regression models."""
        results = trainer.train_all_models(
            sample_data,
            target_column='points',
            test_size=0.2,
            scale_features=True,
            select_features=True,
            n_features=10
        )
        
        assert 'models' in results
        assert 'linear' in results['models']
        assert 'ridge' in results['models']
        assert 'comparison' in results
        assert 'best_model' in results['comparison']
    
    def test_model_saving(self, trainer, sample_data, tmp_path):
        """Test saving trained models."""
        results = trainer.train_all_models(sample_data, target_column='points', test_size=0.2)
        
        saved_paths = trainer.save_all_models()
        
        assert 'linear' in saved_paths
        assert 'ridge' in saved_paths
        assert Path(saved_paths['linear']).exists()
        assert Path(saved_paths['ridge']).exists()
    
    def test_model_loading(self, trainer, sample_data, tmp_path):
        """Test loading saved models."""
        # Train and save
        trainer.train_all_models(sample_data, target_column='points', test_size=0.2)
        saved_paths = trainer.save_all_models()
        
        # Load
        loaded_model = trainer.load_model(saved_paths['linear'])
        
        assert loaded_model is not None


class TestClusteringModels:
    """Test cases for clustering models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for clustering testing."""
        ingestion = DataIngestion()
        df = ingestion.generate_mock_data(n_players=200)
        return ingestion.clean_data(df)
    
    @pytest.fixture
    def trainer(self, tmp_path):
        """Create ClusteringTrainer instance."""
        return ClusteringTrainer(model_dir=str(tmp_path))
    
    def test_kmeans_training(self, trainer, sample_data):
        """Test K-Means clustering training."""
        X = trainer.feature_engineer.prepare_clustering_features(sample_data)
        
        model, metrics, labels = trainer.train_kmeans(X, n_clusters=3, scale_features=True)
        
        assert model is not None
        assert 'silhouette_score' in metrics
        assert len(labels) == len(sample_data)
        assert len(np.unique(labels)) == 3
    
    def test_find_optimal_clusters(self, trainer, sample_data):
        """Test finding optimal number of clusters."""
        X = trainer.feature_engineer.prepare_clustering_features(sample_data)
        
        result = trainer.find_optimal_clusters(X, max_clusters=5, scale_features=True)
        
        assert 'optimal_n_clusters' in result
        assert 2 <= result['optimal_n_clusters'] <= 5
        assert 'evaluation_results' in result
    
    def test_label_clusters(self, trainer, sample_data):
        """Test cluster labeling."""
        X = trainer.feature_engineer.prepare_clustering_features(sample_data)
        model, metrics, labels = trainer.train_kmeans(X, n_clusters=3, scale_features=True)
        
        cluster_names = trainer.label_clusters(sample_data, labels)
        
        assert len(cluster_names) == 3
        assert all(isinstance(name, str) for name in cluster_names.values())
        assert any('Premium' in name or 'Budget' in name or 'Avoid' in name 
                  for name in cluster_names.values())


class TestCSPOptimizer:
    """Test cases for CSP squad optimizer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for CSP testing."""
        ingestion = DataIngestion()
        df = ingestion.generate_mock_data(n_players=300)
        return ingestion.clean_data(df)
    
    @pytest.fixture
    def predicted_points(self, sample_data):
        """Create mock predicted points."""
        # Simple prediction: points = cost * 10 + noise
        np.random.seed(42)
        predictions = {}
        for _, player in sample_data.iterrows():
            predictions[player['id']] = player['cost'] * 10 + np.random.normal(0, 5)
        return predictions
    
    @pytest.fixture
    def optimizer(self):
        """Create CSPOptimizer instance."""
        return CSPOptimizer()
    
    def test_validate_solution_valid(self, optimizer, sample_data, predicted_points):
        """Test validation of a valid squad solution."""
        # Create a valid squad manually
        df_sorted = sample_data.sort_values('cost')
        valid_squad = []
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        club_counts = {}
        budget = 100.0
        
        for _, player in df_sorted.iterrows():
            if len(valid_squad) >= 15:
                break
            
            pos = player['position']
            club = player['team']
            cost = player['cost']
            
            if position_counts[pos] < {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}[pos]:
                if club_counts.get(club, 0) < 3:
                    if budget >= cost:
                        valid_squad.append(player['id'])
                        position_counts[pos] += 1
                        club_counts[club] = club_counts.get(club, 0) + 1
                        budget -= cost
        
        is_valid, violations = optimizer.validate_solution(valid_squad, sample_data, predicted_points)
        
        assert is_valid is True
        assert len(violations) == 0
    
    def test_validate_solution_invalid_budget(self, optimizer, sample_data):
        """Test validation catches budget violations."""
        # Create squad that exceeds budget
        expensive_players = sample_data.nlargest(15, 'cost')
        invalid_squad = expensive_players['id'].tolist()
        
        is_valid, violations = optimizer.validate_solution(invalid_squad, sample_data)
        
        assert is_valid is False
        assert any('budget' in v.lower() or 'cost' in v.lower() for v in violations)
    
    def test_greedy_search(self, optimizer, sample_data, predicted_points):
        """Test greedy search algorithm."""
        solution = optimizer.greedy_search(sample_data, predicted_points)
        
        assert solution is not None
        assert len(solution.player_ids) == 15
        assert solution.total_cost <= 100.0
    
    def test_optimize_squad(self, optimizer, sample_data, predicted_points):
        """Test full squad optimization."""
        solution = optimizer.optimize_squad(sample_data, predicted_points, use_hill_climbing=False)
        
        assert solution is not None
        assert len(solution.player_ids) == 15
        assert solution.total_cost <= 100.0
        assert solution.position_distribution['GK'] == 2
        assert solution.position_distribution['DEF'] == 5
        assert solution.position_distribution['MID'] == 5
        assert solution.position_distribution['FWD'] == 3
    
    def test_custom_constraints(self, sample_data, predicted_points):
        """Test optimizer with custom constraints."""
        constraints = SquadConstraints(
            total_players=15,
            budget=90.0,
            positions={'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3},
            max_per_club=2
        )
        
        optimizer = CSPOptimizer(constraints)
        solution = optimizer.optimize_squad(sample_data, predicted_points, use_hill_climbing=False)
        
        assert solution.total_cost <= 90.0
        assert all(count <= 2 for count in solution.club_distribution.values())

