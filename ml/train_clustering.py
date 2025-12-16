"""
Clustering Model Training Module for FPL Manager Agent

This module trains K-Means clustering models to segment players into groups:
- Premiums (high cost, high points)
- Budget Gems (low cost, high points)
- Avoid (low performance)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import joblib
import json
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from ml.feature_engineering import FeatureEngineering


class ClusteringTrainer:
    """
    Trains and evaluates K-Means clustering models for player segmentation.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the clustering trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.feature_engineer = FeatureEngineering()
        self.models = {}
        self.metrics = {}
        self.cluster_labels = {}
        self.experiments = []
        
        logger.info(f"Clustering trainer initialized with model_dir: {model_dir}")
    
    def train_kmeans(
        self,
        X: pd.DataFrame,
        n_clusters: int = 3,
        random_state: int = 42,
        scale_features: bool = True
    ) -> Tuple[KMeans, Dict[str, float], np.ndarray]:
        """
        Train a K-Means clustering model.
        
        Args:
            X: Feature DataFrame
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
            scale_features: Whether to scale features before clustering
            
        Returns:
            Tuple of (trained_model, metrics_dict, cluster_labels)
        """
        logger.info(f"Training K-Means with {n_clusters} clusters...")
        
        X_processed = X.copy()
        
        # Scale features if requested
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_processed = pd.DataFrame(
                scaler.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
            logger.info("Features scaled for clustering")
        
        # Train K-Means
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = model.fit_predict(X_processed)
        
        # Calculate metrics
        silhouette = silhouette_score(X_processed, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_processed, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_processed, cluster_labels)
        
        # Cluster sizes
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        
        metrics = {
            'model_name': 'KMeans',
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'calinski_harabasz_score': float(calinski_harabasz),
            'cluster_sizes': cluster_sizes,
            'scale_features': scale_features
        }
        
        logger.info(f"K-Means - Silhouette Score: {silhouette:.4f}")
        logger.info(f"Cluster sizes: {cluster_sizes}")
        
        # Store scaler with model for inference
        if scaler is not None:
            model.scaler = scaler
        
        return model, metrics, cluster_labels
    
    def find_optimal_clusters(
        self,
        X: pd.DataFrame,
        max_clusters: int = 10,
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            X: Feature DataFrame
            max_clusters: Maximum number of clusters to test
            scale_features: Whether to scale features
            
        Returns:
            Dictionary with optimal cluster count and evaluation results
        """
        logger.info(f"Finding optimal number of clusters (testing 2 to {max_clusters})...")
        
        X_processed = X.copy()
        if scale_features:
            scaler = StandardScaler()
            X_processed = pd.DataFrame(
                scaler.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        results = {
            'n_clusters': [],
            'silhouette_scores': [],
            'davies_bouldin_scores': [],
            'inertias': []
        }
        
        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_processed)
            
            silhouette = silhouette_score(X_processed, labels)
            davies_bouldin = davies_bouldin_score(X_processed, labels)
            
            results['n_clusters'].append(n)
            results['silhouette_scores'].append(float(silhouette))
            results['davies_bouldin_scores'].append(float(davies_bouldin))
            results['inertias'].append(float(kmeans.inertia_))
        
        # Find optimal cluster count (highest silhouette score)
        best_idx = np.argmax(results['silhouette_scores'])
        optimal_n = results['n_clusters'][best_idx]
        
        logger.info(f"Optimal number of clusters: {optimal_n} (Silhouette: {results['silhouette_scores'][best_idx]:.4f})")
        
        return {
            'optimal_n_clusters': optimal_n,
            'evaluation_results': results
        }
    
    def label_clusters(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        cost_column: str = 'cost',
        points_column: str = 'points'
    ) -> Dict[int, str]:
        """
        Label clusters based on player characteristics.
        
        Labels:
        - Premiums: High cost, high points
        - Budget Gems: Low cost, high points
        - Avoid: Low performance
        
        Args:
            df: DataFrame with player data
            cluster_labels: Cluster assignments
            cost_column: Name of cost column
            points_column: Name of points column
            
        Returns:
            Dictionary mapping cluster ID to label
        """
        cluster_labels_dict = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df[cluster_mask]
            
            avg_cost = cluster_data[cost_column].mean() if cost_column in cluster_data.columns else 0
            avg_points = cluster_data[points_column].mean() if points_column in cluster_data.columns else 0
            
            # Define thresholds
            high_cost_threshold = df[cost_column].quantile(0.7)
            high_points_threshold = df[points_column].quantile(0.7)
            low_cost_threshold = df[cost_column].quantile(0.3)
            low_points_threshold = df[points_column].quantile(0.3)
            
            # Label clusters
            if avg_cost >= high_cost_threshold and avg_points >= high_points_threshold:
                label = "Premiums"
            elif avg_cost <= low_cost_threshold and avg_points >= high_points_threshold:
                label = "Budget Gems"
            elif avg_points <= low_points_threshold:
                label = "Avoid"
            elif avg_cost >= high_cost_threshold and avg_points < high_points_threshold:
                label = "Overpriced"
            else:
                label = "Mid-range"
            
            cluster_labels_dict[int(cluster_id)] = label
            logger.info(f"Cluster {cluster_id}: {label} (Avg Cost: {avg_cost:.2f}, Avg Points: {avg_points:.2f})")
        
        return cluster_labels_dict
    
    def train_all_models(
        self,
        df: pd.DataFrame,
        n_clusters: Optional[int] = None,
        find_optimal: bool = True,
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Train clustering models and label clusters.
        
        Args:
            df: Input DataFrame with player data
            n_clusters: Number of clusters (if None, will find optimal)
            find_optimal: Whether to find optimal number of clusters
            scale_features: Whether to scale features
            
        Returns:
            Dictionary with trained models, metrics, and cluster labels
        """
        logger.info("Starting clustering model training pipeline...")
        
        # Prepare features
        X = self.feature_engineer.prepare_clustering_features(df)
        
        logger.info(f"Prepared {len(X.columns)} features for clustering")
        
        # Find optimal number of clusters if requested
        if find_optimal and n_clusters is None:
            optimal_result = self.find_optimal_clusters(X, max_clusters=8, scale_features=scale_features)
            n_clusters = optimal_result['optimal_n_clusters']
            self.experiments.append({
                'timestamp': datetime.now().isoformat(),
                'optimal_clusters_analysis': optimal_result
            })
        
        if n_clusters is None:
            n_clusters = 3  # Default
        
        # Train K-Means
        model, metrics, cluster_labels = self.train_kmeans(
            X, n_clusters=n_clusters, scale_features=scale_features
        )
        
        self.models['kmeans'] = model
        self.metrics['kmeans'] = metrics
        self.cluster_labels['kmeans'] = cluster_labels
        
        # Label clusters
        cluster_names = self.label_clusters(df, cluster_labels)
        metrics['cluster_names'] = cluster_names
        
        # Add cluster labels to original dataframe
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        df_with_clusters['cluster_label'] = df_with_clusters['cluster'].map(cluster_names)
        
        # Log experiment
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'n_clusters': n_clusters,
            'scale_features': scale_features,
            'metrics': metrics,
            'feature_columns': list(X.columns)
        }
        self.experiments.append(experiment)
        
        logger.info("Clustering model training completed!")
        
        return {
            'model': model,
            'metrics': metrics,
            'cluster_labels': cluster_labels,
            'cluster_names': cluster_names,
            'df_with_clusters': df_with_clusters,
            'feature_engineer': self.feature_engineer,
            'experiment': experiment
        }
    
    def save_model(self, model: KMeans, metrics: Dict[str, float], cluster_names: Dict[int, str]) -> str:
        """
        Save a trained clustering model and its metadata.
        
        Args:
            model: Trained KMeans model
            metrics: Model metrics dictionary
            cluster_names: Cluster label mapping
            
        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"kmeans_{timestamp}.pkl"
        metadata_filename = f"kmeans_{timestamp}_metadata.json"
        
        model_path = self.model_dir / model_filename
        metadata_path = self.model_dir / metadata_filename
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save metadata (metrics and cluster names)
        metadata = {
            'metrics': metrics,
            'cluster_names': {str(k): v for k, v in cluster_names.items()}
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str, metadata_path: Optional[str] = None) -> Tuple[KMeans, Dict[str, Any]]:
        """
        Load a saved clustering model and its metadata.
        
        Args:
            model_path: Path to saved model file
            metadata_path: Path to metadata file (optional)
            
        Returns:
            Tuple of (loaded_model, metadata_dict)
        """
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        metadata = {}
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
        
        return model, metadata
    
    def predict_cluster(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster assignments for new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of cluster labels
        """
        if 'kmeans' not in self.models:
            raise ValueError("K-Means model not found. Train a model first.")
        
        model = self.models['kmeans']
        X_processed = X.copy()
        
        # Scale if model was trained with scaling
        if hasattr(model, 'scaler'):
            X_processed = pd.DataFrame(
                model.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        cluster_labels = model.predict(X_processed)
        return cluster_labels


if __name__ == "__main__":
    # Example usage
    from ml.data_ingestion import DataIngestion
    
    # Load and prepare data
    ingestion = DataIngestion()
    df = ingestion.generate_mock_data(n_players=500)
    df = ingestion.clean_data(df)
    
    # Train clustering model
    trainer = ClusteringTrainer()
    results = trainer.train_all_models(
        df,
        n_clusters=3,
        find_optimal=False,
        scale_features=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("CLUSTERING RESULTS")
    print("="*50)
    print(f"Silhouette Score: {results['metrics']['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Score: {results['metrics']['davies_bouldin_score']:.4f}")
    print(f"\nCluster Labels:")
    for cluster_id, label in results['cluster_names'].items():
        count = (results['cluster_labels'] == int(cluster_id)).sum()
        print(f"  Cluster {cluster_id} ({label}): {count} players")
    
    # Save model
    model_path = trainer.save_model(
        results['model'],
        results['metrics'],
        results['cluster_names']
    )
    print(f"\nModel saved to: {model_path}")

