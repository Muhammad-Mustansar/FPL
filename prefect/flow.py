"""
Prefect Workflow Orchestration for FPL Manager Agent

Implements a complete ML pipeline workflow with:
- Data ingestion
- Data validation
- Feature engineering
- Model training (regression & clustering)
- Model evaluation
- Model saving & versioning
- Retries and failure handling
"""

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.task_runners import ConcurrentTaskRunner
from datetime import timedelta
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_ingestion import DataIngestion
from ml.feature_engineering import FeatureEngineering
from ml.train_regression import RegressionTrainer
from ml.train_clustering import ClusteringTrainer


@task(
    name="ingest_data",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24)
)
def ingest_data_task(data_dir: str = "data", n_players: int = 500) -> pd.DataFrame:
    """
    Task: Ingest player data from CSV or generate mock data.
    
    Args:
        data_dir: Directory for data files
        n_players: Number of players to generate (if using mock data)
        
    Returns:
        DataFrame with player data
    """
    logger = get_run_logger()
    logger.info(f"Starting data ingestion (n_players={n_players})...")
    
    try:
        ingestion = DataIngestion(data_dir)
        
        # Try to load from CSV first
        data_file = Path(data_dir) / "fpl_players.csv"
        if data_file.exists():
            logger.info(f"Loading data from {data_file}")
            df = ingestion.load_from_csv(str(data_file))
        else:
            logger.info("No existing data file found. Generating mock data...")
            df = ingestion.generate_mock_data(n_players=n_players)
            ingestion.save_data(df, "fpl_players.csv")
        
        logger.info(f"Data ingestion completed: {len(df)} players")
        return df
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise


@task(
    name="validate_data",
    retries=2,
    retry_delay_seconds=10
)
def validate_data_task(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Task: Validate ingested data for quality and completeness.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    logger = get_run_logger()
    logger.info("Starting data validation...")
    
    try:
        ingestion = DataIngestion()
        validation_results = ingestion.validate_data(df)
        
        if not validation_results['is_valid']:
            logger.warning(f"Data validation found errors: {validation_results['errors']}")
        else:
            logger.info("Data validation passed")
        
        logger.info(f"Validation stats: {validation_results['stats']}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise


@task(
    name="clean_data",
    retries=2,
    retry_delay_seconds=10
)
def clean_data_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task: Clean and preprocess data.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger = get_run_logger()
    logger.info("Starting data cleaning...")
    
    try:
        ingestion = DataIngestion()
        df_clean = ingestion.clean_data(df)
        
        logger.info(f"Data cleaning completed: {len(df_clean)} records")
        return df_clean
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {str(e)}")
        raise


@task(
    name="train_regression_models",
    retries=2,
    retry_delay_seconds=60,
    timeout_seconds=3600
)
def train_regression_task(
    df: pd.DataFrame,
    model_dir: str = "models",
    target_column: str = "points"
) -> Dict[str, Any]:
    """
    Task: Train regression models (Linear and Ridge).
    
    Args:
        df: Cleaned DataFrame
        model_dir: Directory to save models
        target_column: Target column name
        
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    logger.info("Starting regression model training...")
    
    try:
        trainer = RegressionTrainer(model_dir)
        results = trainer.train_all_models(
            df,
            target_column=target_column,
            test_size=0.2,
            scale_features=True,
            select_features=True,
            n_features=15
        )
        
        # Save models
        saved_paths = trainer.save_all_models()
        
        logger.info(f"Regression training completed. Best model: {results['comparison']['best_model']}")
        logger.info(f"Models saved to: {saved_paths}")
        
        return {
            'metrics': results['metrics'],
            'best_model': results['comparison']['best_model'],
            'saved_paths': saved_paths
        }
        
    except Exception as e:
        logger.error(f"Regression training failed: {str(e)}")
        raise


@task(
    name="train_clustering_model",
    retries=2,
    retry_delay_seconds=60,
    timeout_seconds=1800
)
def train_clustering_task(
    df: pd.DataFrame,
    model_dir: str = "models",
    n_clusters: int = 3
) -> Dict[str, Any]:
    """
    Task: Train K-Means clustering model.
    
    Args:
        df: Cleaned DataFrame
        model_dir: Directory to save models
        n_clusters: Number of clusters
        
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    logger.info(f"Starting clustering model training (n_clusters={n_clusters})...")
    
    try:
        trainer = ClusteringTrainer(model_dir)
        results = trainer.train_all_models(
            df,
            n_clusters=n_clusters,
            find_optimal=False,
            scale_features=True
        )
        
        # Save model
        model_path = trainer.save_model(
            results['model'],
            results['metrics'],
            results['cluster_names']
        )
        
        logger.info(f"Clustering training completed. Silhouette score: {results['metrics']['silhouette_score']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        
        return {
            'metrics': results['metrics'],
            'cluster_names': results['cluster_names'],
            'model_path': model_path
        }
        
    except Exception as e:
        logger.error(f"Clustering training failed: {str(e)}")
        raise


@task(
    name="evaluate_models",
    retries=1,
    retry_delay_seconds=30
)
def evaluate_models_task(
    regression_results: Dict[str, Any],
    clustering_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Task: Evaluate and compare model performance.
    
    Args:
        regression_results: Results from regression training
        clustering_results: Results from clustering training
        
    Returns:
        Dictionary with evaluation summary
    """
    logger = get_run_logger()
    logger.info("Starting model evaluation...")
    
    try:
        evaluation = {
            'regression': {
                'best_model': regression_results['best_model'],
                'metrics': regression_results['metrics']
            },
            'clustering': {
                'silhouette_score': clustering_results['metrics']['silhouette_score'],
                'cluster_names': clustering_results['cluster_names']
            },
            'status': 'success'
        }
        
        logger.info("Model evaluation completed")
        logger.info(f"Best regression model: {evaluation['regression']['best_model']}")
        logger.info(f"Clustering silhouette score: {evaluation['clustering']['silhouette_score']:.4f}")
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise


@task(
    name="send_notification",
    retries=1,
    retry_delay_seconds=10
)
def send_notification_task(
    message: str,
    status: str = "success"
) -> bool:
    """
    Task: Send notification (placeholder for Slack/Discord integration).
    
    Args:
        message: Notification message
        status: Status (success, failure, warning)
        
    Returns:
        True if notification sent successfully
    """
    logger = get_run_logger()
    
    # Placeholder for actual notification service
    # In production, integrate with Slack/Discord webhook
    
    logger.info(f"[NOTIFICATION] Status: {status}, Message: {message}")
    
    # Simulate notification
    if status == "failure":
        logger.error(f"⚠️ Pipeline failed: {message}")
    elif status == "success":
        logger.info(f"✅ Pipeline completed: {message}")
    else:
        logger.warning(f"⚠️ Pipeline warning: {message}")
    
    return True


@flow(
    name="fpl_ml_pipeline",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def ml_pipeline_flow(
    data_dir: str = "data",
    model_dir: str = "models",
    n_players: int = 500,
    n_clusters: int = 3,
    send_notifications: bool = True
) -> Dict[str, Any]:
    """
    Main Prefect Flow for FPL ML Pipeline.
    
    Orchestrates the complete ML workflow:
    1. Data ingestion
    2. Data validation
    3. Data cleaning
    4. Feature engineering (implicit in training)
    5. Model training (regression & clustering)
    6. Model evaluation
    7. Model saving & versioning
    8. Notifications
    
    Args:
        data_dir: Directory for data files
        model_dir: Directory for model files
        n_players: Number of players to generate (if using mock data)
        n_clusters: Number of clusters for K-Means
        send_notifications: Whether to send notifications
        
    Returns:
        Dictionary with pipeline results
    """
    logger = get_run_logger()
    logger.info("="*60)
    logger.info("Starting FPL ML Pipeline")
    logger.info("="*60)
    
    try:
        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        df_raw = ingest_data_task(data_dir, n_players)
        
        # Step 2: Data Validation
        logger.info("Step 2: Data Validation")
        validation_results = validate_data_task(df_raw)
        
        if not validation_results['is_valid']:
            error_msg = f"Data validation failed: {validation_results['errors']}"
            logger.error(error_msg)
            if send_notifications:
                send_notification_task(error_msg, status="failure")
            raise ValueError(error_msg)
        
        # Step 3: Data Cleaning
        logger.info("Step 3: Data Cleaning")
        df_clean = clean_data_task(df_raw)
        
        # Step 4 & 5: Model Training (run in parallel)
        logger.info("Step 4-5: Model Training (Regression & Clustering)")
        regression_results = train_regression_task(df_clean, model_dir, target_column="points")
        clustering_results = train_clustering_task(df_clean, model_dir, n_clusters=n_clusters)
        
        # Step 6: Model Evaluation
        logger.info("Step 6: Model Evaluation")
        evaluation = evaluate_models_task(regression_results, clustering_results)
        
        # Step 7: Success Notification
        if send_notifications:
            success_msg = (
                f"Pipeline completed successfully. "
                f"Best regression model: {evaluation['regression']['best_model']}, "
                f"Clustering silhouette: {evaluation['clustering']['silhouette_score']:.4f}"
            )
            send_notification_task(success_msg, status="success")
        
        logger.info("="*60)
        logger.info("FPL ML Pipeline Completed Successfully")
        logger.info("="*60)
        
        return {
            'status': 'success',
            'data_stats': validation_results['stats'],
            'regression_results': regression_results,
            'clustering_results': clustering_results,
            'evaluation': evaluation
        }
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg)
        
        if send_notifications:
            send_notification_task(error_msg, status="failure")
        
        raise


@flow(
    name="fpl_retrain_regression",
    log_prints=True
)
def retrain_regression_flow(
    data_dir: str = "data",
    model_dir: str = "models"
) -> Dict[str, Any]:
    """
    Flow to retrain only regression models (faster than full pipeline).
    
    Args:
        data_dir: Directory for data files
        model_dir: Directory for model files
        
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    logger.info("Starting regression model retraining...")
    
    try:
        # Load existing data
        ingestion = DataIngestion(data_dir)
        data_file = Path(data_dir) / "fpl_players.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = ingestion.load_from_csv(str(data_file))
        df = ingestion.clean_data(df)
        
        # Train regression models
        results = train_regression_task(df, model_dir, target_column="points")
        
        logger.info("Regression retraining completed")
        return results
        
    except Exception as e:
        logger.error(f"Regression retraining failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the pipeline
    result = ml_pipeline_flow(
        data_dir="data",
        model_dir="models",
        n_players=500,
        n_clusters=3,
        send_notifications=True
    )
    print("\nPipeline Results:")
    print(result)

