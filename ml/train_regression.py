"""
Regression Model Training Module for FPL Manager Agent

This module trains and evaluates regression models to predict Expected Points (xP)
for FPL players. Includes Linear Regression (baseline) and Ridge Regression.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from loguru import logger
import joblib
import json
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ml.feature_engineering import FeatureEngineering


class RegressionTrainer:
    """
    Trains and evaluates regression models for predicting player Expected Points (xP).
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.feature_engineer = FeatureEngineering()
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.experiments = []

        # 🔐 Persisted metadata
        self.final_feature_columns: Optional[list] = None

        logger.info(f"Regression trainer initialized with model_dir: {model_dir}")

    # ------------------------------------------------------------------
    # MODEL TRAINING METHODS
    # ------------------------------------------------------------------

    def train_linear_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Tuple[LinearRegression, Dict[str, float]]:

        logger.info("Training Linear Regression model...")

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)

        metrics = {
            "model_name": "LinearRegression",
            "train_mse": float(train_mse),
            "train_rmse": float(train_rmse),
            "train_r2": float(train_r2),
            "train_mae": float(train_mae),
        }

        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            metrics.update({
                "val_mse": float(mean_squared_error(y_val, y_val_pred)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
                "val_r2": float(r2_score(y_val, y_val_pred)),
                "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
            })

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        metrics["cv_r2_mean"] = float(cv_scores.mean())
        metrics["cv_r2_std"] = float(cv_scores.std())

        return model, metrics

    def train_ridge_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        alpha: float = 1.0
    ) -> Tuple[Ridge, Dict[str, float]]:

        logger.info(f"Training Ridge Regression (alpha={alpha})")

        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)

        metrics = {
            "model_name": "RidgeRegression",
            "alpha": float(alpha),
            "train_mse": float(mean_squared_error(y_train, y_train_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        }

        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            metrics.update({
                "val_mse": float(mean_squared_error(y_val, y_val_pred)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
                "val_r2": float(r2_score(y_val, y_val_pred)),
                "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
            })

        return model, metrics

    # ------------------------------------------------------------------
    # FULL TRAINING PIPELINE
    # ------------------------------------------------------------------

    def train_all_models(
        self,
        df: pd.DataFrame,
        target_column: str = "points",
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
        select_features: bool = True,
        n_features: int = 20
    ) -> Dict[str, Any]:

        logger.info("Starting regression training pipeline")

        X, y = self.feature_engineer.prepare_regression_features(df, target_column)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if scale_features:
            X_train = self.feature_engineer.scale_features(X_train, fit=True)
            X_test = self.feature_engineer.scale_features(X_test, fit=False)

        if select_features:
            X_train = self.feature_engineer.select_features(X_train, y_train, k=n_features, fit=True)
            X_test = self.feature_engineer.select_features(X_test, y_test, k=n_features, fit=False)

        # 🔐 SAVE FINAL FEATURE LIST
        self.final_feature_columns = list(X_train.columns)

        linear_model, linear_metrics = self.train_linear_regression(
            X_train, y_train, X_test, y_test
        )

        ridge_model, ridge_metrics = self.train_ridge_regression(
            X_train, y_train, X_test, y_test, alpha=1.0
        )

        self.models["linear"] = linear_model
        self.models["ridge"] = ridge_model
        self.metrics["linear"] = linear_metrics
        self.metrics["ridge"] = ridge_metrics

        return {
            "models": self.models,
            "metrics": self.metrics,
            "feature_columns": self.final_feature_columns,
            "feature_engineer": self.feature_engineer,
        }

    # ------------------------------------------------------------------
    # PERSISTENCE (🔥 CRITICAL FIX)
    # ------------------------------------------------------------------

    def save_all_models(self) -> Dict[str, str]:
        saved = {}

        # Save models
        for name, model in self.models.items():
            path = self.model_dir / f"{name}_model.joblib"
            joblib.dump(model, path)
            saved[name] = str(path)

        # 🔐 Save feature pipeline
        joblib.dump(self.feature_engineer, self.model_dir / "feature_engineer.joblib")
        joblib.dump(self.final_feature_columns, self.model_dir / "feature_columns.joblib")

        logger.info("Saved models, feature engineer, and feature columns")

        return saved


# ------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ------------------------------------------------------------------

if __name__ == "__main__":
    from ml.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    df = ingestion.generate_mock_data(n_players=500)
    df = ingestion.clean_data(df)

    trainer = RegressionTrainer()
    trainer.train_all_models(
        df,
        target_column="points",
        scale_features=True,
        select_features=True,
        n_features=15,
    )

    trainer.save_all_models()
