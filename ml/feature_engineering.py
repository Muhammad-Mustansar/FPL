"""
Feature Engineering Module for FPL Manager Agent

This module handles feature creation, transformation, and selection
for machine learning models (regression and clustering).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression


class FeatureEngineering:
    """
    Handles feature engineering for FPL player data.
    Creates derived features, handles encoding, and prepares data for ML models.
    """
    
    def __init__(self):
        """Initialize the feature engineering module."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.selected_features = None
        logger.info("Feature engineering module initialized")
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw player data.
        """
        df_features = df.copy()
        
        # Avoid division by zero
        minutes = df_features['minutes'].replace(0, 1)
        
        # 1. Efficiency Metrics (Per 90 stats)
        if 'goals' in df_features.columns:
            df_features['goals_per_90'] = (df_features['goals'] / minutes) * 90
            
        if 'assists' in df_features.columns:
            df_features['assists_per_90'] = (df_features['assists'] / minutes) * 90
            
        if 'clean_sheets' in df_features.columns:
            df_features['clean_sheets_per_90'] = (df_features['clean_sheets'] / minutes) * 90
            
        if 'points' in df_features.columns:
            df_features['points_per_90'] = (df_features['points'] / minutes) * 90

        # 2. Value Metrics
        if 'cost' in df_features.columns and 'points' in df_features.columns:
            df_features['value_season'] = df_features['points'] / df_features['cost'].replace(0, 1)

        # 3. Form & Availability
        if 'form' in df_features.columns:
            # THIS CREATES A CATEGORICAL COLUMN
            df_features['form_category'] = pd.cut(
                df_features['form'],
                bins=[-1, 2, 5, 100],
                labels=['Poor', 'Average', 'Hot']
            )

        # 4. Position Ratios (Interaction Features)
        if 'position' in df_features.columns:
            # GKs
            mask_gk = df_features['position'] == 'GK'
            if 'saves' in df_features.columns:
                df_features.loc[mask_gk, 'gk_save_ratio'] = df_features.loc[mask_gk, 'saves'] / minutes.loc[mask_gk] * 90

            # DEFs
            mask_def = df_features['position'] == 'DEF'
            if 'clean_sheets' in df_features.columns:
                df_features.loc[mask_def, 'def_cs_potential'] = df_features.loc[mask_def, 'clean_sheets'] / minutes.loc[mask_def] * 90

            # MIDs/FWDs
            mask_att = df_features['position'].isin(['MID', 'FWD'])
            if 'goals' in df_features.columns and 'assists' in df_features.columns:
                df_features.loc[mask_att, 'attack_involvement'] = (df_features.loc[mask_att, 'goals'] + df_features.loc[mask_att, 'assists']) / minutes.loc[mask_att] * 90

        # === FIX STARTS HERE ===
        # Do NOT use df_features.fillna(0) blindly, as it breaks Categorical columns (like form_category)
        # Instead, we only fill numeric columns with 0.
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
        # === FIX ENDS HERE ===
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} derived features")
        return df_features
    
    def encode_categorical_features(
        self, 
        df: pd.DataFrame, 
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        """
        df_encoded = df.copy()
        
        if categorical_columns is None:
            # Auto-detect categorical columns
            categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
            # Exclude metadata columns
            categorical_columns = [col for col in categorical_columns if col not in ['name', 'id']]
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Handle unknown categories/NaNs by converting to string first
                    # This safely turns NaNs into "nan" which is then encoded
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # For inference, use existing encoder
                    df_encoded[col] = df_encoded[col].astype(str).map(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        return df_encoded
    
    def prepare_regression_features(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'points',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for regression models.
        """
        # 1. Create Features
        df_features = self.create_derived_features(df)
        
        # 2. Encode
        df_encoded = self.encode_categorical_features(df_features)
        
        # 3. Define Data Leakage Columns
        leaky_cols = [
            'goals', 'assists', 'clean_sheets', 'saves', 'bonus', 
            'yellow_cards', 'red_cards', 'penalties_missed', 'goals_conceded'
        ]
        
        # 4. Select Features
        if feature_columns is None:
            exclude_cols = [target_column, 'id', 'name', 'team'] + leaky_cols
            
            feature_columns = [
                col for col in df_encoded.columns 
                if col not in exclude_cols 
                and pd.api.types.is_numeric_dtype(df_encoded[col])
            ]
        
        available_features = [col for col in feature_columns if col in df_encoded.columns]
        
        X = df_encoded[available_features].copy()
        y = df_encoded[target_column].copy() if target_column in df_encoded.columns else None
        
        # Handle NaNs (Safe here because X is purely numeric after encoding)
        X = X.fillna(0)
        if y is not None:
            y = y.fillna(0)
        
        return X, y
    
    def prepare_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering."""
        df_features = self.create_derived_features(df)
        df_encoded = self.encode_categorical_features(df_features)
        
        clustering_cols = [
            'cost', 'points_per_90', 'goals_per_90', 'assists_per_90', 
            'clean_sheets_per_90', 'form', 'selected_by_percent', 'minutes'
        ]
        
        available = [c for c in clustering_cols if c in df_encoded.columns]
        X = df_encoded[available].fillna(0)
        return X
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if X.empty: return X
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20, fit: bool = True) -> pd.DataFrame:
        if fit:
            k = min(k, X.shape[1])
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        else:
            if not self.selected_features: return X
            for col in self.selected_features:
                if col not in X.columns: X[col] = 0
            X_selected = X[self.selected_features].values
            
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)