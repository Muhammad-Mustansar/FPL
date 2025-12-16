"""
Data Ingestion Module for FPL Manager Agent

This module handles data collection from FPL-like sources (CSV files or Official FPL API).
It includes data validation and basic cleaning operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger
import requests
import os
from dotenv import load_dotenv

class DataIngestion:
    """
    Handles data ingestion from various sources (CSV, API, etc.)
    for Fantasy Premier League player data.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data ingestion module.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        load_dotenv()
        
        # OFFICIAL FPL API ENDPOINT (No Key Required for this specific endpoint)
        self.base_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        
        logger.info(f"Data ingestion initialized. Using FPL API: {self.base_url}")
    
    def fetch_live_data(self) -> Optional[pd.DataFrame]:
        """
        Fetches live data from the Official FPL API.
        """
        try:
            logger.info(f"Fetching live data from {self.base_url}...")
            # The official FPL API is public and usually does not require headers/keys
            response = requests.get(self.base_url)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Successfully connected to FPL API!")
                return self.process_fpl_api_data(data)
            else:
                logger.error(f"API Error {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to connect to API: {str(e)}")
            return None

    def process_fpl_api_data(self, data: Dict) -> pd.DataFrame:
        """
        Process Official FPL JSON data into our standard format.
        """
        try:
            # 1. Extract raw lists from JSON
            elements = data['elements']      # Players
            teams = data['teams']            # Teams metadata
            element_types = data['element_types'] # Position metadata
            
            # 2. Create Lookup Dictionaries
            # Map Team ID -> Short Name (e.g., 1 -> 'ARS')
            team_map = {t['id']: t['short_name'] for t in teams}
            
            # Map Element Type ID -> Position Name (e.g., 1 -> 'GK', 2 -> 'DEF')
            # FPL usually uses: 1=GKP, 2=DEF, 3=MID, 4=FWD
            # We map them to our standard: GK, DEF, MID, FWD
            pos_map = {
                1: 'GK', 
                2: 'DEF', 
                3: 'MID', 
                4: 'FWD'
            }
            # Fallback if ID lookup fails
            pos_name_map = {t['id']: t['singular_name_short'] for t in element_types}
            
            processed_players = []
            
            for p in elements:
                # Get mapped values
                team_name = team_map.get(p['team'], 'UNK')
                pos_code = pos_map.get(p['element_type'], pos_name_map.get(p['element_type'], 'UNK'))
                
                # FPL costs are stored as integers (e.g., 100 = 10.0m)
                real_cost = p['now_cost'] / 10.0
                
                player_record = {
                    'id': p['id'],
                    'name': f"{p['first_name']} {p['second_name']}",
                    'web_name': p['web_name'],
                    'team': team_name,
                    'position': pos_code,
                    'cost': real_cost,
                    'points': p['total_points'],
                    'points_per_game': float(p['points_per_game']),
                    'goals': p['goals_scored'],
                    'assists': p['assists'],
                    'clean_sheets': p['clean_sheets'],
                    'saves': p['saves'],
                    'bonus': p['bonus'],
                    'yellow_cards': p['yellow_cards'],
                    'red_cards': p['red_cards'],
                    'minutes': p['minutes'],
                    'form': float(p['form']),
                    'selected_by_percent': float(p['selected_by_percent']),
                    'status': p['status'],  # 'a'=available, 'd'=doubtful, 'u'=unavailable, 'i'=injured
                    'chance_of_playing': p['chance_of_playing_next_round']
                }
                processed_players.append(player_record)
            
            df = pd.DataFrame(processed_players)
            
            # Filter: Keep only players who are reasonably available 
            # (status 'a' = available, or doubtful but with chance > 0)
            # You can adjust this filter if you want to include injured players
            df = df[df['status'] != 'u'] 
            
            logger.info(f"Processed {len(df)} players from FPL API")
            return df
            
        except KeyError as e:
            logger.error(f"JSON Structure mismatch - Key not found: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing FPL data: {e}")
            return pd.DataFrame()

    def generate_mock_data(self, n_players: int = 500) -> pd.DataFrame:
        """
        Generate mock FPL player data for testing and development.
        """
        np.random.seed(42)
        positions = ['GK', 'DEF', 'MID', 'FWD']
        clubs = ['ARS', 'AVL', 'CHE', 'LIV', 'MCI', 'MUN', 'TOT', 'WHU', 'NEW', 'BHA']
        
        data = {
            'id': range(1, n_players + 1),
            'name': [f"Player_{i}" for i in range(1, n_players + 1)],
            'position': np.random.choice(positions, n_players, p=[0.1, 0.35, 0.35, 0.2]),
            'team': np.random.choice(clubs, n_players),
            'cost': np.round(np.random.uniform(4.0, 14.0, n_players), 1),
            'points': np.random.poisson(lam=80, size=n_players),
            'points_per_game': np.round(np.random.uniform(2.0, 9.0, n_players), 1),
            'minutes': np.random.randint(0, 3420, n_players),
            'goals': np.random.poisson(lam=5, size=n_players),
            'assists': np.random.poisson(lam=3, size=n_players),
            'clean_sheets': np.random.poisson(lam=8, size=n_players),
            'saves': np.random.poisson(lam=20, size=n_players), # Added for GKs
            'bonus': np.random.poisson(lam=10, size=n_players),
            'form': np.round(np.random.uniform(0.0, 10.0, n_players), 1),
            'selected_by_percent': np.round(np.random.uniform(0.1, 50.0, n_players), 1),
            'yellow_cards': np.random.poisson(lam=3, size=n_players),
            'red_cards': np.random.poisson(lam=0.2, size=n_players),
            'status': 'a'
        }
        
        df = pd.DataFrame(data)
        
        # Logic fix: Ensure GKs have more saves and fewer goals
        mask_gk = df['position'] == 'GK'
        df.loc[mask_gk, 'goals'] = 0
        df.loc[mask_gk, 'assists'] = np.random.choice([0, 1], size=mask_gk.sum(), p=[0.95, 0.05])
        df.loc[mask_gk, 'saves'] = np.random.randint(50, 150, size=mask_gk.sum())
        
        return df

    def normalize_fpl_2023_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the 2023/24 FPL dataset (CSV format) into the expected schema.
        Useful if you load historical CSV files.
        """
        df = df_raw.copy()

        # Map common CSV column names to our schema
        rename_map = {
            "element": "id",
            "total_points": "points",
            "goals_scored": "goals",
            "goals_conceded": "goals_conceded",
            "xP": "form",  # treat expected points as current form
            "value": "value_raw",
        }
        df = df.rename(columns=rename_map)

        # ID fallback if missing
        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)

        # Cost: FPL exports are usually *10; divide by 10 to get £m
        if "cost" not in df.columns:
            if "value_raw" in df.columns:
                df["cost"] = pd.to_numeric(df["value_raw"], errors="coerce") / 10.0
            elif "value" in df.columns:
                df["cost"] = pd.to_numeric(df["value"], errors="coerce") / 10.0
            else:
                df["cost"] = 5.0  # default fallback

        # Points fallback
        if "points" not in df.columns and "total_points" in df.columns:
            df["points"] = df["total_points"]
        df["points"] = pd.to_numeric(df.get("points", 0), errors="coerce").fillna(0)

        # Ensure numeric columns exist
        numeric_cols = ["goals", "assists", "clean_sheets", "saves", "bonus", "minutes", "yellow_cards", "red_cards", "form", "selected_by_percent"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

        # Keep only the expected columns
        cols_order = [
            "id", "name", "position", "team", "cost", "points", "goals", "assists",
            "clean_sheets", "form", "selected_by_percent", "minutes", "yellow_cards",
            "red_cards", "saves", "bonus"
        ]
        
        # Fill missing string cols
        if "name" not in df.columns: df["name"] = "Unknown"
        if "position" not in df.columns: df["position"] = "MID"
        if "team" not in df.columns: df["team"] = "UNK"

        # Ensure final DataFrame has all columns
        for col in cols_order:
            if col not in df.columns:
                df[col] = 0

        df = df[cols_order]
        logger.info(f"Normalized historical dataset to {len(df)} rows")
        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check if dataframe has required columns."""
        required = ['id', 'name', 'position', 'team', 'cost', 'points']
        missing = [col for col in required if col not in df.columns]
        
        is_valid = len(missing) == 0
        stats = {
            'count': len(df),
            'missing_cols': missing,
            'avg_points': df['points'].mean() if 'points' in df.columns else 0
        }
        return {'is_valid': is_valid, 'stats': stats}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: fill NaNs and ensure types."""
        df = df.fillna(0)
        return df

    def save_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save processed data to CSV.
        """
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return str(filepath)

if __name__ == "__main__":
    # Test the module
    ingestion = DataIngestion()
    
    print("Attempting to fetch LIVE FPL data...")
    df = ingestion.fetch_live_data()
    
    if df is not None and not df.empty:
        print(f"Success! Fetched {len(df)} players.")
        print("\nTop 5 Players by Points:")
        print(df.sort_values('points', ascending=False)[['name', 'team', 'position', 'cost', 'points']].head())
    else:
        print("Live fetch failed. Generating mock data...")
        df = ingestion.generate_mock_data()
        print(f"Generated {len(df)} mock players.")