"""
Unit Tests for Data Ingestion and Validation

Tests data loading, validation, and cleaning functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_ingestion import DataIngestion


class TestDataIngestion:
    """Test cases for DataIngestion class."""
    
    @pytest.fixture
    def ingestion(self, tmp_path):
        """Create DataIngestion instance with temporary directory."""
        return DataIngestion(data_dir=str(tmp_path))
    
    @pytest.fixture
    def sample_data(self):
        """Create sample player data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
            'position': ['GK', 'DEF', 'MID', 'FWD', 'GK'],
            'team': ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Spurs'],
            'cost': [5.0, 6.5, 8.0, 10.5, 4.5],
            'points': [100, 120, 150, 180, 80],
            'goals': [0, 2, 8, 15, 0],
            'assists': [0, 3, 5, 8, 0],
            'clean_sheets': [10, 12, 0, 0, 8],
            'form': [5.0, 6.0, 7.5, 9.0, 4.0],
            'selected_by_percent': [5.0, 10.0, 25.0, 40.0, 3.0],
            'minutes': [2700, 3000, 3200, 3400, 2500]
        })
    
    def test_generate_mock_data(self, ingestion):
        """Test mock data generation."""
        df = ingestion.generate_mock_data(n_players=100)
        
        assert len(df) == 100
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'position' in df.columns
        assert 'cost' in df.columns
        assert 'points' in df.columns
    
    def test_validate_data_success(self, ingestion, sample_data):
        """Test data validation with valid data."""
        validation = ingestion.validate_data(sample_data)
        
        assert validation['is_valid'] is True
        assert len(validation['errors']) == 0
        assert validation['stats']['total_players'] == 5
    
    def test_validate_data_missing_columns(self, ingestion):
        """Test data validation with missing required columns."""
        invalid_df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        validation = ingestion.validate_data(invalid_df)
        
        assert validation['is_valid'] is False
        assert len(validation['errors']) > 0
    
    def test_validate_data_negative_points(self, ingestion):
        """Test data validation with negative points."""
        invalid_df = pd.DataFrame({
            'id': [1],
            'name': ['Player1'],
            'position': ['GK'],
            'team': ['Arsenal'],
            'cost': [5.0],
            'points': [-10]  # Invalid
        })
        validation = ingestion.validate_data(invalid_df)
        
        assert validation['is_valid'] is False
        assert any('negative points' in error.lower() for error in validation['errors'])
    
    def test_clean_data_handles_nulls(self, ingestion):
        """Test data cleaning handles null values."""
        df_with_nulls = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', None],
            'position': ['GK', None, 'MID'],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'],
            'cost': [5.0, None, 8.0],
            'points': [100, 120, 150]
        })
        
        df_clean = ingestion.clean_data(df_with_nulls)
        
        # Check that nulls are filled
        assert df_clean['cost'].isnull().sum() == 0
        assert df_clean['name'].isnull().sum() == 0
    
    def test_clean_data_removes_duplicates(self, ingestion):
        """Test data cleaning removes duplicates."""
        df_with_duplicates = pd.DataFrame({
            'id': [1, 1, 2, 3],
            'name': ['A', 'A', 'B', 'C'],
            'position': ['GK', 'GK', 'DEF', 'MID'],
            'team': ['Arsenal', 'Arsenal', 'Chelsea', 'Liverpool'],
            'cost': [5.0, 5.0, 6.0, 7.0],
            'points': [100, 100, 120, 150]
        })
        
        df_clean = ingestion.clean_data(df_with_duplicates)
        
        assert len(df_clean) == 3  # One duplicate removed
        assert df_clean['id'].nunique() == 3
    
    def test_save_and_load_data(self, ingestion, sample_data, tmp_path):
        """Test saving and loading data."""
        filename = "test_players.csv"
        filepath = ingestion.save_data(sample_data, filename)
        
        assert Path(filepath).exists()
        
        df_loaded = ingestion.load_from_csv(filepath)
        
        assert len(df_loaded) == len(sample_data)
        assert list(df_loaded.columns) == list(sample_data.columns)


class TestDataValidation:
    """Test cases for data validation logic."""
    
    @pytest.fixture
    def ingestion(self):
        """Create DataIngestion instance."""
        return DataIngestion()
    
    def test_validate_cost_range(self, ingestion):
        """Test validation of cost range."""
        df_valid = pd.DataFrame({
            'id': [1, 2],
            'name': ['A', 'B'],
            'position': ['GK', 'DEF'],
            'team': ['Arsenal', 'Chelsea'],
            'cost': [5.0, 10.0],
            'points': [100, 120]
        })
        
        validation = ingestion.validate_data(df_valid)
        assert validation['is_valid'] is True
        
        df_invalid = pd.DataFrame({
            'id': [1],
            'name': ['A'],
            'position': ['GK'],
            'team': ['Arsenal'],
            'cost': [20.0],  # Out of range
            'points': [100]
        })
        
        validation = ingestion.validate_data(df_invalid)
        assert len(validation['warnings']) > 0
    
    def test_validate_position_distribution(self, ingestion):
        """Test position distribution in validation stats."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'position': ['GK', 'DEF', 'GK'],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'],
            'cost': [5.0, 6.0, 7.0],
            'points': [100, 120, 130]
        })
        
        validation = ingestion.validate_data(df)
        
        assert 'position_distribution' in validation['stats']
        assert validation['stats']['position_distribution']['GK'] == 2
        assert validation['stats']['position_distribution']['DEF'] == 1

