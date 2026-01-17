"""Data ingestion module - Load and validate data"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple
import yaml


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load dataset from file path
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        DataFrame with loaded data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Support for CSV, Excel, or other formats
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate data integrity
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data is valid, raises exception otherwise
    """
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if df.isnull().all().any():
        raise ValueError("Dataset contains columns with all null values")
    
    print(f"Data validation passed. Shape: {df.shape}")
    return True


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Example usage
    config = load_config()
    raw_data_path = config['data']['raw_path']
    df = load_data(raw_data_path)
    validate_data(df)
    print(df.head())
