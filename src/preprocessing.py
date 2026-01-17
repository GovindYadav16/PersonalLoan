"""Preprocessing module - Feature engineering and data cleaning"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Tuple


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        df: Input DataFrame
        strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop')
        
    Returns:
        DataFrame with handled missing values
    """
    df_processed = df.copy()
    
    if strategy == 'drop':
        df_processed = df_processed.dropna()
    elif strategy == 'mean':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    elif strategy == 'mode':
        df_processed = df_processed.fillna(df_processed.mode().iloc[0])
    
    return df_processed


def encode_categorical_features(df: pd.DataFrame, columns: list = None) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using Label Encoding
    
    Args:
        df: Input DataFrame
        columns: List of categorical columns to encode. If None, encodes all object columns
        
    Returns:
        Tuple of (encoded DataFrame, encoders dictionary)
    """
    df_encoded = df.copy()
    encoders = {}
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   scaler_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features
        scaler_path: Path to save/load scaler
        
    Returns:
        Tuple of (scaled X_train, scaled X_test, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_features_target(df: pd.DataFrame, target_column: str, 
                           exclude_columns: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        exclude_columns: List of columns to exclude from features (e.g., ID columns)
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Columns to drop: target and any excluded columns (like ID)
    columns_to_drop = [target_column]
    if exclude_columns:
        columns_to_drop.extend([col for col in exclude_columns if col in df.columns])
    
    X = df.drop(columns=columns_to_drop)
    y = df[target_column]
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple:
    """
    Split data into train and test sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    # Example usage
    print("Preprocessing module loaded successfully")
