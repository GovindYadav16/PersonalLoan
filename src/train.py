"""Model training module"""

import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yaml
from typing import Dict, Any

from src.preprocessing import (
    handle_missing_values, 
    encode_categorical_features,
    prepare_features_target,
    split_data
)
from src.data_ingestion import load_data, load_config


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                       n_estimators: int = 100, max_depth: int = None,
                       random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the tree
        random_state: Random seed
        
    Returns:
        Trained Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                              random_state: int = 42) -> LogisticRegression:
    """
    Train a Logistic Regression classifier
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Trained Logistic Regression model
    """
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, model_path: str) -> None:
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        model_path: Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def load_model(model_path: str) -> Any:
    """
    Load trained model from disk
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model object
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def train_model(config: Dict = None) -> Any:
    """
    Main training pipeline
    
    Args:
        config: Configuration dictionary. If None, loads from config.yaml
        
    Returns:
        Trained model
    """
    if config is None:
        config = load_config()
    
    # Load data
    raw_data_path = config['data']['raw_path']
    df = load_data(raw_data_path)
    
    # Preprocessing
    df = handle_missing_values(df, strategy=config.get('preprocessing', {}).get('missing_strategy', 'mean'))
    
    target_col = config['model']['target_column']
    exclude_cols = config.get('preprocessing', {}).get('exclude_columns', [])
    X, y = prepare_features_target(df, target_col, exclude_columns=exclude_cols)
    
    # Encode categorical features
    X, encoders = encode_categorical_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y, 
        test_size=config.get('model', {}).get('test_size', 0.2),
        random_state=config.get('model', {}).get('random_state', 42)
    )
    
    # Train model
    model_type = config['model'].get('type', 'random_forest')
    
    if model_type == 'random_forest':
        model = train_random_forest(
            X_train, y_train,
            n_estimators=config['model'].get('n_estimators', 100),
            max_depth=config['model'].get('max_depth', None),
            random_state=config['model'].get('random_state', 42)
        )
    elif model_type == 'logistic_regression':
        model = train_logistic_regression(X_train, y_train, 
                                         random_state=config['model'].get('random_state', 42))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save model
    model_path = config['model']['model_path']
    save_model(model, model_path)
    
    print(f"Model training completed. Model saved to {model_path}")
    return model


if __name__ == "__main__":
    model = train_model()
    print("Training completed successfully!")
