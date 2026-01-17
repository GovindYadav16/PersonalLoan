"""Prediction module - Inference for new data"""

import pandas as pd
import numpy as np
import pickle
from typing import Union, List, Dict, Any
import os

from src.train import load_model
from src.preprocessing import handle_missing_values, encode_categorical_features
from src.data_ingestion import load_config


def predict_single(model: Any, features: pd.DataFrame, 
                   return_proba: bool = False) -> Union[int, float, tuple]:
    """
    Make prediction for a single instance
    
    Args:
        model: Trained model
        features: Feature vector as DataFrame
        return_proba: If True, also return prediction probabilities
        
    Returns:
        Prediction (and probabilities if return_proba=True)
    """
    prediction = model.predict(features)[0]
    
    if return_proba and hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features)[0]
        return prediction, probability
    else:
        return prediction


def predict_batch(model: Any, features: pd.DataFrame,
                  return_proba: bool = False) -> Union[np.ndarray, tuple]:
    """
    Make predictions for multiple instances
    
    Args:
        model: Trained model
        features: Feature DataFrame
        return_proba: If True, also return prediction probabilities
        
    Returns:
        Predictions array (and probabilities if return_proba=True)
    """
    predictions = model.predict(features)
    
    if return_proba and hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)
        return predictions, probabilities
    else:
        return predictions


def preprocess_new_data(df: pd.DataFrame, config: Dict = None,
                       encoders_path: str = None) -> pd.DataFrame:
    """
    Preprocess new data for prediction (should match training preprocessing)
    
    Args:
        df: Raw input DataFrame
        config: Configuration dictionary
        encoders_path: Path to saved encoders (optional)
        
    Returns:
        Preprocessed DataFrame with features only (excludes ID and target columns)
    """
    if config is None:
        config = load_config()
    
    # Handle missing values
    strategy = config.get('preprocessing', {}).get('missing_strategy', 'mean')
    df_processed = handle_missing_values(df, strategy=strategy)
    
    # Exclude the same columns that were excluded during training
    # This ensures feature names match what the model expects
    exclude_cols = config.get('preprocessing', {}).get('exclude_columns', [])
    target_col = config.get('model', {}).get('target_column', 'Personal_Loan')
    
    # Create a copy and drop excluded columns
    df_features = df_processed.copy()
    columns_to_drop = []
    
    # Add excluded columns (like ID)
    if exclude_cols:
        columns_to_drop.extend([col for col in exclude_cols if col in df_features.columns])
    
    # Add target column if it exists (it might not be in prediction data)
    if target_col in df_features.columns:
        columns_to_drop.append(target_col)
    
    # Drop columns that shouldn't be features
    if columns_to_drop:
        df_features = df_features.drop(columns=columns_to_drop)
    
    # Encode categorical features
    # In production, you should load the encoders saved during training
    # For now, we'll create new encoders (this should match training logic)
    df_encoded, _ = encode_categorical_features(df_features)
    
    return df_encoded


def predict_from_file(model_path: str, data_path: str, 
                     config: Dict = None,
                     output_path: str = None) -> pd.DataFrame:
    """
    Load model, load data from file, and make predictions
    
    Args:
        model_path: Path to saved model
        data_path: Path to input data file
        config: Configuration dictionary
        output_path: Path to save predictions (optional)
        
    Returns:
        DataFrame with predictions
    """
    # Load model
    model = load_model(model_path)
    
    # Load data
    from src.data_ingestion import load_data
    df = load_data(data_path)
    
    # Preprocess
    df_processed = preprocess_new_data(df, config)
    
    # Make predictions
    predictions, probabilities = predict_batch(model, df_processed, return_proba=True)
    
    # Create results DataFrame
    results = df.copy()
    results['prediction'] = predictions
    if probabilities is not None:
        results['probability'] = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities[:, 0]
    
    # Save if output path provided
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    config = load_config()
    model_path = config['model']['model_path']
    
    # Example: predict from a CSV file
    # results = predict_from_file(model_path, "data/processed/test_data.csv")
    # print(results.head())
    
    print("Prediction module loaded successfully")
