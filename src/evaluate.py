"""Model evaluation module"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import pickle
import os

from src.train import load_model
from src.preprocessing import (
    handle_missing_values, 
    encode_categorical_features, 
    prepare_features_target,
    split_data
)
from src.data_ingestion import load_data, load_config


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    # Add ROC AUC if probabilities are available
    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            pass
    
    return metrics


def print_evaluation_metrics(metrics: Dict[str, float]) -> None:
    """Print evaluation metrics in a formatted way"""
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    print("="*50 + "\n")


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, 
                         save_path: str = None) -> None:
    """
    Plot and optionally save confusion matrix
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot. If None, only displays
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Loan', 'Loan'],
                yticklabels=['No Loan', 'Loan'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_on_test_set(config: Dict = None) -> Dict[str, float]:
    """
    Evaluate model on test set using configuration
    
    Args:
        config: Configuration dictionary. If None, loads from config.yaml
        
    Returns:
        Dictionary of evaluation metrics
    """
    if config is None:
        config = load_config()
    
    # Load model
    model_path = config['model']['model_path']
    model = load_model(model_path)
    
    # Load and preprocess test data
    raw_data_path = config['data']['raw_path']
    df = load_data(raw_data_path)
    df = handle_missing_values(df, strategy=config.get('preprocessing', {}).get('missing_strategy', 'mean'))
    
    target_col = config['model']['target_column']
    exclude_cols = config.get('preprocessing', {}).get('exclude_columns', [])
    X, y = prepare_features_target(df, target_col, exclude_columns=exclude_cols)
    X, _ = encode_categorical_features(X)
    
    # Split data (using same random_state as training)
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=config.get('model', {}).get('test_size', 0.2),
        random_state=config.get('model', {}).get('random_state', 42)
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print_evaluation_metrics(metrics)
    
    # Print classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics


if __name__ == "__main__":
    metrics = evaluate_on_test_set()
    print("Evaluation completed!")
