"""Utility functions for the project"""

import os
import logging
from pathlib import Path
from typing import Any
import json
import yaml


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logging.getLogger(__name__)


def ensure_dir(file_path: str) -> None:
    """Ensure directory exists for the given file path"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file"""
    ensure_dir(file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path: str) -> Any:
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def validate_config(config: dict) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises exception otherwise
    """
    required_keys = ['data', 'model']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if 'raw_path' not in config['data']:
        raise ValueError("Missing 'raw_path' in config['data']")
    
    if 'target_column' not in config['model']:
        raise ValueError("Missing 'target_column' in config['model']")
    
    if 'model_path' not in config['model']:
        raise ValueError("Missing 'model_path' in config['model']")
    
    return True


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Utils module loaded successfully")
