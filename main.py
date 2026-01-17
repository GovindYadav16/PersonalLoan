"""Main entry point for Personal Loan ML Project"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train import train_model
from src.evaluate import evaluate_on_test_set
from src.predict import predict_from_file
from src.data_ingestion import load_config
from src.utils import setup_logging, validate_config


def main():
    """Main function to run training, evaluation, or prediction"""
    parser = argparse.ArgumentParser(description='Personal Loan ML Project')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate', 'predict'],
                       help='Mode: train, evaluate, or predict')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data file (for predict mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predictions (for predict mode)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Running in {args.mode} mode")
    
    # Load and validate config
    config = load_config(args.config)
    validate_config(config)
    
    if args.mode == 'train':
        logger.info("Starting model training...")
        model = train_model(config)
        logger.info("Training completed successfully!")
        
    elif args.mode == 'evaluate':
        logger.info("Starting model evaluation...")
        metrics = evaluate_on_test_set(config)
        logger.info("Evaluation completed!")
        
    elif args.mode == 'predict':
        if args.data is None:
            logger.error("--data argument is required for predict mode")
            sys.exit(1)
        
        logger.info(f"Making predictions on {args.data}...")
        model_path = config['model']['model_path']
        results = predict_from_file(model_path, args.data, config, args.output)
        logger.info(f"Predictions completed! Shape: {results.shape}")
        print(results.head())


if __name__ == "__main__":
    main()
