#!/usr/bin/env python3
"""
Inference script for the MLOps artifact pipeline.
Handles model loading and prediction serving.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd
import joblib

from utils import setup_logging, load_config, load_artifacts


class ModelPredictor:
    """Model predictor class for handling inference."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """Initialize the predictor with a trained model."""
        self.setup_logging()
        self.model_path = model_path
        self.config_path = config_path or "config/config.json"
        
        # Load model and config
        self.model = self._load_model()
        self.config = self._load_config()
        
        logging.info("🎯 Model predictor initialized successfully!")
    
    def setup_logging(self):
        """Setup logging for the predictor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_model(self):
        """Load the trained model from disk."""
        logging.info(f"Loading model from {self.model_path}")
        try:
            model = joblib.load(self.model_path)
            logging.info("✅ Model loaded successfully!")
            return model
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_config(self):
        """Load configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logging.error(f"❌ Failed to load config: {e}")
            return {}
    
    def preprocess_input(self, data: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Preprocess input data for prediction."""
        logging.info("Preprocessing input data...")
        
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        # Add any specific preprocessing logic here
        # For now, just ensure it's a 2D array
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        logging.info(f"Preprocessed data shape: {data.shape}")
        return data
    
    def predict(self, data: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the loaded model."""
        logging.info("Making predictions...")
        
        # Preprocess input
        processed_data = self.preprocess_input(data)
        
        # Make prediction
        predictions = self.model.predict(processed_data)
        
        logging.info(f"✅ Made {len(predictions)} predictions")
        return predictions
    
    def predict_proba(self, data: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make probability predictions using the loaded model."""
        logging.info("Making probability predictions...")
        
        # Preprocess input
        processed_data = self.preprocess_input(data)
        
        # Make probability prediction
        probabilities = self.model.predict_proba(processed_data)
        
        logging.info(f"✅ Made {len(probabilities)} probability predictions")
        return probabilities


def main():
    """Main inference function for testing."""
    setup_logging()
    logging.info("🚀 Starting inference pipeline...")
    
    # Initialize predictor
    model_path = "artifacts/model.pkl"
    predictor = ModelPredictor(model_path)
    
    # Example prediction (replace with your actual data)
    sample_data = [[1, 2, 3, 4, 5]]  # Placeholder data
    
    try:
        # Make prediction
        predictions = predictor.predict(sample_data)
        logging.info(f"Predictions: {predictions}")
        
        # Make probability prediction
        probabilities = predictor.predict_proba(sample_data)
        logging.info(f"Probabilities: {probabilities}")
        
        logging.info("✅ Inference pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Inference failed: {e}")


if __name__ == "__main__":
    main() 