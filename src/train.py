#!/usr/bin/env python3
"""
Training script for the MLOps artifact pipeline.
Handles model training, validation, and artifact storage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from utils import setup_logging, load_config, save_artifacts


def load_data(data_path: str) -> tuple:
    """Load and prepare training data."""
    logging.info(f"Loading data from {data_path}")
    # Placeholder for data loading logic
    # In a real scenario, you'd load your actual dataset here
    return None, None


def train_model(X_train, y_train, config: Dict[str, Any]):
    """Train the machine learning model."""
    logging.info("Starting model training...")
    
    # Initialize model with config parameters
    model = RandomForestClassifier(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', None),
        random_state=config.get('random_state', 42)
    )
    
    # Train the model
    model.fit(X_train, y_train)
    logging.info("Model training completed!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    logging.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logging.info(f"Model accuracy: {accuracy:.4f}")
    logging.info(f"Classification report:\n{report}")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': y_pred
    }


def main():
    """Main training function."""
    # Setup logging
    setup_logging()
    logging.info("🚀 Starting training pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(config['data_path'])
    
    # Train model
    model = train_model(X_train, y_train, config['model'])
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save artifacts
    artifacts = {
        'model': model,
        'metrics': metrics,
        'config': config
    }
    save_artifacts(artifacts, config['artifacts_path'])
    
    logging.info("✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 