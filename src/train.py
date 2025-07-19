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

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from utils import setup_logging, load_config, save_artifacts


def load_data():
    """Load the digits dataset."""
    logging.info("Loading digits dataset...")
    
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    logging.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logging.info(f"Target classes: {len(np.unique(y))}")
    
    return X, y


def train_model(X_train, y_train, config: Dict[str, Any]):
    """Train the LogisticRegression model."""
    logging.info("Starting model training...")
    
    # Initialize LogisticRegression model with config parameters
    model = LogisticRegression(
        C=config.get('C', 1.0),
        max_iter=config.get('max_iter', 100),
        random_state=config.get('random_state', 42),
        solver=config.get('solver', 'lbfgs'),
        multi_class=config.get('multi_class', 'auto')
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
        'predictions': y_pred.tolist()
    }


def save_model(model, model_path: str):
    """Save the model using joblib."""
    logging.info(f"Saving model to {model_path}")
    
    try:
        # Create directory if it doesn't exist and path has a directory
        dir_path = os.path.dirname(model_path)
        if dir_path:  # Only create directory if there's a path
            os.makedirs(dir_path, exist_ok=True)
        
        # Save model using joblib
        joblib.dump(model, model_path)
        logging.info(f"✅ Model saved successfully to {model_path}")
        
    except Exception as e:
        logging.error(f"❌ Failed to save model: {e}")
        raise


def main():
    """Main training function."""
    # Setup logging
    setup_logging()
    logging.info("🚀 Starting training pipeline...")
    
    # Load configuration with correct path
    config_path = "../config/config.json" if os.path.exists("../config/config.json") else "config/config.json"
    config = load_config(config_path)
    
    # Load data
    X, y = load_data()
    
    # Split data into train and test sets
    test_size = config.get('test_size', 0.2)
    random_state = config.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logging.info(f"Training set: {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = train_model(X_train, y_train, config['model'])
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model as model_train.pkl
    model_path = "model_train.pkl"
    save_model(model, model_path)
    
    # Save additional artifacts
    artifacts = {
        'model': model,
        'metrics': metrics,
        'config': config
    }
    artifacts_path = config.get('artifacts_path', 'artifacts')
    if not os.path.isabs(artifacts_path):
        artifacts_path = os.path.join("..", artifacts_path)
    save_artifacts(artifacts, artifacts_path)
    
    logging.info("✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 