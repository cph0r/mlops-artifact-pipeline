#!/usr/bin/env python3
"""
Inference script for digit classification model.

This script loads a trained model and generates predictions on the digits dataset.
"""

import os
import sys
import logging
import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from utils import setup_logging, load_config


def load_model(model_path: str):
    """
    Load the trained model from pickle file.

    Args:
        model_path (str): Path to the saved model file

    Returns:
        The loaded model object

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)

        logging.info(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {str(e)}")
        raise


def load_test_data():
    """
    Load the digits dataset for inference.

    Returns:
        tuple: (X_test, y_test) test features and labels
    """
    try:
        # Load the digits dataset
        X, y = load_digits(return_X_y=True)

        # For inference, we'll use a subset of the data
        # In a real scenario, this would be new unseen data
        test_size = min(100, len(X))  # Use 100 samples for inference
        X_test = X[:test_size]
        y_test = y[:test_size]

        logging.info(
            f"Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features"
        )
        return X_test, y_test

    except Exception as e:
        logging.error(f"Failed to load test data: {str(e)}")
        raise


def generate_predictions(model, X_test):
    """
    Generate predictions using the trained model.

    Args:
        model: The trained model
        X_test: Test features

    Returns:
        numpy.ndarray: Predicted labels
    """
    try:
        predictions = model.predict(X_test)
        logging.info(f"Generated predictions for {len(predictions)} samples")
        return predictions

    except Exception as e:
        logging.error(f"Failed to generate predictions: {str(e)}")
        raise


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate the model predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        dict: Evaluation metrics
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        logging.info(f"Model accuracy: {accuracy:.4f}")
        logging.info(f"Classification report:\n{classification_report(y_true, y_pred)}")

        return {"accuracy": accuracy, "classification_report": report}

    except Exception as e:
        logging.error(f"Failed to evaluate predictions: {str(e)}")
        raise


def save_predictions(predictions, output_path: str):
    """
    Save predictions to a file.

    Args:
        predictions: Model predictions
        output_path: Path to save predictions
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save predictions as numpy array
        np.save(output_path, predictions)

        logging.info(f"Predictions saved to {output_path}")

    except Exception as e:
        logging.error(f"Failed to save predictions: {str(e)}")
        raise


def main():
    """
    Main inference function.
    """
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting inference pipeline")

        # Load configuration
        config = load_config("config/config.json")
        model_config = config["model"]
        artifacts_config = config["artifacts"]

        # Define paths
        model_path = artifacts_config.get("model_path", "src/model_train.pkl")
        predictions_path = artifacts_config.get(
            "predictions_path", "artifacts/predictions.npy"
        )

        # Load the trained model
        logging.info("Loading trained model...")
        model = load_model(model_path)

        # Load test data
        logging.info("Loading test data...")
        X_test, y_test = load_test_data()

        # Generate predictions
        logging.info("Generating predictions...")
        predictions = generate_predictions(model, X_test)

        # Evaluate predictions
        logging.info("Evaluating predictions...")
        metrics = evaluate_predictions(y_test, predictions)

        # Save predictions
        logging.info("Saving predictions...")
        save_predictions(predictions, predictions_path)

        # Print summary
        print("\n" + "=" * 50)
        print("INFERENCE PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Model loaded from: {model_path}")
        print(f"Test samples: {len(X_test)}")
        print(f"Model accuracy: {metrics['accuracy']:.4f}")
        print(f"Predictions saved to: {predictions_path}")
        print("=" * 50)

        logging.info("Inference pipeline completed successfully")
        return True

    except Exception as e:
        logging.error(f"Inference pipeline failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
