#!/usr/bin/env python3
"""
Unit tests for the training pipeline.
Tests configuration loading, model creation, and accuracy validation.
"""

import unittest
import tempfile
import os
import json
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from train import train_model, evaluate_model, load_data, save_model, main
from utils import setup_logging, load_config, save_artifacts, load_artifacts


class TestConfigurationLoading(unittest.TestCase):
    """Test cases for configuration file loading."""

    def setUp(self):
        """Set up test fixtures."""
        # Fix path to work from both root and tests directory
        import os

        if os.path.exists("config/config.json"):
            self.config_path = "config/config.json"
        else:
            self.config_path = "../config/config.json"

    def test_config_file_loads_successfully(self):
        """Test that the configuration file loads successfully."""
        # Test loading config
        config = load_config(self.config_path)

        # Check if config is loaded
        self.assertIsInstance(config, dict)
        self.assertIn("model", config)
        self.assertIn("data", config)
        self.assertIn("training", config)

    def test_required_hyperparameters_exist(self):
        """Check that all required hyperparameters exist in the configuration."""
        config = load_config(self.config_path)
        model_config = config["model"]

        # Check required hyperparameters
        required_params = ["C", "solver", "max_iter"]
        for param in required_params:
            self.assertIn(param, model_config, f"Missing required parameter: {param}")

    def test_hyperparameter_data_types(self):
        """Check that the values have the correct data types."""
        config = load_config(self.config_path)
        model_config = config["model"]

        # Check data types
        self.assertIsInstance(model_config["C"], (int, float), "C should be float/int")
        self.assertIsInstance(model_config["solver"], str, "solver should be string")
        self.assertIsInstance(model_config["max_iter"], int, "max_iter should be int")

        # Check specific values
        self.assertGreater(model_config["C"], 0, "C should be positive")
        self.assertIn(
            model_config["solver"],
            ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            "solver should be a valid option",
        )
        self.assertGreater(model_config["max_iter"], 0, "max_iter should be positive")


class TestModelCreation(unittest.TestCase):
    """Test cases for model creation and training."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "C": 1.0,
            "max_iter": 100,
            "solver": "lbfgs",
            "multi_class": "auto",
            "random_state": 42,
        }

        # Create dummy data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 64)  # 64 features like digits dataset
        self.y_train = np.random.randint(0, 10, 100)  # 10 classes like digits dataset

    def test_train_model_returns_logistic_regression(self):
        """Test that train_model returns a LogisticRegression object."""
        from sklearn.linear_model import LogisticRegression

        # Call training function
        model = train_model(self.X_train, self.y_train, self.config)

        # Verify it returns LogisticRegression object
        self.assertIsInstance(model, LogisticRegression)

    def test_model_has_been_fitted(self):
        """Confirm the object has been fitted by checking attributes."""
        model = train_model(self.X_train, self.y_train, self.config)

        # Check fitted attributes
        self.assertTrue(hasattr(model, "coef_"), "Model should have coef_ attribute")
        self.assertTrue(
            hasattr(model, "classes_"), "Model should have classes_ attribute"
        )
        self.assertTrue(
            hasattr(model, "intercept_"), "Model should have intercept_ attribute"
        )

        # Check that attributes are not None
        self.assertIsNotNone(model.coef_, "coef_ should not be None")
        self.assertIsNotNone(model.classes_, "classes_ should not be None")
        self.assertIsNotNone(model.intercept_, "intercept_ should not be None")

        # Check shapes
        self.assertEqual(
            model.coef_.shape[0],
            len(model.classes_),
            "coef_ should have same number of rows as classes",
        )
        self.assertEqual(
            model.coef_.shape[1],
            self.X_train.shape[1],
            "coef_ should have same number of columns as features",
        )

    def test_model_config_parameters_applied(self):
        """Test that model configuration parameters are correctly applied."""
        model = train_model(self.X_train, self.y_train, self.config)

        # Check that config parameters are applied
        self.assertEqual(model.C, self.config["C"])
        self.assertEqual(model.max_iter, self.config["max_iter"])
        self.assertEqual(model.solver, self.config["solver"])
        self.assertEqual(model.random_state, self.config["random_state"])


class TestModelAccuracy(unittest.TestCase):
    """Test cases for model accuracy validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "multi_class": "auto",
            "random_state": 42,
        }

    def test_model_accuracy_on_digits_dataset(self):
        """Train the model on the digits dataset and check accuracy."""
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        # Load digits dataset
        digits = load_digits()
        X = digits.data
        y = digits.target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        model = train_model(X_train, y_train, self.config)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Check accuracy threshold (digits dataset should achieve >90% accuracy)
        accuracy = metrics["accuracy"]
        self.assertGreater(
            accuracy,
            0.90,
            f"Model accuracy {accuracy:.4f} should be above 90% for digits dataset",
        )

        # Check that metrics contain required keys
        self.assertIn("accuracy", metrics)
        self.assertIn("classification_report", metrics)
        self.assertIn("predictions", metrics)

        # Check predictions length
        self.assertEqual(len(metrics["predictions"]), len(y_test))

    def test_model_accuracy_consistency(self):
        """Test that model accuracy is consistent across multiple runs."""
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        # Load digits dataset
        digits = load_digits()
        X = digits.data
        y = digits.target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        accuracies = []

        # Run multiple times with same random state
        for _ in range(3):
            model = train_model(X_train, y_train, self.config)
            metrics = evaluate_model(model, X_test, y_test)
            accuracies.append(metrics["accuracy"])

        # Check consistency (should be very similar due to fixed random state)
        max_diff = max(accuracies) - min(accuracies)
        self.assertLess(
            max_diff,
            0.01,
            f"Accuracy should be consistent, max difference: {max_diff:.4f}",
        )


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading functionality."""

    def test_load_data_returns_correct_shapes(self):
        """Test that load_data returns data with correct shapes."""
        X, y = load_data()

        # Check data shapes
        self.assertEqual(X.shape[1], 64, "Digits dataset should have 64 features")
        self.assertEqual(len(np.unique(y)), 10, "Digits dataset should have 10 classes")
        self.assertEqual(
            X.shape[0], y.shape[0], "X and y should have same number of samples"
        )

        # Check data types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

        # Check value ranges
        self.assertTrue(np.all(X >= 0), "All pixel values should be non-negative")
        self.assertTrue(np.all(y >= 0) and np.all(y < 10), "Labels should be 0-9")


class TestModelSaving(unittest.TestCase):
    """Test cases for model saving functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")

        # Create a simple model
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X, y)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_save_model_creates_file(self):
        """Test that save_model creates the model file."""
        # Save model
        save_model(self.model, self.model_path)

        # Check if file exists
        self.assertTrue(os.path.exists(self.model_path))

        # Check file size
        file_size = os.path.getsize(self.model_path)
        self.assertGreater(file_size, 0, "Model file should not be empty")

    def test_saved_model_can_be_loaded(self):
        """Test that saved model can be loaded and used."""
        import joblib

        # Save model
        save_model(self.model, self.model_path)

        # Load model
        loaded_model = joblib.load(self.model_path)

        # Check it's the same type
        self.assertIsInstance(loaded_model, type(self.model))

        # Check it can make predictions
        test_data = np.random.randn(5, 10)
        predictions = loaded_model.predict(test_data)
        self.assertEqual(len(predictions), 5)


class TestUtilsFunctions(unittest.TestCase):
    """Test cases for utility functions to increase coverage."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_setup_logging(self):
        """Test logging setup function."""
        # Test default logging setup
        setup_logging()  # Should not raise any exception

        # Test with different log levels
        setup_logging("DEBUG")  # Should not raise any exception
        setup_logging("ERROR")  # Should not raise any exception
        setup_logging("INFO")  # Should not raise any exception

        # Test that logging is working
        import logging

        logger = logging.getLogger()
        self.assertIsNotNone(logger)

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_file.json")

    def test_save_and_load_artifacts(self):
        """Test artifact saving and loading functionality."""
        from sklearn.ensemble import RandomForestClassifier

        # Create test artifacts
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(np.random.randn(10, 3), np.random.randint(0, 2, 10))

        artifacts = {
            "model": model,
            "metrics": {"accuracy": 0.85},
            "config": {"test": True},
        }

        artifacts_path = os.path.join(self.temp_dir, "artifacts")

        # Save artifacts
        save_artifacts(artifacts, artifacts_path)

        # Check if files exist
        self.assertTrue(os.path.exists(os.path.join(artifacts_path, "model.pkl")))
        self.assertTrue(os.path.exists(os.path.join(artifacts_path, "metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(artifacts_path, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(artifacts_path, "metadata.json")))

        # Load artifacts
        loaded_artifacts = load_artifacts(artifacts_path)

        # Check if artifacts are loaded
        self.assertIn("model", loaded_artifacts)
        self.assertIn("metrics", loaded_artifacts)
        self.assertIn("config", loaded_artifacts)

        # Check if model can predict
        predictions = loaded_artifacts["model"].predict(np.random.randn(5, 3))
        self.assertEqual(len(predictions), 5)

    def test_save_artifacts_with_custom_path(self):
        """Test saving artifacts with custom path."""
        artifacts = {"test_data": np.array([1, 2, 3]), "test_dict": {"key": "value"}}

        custom_path = os.path.join(self.temp_dir, "custom_artifacts")
        save_artifacts(artifacts, custom_path)

        # Check if directory was created
        self.assertTrue(os.path.exists(custom_path))

        # Check if files were saved (adjust based on actual save_artifacts implementation)
        # The actual file names depend on how save_artifacts handles different data types
        files_in_dir = os.listdir(custom_path)
        self.assertGreater(len(files_in_dir), 0, "Should have saved at least one file")

        # Check that we have some artifact files
        has_json_files = any(f.endswith(".json") for f in files_in_dir)
        self.assertTrue(has_json_files, "Should have saved JSON files")

    def test_load_artifacts_empty_directory(self):
        """Test loading artifacts from empty directory."""
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)

        # Should not raise error for empty directory
        artifacts = load_artifacts(empty_dir)
        self.assertIsInstance(artifacts, dict)


class TestTrainMainFunction(unittest.TestCase):
    """Test cases for the main function in train.py."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("train.load_data")
    @patch("train.load_config")
    @patch("train.train_model")
    @patch("train.evaluate_model")
    @patch("train.save_model")
    @patch("train.save_artifacts")
    def test_main_function_success(
        self,
        mock_save_artifacts,
        mock_save_model,
        mock_evaluate,
        mock_train,
        mock_load_config,
        mock_load_data,
    ):
        """Test the main function executes successfully."""
        # Mock return values - load_data returns X, y, not X_train, X_test, y_train, y_test
        X, y = np.random.randn(100, 64), np.random.randint(0, 10, 100)
        mock_load_data.return_value = (X, y)

        config = {
            "model": {"C": 1.0, "solver": "lbfgs", "max_iter": 100},
            "training": {"test_size": 0.2},
            "artifacts": {"path": self.temp_dir},
        }
        mock_load_config.return_value = config

        from sklearn.linear_model import LogisticRegression

        mock_model = LogisticRegression()
        mock_train.return_value = mock_model

        mock_metrics = {"accuracy": 0.95, "predictions": [0, 1, 2]}
        mock_evaluate.return_value = mock_metrics

        # Call main function
        main()

        # Verify all functions were called
        mock_load_data.assert_called_once()
        mock_load_config.assert_called_once()
        mock_train.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_save_model.assert_called_once()
        mock_save_artifacts.assert_called_once()

    @patch("train.load_data")
    def test_main_function_data_loading_error(self, mock_load_data):
        """Test main function handles data loading errors."""
        mock_load_data.side_effect = Exception("Data loading failed")

        # Should not crash, but log the error
        with self.assertRaises(Exception):
            main()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_train_model_with_empty_data(self):
        """Test model training with empty data."""
        with self.assertRaises(ValueError):
            train_model(np.array([]), np.array([]), {"C": 1.0, "solver": "lbfgs"})

    def test_evaluate_model_with_none_model(self):
        """Test model evaluation with None model."""
        X_test = np.random.randn(10, 64)
        y_test = np.random.randint(0, 10, 10)

        with self.assertRaises(AttributeError):
            evaluate_model(None, X_test, y_test)

    def test_save_model_with_invalid_path(self):
        """Test saving model with invalid path."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()

        # The save_model function creates directories automatically, so we need to test a different scenario
        # Let's test with a path that would cause an error during actual saving
        # Create a file with the same name as the directory we want to create
        conflict_path = os.path.join(self.temp_dir, "conflict")
        with open(conflict_path, "w") as f:
            f.write("test")

        # Now try to save to a path that would require creating a directory with the same name as a file
        invalid_path = os.path.join(conflict_path, "model.pkl")

        # This should fail because we can't create a directory with the same name as an existing file
        with self.assertRaises((OSError, FileNotFoundError)):
            save_model(model, invalid_path)

    def test_load_config_with_invalid_json(self):
        """Test loading config with invalid JSON."""
        # Create temporary invalid JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')
            invalid_config_path = f.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                load_config(invalid_config_path)
        finally:
            os.unlink(invalid_config_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete training pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_complete_training_pipeline(self):
        """Test the complete training pipeline end-to-end."""
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        # Load data
        digits = load_digits()
        X = digits.data
        y = digits.target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Load config
        import os

        if os.path.exists("config/config.json"):
            config = load_config("config/config.json")
        else:
            config = load_config("../config/config.json")

        # Train model
        model = train_model(X_train, y_train, config["model"])

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Save model
        model_path = os.path.join(self.temp_dir, "integration_model.pkl")
        save_model(model, model_path)

        # Save artifacts
        artifacts = {"model": model, "metrics": metrics, "config": config}
        artifacts_path = os.path.join(self.temp_dir, "artifacts")
        save_artifacts(artifacts, artifacts_path)

        # Verify results
        self.assertGreater(metrics["accuracy"], 0.90)
        self.assertTrue(os.path.exists(model_path))
        self.assertIsInstance(model, type(model))

        # Test loading artifacts
        loaded_artifacts = load_artifacts(artifacts_path)
        self.assertIn("model", loaded_artifacts)
        self.assertIn("metrics", loaded_artifacts)
        self.assertIn("config", loaded_artifacts)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
