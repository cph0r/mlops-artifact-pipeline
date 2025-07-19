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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import train_model, evaluate_model, load_data, save_model
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
        self.assertIn('model', config)
        self.assertIn('data', config)
        self.assertIn('training', config)
        
    def test_required_hyperparameters_exist(self):
        """Check that all required hyperparameters exist in the configuration."""
        config = load_config(self.config_path)
        model_config = config['model']
        
        # Check required hyperparameters
        required_params = ['C', 'solver', 'max_iter']
        for param in required_params:
            self.assertIn(param, model_config, f"Missing required parameter: {param}")
    
    def test_hyperparameter_data_types(self):
        """Check that the values have the correct data types."""
        config = load_config(self.config_path)
        model_config = config['model']
        
        # Check data types
        self.assertIsInstance(model_config['C'], (int, float), "C should be float/int")
        self.assertIsInstance(model_config['solver'], str, "solver should be string")
        self.assertIsInstance(model_config['max_iter'], int, "max_iter should be int")
        
        # Check specific values
        self.assertGreater(model_config['C'], 0, "C should be positive")
        self.assertIn(model_config['solver'], ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], 
                     "solver should be a valid option")
        self.assertGreater(model_config['max_iter'], 0, "max_iter should be positive")


class TestModelCreation(unittest.TestCase):
    """Test cases for model creation and training."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'C': 1.0,
            'max_iter': 100,
            'solver': 'lbfgs',
            'multi_class': 'auto',
            'random_state': 42
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
        self.assertTrue(hasattr(model, 'coef_'), "Model should have coef_ attribute")
        self.assertTrue(hasattr(model, 'classes_'), "Model should have classes_ attribute")
        self.assertTrue(hasattr(model, 'intercept_'), "Model should have intercept_ attribute")
        
        # Check that attributes are not None
        self.assertIsNotNone(model.coef_, "coef_ should not be None")
        self.assertIsNotNone(model.classes_, "classes_ should not be None")
        self.assertIsNotNone(model.intercept_, "intercept_ should not be None")
        
        # Check shapes
        self.assertEqual(model.coef_.shape[0], len(model.classes_), 
                        "coef_ should have same number of rows as classes")
        self.assertEqual(model.coef_.shape[1], self.X_train.shape[1], 
                        "coef_ should have same number of columns as features")
    
    def test_model_config_parameters_applied(self):
        """Test that model configuration parameters are correctly applied."""
        model = train_model(self.X_train, self.y_train, self.config)
        
        # Check that config parameters are applied
        self.assertEqual(model.C, self.config['C'])
        self.assertEqual(model.max_iter, self.config['max_iter'])
        self.assertEqual(model.solver, self.config['solver'])
        self.assertEqual(model.random_state, self.config['random_state'])


class TestModelAccuracy(unittest.TestCase):
    """Test cases for model accuracy validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'multi_class': 'auto',
            'random_state': 42
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
        accuracy = metrics['accuracy']
        self.assertGreater(accuracy, 0.90, 
                          f"Model accuracy {accuracy:.4f} should be above 90% for digits dataset")
        
        # Check that metrics contain required keys
        self.assertIn('accuracy', metrics)
        self.assertIn('classification_report', metrics)
        self.assertIn('predictions', metrics)
        
        # Check predictions length
        self.assertEqual(len(metrics['predictions']), len(y_test))
    
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
            accuracies.append(metrics['accuracy'])
        
        # Check consistency (should be very similar due to fixed random state)
        max_diff = max(accuracies) - min(accuracies)
        self.assertLess(max_diff, 0.01, 
                       f"Accuracy should be consistent, max difference: {max_diff:.4f}")


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading functionality."""
    
    def test_load_data_returns_correct_shapes(self):
        """Test that load_data returns data with correct shapes."""
        X, y = load_data()
        
        # Check data shapes
        self.assertEqual(X.shape[1], 64, "Digits dataset should have 64 features")
        self.assertEqual(len(np.unique(y)), 10, "Digits dataset should have 10 classes")
        self.assertEqual(X.shape[0], y.shape[0], "X and y should have same number of samples")
        
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
        
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
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
        model = train_model(X_train, y_train, config['model'])
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        model_path = os.path.join(self.temp_dir, "integration_model.pkl")
        save_model(model, model_path)
        
        # Verify results
        self.assertGreater(metrics['accuracy'], 0.90)
        self.assertTrue(os.path.exists(model_path))
        self.assertIsInstance(model, type(model))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 