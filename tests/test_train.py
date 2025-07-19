#!/usr/bin/env python3
"""
Unit tests for the training module.
Tests model training, evaluation, and artifact saving functionality.
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

from train import train_model, evaluate_model
from utils import setup_logging, load_config, save_artifacts, load_artifacts


class TestTraining(unittest.TestCase):
    """Test cases for training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            }
        }
        
        # Create dummy data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.X_test = np.random.randn(20, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.y_test = np.random.randint(0, 2, 20)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_train_model(self):
        """Test model training functionality."""
        # Test training with valid data
        model = train_model(self.X_train, self.y_train, self.config['model'])
        
        # Check if model is trained
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'fit'))
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_train_model_with_invalid_data(self):
        """Test model training with invalid data."""
        # Test with empty data
        with self.assertRaises(Exception):
            train_model([], [], self.config['model'])
    
    def test_evaluate_model(self):
        """Test model evaluation functionality."""
        # Train a model first
        model = train_model(self.X_train, self.y_train, self.config['model'])
        
        # Evaluate the model
        metrics = evaluate_model(model, self.X_test, self.y_test)
        
        # Check if metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('classification_report', metrics)
        self.assertIn('predictions', metrics)
        
        # Check accuracy is between 0 and 1
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        
        # Check predictions length
        self.assertEqual(len(metrics['predictions']), len(self.y_test))
    
    def test_evaluate_model_with_invalid_model(self):
        """Test model evaluation with invalid model."""
        with self.assertRaises(Exception):
            evaluate_model(None, self.X_test, self.y_test)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self.artifacts_path = os.path.join(self.temp_dir, 'artifacts')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_setup_logging(self):
        """Test logging setup."""
        # Should not raise any exception
        setup_logging()
        setup_logging("DEBUG")
        setup_logging("ERROR")
    
    def test_save_and_load_config(self):
        """Test config saving and loading."""
        test_config = {
            'model': {'type': 'RandomForest'},
            'data': {'path': '/data'}
        }
        
        # Save config
        save_config(test_config, self.config_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(self.config_path))
        
        # Load config
        loaded_config = load_config(self.config_path)
        
        # Check if config matches
        self.assertEqual(test_config, loaded_config)
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with self.assertRaises(FileNotFoundError):
            load_config('non_existent_file.json')
    
    def test_save_and_load_artifacts(self):
        """Test artifact saving and loading."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create test artifacts
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(np.random.randn(10, 3), np.random.randint(0, 2, 10))
        
        artifacts = {
            'model': model,
            'metrics': {'accuracy': 0.85},
            'config': {'test': True}
        }
        
        # Save artifacts
        save_artifacts(artifacts, self.artifacts_path)
        
        # Check if files exist
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_path, 'model.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_path, 'metrics.json')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_path, 'config.json')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_path, 'metadata.json')))
        
        # Load artifacts
        loaded_artifacts = load_artifacts(self.artifacts_path)
        
        # Check if artifacts are loaded
        self.assertIn('model', loaded_artifacts)
        self.assertIn('metrics', loaded_artifacts)
        self.assertIn('config', loaded_artifacts)
        
        # Check if model can predict
        predictions = loaded_artifacts['model'].predict(np.random.randn(5, 3))
        self.assertEqual(len(predictions), 5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config
        self.config = {
            'data_path': 'test_data.csv',
            'model': {
                'n_estimators': 5,
                'max_depth': 3,
                'random_state': 42
            },
            'artifacts_path': self.temp_dir
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('train.load_data')
    def test_full_training_pipeline(self, mock_load_data):
        """Test the complete training pipeline."""
        # Mock data loading
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        X_test = np.random.randn(10, 3)
        y_train = np.random.randint(0, 2, 50)
        y_test = np.random.randint(0, 2, 10)
        
        mock_load_data.return_value = (X_train, X_test, y_train, y_test)
        
        # This would test the full pipeline if we had the main function
        # For now, we'll test the components individually
        model = train_model(X_train, y_train, self.config['model'])
        metrics = evaluate_model(model, X_test, y_test)
        
        # Verify results
        self.assertIsNotNone(model)
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 