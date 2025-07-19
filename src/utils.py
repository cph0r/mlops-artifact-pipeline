#!/usr/bin/env python3
"""
Utility functions for the MLOps artifact pipeline.
Common functionality used across training and inference.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import joblib


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
    )


def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logging.info(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"❌ Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"❌ Invalid JSON in config file: {e}")
        raise


def save_config(
    config: Dict[str, Any], config_path: str = "config/config.json"
) -> None:
    """Save configuration to JSON file."""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logging.info(f"✅ Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save config: {e}")
        raise


def load_artifacts(artifacts_path: str) -> Dict[str, Any]:
    """Load model artifacts from disk."""
    artifacts = {}

    try:
        # Load model
        model_path = os.path.join(artifacts_path, "model.pkl")
        if os.path.exists(model_path):
            artifacts["model"] = joblib.load(model_path)
            logging.info(f"✅ Model loaded from {model_path}")

        # Load metrics
        metrics_path = os.path.join(artifacts_path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                artifacts["metrics"] = json.load(f)
            logging.info(f"✅ Metrics loaded from {metrics_path}")

        # Load config
        config_path = os.path.join(artifacts_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                artifacts["config"] = json.load(f)
            logging.info(f"✅ Config loaded from {config_path}")

        return artifacts

    except Exception as e:
        logging.error(f"❌ Failed to load artifacts: {e}")
        raise


def save_artifacts(artifacts: Dict[str, Any], artifacts_path: str) -> None:
    """Save model artifacts to disk."""
    try:
        os.makedirs(artifacts_path, exist_ok=True)

        # Save model
        if "model" in artifacts:
            model_path = os.path.join(artifacts_path, "model.pkl")
            joblib.dump(artifacts["model"], model_path)
            logging.info(f"✅ Model saved to {model_path}")

        # Save metrics
        if "metrics" in artifacts:
            metrics_path = os.path.join(artifacts_path, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(artifacts["metrics"], f, indent=2)
            logging.info(f"✅ Metrics saved to {metrics_path}")

        # Save config
        if "config" in artifacts:
            config_path = os.path.join(artifacts_path, "config.json")
            with open(config_path, "w") as f:
                json.dump(artifacts["config"], f, indent=2)
            logging.info(f"✅ Config saved to {config_path}")

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "artifacts": list(artifacts.keys()),
        }
        metadata_path = os.path.join(artifacts_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"✅ Metadata saved to {metadata_path}")

    except Exception as e:
        logging.error(f"❌ Failed to save artifacts: {e}")
        raise


def create_experiment_dir(base_path: str = "experiments") -> str:
    """Create a new experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_path, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    logging.info(f"✅ Created experiment directory: {experiment_dir}")
    return experiment_dir


def validate_data_path(data_path: str) -> bool:
    """Validate if data path exists and is accessible."""
    if not os.path.exists(data_path):
        logging.error(f"❌ Data path does not exist: {data_path}")
        return False

    if not os.access(data_path, os.R_OK):
        logging.error(f"❌ Data path is not readable: {data_path}")
        return False

    logging.info(f"✅ Data path validated: {data_path}")
    return True


def get_file_size(file_path: str) -> Optional[int]:
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except OSError:
        logging.warning(f"⚠️ Could not get file size for: {file_path}")
        return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"
