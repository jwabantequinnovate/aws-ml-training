"""
ML Toolkit - Production-ready utilities for AWS SageMaker ML workflows

This package provides battle-tested components for:
- Data preprocessing and feature engineering
- Model evaluation and monitoring
- SageMaker deployment and management
- Text processing for NLP tasks
- Model lineage tracking
- Training debugging and profiling
- MLflow experiment tracking
- DVC data versioning
"""

from ml_toolkit.preprocessing import DataPreprocessor, TextPreprocessor
from ml_toolkit.evaluation import ModelEvaluator
from ml_toolkit.sagemaker_utils import SageMakerDeployment
from ml_toolkit.config import AWSConfig
from ml_toolkit.lineage import ModelLineageTracker
from ml_toolkit.debugger import create_debugger_config, check_training_issues
from ml_toolkit.mlflow_tracking import MLflowExperimentTracker
from ml_toolkit.dvc_manager import DVCManager

__version__ = "1.0.0"
__all__ = [
    "DataPreprocessor",
    "TextPreprocessor",
    "ModelEvaluator",
    "SageMakerDeployment",
    "AWSConfig",
    "ModelLineageTracker",
    "create_debugger_config",
    "check_training_issues",
    "MLflowExperimentTracker",
    "DVCManager",
]
