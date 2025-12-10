"""
MLflow Integration for Experiment Tracking

Track experiments, parameters, metrics, and models with MLflow.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class MLflowExperimentTracker:
    """
    Wrapper for MLflow experiment tracking
    
    Simplifies tracking of ML experiments with automatic logging
    of parameters, metrics, artifacts, and models.
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: S3 or local path for artifacts
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI (local or remote server)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        self.experiment = mlflow.set_experiment(
            experiment_name,
            artifact_location=artifact_location
        )
        
        self.client = MlflowClient()
        self.run_id = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Human-readable run name
            tags: Dictionary of tags (e.g., {"model_type": "xgboost"})
        """
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        
        # Log tags
        if tags:
            mlflow.set_tags(tags)
        
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metric values
            step: Training step/epoch (for tracking over time)
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        signature=None,
        input_example=None
    ):
        """
        Log trained model
        
        Args:
            model: Trained model object
            artifact_path: Path within run artifacts
            signature: MLflow model signature
            input_example: Sample input for inference
        """
        # Auto-detect model type and log appropriately
        model_type = type(model).__name__
        
        if 'XGBoost' in model_type or 'xgb' in str(type(model)).lower():
            mlflow.xgboost.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example
            )
        elif 'sklearn' in str(type(model).__module__).lower():
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example
            )
        else:
            # Generic Python model
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                signature=signature,
                input_example=input_example
            )
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log file or directory as artifact
        
        Args:
            local_path: Path to file/directory
            artifact_path: Destination path in MLflow
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_figure(self, figure, artifact_file: str):
        """
        Log matplotlib figure
        
        Args:
            figure: Matplotlib figure object
            artifact_file: Filename (e.g., 'confusion_matrix.png')
        """
        mlflow.log_figure(figure, artifact_file)
    
    def log_confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Log confusion matrix as artifact
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
        """
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        self.log_figure(fig, 'confusion_matrix.png')
        plt.close()
    
    def log_feature_importance(self, model, feature_names):
        """
        Log feature importance plot
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
        """
        import matplotlib.pyplot as plt
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(indices)), importances[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title('Top 20 Feature Importances')
            ax.invert_yaxis()
            
            self.log_figure(fig, 'feature_importance.png')
            plt.close()
    
    def log_training_curves(self, history: Dict[str, list]):
        """
        Log training/validation curves
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', etc.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Validation')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
        
        # Accuracy curve
        if 'train_accuracy' in history:
            axes[1].plot(history['train_accuracy'], label='Train')
            if 'val_accuracy' in history:
                axes[1].plot(history['val_accuracy'], label='Validation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training Accuracy')
            axes[1].legend()
        
        self.log_figure(fig, 'training_curves.png')
        plt.close()
    
    def end_run(self):
        """End the current run"""
        mlflow.end_run()
    
    def compare_runs(self, metric: str = "accuracy", top_n: int = 5) -> pd.DataFrame:
        """
        Compare runs within the experiment
        
        Args:
            metric: Metric to sort by
            top_n: Number of top runs to return
            
        Returns:
            DataFrame with run comparisons
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n
        )
        
        return runs[['run_id', 'start_time', 'params', 'metrics']]
    
    def get_best_run(self, metric: str = "accuracy") -> str:
        """
        Get run ID with best metric value
        
        Args:
            metric: Metric to optimize
            
        Returns:
            Best run ID
        """
        runs = self.compare_runs(metric=metric, top_n=1)
        return runs.iloc[0]['run_id']
    
    def load_model(self, run_id: str, artifact_path: str = "model"):
        """
        Load model from a specific run
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to model artifact
            
        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.pyfunc.load_model(model_uri)


# Example: Complete ML experiment tracking
def example_fraud_detection_with_mlflow():
    """
    Complete example of tracking fraud detection experiment
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Initialize tracker
    tracker = MLflowExperimentTracker(
        experiment_name="fraud-detection",
        artifact_location="s3://my-bucket/mlflow-artifacts"  # Or local path
    )
    
    # Load data (example)
    # X, y = load_fraud_data()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Start run
    tracker.start_run(
        run_name="random-forest-baseline",
        tags={
            "model_type": "random_forest",
            "dataset": "fraud_v2",
            "engineer": "data_science_team"
        }
    )
    
    # Log parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced"
    }
    tracker.log_params(params)
    
    # Train model
    model = RandomForestClassifier(**params, random_state=42)
    # model.fit(X_train, y_train)
    
    # Predictions
    # y_pred = model.predict(X_test)
    
    # Log metrics
    # metrics = {
    #     "accuracy": accuracy_score(y_test, y_pred),
    #     "precision": precision_score(y_test, y_pred),
    #     "recall": recall_score(y_test, y_pred),
    #     "f1": f1_score(y_test, y_pred)
    # }
    # tracker.log_metrics(metrics)
    
    # Log visualizations
    # tracker.log_confusion_matrix(y_test, y_pred, labels=['Normal', 'Fraud'])
    # tracker.log_feature_importance(model, feature_names)
    
    # Log model
    # tracker.log_model(model, signature=signature, input_example=X_test[:5])
    
    # End run
    tracker.end_run()
    
    print("✅ Experiment tracked successfully!")
    print(f"View at: http://localhost:5000 (run `mlflow ui` to start server)")


# MLflow Model Registry functions
def register_model(model_uri: str, model_name: str, description: str = ""):
    """
    Register model in MLflow Model Registry
    
    Args:
        model_uri: URI of the model (e.g., runs:/run_id/model)
        model_name: Name for the registered model
        description: Model description
    """
    result = mlflow.register_model(model_uri, model_name)
    
    # Update description
    client = MlflowClient()
    client.update_registered_model(
        name=model_name,
        description=description
    )
    
    return result


def promote_model_to_production(model_name: str, version: int):
    """
    Promote model version to production
    
    Args:
        model_name: Registered model name
        version: Model version number
    """
    client = MlflowClient()
    
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    
    print(f"✅ Model {model_name} v{version} promoted to Production")


def get_production_model(model_name: str):
    """
    Load current production model
    
    Args:
        model_name: Registered model name
        
    Returns:
        Production model
    """
    model_uri = f"models:/{model_name}/Production"
    return mlflow.pyfunc.load_model(model_uri)
