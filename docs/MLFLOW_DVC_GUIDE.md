# MLflow & DVC Integration Examples

Comprehensive examples for experiment tracking and data versioning.

## MLflow - Experiment Tracking

### Quick Start

```bash
# Install (already in pyproject.toml)
poetry add mlflow

# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

### Track Your First Experiment

```python
from ml_toolkit.mlflow_tracking import MLflowExperimentTracker

# Initialize
tracker = MLflowExperimentTracker(
    experiment_name="fraud-detection-experiments",
    tracking_uri="http://localhost:5000",  # Or leave empty for local
    artifact_location="s3://my-bucket/mlflow"  # Optional S3 storage
)

# Start run
tracker.start_run(
    run_name="xgboost-baseline",
    tags={"model_type": "xgboost", "dataset": "v2"}
)

# Log everything
tracker.log_params({
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1
})

tracker.log_metrics({
    "accuracy": 0.95,
    "precision": 0.88,
    "recall": 0.92,
    "f1": 0.90
})

tracker.log_model(model)
tracker.log_confusion_matrix(y_test, y_pred)
tracker.log_feature_importance(model, feature_names)

tracker.end_run()
```

### Compare Experiments

```python
# Compare all runs
comparison = tracker.compare_runs(metric="f1", top_n=10)
print(comparison)

# Get best model
best_run_id = tracker.get_best_run(metric="f1")
best_model = tracker.load_model(best_run_id)
```

### Model Registry

```python
from ml_toolkit.mlflow_tracking import register_model, promote_model_to_production

# Register model
register_model(
    model_uri=f"runs:/{run_id}/model",
    model_name="fraud-detector",
    description="XGBoost fraud detection model"
)

# Promote to production
promote_model_to_production("fraud-detector", version=3)

# Load production model
from ml_toolkit.mlflow_tracking import get_production_model
prod_model = get_production_model("fraud-detector")
```

## DVC - Data Version Control

### Quick Start

```bash
# Initialize DVC
dvc init

# Configure S3 remote
dvc remote add -d s3storage s3://my-bucket/dvc-cache
dvc remote modify s3storage region us-east-1

# Track data
dvc add data/transactions.csv
git add data/transactions.csv.dvc .gitignore
git commit -m "Track transactions dataset"

# Push to S3
dvc push
```

### Using DVC Manager

```python
from ml_toolkit.dvc_manager import DVCManager

dvc = DVCManager()

# Track datasets
dvc.add("data/raw/transactions.csv")
dvc.add("data/processed/")

# Configure S3 remote
dvc.add_remote(
    name="s3storage",
    url="s3://my-ml-bucket/dvc-data",
    default=True
)

# Push/pull data
dvc.push()
dvc.pull()
```

### Create ML Pipeline

Create `params.yaml`:

```yaml
preprocess:
  test_size: 0.2
  random_state: 42

train:
  model_type: xgboost
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
```

Define pipeline in code:

```python
dvc = DVCManager()

# Stage 1: Preprocess
dvc.create_pipeline(
    name="preprocess",
    command="python scripts/preprocess.py",
    deps=["data/raw/transactions.csv", "scripts/preprocess.py"],
    outputs=["data/processed/train.csv", "data/processed/test.csv"],
    params=["params.yaml:preprocess"]
)

# Stage 2: Train
dvc.create_pipeline(
    name="train",
    command="python scripts/train.py",
    deps=["data/processed/train.csv", "scripts/train.py"],
    outputs=["models/model.pkl"],
    params=["params.yaml:train"],
    metrics=["metrics/train.json"]
)

# Run pipeline
dvc.run_pipeline()
```

Or use `dvc.yaml`:

```yaml
stages:
  preprocess:
    cmd: python scripts/preprocess.py
    deps:
      - data/raw/transactions.csv
      - scripts/preprocess.py
    params:
      - preprocess
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
  
  train:
    cmd: python scripts/train.py
    deps:
      - data/processed/train.csv
      - scripts/train.py
    params:
      - train
    outs:
      - models/model.pkl
    metrics:
      - metrics/train.json:
          cache: false
```

Run with:

```bash
dvc repro
```

### Experiment Tracking with DVC

```bash
# Run experiment
dvc exp run

# Run with different params
dvc exp run --set-param train.learning_rate=0.05

# Compare experiments
dvc exp show

# Apply best experiment
dvc exp apply exp-abc123
```

### Compare Metrics

```python
dvc = DVCManager()

# Show all metrics
metrics = dvc.show_metrics()
print(metrics)

# Compare experiments
dvc.compare_experiments()
```

## Combined MLflow + DVC Workflow

Use DVC for data/pipeline management and MLflow for experiment tracking:

```python
from ml_toolkit.mlflow_tracking import MLflowExperimentTracker
from ml_toolkit.dvc_manager import DVCManager

# Setup DVC
dvc = DVCManager()
dvc.add("data/raw/")
dvc.push()

# Setup MLflow
tracker = MLflowExperimentTracker("fraud-detection")

# Train with both
tracker.start_run(run_name="experiment-1")

# DVC tracks data versions
dvc.run_pipeline()

# MLflow tracks model and metrics
tracker.log_params(params)
tracker.log_metrics(metrics)
tracker.log_model(model)

tracker.end_run()

# Commit everything
# git add dvc.lock params.yaml
# git commit -m "Experiment 1: baseline model"
# dvc push  # Data to S3
```

## Best Practices

### DVC
- Track large files (>10MB) with DVC, not Git
- Use `.dvcignore` for temporary files
- Regular `dvc push` to backup data
- Use `params.yaml` for all hyperparameters
- Track metrics in JSON for easy comparison

### MLflow
- Consistent naming conventions for experiments
- Always log hyperparameters, even defaults
- Use tags for organization (model_type, dataset_version)
- Log artifacts (plots, configs, preprocessors)
- Use Model Registry for production models

### Integration
- DVC for data + code → MLflow for results
- DVC pipeline → MLflow auto-logging
- Git commits link DVC and MLflow versions
- Use both for complete reproducibility

## Remote Storage Setup

### S3 for DVC

```bash
# Configure
dvc remote add -d s3storage s3://bucket/dvc-cache
dvc remote modify s3storage region us-east-1

# Use AWS credentials (automatic with AWS CLI configured)
# Or set explicitly:
dvc remote modify --local s3storage access_key_id YOUR_KEY
dvc remote modify --local s3storage secret_access_key YOUR_SECRET
```

### S3 for MLflow

```python
tracker = MLflowExperimentTracker(
    experiment_name="my-experiment",
    artifact_location="s3://bucket/mlflow-artifacts"
)
```

Or configure MLflow server:

```bash
mlflow server \
    --backend-store-uri postgresql://user:pass@host/db \
    --default-artifact-root s3://bucket/mlflow-artifacts \
    --host 0.0.0.0
```

## Troubleshooting

### DVC
```bash
# Check status
dvc status

# Verify remote connection
dvc remote list

# Force push
dvc push --force

# Pull specific file
dvc pull data/file.csv.dvc
```

### MLflow
```bash
# Check tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Clean local cache
rm -rf mlruns/

# Check registered models
mlflow models list
```
