"""
DVC Integration for Data Version Control

Version datasets, track data pipelines, and manage large files with DVC.
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
import yaml


class DVCManager:
    """
    Wrapper for DVC (Data Version Control) operations
    
    Manages data versioning, pipeline tracking, and remote storage.
    """
    
    def __init__(self, repo_path: str = "."):
        """
        Initialize DVC manager
        
        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)
        self.dvc_dir = self.repo_path / ".dvc"
        
        if not self.dvc_dir.exists():
            print("⚠️  DVC not initialized. Run `dvc init` first.")
    
    def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run DVC command"""
        return subprocess.run(
            command,
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
    
    def init(self):
        """Initialize DVC in the repository"""
        result = self._run_command(["dvc", "init"])
        if result.returncode == 0:
            print("✅ DVC initialized")
        else:
            print(f"❌ Error: {result.stderr}")
    
    def add(self, path: str) -> bool:
        """
        Track file or directory with DVC
        
        Args:
            path: Path to file or directory
            
        Returns:
            Success status
        """
        result = self._run_command(["dvc", "add", path])
        
        if result.returncode == 0:
            print(f"✅ Added {path} to DVC")
            print(f"   Remember to: git add {path}.dvc .gitignore")
            return True
        else:
            print(f"❌ Error adding {path}: {result.stderr}")
            return False
    
    def push(self, remote: Optional[str] = None):
        """
        Push tracked files to remote storage
        
        Args:
            remote: Remote name (default: configured remote)
        """
        cmd = ["dvc", "push"]
        if remote:
            cmd.extend(["-r", remote])
        
        result = self._run_command(cmd)
        
        if result.returncode == 0:
            print("✅ Pushed data to remote storage")
        else:
            print(f"❌ Error: {result.stderr}")
    
    def pull(self, remote: Optional[str] = None):
        """
        Pull tracked files from remote storage
        
        Args:
            remote: Remote name (default: configured remote)
        """
        cmd = ["dvc", "pull"]
        if remote:
            cmd.extend(["-r", remote])
        
        result = self._run_command(cmd)
        
        if result.returncode == 0:
            print("✅ Pulled data from remote storage")
        else:
            print(f"❌ Error: {result.stderr}")
    
    def add_remote(self, name: str, url: str, default: bool = False):
        """
        Configure remote storage
        
        Args:
            name: Remote name (e.g., 's3-storage')
            url: Storage URL (e.g., 's3://bucket/path')
            default: Set as default remote
        """
        # Add remote
        result = self._run_command(["dvc", "remote", "add", name, url])
        
        if result.returncode != 0:
            print(f"❌ Error adding remote: {result.stderr}")
            return
        
        # Set as default
        if default:
            self._run_command(["dvc", "remote", "default", name])
        
        print(f"✅ Added remote '{name}': {url}")
        
        # For S3, remind about credentials
        if url.startswith("s3://"):
            print("\n💡 S3 credentials:")
            print("   Option 1: AWS CLI credentials (recommended)")
            print("   Option 2: dvc remote modify --local s3-storage access_key_id YOUR_KEY")
            print("   Option 3: IAM role for SageMaker/EC2")
    
    def status(self) -> Dict:
        """
        Check status of tracked files
        
        Returns:
            Status information
        """
        result = self._run_command(["dvc", "status", "--json"])
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"❌ Error: {result.stderr}")
            return {}
    
    def create_pipeline(
        self,
        name: str,
        command: str,
        deps: List[str],
        outputs: List[str],
        params: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Create DVC pipeline stage
        
        Args:
            name: Stage name
            command: Command to run
            deps: List of dependencies (files/dirs)
            outputs: List of output files/dirs
            params: Parameter files (e.g., params.yaml)
            metrics: Metric files (e.g., metrics.json)
        """
        cmd = [
            "dvc", "stage", "add",
            "-n", name,
            "-d", *deps,
            "-o", *outputs
        ]
        
        if params:
            for param in params:
                cmd.extend(["-p", param])
        
        if metrics:
            for metric in metrics:
                cmd.extend(["-M", metric])
        
        cmd.append(command)
        
        result = self._run_command(cmd)
        
        if result.returncode == 0:
            print(f"✅ Added pipeline stage: {name}")
        else:
            print(f"❌ Error: {result.stderr}")
    
    def run_pipeline(self):
        """Run DVC pipeline"""
        result = self._run_command(["dvc", "repro"])
        
        if result.returncode == 0:
            print("✅ Pipeline executed successfully")
        else:
            print(f"❌ Error: {result.stderr}")
    
    def show_metrics(self) -> Dict:
        """
        Show metrics from all experiments
        
        Returns:
            Metrics dictionary
        """
        result = self._run_command(["dvc", "metrics", "show", "--json"])
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"❌ Error: {result.stderr}")
            return {}
    
    def compare_experiments(self, experiments: Optional[List[str]] = None):
        """
        Compare experiments
        
        Args:
            experiments: List of experiment names (default: all)
        """
        cmd = ["dvc", "exp", "show", "--json"]
        
        if experiments:
            cmd.extend(experiments)
        
        result = self._run_command(cmd)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Pretty print comparison
            self._print_experiment_table(data)
        else:
            print(f"❌ Error: {result.stderr}")
    
    def _print_experiment_table(self, data: Dict):
        """Print experiments in table format"""
        import pandas as pd
        
        rows = []
        for exp_name, exp_data in data.items():
            metrics = exp_data.get('metrics', {})
            params = exp_data.get('params', {})
            
            row = {'experiment': exp_name}
            row.update(metrics)
            row.update(params)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        print("\n📊 Experiment Comparison:")
        print(df.to_string(index=False))


# Example: Complete DVC setup for fraud detection project
def setup_fraud_detection_dvc():
    """
    Example: Setup DVC for fraud detection project
    """
    dvc = DVCManager()
    
    # Initialize DVC
    dvc.init()
    
    # Add S3 remote storage
    dvc.add_remote(
        name="s3-storage",
        url="s3://my-ml-bucket/fraud-detection-data",
        default=True
    )
    
    # Track raw data
    print("\n📦 Tracking datasets...")
    dvc.add("data/raw/transactions.csv")
    dvc.add("data/processed/")
    
    # Create data processing pipeline
    print("\n🔧 Creating pipeline...")
    
    # Stage 1: Data preprocessing
    dvc.create_pipeline(
        name="preprocess",
        command="python scripts/preprocess.py",
        deps=["data/raw/transactions.csv", "scripts/preprocess.py"],
        outputs=["data/processed/train.csv", "data/processed/test.csv"],
        params=["params.yaml:preprocess"]
    )
    
    # Stage 2: Feature engineering
    dvc.create_pipeline(
        name="features",
        command="python scripts/feature_engineering.py",
        deps=["data/processed/train.csv", "scripts/feature_engineering.py"],
        outputs=["data/features/X_train.pkl", "data/features/y_train.pkl"],
        params=["params.yaml:features"]
    )
    
    # Stage 3: Model training
    dvc.create_pipeline(
        name="train",
        command="python scripts/train.py",
        deps=["data/features/X_train.pkl", "data/features/y_train.pkl", "scripts/train.py"],
        outputs=["models/fraud_model.pkl"],
        params=["params.yaml:train"],
        metrics=["metrics/train_metrics.json"]
    )
    
    # Stage 4: Model evaluation
    dvc.create_pipeline(
        name="evaluate",
        command="python scripts/evaluate.py",
        deps=["models/fraud_model.pkl", "data/processed/test.csv"],
        outputs=["results/predictions.csv"],
        metrics=["metrics/eval_metrics.json"]
    )
    
    print("\n✅ DVC pipeline created!")
    print("\n📝 Next steps:")
    print("   1. Create params.yaml with hyperparameters")
    print("   2. Run pipeline: dvc repro")
    print("   3. Track changes: git add dvc.yaml dvc.lock")
    print("   4. Push data: dvc push")


# Example: params.yaml structure
EXAMPLE_PARAMS_YAML = """
preprocess:
  test_size: 0.2
  random_state: 42
  handle_missing: 'mean'

features:
  scaling_method: 'standard'
  create_interactions: true
  polynomial_degree: 2

train:
  model_type: 'xgboost'
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  class_weight: 'balanced'
  random_state: 42
"""


def create_example_params():
    """Create example params.yaml file"""
    with open("params.yaml", "w") as f:
        f.write(EXAMPLE_PARAMS_YAML)
    print("✅ Created params.yaml")


# Git integration helpers
def dvc_git_workflow():
    """
    Example workflow combining DVC and Git
    """
    print("""
    🔄 DVC + Git Workflow:
    
    1. Track data with DVC:
       $ dvc add data/transactions.csv
       $ git add data/transactions.csv.dvc .gitignore
       $ git commit -m "Track transactions dataset"
    
    2. Make changes to data or code:
       $ python preprocess.py
       $ dvc add data/processed/
    
    3. Run pipeline and track:
       $ dvc repro
       $ git add dvc.lock metrics/
       $ git commit -m "Update model with new hyperparameters"
    
    4. Push everything:
       $ git push origin main
       $ dvc push
    
    5. Collaborator pulls:
       $ git pull origin main
       $ dvc pull
    
    6. Compare experiments:
       $ dvc exp show
       $ dvc metrics diff
    """)
