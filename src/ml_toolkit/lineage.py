"""
Model Lineage Tracking with SageMaker

Track the complete lifecycle of ML models from data to deployment.
"""

import boto3
from sagemaker.lineage import context, artifact, association, action
from sagemaker.lineage.visualizer import LineageTableVisualizer
from datetime import datetime
from typing import Dict, List, Optional


class ModelLineageTracker:
    """
    Track model lineage throughout the ML lifecycle
    
    Captures:
    - Training data versions
    - Code versions
    - Hyperparameters
    - Model artifacts
    - Evaluation metrics
    - Deployment history
    """
    
    def __init__(self, sagemaker_session=None):
        """Initialize lineage tracker"""
        from sagemaker import Session
        
        self.session = sagemaker_session or Session()
        self.sm_client = boto3.client('sagemaker')
    
    def create_training_context(
        self,
        model_name: str,
        description: str = ""
    ) -> str:
        """
        Create a context for tracking training lineage
        
        Returns:
            Context ARN
        """
        context_name = f"{model_name}-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        training_context = context.Context.create(
            context_name=context_name,
            source_uri=f"training/{model_name}",
            context_type="Training",
            description=description or f"Training context for {model_name}",
            properties={
                "model_name": model_name,
                "created_at": datetime.now().isoformat()
            }
        )
        
        return training_context.context_arn
    
    def track_dataset(
        self,
        dataset_uri: str,
        dataset_name: str,
        context_arn: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Track dataset used for training
        
        Args:
            dataset_uri: S3 URI of dataset
            dataset_name: Human-readable name
            context_arn: Training context ARN
            metadata: Additional metadata (rows, columns, etc.)
        """
        dataset_artifact = artifact.Artifact.create(
            artifact_name=f"dataset-{dataset_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            source_uri=dataset_uri,
            artifact_type="Dataset",
            properties=metadata or {}
        )
        
        # Associate with context
        association.Association.create(
            source_arn=dataset_artifact.artifact_arn,
            destination_arn=context_arn,
            association_type="ContributedTo"
        )
        
        return dataset_artifact.artifact_arn
    
    def track_model_artifact(
        self,
        model_uri: str,
        model_name: str,
        context_arn: str,
        metrics: Optional[Dict] = None
    ) -> str:
        """
        Track trained model artifact
        
        Args:
            model_uri: S3 URI of model.tar.gz
            model_name: Model identifier
            context_arn: Training context ARN
            metrics: Model performance metrics
        """
        model_artifact = artifact.Artifact.create(
            artifact_name=f"model-{model_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            source_uri=model_uri,
            artifact_type="Model",
            properties={
                "model_name": model_name,
                "metrics": metrics or {},
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Associate with training context
        association.Association.create(
            source_arn=context_arn,
            destination_arn=model_artifact.artifact_arn,
            association_type="Produced"
        )
        
        return model_artifact.artifact_arn
    
    def track_training_job(
        self,
        training_job_name: str,
        context_arn: str
    ) -> str:
        """
        Link SageMaker training job to context
        
        Automatically captures:
        - Hyperparameters
        - Instance types
        - Training duration
        - Final metrics
        """
        # Get training job details
        job_details = self.sm_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        
        # Create training action
        training_action = action.Action.create(
            action_name=f"training-{training_job_name}",
            source_uri=job_details['TrainingJobArn'],
            action_type="Training",
            properties={
                "training_job_name": training_job_name,
                "hyperparameters": job_details.get('HyperParameters', {}),
                "instance_type": job_details['ResourceConfig']['InstanceType'],
                "instance_count": job_details['ResourceConfig']['InstanceCount'],
                "training_time_seconds": job_details.get('TrainingTimeInSeconds', 0),
                "billable_time_seconds": job_details.get('BillableTimeInSeconds', 0),
                "final_metrics": {
                    m['MetricName']: m['Value'] 
                    for m in job_details.get('FinalMetricDataList', [])
                }
            }
        )
        
        # Associate with context
        association.Association.create(
            source_arn=training_action.action_arn,
            destination_arn=context_arn,
            association_type="AssociatedWith"
        )
        
        return training_action.action_arn
    
    def track_deployment(
        self,
        endpoint_name: str,
        model_artifact_arn: str
    ) -> str:
        """
        Track model deployment to endpoint
        
        Args:
            endpoint_name: SageMaker endpoint name
            model_artifact_arn: Model artifact ARN
        """
        # Get endpoint details
        endpoint_details = self.sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        
        # Create deployment action
        deployment_action = action.Action.create(
            action_name=f"deployment-{endpoint_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            source_uri=endpoint_details['EndpointArn'],
            action_type="Deployment",
            properties={
                "endpoint_name": endpoint_name,
                "endpoint_status": endpoint_details['EndpointStatus'],
                "instance_type": endpoint_details['ProductionVariants'][0]['InstanceType'],
                "deployed_at": datetime.now().isoformat()
            }
        )
        
        # Link model to deployment
        association.Association.create(
            source_arn=model_artifact_arn,
            destination_arn=deployment_action.action_arn,
            association_type="DerivedFrom"
        )
        
        return deployment_action.action_arn
    
    def get_model_lineage(
        self,
        model_artifact_arn: str,
        max_depth: int = 10
    ) -> Dict:
        """
        Get complete lineage for a model
        
        Returns lineage graph showing:
        - Source datasets
        - Training jobs
        - Hyperparameters
        - Metrics
        - Deployments
        """
        lineage = {
            "model_arn": model_artifact_arn,
            "datasets": [],
            "training_jobs": [],
            "deployments": [],
            "metrics": {}
        }
        
        # Query lineage
        query = {
            "StartArns": [model_artifact_arn],
            "Direction": "Ascendants",  # Go backwards to find sources
            "IncludeEdges": True,
            "MaxDepth": max_depth
        }
        
        response = self.sm_client.query_lineage(**query)
        
        # Parse results
        for vertex in response.get('Vertices', []):
            if vertex['Type'] == 'Dataset':
                lineage['datasets'].append({
                    'arn': vertex['Arn'],
                    'source_uri': vertex.get('SourceUri', ''),
                    'properties': vertex.get('Properties', {})
                })
            elif vertex['Type'] == 'TrainingJob':
                lineage['training_jobs'].append({
                    'arn': vertex['Arn'],
                    'name': vertex.get('Properties', {}).get('training_job_name'),
                    'hyperparameters': vertex.get('Properties', {}).get('hyperparameters', {}),
                    'metrics': vertex.get('Properties', {}).get('final_metrics', {})
                })
        
        # Get forward lineage (deployments)
        forward_query = {
            "StartArns": [model_artifact_arn],
            "Direction": "Descendants",
            "MaxDepth": max_depth
        }
        
        forward_response = self.sm_client.query_lineage(**forward_query)
        
        for vertex in forward_response.get('Vertices', []):
            if vertex['Type'] == 'Deployment':
                lineage['deployments'].append({
                    'arn': vertex['Arn'],
                    'endpoint': vertex.get('Properties', {}).get('endpoint_name'),
                    'deployed_at': vertex.get('Properties', {}).get('deployed_at')
                })
        
        return lineage
    
    def visualize_lineage(
        self,
        model_artifact_arn: str,
        output_path: str = "lineage.html"
    ):
        """
        Generate HTML visualization of model lineage
        
        Creates an interactive graph showing the full model lifecycle.
        """
        try:
            visualizer = LineageTableVisualizer(self.session)
            
            # Generate visualization
            visualizer_df = visualizer.show(
                start_arn=model_artifact_arn,
                direction="Both",
                max_depth=10
            )
            
            # Save to HTML
            visualizer_df.to_html(output_path)
            print(f"✅ Lineage visualization saved to {output_path}")
            
        except Exception as e:
            print(f"⚠️  Could not generate visualization: {e}")
            print("Lineage data available via get_model_lineage() method")


# Example usage
def example_track_full_lifecycle():
    """
    Complete example of tracking model lifecycle
    """
    tracker = ModelLineageTracker()
    
    # 1. Create training context
    context_arn = tracker.create_training_context(
        model_name="fraud-detection-v2",
        description="Fraud detection model with improved features"
    )
    
    # 2. Track dataset
    dataset_arn = tracker.track_dataset(
        dataset_uri="s3://bucket/data/fraud_transactions.csv",
        dataset_name="fraud-transactions",
        context_arn=context_arn,
        metadata={
            "rows": 100000,
            "fraud_ratio": 0.01,
            "date_range": "2023-01-01 to 2023-12-31"
        }
    )
    
    # 3. Track training job
    training_arn = tracker.track_training_job(
        training_job_name="fraud-detection-training-2024-01-15",
        context_arn=context_arn
    )
    
    # 4. Track model artifact
    model_arn = tracker.track_model_artifact(
        model_uri="s3://bucket/models/fraud-model-v2.tar.gz",
        model_name="fraud-detection-v2",
        context_arn=context_arn,
        metrics={
            "accuracy": 0.95,
            "precision": 0.88,
            "recall": 0.92,
            "auc": 0.96
        }
    )
    
    # 5. Track deployment
    deployment_arn = tracker.track_deployment(
        endpoint_name="fraud-detection-endpoint",
        model_artifact_arn=model_arn
    )
    
    # 6. Retrieve and visualize lineage
    lineage = tracker.get_model_lineage(model_arn)
    print("Model Lineage:", lineage)
    
    tracker.visualize_lineage(model_arn)
