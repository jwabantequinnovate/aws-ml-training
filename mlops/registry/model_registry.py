"""
Model Registry Management for SageMaker

This module provides utilities for managing models in SageMaker Model Registry,
including registration, versioning, approval workflows, and lineage tracking.
"""

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from datetime import datetime
from typing import Dict, List, Optional
import json


class ModelRegistry:
    """
    Manage models in SageMaker Model Registry
    """
    
    def __init__(self, role: str = None):
        """
        Initialize Model Registry manager
        
        Args:
            role: IAM role for SageMaker (auto-detected if None)
        """
        self.session = sagemaker.Session()
        self.region = boto3.Session().region_name
        
        if role is None:
            self.role = sagemaker.get_execution_role()
        else:
            self.role = role
        
        self.sagemaker_client = boto3.client('sagemaker')
    
    def create_model_package_group(
        self,
        group_name: str,
        description: str = None
    ) -> Dict:
        """
        Create a model package group
        
        Args:
            group_name: Name for the model package group
            description: Description of the model package group
            
        Returns:
            Model package group ARN
        """
        try:
            response = self.sagemaker_client.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription=description or f"Model package group for {group_name}"
            )
            print(f"✓ Model package group created: {group_name}")
            return response
        except self.sagemaker_client.exceptions.ResourceInUse:
            print(f"Model package group already exists: {group_name}")
            response = self.sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            return response
    
    def register_model(
        self,
        model_package_group_name: str,
        model_data: str,
        image_uri: str,
        model_metrics: Dict = None,
        inference_instances: List[str] = None,
        transform_instances: List[str] = None,
        approval_status: str = 'PendingManualApproval',
        description: str = None,
        tags: Dict = None
    ) -> Dict:
        """
        Register a model in the model registry
        
        Args:
            model_package_group_name: Model package group name
            model_data: S3 URI of model artifacts
            image_uri: Container image URI
            model_metrics: Dictionary of model metrics
            inference_instances: List of instance types for inference
            transform_instances: List of instance types for batch transform
            approval_status: Approval status (PendingManualApproval, Approved, Rejected)
            description: Model description
            tags: Dictionary of tags
            
        Returns:
            Model package ARN
        """
        if inference_instances is None:
            inference_instances = ['ml.m5.xlarge', 'ml.m5.2xlarge']
        
        if transform_instances is None:
            transform_instances = ['ml.m5.xlarge']
        
        # Prepare model metrics
        model_metrics_dict = {}
        if model_metrics:
            model_metrics_dict = {
                'ModelQuality': {
                    'Statistics': {
                        'ContentType': 'application/json',
                        'S3Uri': model_metrics.get('statistics_s3_uri', '')
                    }
                }
            }
        
        # Prepare tags
        tag_list = []
        if tags:
            tag_list = [{'Key': k, 'Value': str(v)} for k, v in tags.items()]
        
        # Register model
        response = self.sagemaker_client.create_model_package(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageDescription=description or f"Model registered at {datetime.now().isoformat()}",
            InferenceSpecification={
                'Containers': [
                    {
                        'Image': image_uri,
                        'ModelDataUrl': model_data
                    }
                ],
                'SupportedContentTypes': ['application/json', 'text/csv'],
                'SupportedResponseMIMETypes': ['application/json'],
                'SupportedRealtimeInferenceInstanceTypes': inference_instances,
                'SupportedTransformInstanceTypes': transform_instances
            },
            ModelApprovalStatus=approval_status,
            ModelMetrics=model_metrics_dict if model_metrics_dict else None,
            Tags=tag_list
        )
        
        model_package_arn = response['ModelPackageArn']
        print(f"✓ Model registered: {model_package_arn}")
        
        return response
    
    def update_model_approval_status(
        self,
        model_package_arn: str,
        approval_status: str,
        approval_description: str = None
    ) -> Dict:
        """
        Update model approval status
        
        Args:
            model_package_arn: Model package ARN
            approval_status: New approval status (Approved, Rejected)
            approval_description: Description for approval/rejection
            
        Returns:
            Update response
        """
        response = self.sagemaker_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus=approval_status,
            ApprovalDescription=approval_description
        )
        
        print(f"✓ Model approval status updated to: {approval_status}")
        return response
    
    def list_model_packages(
        self,
        model_package_group_name: str,
        approval_status: str = None,
        sort_by: str = 'CreationTime',
        sort_order: str = 'Descending',
        max_results: int = 10
    ) -> List[Dict]:
        """
        List model packages in a group
        
        Args:
            model_package_group_name: Model package group name
            approval_status: Filter by approval status
            sort_by: Sort field
            sort_order: Sort order (Ascending, Descending)
            max_results: Maximum results to return
            
        Returns:
            List of model packages
        """
        params = {
            'ModelPackageGroupName': model_package_group_name,
            'SortBy': sort_by,
            'SortOrder': sort_order,
            'MaxResults': max_results
        }
        
        if approval_status:
            params['ModelApprovalStatus'] = approval_status
        
        response = self.sagemaker_client.list_model_packages(**params)
        return response['ModelPackageSummaryList']
    
    def get_latest_approved_model(
        self,
        model_package_group_name: str
    ) -> Optional[Dict]:
        """
        Get the latest approved model from a group
        
        Args:
            model_package_group_name: Model package group name
            
        Returns:
            Latest approved model package or None
        """
        models = self.list_model_packages(
            model_package_group_name=model_package_group_name,
            approval_status='Approved',
            sort_by='CreationTime',
            sort_order='Descending',
            max_results=1
        )
        
        if models:
            return models[0]
        return None
    
    def describe_model_package(
        self,
        model_package_arn: str
    ) -> Dict:
        """
        Get detailed information about a model package
        
        Args:
            model_package_arn: Model package ARN
            
        Returns:
            Model package details
        """
        response = self.sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )
        return response
    
    def compare_model_versions(
        self,
        model_package_group_name: str,
        version1_arn: str,
        version2_arn: str
    ) -> Dict:
        """
        Compare two model versions
        
        Args:
            model_package_group_name: Model package group name
            version1_arn: First model version ARN
            version2_arn: Second model version ARN
            
        Returns:
            Comparison dictionary
        """
        model1 = self.describe_model_package(version1_arn)
        model2 = self.describe_model_package(version2_arn)
        
        comparison = {
            'model_package_group': model_package_group_name,
            'version1': {
                'arn': version1_arn,
                'status': model1['ModelApprovalStatus'],
                'created': model1['CreationTime'].isoformat(),
                'metrics': model1.get('ModelMetrics', {})
            },
            'version2': {
                'arn': version2_arn,
                'status': model2['ModelApprovalStatus'],
                'created': model2['CreationTime'].isoformat(),
                'metrics': model2.get('ModelMetrics', {})
            }
        }
        
        return comparison
    
    def deploy_model_from_registry(
        self,
        model_package_arn: str,
        endpoint_name: str,
        instance_type: str = 'ml.m5.xlarge',
        instance_count: int = 1
    ) -> str:
        """
        Deploy a model from the registry to an endpoint
        
        Args:
            model_package_arn: Model package ARN
            endpoint_name: Name for the endpoint
            instance_type: Instance type
            instance_count: Number of instances
            
        Returns:
            Endpoint name
        """
        model = Model(
            model_data=model_package_arn,
            role=self.role,
            sagemaker_session=self.session
        )
        
        print(f"Deploying model to endpoint: {endpoint_name}")
        
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name
        )
        
        print(f"✓ Model deployed to: {endpoint_name}")
        return endpoint_name
    
    def create_model_lineage(
        self,
        model_package_arn: str,
        training_job_arn: str = None,
        dataset_arns: List[str] = None
    ) -> None:
        """
        Create lineage associations for a model
        
        Args:
            model_package_arn: Model package ARN
            training_job_arn: Training job ARN
            dataset_arns: List of dataset ARNs
        """
        lineage = boto3.client('sagemaker')
        
        # Associate with training job
        if training_job_arn:
            lineage.add_association(
                SourceArn=training_job_arn,
                DestinationArn=model_package_arn,
                AssociationType='DerivedFrom'
            )
            print(f"✓ Associated with training job")
        
        # Associate with datasets
        if dataset_arns:
            for dataset_arn in dataset_arns:
                lineage.add_association(
                    SourceArn=dataset_arn,
                    DestinationArn=model_package_arn,
                    AssociationType='DerivedFrom'
                )
            print(f"✓ Associated with {len(dataset_arns)} datasets")
    
    def delete_model_package(
        self,
        model_package_arn: str
    ) -> None:
        """
        Delete a model package
        
        Args:
            model_package_arn: Model package ARN
        """
        self.sagemaker_client.delete_model_package(
            ModelPackageName=model_package_arn
        )
        print(f"✓ Model package deleted: {model_package_arn}")


def example_workflow():
    """
    Example workflow for model registry
    """
    registry = ModelRegistry()
    
    # 1. Create model package group
    print("Step 1: Create model package group")
    registry.create_model_package_group(
        group_name='fraud-detection-models',
        description='Fraud detection model versions'
    )
    
    # 2. Register a model
    print("\nStep 2: Register model")
    model_metrics = {
        'statistics_s3_uri': 's3://bucket/metrics/statistics.json'
    }
    
    response = registry.register_model(
        model_package_group_name='fraud-detection-models',
        model_data='s3://bucket/models/model.tar.gz',
        image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn:1.2-1',
        model_metrics=model_metrics,
        approval_status='PendingManualApproval',
        description='XGBoost fraud detection model v1',
        tags={'version': '1.0', 'algorithm': 'xgboost'}
    )
    
    model_package_arn = response['ModelPackageArn']
    
    # 3. List models
    print("\nStep 3: List models in group")
    models = registry.list_model_packages(
        model_package_group_name='fraud-detection-models'
    )
    print(f"Found {len(models)} models")
    
    # 4. Approve model
    print("\nStep 4: Approve model")
    registry.update_model_approval_status(
        model_package_arn=model_package_arn,
        approval_status='Approved',
        approval_description='Model passed validation tests'
    )
    
    # 5. Get latest approved model
    print("\nStep 5: Get latest approved model")
    latest_model = registry.get_latest_approved_model(
        model_package_group_name='fraud-detection-models'
    )
    if latest_model:
        print(f"Latest approved: {latest_model['ModelPackageArn']}")
    
    # 6. Deploy approved model
    print("\nStep 6: Deploy model")
    registry.deploy_model_from_registry(
        model_package_arn=model_package_arn,
        endpoint_name='fraud-detection-endpoint',
        instance_type='ml.m5.xlarge',
        instance_count=1
    )
    
    print("\n✓ Model registry workflow complete!")


if __name__ == '__main__':
    example_workflow()
