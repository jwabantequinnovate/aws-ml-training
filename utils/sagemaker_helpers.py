"""
SageMaker helper utilities

Common functions for working with AWS SageMaker across all modules.
"""

import boto3
import sagemaker
from sagemaker import get_execution_role
import json
import time
from typing import Dict, Optional, List
import pandas as pd


class SageMakerHelper:
    """
    Helper class for SageMaker operations
    """
    
    def __init__(self, role: Optional[str] = None):
        """
        Initialize SageMaker helper
        
        Args:
            role: IAM role for SageMaker (auto-detected if None)
        """
        self.session = sagemaker.Session()
        self.boto_session = boto3.Session()
        self.region = self.boto_session.region_name
        
        try:
            self.role = role or get_execution_role()
        except ValueError:
            # Not running in SageMaker environment
            self.role = role
            
        self.bucket = self.session.default_bucket()
        self.sagemaker_client = boto3.client('sagemaker')
        self.runtime_client = boto3.client('sagemaker-runtime')
    
    def upload_to_s3(
        self, 
        local_path: str, 
        s3_prefix: str
    ) -> str:
        """
        Upload file to S3
        
        Args:
            local_path: Local file path
            s3_prefix: S3 prefix
            
        Returns:
            S3 URI
        """
        s3_uri = self.session.upload_data(
            path=local_path,
            bucket=self.bucket,
            key_prefix=s3_prefix
        )
        return s3_uri
    
    def download_from_s3(
        self, 
        s3_uri: str, 
        local_path: str
    ) -> None:
        """
        Download file from S3
        
        Args:
            s3_uri: S3 URI
            local_path: Local destination path
        """
        self.session.download_data(
            path=local_path,
            bucket=s3_uri.split('/')[2],
            key_prefix='/'.join(s3_uri.split('/')[3:])
        )
    
    def create_model(
        self,
        model_name: str,
        model_data: str,
        image_uri: str,
        role: Optional[str] = None
    ) -> Dict:
        """
        Create SageMaker model
        
        Args:
            model_name: Name for the model
            model_data: S3 URI of model artifacts
            image_uri: Docker image URI
            role: IAM role (uses default if None)
            
        Returns:
            Model creation response
        """
        role = role or self.role
        
        response = self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data
            },
            ExecutionRoleArn=role
        )
        
        return response
    
    def create_endpoint_config(
        self,
        config_name: str,
        model_name: str,
        instance_type: str = 'ml.m5.xlarge',
        instance_count: int = 1,
        variant_name: str = 'AllTraffic'
    ) -> Dict:
        """
        Create endpoint configuration
        
        Args:
            config_name: Configuration name
            model_name: Model name
            instance_type: Instance type
            instance_count: Number of instances
            variant_name: Variant name
            
        Returns:
            Configuration creation response
        """
        response = self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': variant_name,
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': instance_count,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        return response
    
    def create_endpoint(
        self,
        endpoint_name: str,
        config_name: str,
        wait: bool = True
    ) -> Dict:
        """
        Create SageMaker endpoint
        
        Args:
            endpoint_name: Endpoint name
            config_name: Configuration name
            wait: Whether to wait for endpoint creation
            
        Returns:
            Endpoint creation response
        """
        response = self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        if wait:
            print(f"Creating endpoint {endpoint_name}...")
            self.wait_for_endpoint(endpoint_name)
        
        return response
    
    def wait_for_endpoint(
        self,
        endpoint_name: str,
        max_wait_time: int = 1800
    ) -> None:
        """
        Wait for endpoint to be in service
        
        Args:
            endpoint_name: Endpoint name
            max_wait_time: Maximum wait time in seconds
        """
        start_time = time.time()
        
        while True:
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            status = response['EndpointStatus']
            
            if status == 'InService':
                print(f"✓ Endpoint {endpoint_name} is in service")
                break
            elif status == 'Failed':
                raise Exception(f"Endpoint creation failed: {response['FailureReason']}")
            
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                raise Exception(f"Endpoint creation timed out after {max_wait_time}s")
            
            print(f"Status: {status}, elapsed: {int(elapsed_time)}s")
            time.sleep(30)
    
    def invoke_endpoint(
        self,
        endpoint_name: str,
        payload: Dict,
        content_type: str = 'application/json'
    ) -> Dict:
        """
        Invoke SageMaker endpoint
        
        Args:
            endpoint_name: Endpoint name
            payload: Input data
            content_type: Content type
            
        Returns:
            Prediction response
        """
        response = self.runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        return result
    
    def delete_endpoint(
        self,
        endpoint_name: str,
        delete_config: bool = True,
        delete_model: bool = True
    ) -> None:
        """
        Delete endpoint and optionally its configuration and model
        
        Args:
            endpoint_name: Endpoint name
            delete_config: Whether to delete endpoint config
            delete_model: Whether to delete model
        """
        # Get endpoint details
        try:
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            config_name = response['EndpointConfigName']
        except self.sagemaker_client.exceptions.ClientError as e:
            print(f"Endpoint {endpoint_name} not found: {e}")
            return
        
        # Delete endpoint
        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✓ Endpoint {endpoint_name} deleted")
        
        # Delete endpoint config
        if delete_config:
            try:
                config_response = self.sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=config_name
                )
                model_name = config_response['ProductionVariants'][0]['ModelName']
                
                self.sagemaker_client.delete_endpoint_config(
                    EndpointConfigName=config_name
                )
                print(f"✓ Endpoint config {config_name} deleted")
                
                # Delete model
                if delete_model:
                    self.sagemaker_client.delete_model(ModelName=model_name)
                    print(f"✓ Model {model_name} deleted")
            except Exception as e:
                print(f"Warning: Could not delete config/model: {e}")
    
    def create_batch_transform_job(
        self,
        job_name: str,
        model_name: str,
        input_s3_uri: str,
        output_s3_uri: str,
        instance_type: str = 'ml.m5.xlarge',
        instance_count: int = 1,
        max_payload_mb: int = 6,
        wait: bool = True
    ) -> Dict:
        """
        Create batch transform job
        
        Args:
            job_name: Job name
            model_name: Model name
            input_s3_uri: Input data S3 URI
            output_s3_uri: Output S3 URI
            instance_type: Instance type
            instance_count: Number of instances
            max_payload_mb: Max payload size in MB
            wait: Whether to wait for completion
            
        Returns:
            Job creation response
        """
        response = self.sagemaker_client.create_transform_job(
            TransformJobName=job_name,
            ModelName=model_name,
            TransformInput={
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_s3_uri
                    }
                },
                'ContentType': 'text/csv',
                'SplitType': 'Line'
            },
            TransformOutput={
                'S3OutputPath': output_s3_uri,
                'AssembleWith': 'Line'
            },
            TransformResources={
                'InstanceType': instance_type,
                'InstanceCount': instance_count
            },
            MaxPayloadInMB=max_payload_mb
        )
        
        if wait:
            print(f"Running batch transform job {job_name}...")
            self.wait_for_transform_job(job_name)
        
        return response
    
    def wait_for_transform_job(
        self,
        job_name: str,
        max_wait_time: int = 3600
    ) -> None:
        """
        Wait for batch transform job to complete
        
        Args:
            job_name: Job name
            max_wait_time: Maximum wait time in seconds
        """
        start_time = time.time()
        
        while True:
            response = self.sagemaker_client.describe_transform_job(
                TransformJobName=job_name
            )
            status = response['TransformJobStatus']
            
            if status == 'Completed':
                print(f"✓ Transform job {job_name} completed")
                break
            elif status == 'Failed':
                raise Exception(f"Transform job failed: {response['FailureReason']}")
            
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                raise Exception(f"Transform job timed out after {max_wait_time}s")
            
            print(f"Status: {status}, elapsed: {int(elapsed_time)}s")
            time.sleep(60)


def get_sagemaker_image_uri(
    framework: str,
    region: Optional[str] = None,
    version: Optional[str] = None
) -> str:
    """
    Get SageMaker container image URI
    
    Args:
        framework: Framework name (sklearn, xgboost, pytorch, etc.)
        region: AWS region
        version: Framework version
        
    Returns:
        Image URI
    """
    if region is None:
        region = boto3.Session().region_name
    
    # Use sagemaker image_uris helper
    from sagemaker import image_uris
    
    return image_uris.retrieve(
        framework=framework,
        region=region,
        version=version
    )
