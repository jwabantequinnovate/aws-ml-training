"""
AWS Configuration Management

Handles environment-specific configurations and best practices.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AWSConfig:
    """AWS configuration for SageMaker and S3"""
    
    region: str = os.getenv("AWS_REGION", "us-east-1")
    bucket_prefix: str = "ml-training"
    
    # Instance types by use case
    TRAINING_INSTANCE_SMALL = "ml.m5.large"
    TRAINING_INSTANCE_MEDIUM = "ml.m5.xlarge"
    TRAINING_INSTANCE_GPU = "ml.p3.2xlarge"
    
    ENDPOINT_INSTANCE_SMALL = "ml.t2.medium"
    ENDPOINT_INSTANCE_MEDIUM = "ml.m5.large"
    ENDPOINT_INSTANCE_GPU = "ml.g4dn.xlarge"
    
    # Cost optimization settings
    SPOT_INSTANCES_ENABLED = True
    MAX_SPOT_WAIT_TIME = 3600  # 1 hour
    
    # Model monitoring
    MONITORING_SCHEDULE_CRON = "cron(0 * ? * * *)"  # Hourly
    DRIFT_THRESHOLD = 0.05  # p-value threshold
    
    # Feature Store
    FEATURE_STORE_ONLINE_ENABLED = False  # Disable to reduce costs
    
    def get_training_instance(self, use_gpu: bool = False, size: str = "small") -> str:
        """Get appropriate training instance type"""
        if use_gpu:
            return self.TRAINING_INSTANCE_GPU
        return self.TRAINING_INSTANCE_MEDIUM if size == "medium" else self.TRAINING_INSTANCE_SMALL
    
    def get_endpoint_instance(self, use_gpu: bool = False, size: str = "small") -> str:
        """Get appropriate endpoint instance type"""
        if use_gpu:
            return self.ENDPOINT_INSTANCE_GPU
        return self.ENDPOINT_INSTANCE_MEDIUM if size == "medium" else self.ENDPOINT_INSTANCE_SMALL
