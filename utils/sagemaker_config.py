"""
Universal SageMaker Configuration Helper

This module provides compatibility helpers for SageMaker Studio, Notebook Instances,
and local development environments.

Usage:
    from sagemaker_config import get_sagemaker_config
    
    config = get_sagemaker_config()
    role = config['role']
    session = config['session']
    bucket = config['bucket']
"""

import sagemaker
import boto3
import os
import sys
from typing import Dict, Optional


def get_execution_role_safe() -> str:
    """
    Safely get SageMaker execution role.
    
    Works in:
    - SageMaker Studio
    - SageMaker Notebook Instance
    - Local environment (with SAGEMAKER_ROLE env var)
    
    Returns:
        str: IAM role ARN
        
    Raises:
        ValueError: If role cannot be determined
    """
    # Method 1: Try standard SageMaker method
    try:
        role = sagemaker.get_execution_role()
        print(f"✅ Retrieved role via get_execution_role()")
        return role
    except ValueError:
        pass
    
    # Method 2: Check environment variable
    role = os.environ.get('SAGEMAKER_ROLE')
    if role:
        print(f"✅ Retrieved role from SAGEMAKER_ROLE environment variable")
        return role
    
    # Method 3: Try to get from IAM (if role name is known)
    role_name = os.environ.get('SAGEMAKER_ROLE_NAME', 'SageMakerExecutionRole')
    try:
        iam = boto3.client('iam')
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"✅ Retrieved role from IAM: {role_name}")
        return role
    except Exception as e:
        pass
    
    # Method 4: Try common role names
    common_role_names = [
        'SageMakerExecutionRole',
        'SageMakerNotebookRole',
        'SageMakerRole',
        'AmazonSageMaker-ExecutionRole'
    ]
    
    iam = boto3.client('iam')
    for role_name in common_role_names:
        try:
            response = iam.get_role(RoleName=role_name)
            role = response['Role']['Arn']
            print(f"✅ Found role: {role_name}")
            print(f"⚠️  Consider setting SAGEMAKER_ROLE environment variable:")
            print(f"   export SAGEMAKER_ROLE={role}")
            return role
        except:
            continue
    
    # If all methods fail, provide helpful error
    error_msg = """
❌ Could not determine SageMaker execution role.

Please set one of the following:

Option 1 - Environment variable (recommended):
    export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/YourRoleName

Option 2 - Set role name:
    export SAGEMAKER_ROLE_NAME=YourRoleName

Option 3 - For Notebook Instance:
    1. Go to SageMaker Console → Notebook instances
    2. Find your instance and note the IAM role
    3. Set SAGEMAKER_ROLE to that ARN

Option 4 - Create a new role:
    aws iam create-role --role-name SageMakerExecutionRole \\
        --assume-role-policy-document file://trust-policy.json
    """
    
    print(error_msg)
    raise ValueError("Cannot determine SageMaker execution role")


def get_sagemaker_config(
    s3_prefix: str = 'ml-training',
    default_bucket: Optional[str] = None
) -> Dict[str, any]:
    """
    Get complete SageMaker configuration.
    
    Args:
        s3_prefix: Prefix for S3 paths (default: 'ml-training')
        default_bucket: Override default bucket name (optional)
        
    Returns:
        dict: Configuration dictionary with keys:
            - role: IAM role ARN
            - session: SageMaker session
            - region: AWS region
            - bucket: S3 bucket name
            - s3_prefix: S3 prefix for data
            - account_id: AWS account ID
    """
    print("🔧 Configuring SageMaker environment...")
    print("-" * 60)
    
    # Get execution role
    role = get_execution_role_safe()
    
    # Create SageMaker session
    try:
        session = sagemaker.Session()
        print(f"✅ Created SageMaker session")
    except Exception as e:
        print(f"❌ Error creating SageMaker session: {e}")
        raise
    
    # Get region
    region = session.boto_region_name
    print(f"✅ Region: {region}")
    
    # Get or create bucket
    if default_bucket:
        bucket = default_bucket
        print(f"✅ Using specified bucket: {bucket}")
    else:
        bucket = session.default_bucket()
        print(f"✅ Using default bucket: {bucket}")
    
    # Get account ID
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    print(f"✅ Account ID: {account_id}")
    
    # Verify S3 access
    try:
        s3 = boto3.client('s3')
        s3.head_bucket(Bucket=bucket)
        print(f"✅ S3 bucket accessible: s3://{bucket}")
    except Exception as e:
        print(f"⚠️  Warning: Cannot access bucket: {e}")
        print(f"   The bucket may need to be created or permissions updated")
    
    config = {
        'role': role,
        'session': session,
        'region': region,
        'bucket': bucket,
        's3_prefix': s3_prefix,
        'account_id': account_id
    }
    
    print("-" * 60)
    print("✅ SageMaker configuration complete!")
    print()
    print("📋 Configuration Summary:")
    print(f"   Role: {role}")
    print(f"   Region: {region}")
    print(f"   Bucket: s3://{bucket}/{s3_prefix}")
    print(f"   Account: {account_id}")
    print()
    
    return config


def verify_sagemaker_access() -> bool:
    """
    Verify SageMaker API access.
    
    Returns:
        bool: True if access is working, False otherwise
    """
    print("🔍 Verifying SageMaker API access...")
    
    try:
        sm = boto3.client('sagemaker')
        
        # Try to list training jobs (should work with minimal permissions)
        response = sm.list_training_jobs(MaxResults=1)
        print("✅ SageMaker API access verified")
        return True
        
    except Exception as e:
        print(f"❌ SageMaker API access failed: {e}")
        print()
        print("Please verify IAM permissions include:")
        print("  - sagemaker:ListTrainingJobs")
        print("  - sagemaker:CreateTrainingJob")
        print("  - sagemaker:DescribeTrainingJob")
        print("  - And other SageMaker permissions")
        return False


def print_available_instances():
    """Print available SageMaker instance types for reference."""
    print("📊 Common SageMaker Instance Types:")
    print()
    print("Training Instances:")
    print("  CPU:")
    print("    - ml.m5.large      : 2 vCPU, 8 GB   (~$0.14/hr)")
    print("    - ml.m5.xlarge     : 4 vCPU, 16 GB  (~$0.28/hr)")
    print("    - ml.m5.2xlarge    : 8 vCPU, 32 GB  (~$0.55/hr)")
    print("  GPU:")
    print("    - ml.p3.2xlarge    : 8 vCPU, 61 GB, 1 V100  (~$3.83/hr)")
    print("    - ml.g4dn.xlarge   : 4 vCPU, 16 GB, 1 T4    (~$0.74/hr)")
    print()
    print("Inference Instances:")
    print("  - ml.t2.medium     : 2 vCPU, 4 GB   (~$0.06/hr)")
    print("  - ml.m5.large      : 2 vCPU, 8 GB   (~$0.12/hr)")
    print("  - ml.c5.xlarge     : 4 vCPU, 8 GB   (~$0.20/hr)")
    print()
    print("💡 Tip: Use ml.t2/t3 instances for development, ml.m5 for production")
    print()


def setup_notebook_environment():
    """
    Complete setup for a new notebook.
    Should be called at the beginning of each notebook.
    
    Returns:
        dict: SageMaker configuration
    """
    print("=" * 60)
    print("🚀 Setting up SageMaker Notebook Environment")
    print("=" * 60)
    print()
    
    # Get configuration
    config = get_sagemaker_config()
    
    # Verify access
    verify_sagemaker_access()
    
    # Print instance info
    print_available_instances()
    
    # Set up common directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    print("✅ Created local directories: data/, models/, outputs/")
    print()
    
    return config


# Convenience function for quick setup
def quick_setup(s3_prefix: str = 'ml-training') -> tuple:
    """
    Quick setup returning commonly used variables.
    
    Args:
        s3_prefix: S3 prefix for data
        
    Returns:
        tuple: (role, session, bucket, region)
    """
    config = get_sagemaker_config(s3_prefix=s3_prefix)
    return (
        config['role'],
        config['session'],
        config['bucket'],
        config['region']
    )


if __name__ == '__main__':
    """Test the configuration when run as a script."""
    print("Testing SageMaker configuration helper...")
    print()
    
    try:
        config = setup_notebook_environment()
        print("✅ Configuration test successful!")
        print()
        print("You can now use:")
        print("  from sagemaker_config import get_sagemaker_config")
        print("  config = get_sagemaker_config()")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        sys.exit(1)
