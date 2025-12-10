"""
SageMaker Deployment Script for Fraud Detection Model

This script demonstrates how to deploy a fraud detection model to SageMaker
with real-time inference endpoint.

Compatible with:
- SageMaker Studio
- SageMaker Notebook Instances
- Local environment (with proper AWS credentials)
"""

import sys
import os

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import joblib
import json
from datetime import datetime

# Import universal config helper
try:
    from utils.sagemaker_config import get_sagemaker_config
    config = get_sagemaker_config(s3_prefix='fraud-detection')
    ROLE = config['role']
    REGION = config['region']
    BUCKET = config['bucket']
    SESSION = config['session']
except ImportError:
    print("⚠️  Warning: Could not import sagemaker_config, using fallback method")
    # Fallback to standard method
    try:
        ROLE = sagemaker.get_execution_role()
    except ValueError:
        ROLE = os.environ.get('SAGEMAKER_ROLE')
        if not ROLE:
            raise ValueError("Please set SAGEMAKER_ROLE environment variable")
    
    SESSION = sagemaker.Session()
    REGION = SESSION.boto_region_name
    BUCKET = SESSION.default_bucket()

# Configuration
MODEL_NAME = "fraud-detection-model"


def create_inference_script():
    """
    Create inference.py script for SageMaker endpoint
    """
    inference_code = '''
import joblib
import json
import numpy as np
import pandas as pd
import os

def model_fn(model_dir):
    """Load model artifacts"""
    model = joblib.load(os.path.join(model_dir, "fraud_detection_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "fraud_detection_scaler.pkl"))
    encoder = joblib.load(os.path.join(model_dir, "fraud_detection_encoder.pkl"))
    features = joblib.load(os.path.join(model_dir, "fraud_detection_features.pkl"))
    
    return {
        'model': model,
        'scaler': scaler,
        'encoder': encoder,
        'features': features
    }

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """Make predictions"""
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    features = model_artifacts['features']
    
    # Ensure all required features are present
    X = input_data[features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction_proba = model.predict_proba(X_scaled)[:, 1]
    prediction = (prediction_proba >= 0.5).astype(int)
    
    return {
        'fraud_probability': float(prediction_proba[0]),
        'is_fraud': int(prediction[0]),
        'threshold': 0.5
    }

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
    
    with open('inference.py', 'w') as f:
        f.write(inference_code)
    
    print("✓ Inference script created")


def package_model_artifacts():
    """
    Package model artifacts for SageMaker
    """
    import tarfile
    
    model_dir = '../../models'
    
    # Create tar.gz archive
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add(f'{model_dir}/fraud_detection_model.pkl', arcname='fraud_detection_model.pkl')
        tar.add(f'{model_dir}/fraud_detection_scaler.pkl', arcname='fraud_detection_scaler.pkl')
        tar.add(f'{model_dir}/fraud_detection_encoder.pkl', arcname='fraud_detection_encoder.pkl')
        tar.add(f'{model_dir}/fraud_detection_features.pkl', arcname='fraud_detection_features.pkl')
        tar.add('inference.py', arcname='inference.py')
    
    print("✓ Model artifacts packaged")
    return 'model.tar.gz'


def upload_to_s3(tar_file, bucket, prefix='fraud-detection'):
    """
    Upload model artifacts to S3
    """
    s3_client = boto3.client('s3')
    key = f'{prefix}/model/{tar_file}'
    
    s3_client.upload_file(tar_file, bucket, key)
    s3_uri = f's3://{bucket}/{key}'
    
    print(f"✓ Model uploaded to {s3_uri}")
    return s3_uri


def deploy_model(model_data, endpoint_name=None):
    """
    Deploy model to SageMaker endpoint
    """
    if endpoint_name is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        endpoint_name = f'{MODEL_NAME}-{timestamp}'
    
    # Create SKLearn model
    model = SKLearnModel(
        model_data=model_data,
        role=ROLE,
        entry_point='inference.py',
        framework_version='1.2-1',
        py_version='py3'
    )
    
    # Deploy to endpoint
    print(f"Deploying model to endpoint: {endpoint_name}")
    print("This may take 5-10 minutes...")
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        endpoint_name=endpoint_name
    )
    
    print(f"✓ Model deployed successfully to endpoint: {endpoint_name}")
    return predictor, endpoint_name


def test_endpoint(endpoint_name):
    """
    Test the deployed endpoint
    """
    runtime_client = boto3.client('sagemaker-runtime')
    
    # Sample test data
    test_data = {
        'transaction_amount': 250.0,
        'hour_of_day': 2,
        'day_of_week': 3,
        'distance_from_home': 150.0,
        'distance_from_last_transaction': 200.0,
        'transaction_velocity': 8,
        'is_weekend': 0,
        'is_night': 1,
        'amount_velocity_ratio': 31.25,
        'distance_ratio': 1.33,
        'merchant_category_encoded': 4
    }
    
    # Invoke endpoint
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(test_data)
    )
    
    result = json.loads(response['Body'].read().decode())
    
    print("\n" + "="*50)
    print("Endpoint Test Results:")
    print("="*50)
    print(f"Input: {json.dumps(test_data, indent=2)}")
    print(f"\nPrediction: {json.dumps(result, indent=2)}")
    print("="*50)
    
    return result


def setup_autoscaling(endpoint_name, min_capacity=1, max_capacity=3):
    """
    Configure auto-scaling for the endpoint
    """
    autoscaling_client = boto3.client('application-autoscaling')
    
    # Register scalable target
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    
    autoscaling_client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity
    )
    
    # Configure target tracking scaling policy
    autoscaling_client.put_scaling_policy(
        PolicyName=f'{endpoint_name}-scaling-policy',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 70.0,  # Target 70% invocations per instance
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
            },
            'ScaleInCooldown': 300,
            'ScaleOutCooldown': 60
        }
    )
    
    print(f"✓ Auto-scaling configured for {endpoint_name}")


def delete_endpoint(endpoint_name):
    """
    Delete the SageMaker endpoint
    """
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✓ Endpoint {endpoint_name} deleted")
    except Exception as e:
        print(f"Error deleting endpoint: {str(e)}")


def main():
    """
    Main deployment workflow
    """
    print("="*60)
    print("SageMaker Fraud Detection Model Deployment")
    print("="*60)
    
    # Step 1: Create inference script
    print("\n1. Creating inference script...")
    create_inference_script()
    
    # Step 2: Package model artifacts
    print("\n2. Packaging model artifacts...")
    tar_file = package_model_artifacts()
    
    # Step 3: Upload to S3
    print("\n3. Uploading to S3...")
    model_data = upload_to_s3(tar_file, BUCKET)
    
    # Step 4: Deploy model
    print("\n4. Deploying model...")
    predictor, endpoint_name = deploy_model(model_data)
    
    # Step 5: Test endpoint
    print("\n5. Testing endpoint...")
    test_endpoint(endpoint_name)
    
    # Step 6: Setup auto-scaling
    print("\n6. Configuring auto-scaling...")
    setup_autoscaling(endpoint_name)
    
    print("\n" + "="*60)
    print("Deployment Complete!")
    print("="*60)
    print(f"\nEndpoint Name: {endpoint_name}")
    print(f"Model Location: {model_data}")
    print("\nTo delete the endpoint later, run:")
    print(f"  delete_endpoint('{endpoint_name}')")
    print("="*60)
    
    return endpoint_name


if __name__ == "__main__":
    endpoint_name = main()
