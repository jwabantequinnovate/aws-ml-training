# Troubleshooting Guide

Common issues and solutions for AWS ML Training

## Setup Issues

### 1. AWS Credentials Not Found

**Error:**
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solutions:**
1. Configure AWS CLI:
   ```bash
   aws configure
   ```
2. Set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_DEFAULT_REGION=us-east-1
   ```
3. Use IAM role (in SageMaker Studio)

### 2. Python Package Installation Fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement package-name
```

**Solutions:**
1. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```
2. Install with specific version:
   ```bash
   pip install package-name==version
   ```
3. Check Python version compatibility

### 3. Jupyter Kernel Not Found

**Error:**
```
Kernel not found
```

**Solutions:**
```bash
pip install ipykernel
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
```

## SageMaker Issues

### 1. Endpoint Creation Failed

**Error:**
```
ClientError: An error occurred (ValidationException) when calling CreateEndpoint
```

**Solutions:**
1. Check CloudWatch logs:
   ```bash
   aws logs tail /aws/sagemaker/Endpoints/endpoint-name --follow
   ```
2. Verify model artifacts in S3
3. Check IAM role permissions
4. Ensure instance type is available in region

### 2. Insufficient Capacity Error

**Error:**
```
CapacityError: Unable to provision requested instance
```

**Solutions:**
1. Try different instance type
2. Try different availability zone
3. Request quota increase in AWS Console
4. Use spot instances as fallback

### 3. Model Artifacts Not Found

**Error:**
```
ClientError: The specified key does not exist
```

**Solutions:**
1. Verify S3 path:
   ```bash
   aws s3 ls s3://bucket/path/to/model.tar.gz
   ```
2. Check model packaging
3. Ensure correct bucket permissions

### 4. Endpoint Invocation Fails

**Error:**
```
ModelError: An error occurred (ModelError) when calling InvokeEndpoint
```

**Solutions:**
1. Check input format matches expected format
2. Review inference.py script
3. Check CloudWatch logs
4. Test with sample data locally first

## Training Issues

### 1. Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size
2. Use larger instance type (ml.p3.2xlarge → ml.p3.8xlarge)
3. Use gradient accumulation
4. Enable mixed precision training

### 2. Training Job Fails

**Error:**
```
AlgorithmError: Training job failed
```

**Solutions:**
1. Check CloudWatch logs:
   ```python
   import boto3
   logs = boto3.client('logs')
   response = logs.get_log_events(
       logGroupName='/aws/sagemaker/TrainingJobs',
       logStreamName='training-job-name/algo-1-1234567890'
   )
   ```
2. Verify training script locally
3. Check data format
4. Review hyperparameters

### 3. Slow Training

**Symptoms:**
- Training takes much longer than expected

**Solutions:**
1. Use GPU instances for deep learning
2. Enable distributed training
3. Optimize data loading:
   ```python
   # Use multiple workers
   train_loader = DataLoader(dataset, num_workers=4, pin_memory=True)
   ```
4. Profile code to find bottlenecks

## Data Issues

### 1. Data Not Found in S3

**Error:**
```
NoSuchKey: The specified key does not exist
```

**Solutions:**
1. Verify S3 path:
   ```bash
   aws s3 ls s3://bucket/path/
   ```
2. Check bucket permissions
3. Ensure data was uploaded successfully

### 2. Data Format Issues

**Error:**
```
ParserError: Error tokenizing data
```

**Solutions:**
1. Verify CSV delimiter
2. Check for corrupted files
3. Validate data schema:
   ```python
   df = pd.read_csv('data.csv')
   print(df.info())
   print(df.head())
   ```

### 3. Imbalanced Dataset Issues

**Symptoms:**
- Model predicts only majority class
- Poor minority class performance

**Solutions:**
1. Apply SMOTE:
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_balanced, y_balanced = smote.fit_resample(X, y)
   ```
2. Use class weights:
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
   ```
3. Use appropriate metrics (F1, AUC-PR instead of accuracy)

## Model Performance Issues

### 1. Overfitting

**Symptoms:**
- High training accuracy, low validation accuracy
- Large gap between training and validation loss

**Solutions:**
1. Increase regularization:
   ```python
   model = XGBClassifier(reg_alpha=0.1, reg_lambda=1.0)
   ```
2. Add dropout (neural networks)
3. Reduce model complexity
4. Get more training data
5. Apply data augmentation

### 2. Underfitting

**Symptoms:**
- Low training and validation accuracy
- Model too simple for data

**Solutions:**
1. Increase model complexity
2. Add more features
3. Reduce regularization
4. Train for more epochs

### 3. Poor Inference Performance

**Symptoms:**
- High latency
- Low throughput

**Solutions:**
1. Optimize model:
   ```python
   # Model quantization
   import torch
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```
2. Use batch inference
3. Enable auto-scaling
4. Use multi-model endpoints
5. Consider model compilation

## Deployment Issues

### 1. High Latency

**Symptoms:**
- Endpoint response time > 1000ms

**Solutions:**
1. Profile inference code
2. Optimize preprocessing
3. Use smaller model (DistilBERT vs BERT)
4. Enable model caching
5. Use faster instance type

### 2. Auto-scaling Not Working

**Symptoms:**
- Endpoint doesn't scale under load

**Solutions:**
1. Verify scaling policy:
   ```bash
   aws application-autoscaling describe-scaling-policies \
       --service-namespace sagemaker
   ```
2. Check CloudWatch metrics
3. Adjust scaling thresholds
4. Ensure proper IAM permissions

### 3. A/B Test Not Routing Traffic

**Symptoms:**
- All traffic goes to one variant

**Solutions:**
1. Verify variant weights:
   ```python
   client = boto3.client('sagemaker')
   response = client.describe_endpoint(EndpointName='endpoint-name')
   print(response['ProductionVariants'])
   ```
2. Check target variant in invocation
3. Allow warm-up time for new variant

## Cost Issues

### 1. Unexpected High Costs

**Symptoms:**
- AWS bill higher than expected

**Solutions:**
1. Check running resources:
   ```bash
   # List all endpoints
   aws sagemaker list-endpoints
   
   # List training jobs
   aws sagemaker list-training-jobs
   
   # List notebook instances
   aws sagemaker list-notebook-instances
   ```
2. Delete unused endpoints:
   ```bash
   aws sagemaker delete-endpoint --endpoint-name endpoint-name
   ```
3. Set up billing alerts
4. Use Cost Explorer to identify services

### 2. Training Costs Too High

**Solutions:**
1. Use managed spot training (up to 90% savings)
2. Optimize training time (early stopping)
3. Use smaller instances for experimentation
4. Implement checkpointing to resume training

## Network Issues

### 1. Timeout Errors

**Error:**
```
ReadTimeoutError: Read timed out
```

**Solutions:**
1. Increase timeout:
   ```python
   import boto3
   from botocore.config import Config
   
   config = Config(
       connect_timeout=5,
       read_timeout=60,
       retries={'max_attempts': 3}
   )
   client = boto3.client('sagemaker', config=config)
   ```
2. Check network connectivity
3. Verify security groups and VPC settings

### 2. VPC Configuration Issues

**Error:**
```
VPCResourceNotFound: The VPC configuration is invalid
```

**Solutions:**
1. Verify VPC and subnet IDs
2. Check security group rules
3. Ensure NAT gateway for internet access
4. Review VPC endpoints

## Permission Issues

### 1. Access Denied

**Error:**
```
AccessDeniedException: User is not authorized to perform action
```

**Solutions:**
1. Check IAM policy:
   ```json
   {
       "Effect": "Allow",
       "Action": [
           "sagemaker:*",
           "s3:*",
           "iam:PassRole"
       ],
       "Resource": "*"
   }
   ```
2. Verify execution role
3. Check resource-based policies

### 2. S3 Access Denied

**Error:**
```
AccessDenied: Access Denied
```

**Solutions:**
1. Check bucket policy
2. Verify IAM permissions
3. Ensure correct bucket name
4. Check bucket region

## Debug Tips

### 1. Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sagemaker')
logger.setLevel(logging.DEBUG)
```

### 2. Check CloudWatch Logs

```python
import boto3

logs = boto3.client('logs')

# List log streams
response = logs.describe_log_streams(
    logGroupName='/aws/sagemaker/Endpoints/endpoint-name',
    orderBy='LastEventTime',
    descending=True,
    limit=5
)

# Get log events
for stream in response['logStreams']:
    events = logs.get_log_events(
        logGroupName='/aws/sagemaker/Endpoints/endpoint-name',
        logStreamName=stream['logStreamName']
    )
    for event in events['events']:
        print(event['message'])
```

### 3. Test Locally First

```python
# Test inference script locally
import joblib

model = joblib.load('model.pkl')
test_data = {...}
prediction = model.predict(test_data)
print(prediction)
```

## Getting Help

### Resources
1. AWS Documentation: https://docs.aws.amazon.com/sagemaker/
2. AWS Forums: https://forums.aws.amazon.com/
3. Stack Overflow: Tag `amazon-sagemaker`
4. GitHub Issues: Create issue in training repository

### Contact
- Training facilitators
- AWS Support (if available)
- Course discussion forum

## Quick Reference

### Useful Commands

```bash
# Check AWS credentials
aws sts get-caller-identity

# List SageMaker resources
aws sagemaker list-endpoints
aws sagemaker list-training-jobs
aws sagemaker list-models

# Check CloudWatch logs
aws logs tail /aws/sagemaker/Endpoints/endpoint-name --follow

# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name name

# Check S3 bucket
aws s3 ls s3://bucket-name/

# Get instance quotas
aws service-quotas list-service-quotas \
    --service-code sagemaker
```

### Useful Python Snippets

```python
# Get SageMaker session
import sagemaker
session = sagemaker.Session()
bucket = session.default_bucket()
role = sagemaker.get_execution_role()

# Upload to S3
s3_uri = session.upload_data(
    path='local_file.csv',
    bucket=bucket,
    key_prefix='data'
)

# Download from S3
session.download_data(
    path='.',
    bucket=bucket,
    key_prefix='data/file.csv'
)
```

---

**Still stuck?** Create an issue with:
- Error message (full traceback)
- Code that reproduces the issue
- Environment details (Python version, library versions)
- Steps already tried
