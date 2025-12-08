# Setup Guide - AWS ML Training

This guide will help you set up your environment for the AWS Machine Learning training program.

## Prerequisites

### Required Accounts & Access
- [ ] AWS Account with SageMaker access
- [ ] IAM user with appropriate permissions
- [ ] AWS CLI installed and configured
- [ ] Python 3.11 or higher installed

### Required Knowledge
- Basic understanding of machine learning concepts
- Familiarity with Python programming
- Basic AWS services knowledge
- Command line proficiency

## Setup Steps

### 1. AWS Account Configuration

#### Create IAM User (if needed)
```bash
# Use AWS Console to create IAM user with policies:
# - AmazonSageMakerFullAccess
# - AmazonS3FullAccess
# - CloudWatchLogsFullAccess
```

#### Configure AWS CLI
```bash
# Install AWS CLI (if not already installed)
# For Linux/Mac:
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# For Windows: Download and run the MSI installer from AWS website

# Configure credentials
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Default output format (json)

# Verify configuration
aws sts get-caller-identity
```

### 2. Local Development Environment

#### Option A: Local Setup (Recommended for Development)

##### Install Python Environment
```bash
# Check Python version (should be 3.11+)
python --version

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

##### Clone Repository
```bash
git clone https://github.com/jwabantequinnovate/aws-ml-training.git
cd aws-ml-training
```

##### Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Verify installations
python -c "import sagemaker; print(f'SageMaker SDK: {sagemaker.__version__}')"
python -c "import boto3; print(f'Boto3: {boto3.__version__}')"
```

#### Option B: SageMaker Studio (Recommended for Training)

1. **Open AWS Console** → Navigate to Amazon SageMaker
2. **SageMaker Studio** → Create a new domain (if not exists)
3. **Launch Studio** → Open Studio interface
4. **Clone Repository**:
   ```bash
   git clone https://github.com/jwabantequinnovate/aws-ml-training.git
   ```
5. **Select Kernel**: Python 3 (Data Science)

### 3. Verify Setup

#### Test AWS Connectivity
```bash
# Test S3 access
aws s3 ls

# Test SageMaker access
aws sagemaker list-notebook-instances
```

#### Test Python Environment
```python
# Create a test script: test_setup.py
import sys
import boto3
import sagemaker
import pandas as pd
import numpy as np
import sklearn
import xgboost
import torch
import transformers

print("Python version:", sys.version)
print("Boto3 version:", boto3.__version__)
print("SageMaker SDK version:", sagemaker.__version__)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("XGBoost version:", xgboost.__version__)
print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)

# Test AWS access
session = boto3.Session()
print("AWS Region:", session.region_name)

# Test SageMaker
sm_session = sagemaker.Session()
print("SageMaker Bucket:", sm_session.default_bucket())

print("\n✓ All checks passed! Environment is ready.")
```

Run the test:
```bash
python test_setup.py
```

### 4. Download NLTK Data (for NLP modules)

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### 5. Set Up Jupyter Notebook

```bash
# Install Jupyter (if not using SageMaker Studio)
pip install jupyter notebook

# Launch Jupyter
jupyter notebook

# Navigate to: http://localhost:8888
```

### 6. Configure SageMaker Execution Role

#### In SageMaker Studio
The execution role is automatically configured.

#### For Local Development
Create a SageMaker execution role in IAM:

1. Go to IAM Console → Roles → Create Role
2. Select "SageMaker" as the service
3. Attach policies:
   - AmazonSageMakerFullAccess
   - AmazonS3FullAccess
4. Name the role: `SageMakerExecutionRole`
5. Copy the role ARN

Update your code to use the role:
```python
import sagemaker

# Replace with your role ARN
role = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole'
```

### 7. Create S3 Bucket for Training

```bash
# Create bucket (use a unique name)
aws s3 mb s3://ml-training-YOUR-NAME-$(date +%s)

# Set bucket as default
export SAGEMAKER_BUCKET=ml-training-YOUR-NAME-TIMESTAMP
```

Or in Python:
```python
import boto3
from datetime import datetime

s3 = boto3.client('s3')
bucket_name = f'ml-training-{datetime.now().strftime("%Y%m%d%H%M%S")}'
s3.create_bucket(Bucket=bucket_name)
print(f"Created bucket: {bucket_name}")
```

## Troubleshooting

### Common Issues

#### 1. AWS Credentials Not Found
```
Error: Unable to locate credentials
```
**Solution**: Run `aws configure` and enter your credentials.

#### 2. Region Not Set
```
Error: You must specify a region
```
**Solution**: Set region in AWS config or use:
```bash
export AWS_DEFAULT_REGION=us-east-1
```

#### 3. Permission Denied
```
Error: User is not authorized to perform: sagemaker:CreateModel
```
**Solution**: Ensure IAM user has SageMaker permissions.

#### 4. Package Installation Fails
```
Error: No matching distribution found for package-name
```
**Solution**: Upgrade pip and try again:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Jupyter Kernel Issues
**Solution**: Install ipykernel in your virtual environment:
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name="Python (ML Training)"
```

## Cost Management

### Estimated Costs
- **SageMaker Studio**: ~$0.05/hour (ml.t3.medium)
- **SageMaker Training**: ~$0.269/hour (ml.m5.xlarge)
- **SageMaker Endpoints**: ~$0.269/hour (ml.m5.xlarge)
- **S3 Storage**: ~$0.023/GB/month

### Cost Optimization Tips
1. **Stop notebooks** when not in use
2. **Delete endpoints** after testing
3. **Use spot instances** for training when possible
4. **Clean up S3** regularly
5. **Set up billing alerts** in AWS Console

### Set Up Billing Alert
```bash
# Create SNS topic for alerts
aws sns create-topic --name ml-training-billing-alerts

# Subscribe to topic
aws sns subscribe \
    --topic-arn arn:aws:sns:REGION:ACCOUNT:ml-training-billing-alerts \
    --protocol email \
    --notification-endpoint your-email@example.com

# Create CloudWatch alarm (in Console)
# Billing → Create Alarm → Estimated Charges > $50
```

## Next Steps

1. ✓ Complete this setup guide
2. → Review [Module 1: Fraud Detection](../01-fraud-detection/README.md)
3. → Complete exercises in order
4. → Join discussion sessions
5. → Build capstone project

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [AWS CLI User Guide](https://docs.aws.amazon.com/cli/latest/userguide/)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)

## Support

For issues or questions:
1. Check [Troubleshooting Guide](./troubleshooting.md)
2. Review module-specific READMEs
3. Create an issue in the repository
4. Contact training facilitators

---

**Ready to start?** Proceed to [Module 1: Fraud Detection](../01-fraud-detection/README.md)
