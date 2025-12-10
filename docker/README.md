# Custom Inference Container for SageMaker

This directory contains custom Docker containers for SageMaker BYOC (Bring Your Own Container).

## Why Custom Containers?

- Add custom preprocessing logic
- Include additional libraries not in standard containers
- Optimize for specific use cases
- Full control over serving infrastructure
- Support both numeric and text data

## Available Containers

### sklearn-custom

**Generic scikit-learn container** with Flask serving - works with:
- ✅ Numeric/tabular data (RandomForest, XGBoost, etc.)
- ✅ Text classification (with TF-IDF vectorizer)
- ✅ Mixed data types

**Build and push:**

```bash
cd docker/sklearn-custom

# Build locally
docker build -t sklearn-custom:latest .

# Test locally
docker run -p 8080:8080 sklearn-custom:latest

# Tag for ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker tag sklearn-custom:latest <account>.dkr.ecr.us-east-1.amazonaws.com/sklearn-custom:latest

# Push to ECR
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/sklearn-custom:latest
```

**Test endpoint:**

```bash
# Health check
curl http://localhost:8080/ping

# Inference - Numeric data (Lab 6 format)
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.2, 3.4, 5.6, 7.8, 9.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}'

# Inference - Text data (with vectorizer)
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"text": ["Sample text to classify"]}'
```

## Using in SageMaker

```python
from sagemaker.model import Model

model = Model(
    image_uri='<account>.dkr.ecr.us-east-1.amazonaws.com/sklearn-custom:latest',
    model_data='s3://bucket/path/to/model.tar.gz',
    role=role
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

## Container Structure

```
/opt/ml/
├── model/              # Model artifacts (loaded from S3)
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
└── code/               # Inference code
    ├── inference.py
    └── serve
```

## Best Practices

1. **Keep containers small** - Use slim base images
2. **Cache dependencies** - Layer requirements.txt early
3. **Health checks** - Implement /ping endpoint
4. **Logging** - Print to stdout for CloudWatch
5. **Error handling** - Return proper HTTP status codes
6. **Testing** - Test locally before pushing to ECR
