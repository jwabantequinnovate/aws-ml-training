# SageMaker Deployment Strategies

This document covers three main deployment patterns for ML models on AWS SageMaker.

## 1. Real-time Inference

### Overview
Real-time inference provides low-latency predictions via HTTPS endpoints, ideal for interactive applications.

### Use Cases
- Fraud detection (immediate transaction scoring)
- Sentiment analysis APIs
- Real-time recommendation systems
- Interactive chatbots

### Architecture
```
Client → API Gateway → SageMaker Endpoint → Response
```

### Configuration
```python
from sagemaker.model import Model

model = Model(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    image_uri=image_uri
)

predictor = model.deploy(
    initial_instance_count=2,
    instance_type='ml.m5.xlarge',
    endpoint_name='realtime-endpoint'
)
```

### Key Features
- **Latency**: <100ms typical
- **Scaling**: Auto-scaling based on traffic
- **Cost**: Pay per hour per instance
- **Best for**: Latency-sensitive applications

### Auto-scaling Example
```python
import boto3

autoscaling = boto3.client('application-autoscaling')

# Register scalable target
autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Target tracking policy
autoscaling.put_scaling_policy(
    PolicyName='target-tracking-policy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

## 2. Batch Transform

### Overview
Batch transform processes large datasets asynchronously, optimized for cost-efficiency.

### Use Cases
- Customer churn scoring (monthly batch)
- Large-scale document classification
- Periodic risk assessments
- Data preprocessing pipelines

### Architecture
```
S3 Input → Batch Transform Job → S3 Output
```

### Configuration
```python
from sagemaker.transformer import Transformer

transformer = Transformer(
    model_name='my-model',
    instance_count=4,
    instance_type='ml.m5.xlarge',
    output_path='s3://bucket/output/',
    sagemaker_session=session
)

transformer.transform(
    data='s3://bucket/input/data.csv',
    content_type='text/csv',
    split_type='Line',
    wait=True
)
```

### Key Features
- **Latency**: Minutes to hours
- **Scaling**: Processes large datasets in parallel
- **Cost**: Pay only during processing
- **Best for**: Periodic bulk predictions

### Optimization Tips
1. **Parallelization**: Use multiple instances
2. **Batch Size**: Adjust max_payload for efficiency
3. **Data Format**: Use efficient formats (Parquet, RecordIO)
4. **Scheduling**: Use EventBridge for periodic runs

### Scheduled Batch Job Example
```python
import boto3

events = boto3.client('events')
lambda_client = boto3.client('lambda')

# Create EventBridge rule for monthly execution
events.put_rule(
    Name='monthly-batch-transform',
    ScheduleExpression='cron(0 0 1 * ? *)',  # First day of month
    State='ENABLED'
)

# Add Lambda target to trigger batch transform
events.put_targets(
    Rule='monthly-batch-transform',
    Targets=[
        {
            'Id': '1',
            'Arn': lambda_function_arn,
            'Input': json.dumps({
                'model_name': 'churn-model',
                'input_path': 's3://bucket/monthly-data/',
                'output_path': 's3://bucket/monthly-predictions/'
            })
        }
    ]
)
```

## 3. Async Inference

### Overview
Asynchronous inference queues requests and processes them with variable latency, ideal for large payloads.

### Use Cases
- Document processing (PDFs, images)
- Video analysis
- Large text classification
- High-throughput NLP tasks

### Architecture
```
Client → S3 Request Queue → Async Endpoint → S3 Response Queue → Client Poll
```

### Configuration
```python
from sagemaker.async_inference import AsyncInferenceConfig

async_config = AsyncInferenceConfig(
    output_path='s3://bucket/async-output/',
    max_concurrent_invocations_per_instance=4,
    notification_config={
        'SuccessTopic': 'arn:aws:sns:region:account:success-topic',
        'ErrorTopic': 'arn:aws:sns:region:account:error-topic'
    }
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    async_inference_config=async_config,
    endpoint_name='async-endpoint'
)
```

### Key Features
- **Latency**: Seconds to minutes
- **Scaling**: Handles variable load efficiently
- **Cost**: Auto-scales to zero when idle
- **Best for**: Large payloads, variable workloads

### Invoking Async Endpoint
```python
import boto3

runtime_client = boto3.client('sagemaker-runtime')

# Upload input to S3
input_location = session.upload_data(
    path='input.json',
    key_prefix='async-input'
)

# Invoke async endpoint
response = runtime_client.invoke_endpoint_async(
    EndpointName='async-endpoint',
    InputLocation=input_location
)

# Get output location
output_location = response['OutputLocation']

# Poll for results or use SNS notification
```

## Comparison Matrix

| Feature | Real-time | Batch Transform | Async Inference |
|---------|-----------|-----------------|-----------------|
| Latency | <100ms | Minutes-Hours | Seconds-Minutes |
| Payload Size | <6MB | Large | Very Large |
| Cost Model | Per hour | Per job | Auto-scale |
| Use Case | Interactive | Periodic bulk | Variable load |
| Scaling | Auto-scale | Manual | Auto-scale to 0 |
| Best For | Low latency | Cost efficiency | Large payloads |

## Multi-Model Endpoints

Deploy multiple models to a single endpoint for cost optimization.

### Configuration
```python
from sagemaker.multidatamodel import MultiDataModel

multi_model = MultiDataModel(
    name='multi-model-endpoint',
    model_data_prefix='s3://bucket/models/',
    image_uri=image_uri,
    role=role
)

multi_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='multi-model-endpoint'
)

# Invoke specific model
response = runtime_client.invoke_endpoint(
    EndpointName='multi-model-endpoint',
    TargetModel='model-v1.tar.gz',
    Body=json.dumps(data)
)
```

## A/B Testing

Deploy multiple model variants for comparison.

### Configuration
```python
# Create endpoint config with multiple variants
sagemaker_client.create_endpoint_config(
    EndpointConfigName='ab-test-config',
    ProductionVariants=[
        {
            'VariantName': 'ModelA',
            'ModelName': 'model-a',
            'InstanceType': 'ml.m5.xlarge',
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 0.7  # 70% traffic
        },
        {
            'VariantName': 'ModelB',
            'ModelName': 'model-b',
            'InstanceType': 'ml.m5.xlarge',
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 0.3  # 30% traffic
        }
    ]
)
```

## Canary Deployment

Gradually roll out new model version.

```python
# Start with 5% traffic to new version
sagemaker_client.update_endpoint_weights_and_capacities(
    EndpointName='my-endpoint',
    DesiredWeightsAndCapacities=[
        {
            'VariantName': 'ModelOld',
            'DesiredWeight': 0.95
        },
        {
            'VariantName': 'ModelNew',
            'DesiredWeight': 0.05
        }
    ]
)

# Monitor metrics, then increase gradually
# 5% → 25% → 50% → 100%
```

## Best Practices

### 1. Monitoring
- Set up CloudWatch alarms for latency, errors
- Monitor invocations per instance
- Track model performance metrics

### 2. Cost Optimization
- Use Savings Plans for predictable workloads
- Auto-scale endpoints based on traffic
- Use batch transform for non-urgent predictions
- Consider spot instances for batch jobs

### 3. Security
- Enable encryption at rest and in transit
- Use VPC endpoints for private deployments
- Implement IAM policies for least privilege
- Enable CloudTrail logging

### 4. Performance
- Use appropriate instance types
- Enable model caching
- Optimize model artifacts (quantization)
- Use multi-model endpoints when applicable

## Next Steps

1. Choose deployment strategy based on requirements
2. Implement monitoring and alerting
3. Set up CI/CD pipeline for model deployment
4. Test deployment with realistic load
5. Document deployment process for team
