# MLOps Architecture - Best Practices

## High-Level ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        End-to-End ML Pipeline                           │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Data       │    │   Feature    │    │    Model     │    │  Deployment  │
│  Ingestion   │───>│  Engineering │───>│   Training   │───>│   & Serving  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      │                    │                    │                    │
      v                    v                    v                    v
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Data Sources │    │ Feature Store│    │ Model        │    │  Monitoring  │
│ (DB, S3, API)│    │  (SageMaker) │    │  Registry    │    │ & Observ.    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    v
                                                             ┌──────────────┐
                                                             │  Retraining  │
                                                             │   Trigger    │
                                                             └──────────────┘
```

## Architecture Components

### 1. Data Layer

#### Data Sources
- **Databases**: RDS, DynamoDB, Redshift
- **Object Storage**: S3
- **Streaming**: Kinesis, Kafka
- **APIs**: REST, GraphQL

#### Data Pipeline
```
Raw Data → Validation → Cleaning → Transformation → Feature Store
```

**Tools:**
- AWS Glue for ETL
- Amazon EMR for big data processing
- AWS Lambda for serverless processing
- Step Functions for orchestration

#### Best Practices
✓ Version control data schemas
✓ Implement data quality checks
✓ Use partitioning for large datasets
✓ Enable encryption at rest and in transit
✓ Set up data lifecycle policies

### 2. Feature Engineering

#### Feature Store Architecture
```
┌─────────────────────────────────────────────────┐
│           SageMaker Feature Store               │
├─────────────────────────────────────────────────┤
│  Online Store          │  Offline Store          │
│  (Real-time features)  │  (Training features)    │
│  Low-latency access    │  Historical features    │
└─────────────────────────────────────────────────┘
```

**Benefits:**
- Consistent features across training and inference
- Feature versioning and lineage
- Feature sharing across teams
- Reduced feature computation cost

#### Implementation
```python
from sagemaker.feature_store.feature_group import FeatureGroup

feature_group = FeatureGroup(
    name="customer-features",
    sagemaker_session=session
)

feature_group.create(
    s3_uri=f"s3://{bucket}/feature-store",
    record_identifier_name="customer_id",
    event_time_feature_name="event_time",
    role_arn=role,
    enable_online_store=True
)
```

#### Best Practices
✓ Separate online and offline feature stores
✓ Version features with metadata
✓ Monitor feature drift
✓ Document feature definitions
✓ Implement feature validation

### 3. Model Training

#### Training Architecture

**Distributed Training:**
```
┌──────────────────────────────────────────┐
│      SageMaker Training Cluster          │
├──────────────────────────────────────────┤
│  Master Node                             │
│    ├─> Worker Node 1                     │
│    ├─> Worker Node 2                     │
│    └─> Worker Node N                     │
└──────────────────────────────────────────┘
```

**Hyperparameter Tuning:**
```python
from sagemaker.tuner import HyperparameterTuner

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='validation:auc',
    hyperparameter_ranges={
        'max_depth': IntegerParameter(3, 10),
        'eta': ContinuousParameter(0.01, 0.3),
        'subsample': ContinuousParameter(0.5, 1.0)
    },
    max_jobs=20,
    max_parallel_jobs=4
)
```

#### Best Practices
✓ Use managed spot training for cost savings
✓ Implement early stopping
✓ Version all training code
✓ Log hyperparameters and metrics
✓ Use distributed training for large datasets

### 4. Model Registry

#### Model Lifecycle
```
Development → Staging → Production → Archived
     │           │          │           │
     v           v          v           v
  Training    Validation  Serving   Retirement
```

**SageMaker Model Registry:**
```python
from sagemaker.model import Model

model = Model(
    image_uri=image_uri,
    model_data=model_data,
    role=role
)

# Register model
model_package = model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name="fraud-detection-models",
    approval_status="PendingManualApproval",
    model_metrics={
        "Metrics": {
            "accuracy": {"Value": 0.92},
            "auc": {"Value": 0.95}
        }
    }
)
```

#### Best Practices
✓ Tag models with metadata
✓ Track model lineage
✓ Implement approval workflows
✓ Version models systematically
✓ Document model performance

### 5. Deployment Strategies

#### Multi-Variant Endpoint (A/B Testing)
```
                    ┌─────────────────┐
                    │   API Gateway   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   Application   │
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
         70% Traffic              30% Traffic
                 │                       │
         ┌───────┴───────┐      ┌───────┴───────┐
         │   Model A     │      │   Model B     │
         │  (Current)    │      │   (New)       │
         └───────────────┘      └───────────────┘
```

#### Canary Deployment
```
Phase 1: 95% Old, 5% New   → Monitor
Phase 2: 75% Old, 25% New  → Monitor
Phase 3: 50% Old, 50% New  → Monitor
Phase 4: 25% Old, 75% New  → Monitor
Phase 5: 0% Old, 100% New  → Promote
```

#### Blue-Green Deployment
```
┌──────────────────────────────────────────┐
│           Traffic Router                 │
└───────────┬──────────────────────────────┘
            │ Switch traffic instantly
     ┌──────┴──────┐
     │             │
┌────┴────┐  ┌────┴────┐
│  Blue   │  │  Green  │
│ (Old)   │  │  (New)  │
└─────────┘  └─────────┘
```

#### Best Practices
✓ Start with shadow mode
✓ Implement circuit breakers
✓ Monitor key metrics continuously
✓ Have rollback plan ready
✓ Automate deployment process

### 6. Monitoring & Observability

#### Monitoring Stack
```
┌──────────────────────────────────────────────────┐
│              CloudWatch Metrics                  │
├──────────────────────────────────────────────────┤
│  • Model Latency       • Error Rates             │
│  • Invocations/sec     • CPU/Memory              │
│  • Model Accuracy      • Cost per prediction     │
└──────────────────────────────────────────────────┘
                         │
                         v
┌──────────────────────────────────────────────────┐
│          SageMaker Model Monitor                 │
├──────────────────────────────────────────────────┤
│  • Data Quality        • Model Quality           │
│  • Bias Detection      • Feature Attribution     │
│  • Data Drift          • Model Drift             │
└──────────────────────────────────────────────────┘
                         │
                         v
┌──────────────────────────────────────────────────┐
│               Alerting (SNS)                     │
├──────────────────────────────────────────────────┤
│  • Performance degradation                       │
│  • Error rate spike                              │
│  • Data drift detected                           │
└──────────────────────────────────────────────────┘
```

#### Model Monitor Setup
```python
from sagemaker.model_monitor import DataCaptureConfig

# Enable data capture
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=f's3://{bucket}/data-capture'
)

# Deploy with monitoring
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    data_capture_config=data_capture_config
)

# Create monitoring schedule
from sagemaker.model_monitor import DefaultModelMonitor

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    max_runtime_in_seconds=3600
)

monitor.create_monitoring_schedule(
    monitor_schedule_name='daily-model-monitor',
    endpoint_input=predictor.endpoint_name,
    output_s3_uri=f's3://{bucket}/monitoring-results',
    statistics=baseline_statistics,
    constraints=baseline_constraints,
    schedule_cron_expression='cron(0 0 * * ? *)'
)
```

#### Best Practices
✓ Monitor both technical and business metrics
✓ Set up alerting for anomalies
✓ Track model drift continuously
✓ Implement automated retraining triggers
✓ Maintain monitoring dashboards

### 7. CI/CD Pipeline

#### Pipeline Stages
```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│   Code   │   │  Build & │   │   Test   │   │  Deploy  │
│  Commit  │──>│   Test   │──>│  Models  │──>│  to Prod │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                      │              │              │
                      v              v              v
                 Unit Tests    Integration    Monitoring
                 Linting       Tests          Validation
                 Security      Performance    Approval
```

#### Implementation Options

**Option 1: AWS CodePipeline**
```yaml
# buildspec.yml (already created)
version: 0.2
phases:
  install:
    runtime-versions:
      python: 3.11
  build:
    commands:
      - pytest tests/
      - python train.py
      - python deploy.py
```

**Option 2: Jenkins Pipeline**
```groovy
# Jenkinsfile (to be created)
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Test') {
            steps {
                sh 'pytest tests/'
            }
        }
        stage('Train') {
            steps {
                sh 'python train.py'
            }
        }
        stage('Deploy') {
            steps {
                sh 'python deploy.py'
            }
        }
    }
}
```

#### Best Practices
✓ Automate everything possible
✓ Run tests before deployment
✓ Use infrastructure as code
✓ Implement proper secrets management
✓ Maintain deployment rollback capability

## Cost Optimization

### Strategies

#### 1. Instance Selection
- Use appropriate instance types
- Right-size for workload
- Consider ARM-based Graviton instances

#### 2. Training Optimization
- Use managed spot training (up to 90% savings)
- Implement early stopping
- Use distributed training efficiently

#### 3. Inference Optimization
- Use auto-scaling
- Multi-model endpoints
- Batch transform for non-urgent predictions
- Async inference for variable workloads

#### 4. Storage Optimization
- Set S3 lifecycle policies
- Use intelligent tiering
- Clean up old model artifacts
- Compress data when possible

### Cost Monitoring
```python
# Set up budget alert
import boto3

budgets = boto3.client('budgets')

budgets.create_budget(
    AccountId='123456789012',
    Budget={
        'BudgetName': 'ML-Training-Budget',
        'BudgetLimit': {
            'Amount': '1000',
            'Unit': 'USD'
        },
        'TimeUnit': 'MONTHLY',
        'BudgetType': 'COST'
    },
    NotificationsWithSubscribers=[
        {
            'Notification': {
                'NotificationType': 'ACTUAL',
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': 80
            },
            'Subscribers': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'team@example.com'
                }
            ]
        }
    ]
)
```

## Security Best Practices

### 1. Data Security
- Encrypt data at rest (S3, EBS)
- Encrypt data in transit (TLS)
- Use VPC endpoints
- Implement least privilege access

### 2. Model Security
- Sign model artifacts
- Scan containers for vulnerabilities
- Use private Docker registries
- Implement model access controls

### 3. Network Security
- Use VPC for SageMaker
- Implement security groups
- Use PrivateLink
- Enable VPC Flow Logs

### 4. Compliance
- Enable CloudTrail logging
- Implement audit trails
- Document data lineage
- Regular security reviews

## Disaster Recovery

### Backup Strategy
- Regular model backups to S3
- Cross-region replication
- Version all artifacts
- Document recovery procedures

### High Availability
- Multi-AZ deployments
- Auto-scaling configuration
- Health checks and monitoring
- Automated failover

## Summary Checklist

### Architecture Review
- [ ] Data pipeline designed
- [ ] Feature store implemented
- [ ] Training pipeline automated
- [ ] Model registry configured
- [ ] Deployment strategy chosen
- [ ] Monitoring established
- [ ] CI/CD pipeline created
- [ ] Security measures implemented
- [ ] Cost optimization applied
- [ ] Disaster recovery planned

### Production Readiness
- [ ] Documentation complete
- [ ] Testing comprehensive
- [ ] Monitoring dashboards ready
- [ ] Alerting configured
- [ ] Runbooks prepared
- [ ] Team trained
- [ ] Compliance verified
- [ ] Performance validated

---

**Next Steps**: Implement this architecture incrementally, starting with core components and adding complexity as needed.
