# AWS ML Training - 2-Day MLOps Workshop

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue)](https://python-poetry.org/)

2-day hands-on training covering AWS SageMaker and production MLOps patterns.

## 🎯 Overview

This workshop provides practical experience with AWS machine learning services through 9 comprehensive labs. You'll build real ML systems handling fraud detection, customer churn, text classification, and generative AI.

## 📚 Workshop Structure (2 Days)

### Day 1: ML Fundamentals + MLOps Tools

| Lab | Topic | Duration | Key Skills |
|-----|-------|----------|------------|
| **Lab 1** | Fraud Detection | 90 min | Imbalanced data, SMOTE, cost-sensitive learning |
| **Lab 2** | Customer Churn | 90 min | Feature engineering, model comparison, batch prediction |
| **Lab 3** | Text Classification | 90 min | TF-IDF, SageMaker Clarify, model explainability |
| **Lab 4** | Sentiment Analysis | 90 min | Multi-class classification, Model Monitor, drift detection |

### Day 2: Advanced MLOps & Deployment

| Lab | Topic | Duration | Key Skills |
|-----|-------|----------|------------|
| **Lab 5** | Feature Store & Registry | 90 min | Feature management, model versioning, lineage tracking |
| **Lab 6** | Advanced Endpoints | 90 min | Serverless inference, async endpoints, multi-model hosting |
| **Lab 7** | SageMaker Pipelines | 90 min | ML automation, CI/CD workflows, pipeline orchestration |
| **Lab 8** | Deployment Strategies | 90 min | Blue/Green, Canary releases, A/B testing |
| **Lab 9** | Generative AI (Bedrock) | 90 min | Claude, Titan, Llama comparison, Guardrails |

## 🏗️ Repository Structure

```
aws-ml-training/
├── 01-fraud-detection/          # Lab 1: Fraud detection with SMOTE
│   ├── exercises/               # Hands-on exercises with TODOs
│   ├── solutions/               # Complete implementations
│   ├── data/                    # Sample fraud dataset
│   └── README.md
│
├── 02-customer-churn/           # Lab 2: Customer churn prediction
├── 03-text-classification/      # Lab 3: TF-IDF + Clarify
├── 04-sentiment-analysis/       # Lab 4: Multi-class + Model Monitor
├── 05-mlops-packaging/          # Lab 5: Feature Store + Registry
├── 06-mlops-deployment/         # Lab 6: Advanced endpoints
├── 07-mlops-pipelines/          # Lab 7: SageMaker Pipelines
├── 08-deployment-strategies/    # Lab 8: Blue/Green + Canary
├── 09-bedrock-genai/            # Lab 9: Generative AI
│
├── src/ml_toolkit/              # Reusable Python package
│   ├── preprocessing.py         # Data preprocessing
│   ├── evaluation.py            # Model evaluation
│   ├── sagemaker_utils.py       # Deployment helpers
│   ├── mlflow_tracking.py       # MLflow integration
│   ├── dvc_manager.py           # DVC integration
│   ├── debugger.py              # SageMaker Debugger
│   └── lineage.py               # Model lineage tracking
│
├── tests/                       # Production-quality tests
│   ├── unit/                    # Fast unit tests
│   ├── integration/             # E2E with SageMaker
│   └── conftest.py              # Pytest fixtures
│
├── docker/                      # Custom inference containers
├── docs/                        # Additional documentation
├── scripts/                     # Utility scripts
├── pyproject.toml              # Poetry dependencies
└── Makefile                    # Quick commands
```

## 🚀 Getting Started

### Prerequisites

- AWS Account with SageMaker access
- Python 3.9+ installed
- AWS CLI configured (`aws configure`)
- 8GB+ RAM recommended

### Setup in SageMaker Studio

```bash
# 1. Clone repository in SageMaker Studio
cd /home/sagemaker-user/
git clone https://github.com/jwabantequinnovate/aws-ml-training.git
cd aws-ml-training

# 2. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 3. Add Poetry to PATH
export PATH="/home/sagemaker-user/.local/bin:$PATH"
echo 'export PATH="/home/sagemaker-user/.local/bin:$PATH"' >> ~/.bashrc

# 4. Verify Poetry installation
poetry --version

# 5. Install dependencies (make sure you're in the project directory!)
cd /home/sagemaker-user/aws-ml-training
poetry install

# 6. Start Jupyter Lab
poetry run jupyter lab
```

### Running Labs

1. Navigate to any lab folder (e.g., `01-fraud-detection/`)
2. Open the **exercise** notebook in Jupyter Lab
3. Follow the TODO sections
4. Check the **solution** notebook if needed

## 🧪 Testing (Optional - For Advanced Users)

Production-ready code includes comprehensive tests:

```bash
poetry run pytest tests/unit/           # Fast unit tests (no AWS required)
poetry run pytest tests/                # Full test suite
poetry run pytest --cov=src/ml_toolkit  # With coverage report
```

The `src/ml_toolkit/` package demonstrates professional testing practices.

## 📖 Lab Descriptions

### 🔴 Lab 1: Fraud Detection
- Handle imbalanced datasets (1:100 fraud ratio)
- SMOTE for handling class imbalance
- Cost-sensitive learning and threshold optimization
- Model comparison (Logistic Regression, XGBoost, LightGBM)
- Feature importance analysis

### 📉 Lab 2: Customer Churn
- Feature engineering from temporal data
- Compare multiple model approaches
- Hyperparameter tuning
- Batch prediction workflows
- Model evaluation and selection

### 📝 Lab 3: Text Classification + Clarify
- TF-IDF text vectorization
- Support ticket classification
- **SageMaker Clarify** for model explainability
- Feature importance analysis
- Regulatory compliance patterns

### 😊 Lab 4: Sentiment Analysis + Monitor
- Multi-class sentiment classification
- **SageMaker Model Monitor** setup
- Data drift detection
- Automated monitoring schedules
- Alerting and notifications

### 📦 Lab 5: Feature Store & Model Registry
- Centralized feature management
- **SageMaker Feature Store** integration
- Model versioning and lineage
- Artifact tracking
- Team collaboration patterns

### 🚀 Lab 6: Advanced Endpoints
- Serverless inference configurations
- Async endpoint patterns
- Multi-model hosting
- **BYOC - Bring Your Own Container** (custom Docker)
- Auto-scaling strategies
- Cost optimization techniques

### 🔄 Lab 7: SageMaker Pipelines
- CI/CD for machine learning
- Automated training workflows
- Model validation gates
- Pipeline orchestration
- Integration with MLOps tools

### 🎯 Lab 8: Deployment Strategies
- Blue/Green deployments
- Canary release patterns
- A/B testing frameworks
- Automated rollbacks
- Traffic shifting strategies

### 🤖 Lab 9: Generative AI with Bedrock
- Compare Claude, Titan, and Llama models
- Prompt engineering techniques
- **Guardrails** for safe AI
- Cost analysis and optimization
- Real-world GenAI applications

## 🛠️ Tech Stack

**AWS Services:**
- SageMaker (Training, Endpoints, Pipelines, Feature Store, Model Monitor, Clarify)
- Amazon Bedrock (Claude, Titan, Llama)
- S3, CloudWatch, IAM

**ML Frameworks:**
- Scikit-learn, XGBoost
- Pandas, NumPy

**MLOps Tools:**
- SageMaker Experiments
- SageMaker Debugger
- SageMaker Model Registry
- SageMaker Feature Store

**Development:**
- Python 3.9+
- Poetry (dependency management)
- Pytest (testing)
- Jupyter Lab

## 🎯 Learning Outcomes

After this workshop, you will:

✅ Train and deploy ML models on AWS SageMaker  
✅ Handle imbalanced datasets with SMOTE  
✅ Implement model explainability with Clarify  
✅ Set up drift detection with Model Monitor  
✅ Build ML pipelines with SageMaker Pipelines  
✅ Deploy with Blue/Green and Canary strategies  
✅ Work with Generative AI using Amazon Bedrock  