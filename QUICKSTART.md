# Quick Start Guide

Get started with AWS ML Training in 5 minutes!

## 1. Prerequisites

- AWS Account with SageMaker access
- Python 3.11+
- AWS CLI configured

## 2. Setup (2 minutes)

```bash
# Clone repository
git clone https://github.com/jwabantequinnovate/aws-ml-training.git
cd aws-ml-training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Configure AWS (1 minute)

```bash
# Configure AWS credentials
aws configure
# Enter your AWS Access Key, Secret Key, and Region (e.g., us-east-1)

# Verify
aws sts get-caller-identity
```

## 4. Start Learning (2 minutes)

### Option A: Jupyter Notebook (Local)
```bash
jupyter notebook
# Navigate to: 01-fraud-detection/exercises/fraud_detection_exercise.ipynb
```

### Option B: SageMaker Studio (Cloud)
1. Open AWS Console → SageMaker Studio
2. Launch Studio
3. Clone this repository in Studio
4. Open: `01-fraud-detection/exercises/fraud_detection_exercise.ipynb`

## 5. Your First Model

Run the fraud detection exercise to:
- ✓ Generate synthetic data
- ✓ Train an XGBoost model
- ✓ Evaluate performance
- ✓ (Optional) Deploy to SageMaker

## Module Progression

1. **Fraud Detection** (4-6 hours)
   - Real-time inference
   - Handling imbalanced data
   - SHAP explainability

2. **Customer Churn** (4-6 hours)
   - Batch inference
   - Model comparison
   - Feature engineering

3. **Text Classification** (5-7 hours)
   - BERT fine-tuning
   - Async inference
   - NLP preprocessing

4. **Sentiment Analysis** (4-6 hours)
   - Transfer learning
   - A/B testing
   - Multi-lingual models

## Key Resources

- **Setup Guide**: [docs/setup-guide.md](docs/setup-guide.md)
- **Troubleshooting**: [docs/troubleshooting.md](docs/troubleshooting.md)
- **Architecture**: [architecture/best-practices.md](architecture/best-practices.md)

## Common Commands

```bash
# Test setup
python -c "import sagemaker; print(sagemaker.__version__)"

# List your endpoints
aws sagemaker list-endpoints

# Clean up (IMPORTANT!)
# Delete endpoints after exercises to avoid costs
aws sagemaker delete-endpoint --endpoint-name YOUR-ENDPOINT-NAME
```

## Cost Management

**Estimated Costs** (per exercise):
- Training: $5-10
- Endpoint (1 hour): $0.27
- Storage: <$1

**Cost Saving Tips**:
1. Delete endpoints after testing
2. Use spot instances when possible
3. Stop notebooks when not in use
4. Set up billing alerts

## Need Help?

1. Check [troubleshooting guide](docs/troubleshooting.md)
2. Review module README files
3. Create a GitHub issue
4. Contact training facilitators

## Next Steps

After completing exercises:
1. Review solutions
2. Explore MLOps components
3. Study deployment strategies
4. Build your own project

---

**Ready to start?** Open your first notebook and begin learning!

🚀 **Happy Learning!**
