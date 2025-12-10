# Module 1: Fraud Detection

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Handle highly imbalanced datasets typical in fraud detection
- Engineer features specific to fraud detection scenarios
- Build and compare multiple fraud detection models
- Deploy fraud detection models with real-time inference on SageMaker
- Implement model explainability for regulatory compliance
- Monitor model performance in production

## 📊 Use Case Overview

Fraud detection is a critical application of machine learning in financial services. This module covers:

- **Problem**: Identifying fraudulent transactions in real-time
- **Challenge**: Highly imbalanced datasets (fraud is rare)
- **Solution**: Advanced ML techniques with proper handling of class imbalance
- **Deployment**: Real-time scoring API with low latency requirements

## 📚 Module Structure

### 1. Exercises (Student Version)
- `fraud_detection_exercise.ipynb` - Main exercise notebook
- `preprocessing_exercise.py` - Data preprocessing challenges
- `model_training_exercise.py` - Model training tasks

### 2. Solutions (Instructor Version)
- `fraud_detection_solution.ipynb` - Complete solution notebook
- `preprocessing_solution.py` - Data preprocessing implementation
- `model_training_solution.py` - Full model training pipeline
- `deploy_solution.py` - SageMaker deployment code

### 3. Data
- Sample fraud detection dataset (synthetic)
- Data dictionary and schema
- Preprocessing utilities

## 🔧 Technical Topics Covered

### Data Preprocessing
- Handling missing values in financial data
- Feature scaling and normalization
- Temporal feature engineering
- Categorical encoding strategies

### Handling Imbalanced Data
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random under-sampling
- Class weight adjustments
- Ensemble methods for imbalanced data

### Model Development
- **Logistic Regression** (baseline)
- **Random Forest** (tree-based)
- **XGBoost** (gradient boosting)
- **LightGBM** (efficient gradient boosting)

### Model Evaluation
- Precision-Recall curves
- ROC-AUC scores
- Cost-sensitive evaluation
- Confusion matrix analysis

### Model Explainability
- SHAP (SHapley Additive exPlanations)
- Feature importance
- Individual prediction explanations
- Regulatory compliance considerations

### SageMaker Deployment
- Creating real-time inference endpoints
- Auto-scaling configuration
- A/B testing setup
- Endpoint monitoring

## 📋 Prerequisites

- Understanding of binary classification
- Basic knowledge of financial transactions
- Familiarity with scikit-learn
- AWS account with SageMaker access

## ⏱️ Estimated Time

- **Exercise**: 2-3 hours
- **Review & Discussion**: 1-2 hours
- **Deployment**: 1-2 hours
- **Total**: 4-6 hours

## 🚀 Getting Started

### Setup

```bash
cd 01-fraud-detection/exercises
jupyter notebook fraud_detection_exercise.ipynb
```

### Data Download

The synthetic fraud detection dataset will be generated as part of the exercise.

```python
# In the notebook
from utils import generate_fraud_dataset
df = generate_fraud_dataset(n_samples=100000, fraud_ratio=0.02)
```

## 💡 Key Concepts

### Evaluation Metrics for Fraud Detection

1. **Precision**: Of all predicted frauds, what % are actual frauds?
   - High precision = Few false alarms
   - Important for customer experience

2. **Recall**: Of all actual frauds, what % did we catch?
   - High recall = Catching most frauds
   - Important for financial loss prevention

3. **F1-Score**: Harmonic mean of precision and recall
   - Balanced metric

4. **Cost-Sensitive Metrics**: 
   - Cost of false positive (investigating legitimate transaction)
   - Cost of false negative (missing a fraudulent transaction)

### Production Considerations

- **Latency Requirements**: < 100ms for transaction approval
- **Model Refresh**: Daily/weekly retraining schedule
- **Feature Store**: Real-time feature computation
- **Monitoring**: Data drift detection, model performance tracking

## 🏗️ Architecture

```
┌─────────────┐
│ Transaction │
│   Stream    │
└──────┬──────┘
       │
       v
┌─────────────┐       ┌──────────────┐
│  Feature    │──────>│  SageMaker   │
│ Engineering │       │   Endpoint   │
└─────────────┘       └──────┬───────┘
                             │
                             v
                      ┌──────────────┐
                      │ Fraud Score  │
                      │  (0.0-1.0)   │
                      └──────────────┘
```

## 📖 Additional Resources

- [Handling Imbalanced Datasets](https://imbalanced-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP for Model Explainability](https://shap.readthedocs.io/)
- [SageMaker Real-time Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)

## ✅ Success Criteria

Students should achieve:
- [ ] AUC-ROC > 0.90
- [ ] Precision > 0.85 at recall of 0.70
- [ ] Successfully deploy model to SageMaker
- [ ] Implement SHAP explainability
- [ ] Set up basic monitoring

## 🤔 Discussion Questions

1. What is the business cost of a false positive vs false negative?
2. How would you handle concept drift in fraud patterns?
3. What features would you add for better fraud detection?
4. How would you explain a fraud prediction to a customer?
5. What are the latency vs accuracy trade-offs?

## 📝 Exercise Tasks

### Part 1: Data Exploration (30 min)
- Load and explore the fraud dataset
- Analyze class imbalance
- Visualize feature distributions
- Identify missing values

### Part 2: Preprocessing (45 min)
- Handle missing values
- Engineer temporal features
- Scale numerical features
- Handle imbalanced classes

### Part 3: Model Training (60 min)
- Train baseline logistic regression
- Train XGBoost model
- Train LightGBM model
- Compare model performance

### Part 4: Model Evaluation (30 min)
- Generate precision-recall curves
- Calculate cost-sensitive metrics
- Analyze feature importance
- Create SHAP explanations

### Part 5: Deployment (45 min)
- Package model for SageMaker
- Create inference endpoint
- Test endpoint with sample data
- Set up monitoring

## 🎓 Next Steps

After completing this module:
1. Review the solution notebook
2. Discuss results with instructor
3. Explore advanced techniques (deep learning, anomaly detection)
4. Move to Module 2: Customer Churn Prediction
