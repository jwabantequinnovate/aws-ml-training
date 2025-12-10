# Module 2: Customer Churn Prediction

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Analyze customer behavior patterns to predict churn
- Build ensemble models for churn prediction
- Implement batch inference patterns with SageMaker Batch Transform
- Compare multiple models systematically
- Handle time-series features in customer data
- Deploy models for periodic batch scoring

## 📊 Use Case Overview

Customer churn prediction helps businesses retain valuable customers by identifying those at risk of leaving.

- **Problem**: Predict which customers will churn in the next 30 days
- **Challenge**: Multiple factors influence churn, requires feature engineering
- **Solution**: Ensemble ML models with customer behavior features
- **Deployment**: Batch transform for monthly customer scoring

## 📚 Module Structure

### 1. Exercises
- `customer_churn_exercise.ipynb` - Main exercise notebook
- `feature_engineering_exercise.py` - Feature engineering challenges
- `model_comparison_exercise.py` - Model comparison framework

### 2. Solutions
- `customer_churn_solution.ipynb` - Complete solution
- `batch_inference.py` - Batch transform implementation
- `model_registry.py` - Model versioning and comparison

### 3. Data
- Sample customer churn dataset
- Customer behavior data
- Subscription history

## 🔧 Technical Topics Covered

### Customer Analytics
- Customer lifetime value (CLV) estimation
- Tenure and engagement analysis
- Service usage patterns
- Payment history analysis

### Feature Engineering
- **Temporal Features**: Account age, days since last activity
- **Aggregate Features**: Average monthly usage, total spend
- **Behavioral Features**: Service usage patterns, support tickets
- **Engagement Scores**: Interaction frequency, feature adoption

### Model Development
- **Random Forest**: Baseline ensemble model
- **Gradient Boosting**: XGBoost and LightGBM
- **Ensemble Stacking**: Combining multiple models
- **Neural Networks**: Deep learning approach (optional)

### Model Comparison Framework
- Cross-validation strategies
- Hyperparameter tuning
- Performance metrics comparison
- Business impact analysis

### SageMaker Batch Transform
- Large-scale batch predictions
- Scheduled batch jobs
- Cost optimization for batch inference
- Output aggregation and storage

## 📋 Prerequisites

- Completion of Module 1 (Fraud Detection)
- Understanding of ensemble methods
- Familiarity with customer analytics
- AWS SageMaker Batch Transform concepts

## ⏱️ Estimated Time

- **Exercise**: 2-3 hours
- **Model Comparison**: 1-2 hours
- **Batch Deployment**: 1-2 hours
- **Total**: 4-6 hours

## 🚀 Getting Started

```bash
cd 02-customer-churn/exercises
jupyter notebook customer_churn_exercise.ipynb
```

## 💡 Key Concepts

### Churn Definition

**Early Churn**: Customer leaves within first 3 months
**Standard Churn**: Customer cancels subscription
**Voluntary Churn**: Customer-initiated cancellation
**Involuntary Churn**: Payment failure, contract expiration

### Evaluation Metrics

1. **Accuracy**: Overall correctness (can be misleading)
2. **Precision**: Of predicted churners, how many actually churn?
3. **Recall**: Of actual churners, how many did we identify?
4. **F1-Score**: Balance between precision and recall
5. **Lift**: How much better than random targeting?

### Business Metrics

- **Retention Rate**: % of customers retained
- **Revenue Impact**: Value of prevented churn
- **Campaign Efficiency**: Cost per retained customer
- **ROI**: Return on retention investment

## 🏗️ Architecture - Batch Inference

```
┌──────────────┐
│  Customer DB │
└──────┬───────┘
       │
       v
┌──────────────┐      ┌─────────────────┐
│ Extract Data │─────>│  S3 Input       │
│   (Monthly)  │      │  Bucket         │
└──────────────┘      └────────┬────────┘
                               │
                               v
                      ┌─────────────────┐
                      │  SageMaker      │
                      │  Batch Transform│
                      └────────┬────────┘
                               │
                               v
                      ┌─────────────────┐
                      │  S3 Output      │
                      │  Bucket         │
                      └────────┬────────┘
                               │
                               v
                      ┌─────────────────┐
                      │  Churn Scores   │
                      │  + Actions      │
                      └─────────────────┘
```

## 📖 Additional Resources

- [Customer Churn Analysis](https://en.wikipedia.org/wiki/Customer_attrition)
- [SageMaker Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
- [Ensemble Learning Methods](https://scikit-learn.org/stable/modules/ensemble.html)

## ✅ Success Criteria

Students should achieve:
- [ ] Accuracy > 85%
- [ ] F1-Score > 0.75
- [ ] Successfully implement batch transform
- [ ] Create model comparison framework
- [ ] Build automated retraining pipeline

## 🤔 Discussion Questions

1. How do you define churn for your business?
2. What is the optimal time window for prediction?
3. How do you balance false positives vs false negatives?
4. What retention strategies can you implement?
5. How often should you retrain the model?

## 📝 Exercise Tasks

### Part 1: Data Exploration (30 min)
- Load customer churn dataset
- Analyze churn patterns
- Identify key features
- Visualize customer segments

### Part 2: Feature Engineering (60 min)
- Create temporal features
- Calculate aggregate statistics
- Build engagement scores
- Handle categorical variables

### Part 3: Model Training (60 min)
- Train multiple models
- Perform hyperparameter tuning
- Create ensemble models
- Compare performance

### Part 4: Model Comparison (30 min)
- Build comparison framework
- Analyze feature importance
- Calculate business metrics
- Select best model

### Part 5: Batch Deployment (45 min)
- Prepare batch transform job
- Deploy to SageMaker
- Process batch predictions
- Schedule periodic runs

## 🎓 Next Steps

After completing this module:
1. Review model comparison results
2. Discuss retention strategies
3. Explore advanced ensemble techniques
4. Move to Module 3: Text Classification
