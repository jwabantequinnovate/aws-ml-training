"""
Pytest configuration and shared fixtures

This file contains fixtures that are available to all test files.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock


@pytest.fixture
def sample_fraud_data() -> pd.DataFrame:
    """Generate realistic fraud detection dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    # Legitimate transactions cluster around certain patterns
    legitimate = pd.DataFrame({
        'amount': np.random.gamma(2, 50, n_samples // 2),
        'transaction_hour': np.random.randint(8, 22, n_samples // 2),
        'days_since_last': np.random.poisson(7, n_samples // 2),
        'merchant_category': np.random.choice(['grocery', 'gas', 'dining'], n_samples // 2),
        'is_fraud': 0
    })
    
    # Fraudulent transactions have different patterns
    fraudulent = pd.DataFrame({
        'amount': np.random.gamma(5, 200, n_samples // 2),
        'transaction_hour': np.random.randint(0, 24, n_samples // 2),
        'days_since_last': np.random.poisson(30, n_samples // 2),
        'merchant_category': np.random.choice(['electronics', 'jewelry', 'online'], n_samples // 2),
        'is_fraud': 1
    })
    
    return pd.concat([legitimate, fraudulent], ignore_index=True).sample(frac=1, random_state=42)


@pytest.fixture
def sample_churn_data() -> pd.DataFrame:
    """Generate realistic customer churn dataset"""
    np.random.seed(42)
    n_samples = 500
    
    return pd.DataFrame({
        'tenure': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'churn': np.random.binomial(1, 0.27, n_samples)
    })


@pytest.fixture
def sample_text_data() -> Tuple[list, list]:
    """Generate realistic support ticket data"""
    tickets = [
        "Cannot log into my account, getting error 403",
        "Application crashes when I click submit button",
        "How do I reset my password?",
        "Need help setting up email notifications",
        "Great product! Works perfectly for our team",
        "Billing question about invoice #12345",
        "Software update failed with error code XYZ",
        "Love the new features in latest release!",
    ]
    
    labels = [
        'technical', 'technical', 'account', 'account',
        'feedback', 'billing', 'technical', 'feedback'
    ]
    
    return tickets, labels


@pytest.fixture
def sample_sentiment_data() -> Tuple[list, list]:
    """Generate realistic product reviews"""
    reviews = [
        "This product exceeded my expectations! Highly recommend.",
        "Terrible experience, waste of money. Do not buy.",
        "It's okay, nothing special but gets the job done.",
        "Absolutely love it! Best purchase I've made this year.",
        "Disappointing quality, broke after one week of use.",
        "Average product, met basic expectations.",
        "Outstanding! Will definitely buy again.",
        "Not worth the price, better alternatives available.",
    ]
    
    sentiments = [
        'positive', 'negative', 'neutral', 'positive',
        'negative', 'neutral', 'positive', 'negative'
    ]
    
    return reviews, sentiments


@pytest.fixture
def mock_sagemaker_session():
    """Mock SageMaker session for testing without AWS calls"""
    mock_session = MagicMock()
    mock_session.default_bucket.return_value = "test-bucket"
    mock_session.boto_region_name = "us-east-1"
    mock_session.upload_data.return_value = "s3://test-bucket/test-path"
    return mock_session


@pytest.fixture
def mock_sagemaker_client():
    """Mock SageMaker boto3 client"""
    mock_client = MagicMock()
    mock_client.describe_endpoint.return_value = {
        'EndpointStatus': 'InService',
        'EndpointArn': 'arn:aws:sagemaker:us-east-1:123456789:endpoint/test'
    }
    return mock_client


@pytest.fixture
def temp_model_dir(tmp_path) -> Path:
    """Create temporary directory for model artifacts"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def sample_trained_model():
    """Create a simple trained model for testing"""
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    return model
