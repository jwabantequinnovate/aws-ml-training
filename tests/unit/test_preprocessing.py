"""
Unit tests for data preprocessing module

These tests verify preprocessing logic without AWS dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from src.ml_toolkit.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""
    
    def test_initialization(self):
        """Test preprocessor initializes correctly"""
        preprocessor = DataPreprocessor()
        assert isinstance(preprocessor.scalers, dict)
        assert isinstance(preprocessor.encoders, dict)
    
    def test_handle_missing_values_mean(self, sample_fraud_data):
        """Test mean imputation strategy"""
        preprocessor = DataPreprocessor()
        df = sample_fraud_data.copy()
        
        # Introduce missing values
        df.loc[0:10, 'amount'] = np.nan
        original_null_count = df['amount'].isnull().sum()
        
        result = preprocessor.handle_missing_values(df, strategy='mean')
        
        assert result['amount'].isnull().sum() == 0
        assert original_null_count > 0
        assert result.shape[0] == df.shape[0]
    
    def test_handle_missing_values_median(self, sample_fraud_data):
        """Test median imputation strategy"""
        preprocessor = DataPreprocessor()
        df = sample_fraud_data.copy()
        df.loc[0:10, 'amount'] = np.nan
        
        result = preprocessor.handle_missing_values(df, strategy='median')
        
        assert result['amount'].isnull().sum() == 0
    
    def test_handle_missing_values_drop(self, sample_fraud_data):
        """Test dropping rows with missing values"""
        preprocessor = DataPreprocessor()
        df = sample_fraud_data.copy()
        df.loc[0:10, 'amount'] = np.nan
        
        result = preprocessor.handle_missing_values(df, strategy='drop')
        
        assert result.shape[0] < df.shape[0]
        assert result.isnull().sum().sum() == 0
    
    def test_encode_categorical_variables(self, sample_churn_data):
        """Test categorical encoding"""
        preprocessor = DataPreprocessor()
        df = sample_churn_data.copy()
        
        categorical_cols = ['contract_type', 'payment_method']
        result = preprocessor.encode_categorical(df, categorical_cols)
        
        # Check that categorical columns are encoded
        for col in categorical_cols:
            assert pd.api.types.is_numeric_dtype(result[col])
    
    def test_scale_numerical_features(self, sample_fraud_data):
        """Test feature scaling"""
        preprocessor = DataPreprocessor()
        df = sample_fraud_data.copy()
        
        numerical_cols = ['amount', 'transaction_hour']
        result = preprocessor.scale_features(df, numerical_cols, method='standard')
        
        # Check scaling applied
        for col in numerical_cols:
            # StandardScaler should produce mean ~0 and std ~1
            assert abs(result[col].mean()) < 0.1
            assert abs(result[col].std() - 1.0) < 0.1
    
    def test_feature_engineering_invalid_data(self):
        """Test handling of invalid input data"""
        preprocessor = DataPreprocessor()
        
        with pytest.raises((ValueError, KeyError)):
            preprocessor.handle_missing_values(pd.DataFrame(), strategy='mean')
    
    @pytest.mark.parametrize("strategy", ["mean", "median", "drop"])
    def test_missing_value_strategies(self, sample_fraud_data, strategy):
        """Test all missing value strategies work"""
        preprocessor = DataPreprocessor()
        df = sample_fraud_data.copy()
        df.loc[0:10, 'amount'] = np.nan
        
        result = preprocessor.handle_missing_values(df, strategy=strategy)
        
        if strategy == 'drop':
            assert result.shape[0] < df.shape[0]
        else:
            assert result.shape[0] == df.shape[0]


class TestTextPreprocessor:
    """Test suite for text preprocessing"""
    
    def test_text_cleaning(self, sample_text_data):
        """Test basic text cleaning operations"""
        from src.ml_toolkit.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        texts, _ = sample_text_data
        
        # Clean text (lowercase, remove special chars, etc)
        cleaned = [preprocessor.clean_text(text) for text in texts]
        
        assert all(isinstance(text, str) for text in cleaned)
        assert all(len(text) > 0 for text in cleaned)
    
    def test_tokenization(self, sample_text_data):
        """Test text tokenization"""
        from src.ml_toolkit.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        texts, _ = sample_text_data
        
        tokens = [preprocessor.tokenize(text) for text in texts]
        
        assert all(isinstance(token_list, list) for token_list in tokens)
        assert all(len(token_list) > 0 for token_list in tokens)
    
    def test_stopword_removal(self):
        """Test stopword removal"""
        from src.ml_toolkit.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        text = "this is a test with many common stopwords"
        
        cleaned = preprocessor.remove_stopwords(text)
        
        assert "test" in cleaned.lower()
        assert cleaned != text  # Should be different after stopword removal


@pytest.mark.unit
class TestFeatureEngineering:
    """Test feature engineering utilities"""
    
    def test_create_interaction_features(self, sample_fraud_data):
        """Test interaction feature creation"""
        preprocessor = DataPreprocessor()
        df = sample_fraud_data.copy()
        
        # Create interaction between amount and hour
        result = preprocessor.create_interactions(
            df, [('amount', 'transaction_hour')]
        )
        
        assert 'amount_x_transaction_hour' in result.columns
        assert result.shape[1] > df.shape[1]
    
    def test_create_polynomial_features(self, sample_fraud_data):
        """Test polynomial feature creation"""
        preprocessor = DataPreprocessor()
        df = sample_fraud_data[['amount']].copy()
        
        result = preprocessor.create_polynomial_features(df, ['amount'], degree=2)
        
        assert result.shape[1] > df.shape[1]
        assert f'amount^2' in result.columns
