"""
Unit tests for preprocessing utilities
"""

import pytest
import pandas as pd
import numpy as np
from utils.preprocessing import DataPreprocessor, TextPreprocessor


class TestDataPreprocessor:
    """Tests for DataPreprocessor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = DataPreprocessor()
        self.sample_df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'category_col': ['A', 'B', 'A', 'C', 'B']
        })
    
    def test_handle_missing_values_mean(self):
        """Test handling missing values with mean strategy"""
        result = self.preprocessor.handle_missing_values(
            self.sample_df, 
            strategy='mean'
        )
        assert result['numeric_col'].isnull().sum() == 0
        assert result['numeric_col'].iloc[2] == pytest.approx(3.0, rel=1e-5)
    
    def test_handle_missing_values_drop(self):
        """Test dropping rows with missing values"""
        result = self.preprocessor.handle_missing_values(
            self.sample_df, 
            strategy='drop'
        )
        assert len(result) == 4
        assert result['numeric_col'].isnull().sum() == 0
    
    def test_scale_features(self):
        """Test feature scaling"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaled = self.preprocessor.scale_features(X, method='standard', fit=True)
        
        # Check that mean is approximately 0 and std is approximately 1
        assert np.abs(scaled.mean()) < 1e-10
        assert np.abs(scaled.std() - 1.0) < 0.1
    
    def test_encode_categorical(self):
        """Test categorical encoding"""
        result = self.preprocessor.encode_categorical(
            self.sample_df,
            columns=['category_col'],
            method='label'
        )
        assert 'category_col_encoded' in result.columns
        assert result['category_col_encoded'].dtype == np.int64


class TestTextPreprocessor:
    """Tests for TextPreprocessor class"""
    
    def test_clean_text(self):
        """Test text cleaning"""
        text = "Check out https://example.com! #ML @user123"
        cleaned = TextPreprocessor.clean_text(text, lowercase=True)
        
        assert 'https://example.com' not in cleaned
        assert '#ML' not in cleaned
        assert '@user123' not in cleaned
    
    def test_clean_text_preserve_case(self):
        """Test text cleaning preserving case"""
        text = "Hello World"
        cleaned = TextPreprocessor.clean_text(text, lowercase=False)
        assert 'Hello' in cleaned
        assert cleaned != cleaned.lower()


def test_train_test_split_stratified():
    """Test stratified train-test split"""
    from utils.preprocessing import train_test_split_stratified
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_val, X_test, y_train, y_val, y_test = \
        train_test_split_stratified(X, y, test_size=0.2, val_size=0.2)
    
    # Check sizes
    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20
    
    # Check stratification (approximate due to rounding)
    train_ratio = y_train.mean()
    val_ratio = y_val.mean()
    test_ratio = y_test.mean()
    
    assert abs(train_ratio - y.mean()) < 0.1
    assert abs(val_ratio - y.mean()) < 0.2
    assert abs(test_ratio - y.mean()) < 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
