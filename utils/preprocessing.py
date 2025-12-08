"""
Common preprocessing utilities for ML training

This module provides reusable preprocessing functions for all training modules.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from typing import List, Tuple, Optional


class DataPreprocessor:
    """
    Base class for data preprocessing across different use cases
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'mean',
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in dataframe
        
        Args:
            df: Input dataframe
            strategy: 'mean', 'median', 'mode', 'drop', or 'fill'
            fill_value: Value to fill if strategy is 'fill'
            
        Returns:
            Dataframe with handled missing values
        """
        df_copy = df.copy()
        
        if strategy == 'drop':
            return df_copy.dropna()
        
        for col in df_copy.columns:
            if df_copy[col].isnull().sum() > 0:
                if strategy == 'mean':
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                elif strategy == 'mode':
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
                elif strategy == 'fill':
                    df_copy[col].fillna(fill_value, inplace=True)
                    
        return df_copy
    
    def scale_features(
        self, 
        X: np.ndarray, 
        method: str = 'standard',
        fit: bool = True
    ) -> np.ndarray:
        """
        Scale numerical features
        
        Args:
            X: Input features
            method: 'standard' or 'minmax'
            fit: Whether to fit the scaler
            
        Returns:
            Scaled features
        """
        if method not in self.scalers:
            if method == 'standard':
                self.scalers[method] = StandardScaler()
            elif method == 'minmax':
                self.scalers[method] = MinMaxScaler()
        
        if fit:
            return self.scalers[method].fit_transform(X)
        else:
            return self.scalers[method].transform(X)
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = 'label'
    ) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input dataframe
            columns: List of categorical columns
            method: 'label' or 'onehot'
            
        Returns:
            Dataframe with encoded categorical variables
        """
        df_copy = df.copy()
        
        for col in columns:
            if method == 'label':
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                df_copy[f'{col}_encoded'] = self.encoders[col].fit_transform(df_copy[col])
            elif method == 'onehot':
                dummies = pd.get_dummies(df_copy[col], prefix=col)
                df_copy = pd.concat([df_copy, dummies], axis=1)
                
        return df_copy
    
    def create_temporal_features(
        self, 
        df: pd.DataFrame, 
        date_column: str
    ) -> pd.DataFrame:
        """
        Create temporal features from date column
        
        Args:
            df: Input dataframe
            date_column: Name of date column
            
        Returns:
            Dataframe with temporal features
        """
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        df_copy['year'] = df_copy[date_column].dt.year
        df_copy['month'] = df_copy[date_column].dt.month
        df_copy['day'] = df_copy[date_column].dt.day
        df_copy['dayofweek'] = df_copy[date_column].dt.dayofweek
        df_copy['quarter'] = df_copy[date_column].dt.quarter
        df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)
        
        return df_copy


class TextPreprocessor:
    """
    Text preprocessing utilities for NLP tasks
    """
    
    @staticmethod
    def clean_text(text: str, lowercase: bool = True) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            lowercase: Whether to convert to lowercase
            
        Returns:
            Cleaned text
        """
        import re
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        if lowercase:
            text = text.lower()
            
        return text
    
    @staticmethod
    def remove_stopwords(text: str, language: str = 'english') -> str:
        """
        Remove stopwords from text
        
        Args:
            text: Input text
            language: Language for stopwords
            
        Returns:
            Text without stopwords
        """
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words(language))
            words = text.split()
            filtered_words = [word for word in words if word not in stop_words]
            return ' '.join(filtered_words)
        except (ImportError, LookupError) as e:
            # If NLTK not available or stopwords not downloaded, return original text
            print(f"Warning: NLTK stopwords not available ({e}). Returning original text.")
            return text


def train_test_split_stratified(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets with stratification
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def detect_outliers(
    df: pd.DataFrame, 
    columns: List[str], 
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers in numerical columns
    
    Args:
        df: Input dataframe
        columns: Columns to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Dataframe with outlier flags
    """
    df_copy = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy[f'{col}_outlier'] = (
                (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
            ).astype(int)
        elif method == 'zscore':
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            z_scores = np.abs((df_copy[col] - mean) / std)
            df_copy[f'{col}_outlier'] = (z_scores > threshold).astype(int)
    
    return df_copy
