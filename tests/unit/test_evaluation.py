"""
Unit tests for model evaluation module

Tests metric calculations, visualizations, and reporting.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.ml_toolkit.evaluation import ModelEvaluator


class TestModelEvaluator:
    """Test suite for ModelEvaluator class"""
    
    def test_calculate_binary_metrics(self):
        """Test metric calculation for binary classification"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, average='binary')
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Verify accuracy calculation
        expected_accuracy = accuracy_score(y_true, y_pred)
        assert abs(metrics['accuracy'] - expected_accuracy) < 1e-6
        
        # All metrics should be between 0 and 1
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} out of range: {value}"
    
    def test_calculate_multiclass_metrics(self):
        """Test metric calculation for multi-class classification"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 2, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 1, 1, 0, 2, 2, 0, 1])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, average='weighted')
        
        assert all(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1'])
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_calculate_metrics_with_probabilities(self):
        """Test metrics with probability predictions"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])
        
        metrics = evaluator.calculate_metrics(
            y_true, y_pred, y_pred_proba=y_proba[:, 1], average='binary'
        )
        
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_confusion_matrix_generation(self):
        """Test confusion matrix calculation"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        cm = evaluator.get_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
    
    def test_classification_report(self):
        """Test classification report generation"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 2, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 0, 2, 2])
        
        report = evaluator.get_classification_report(y_true, y_pred)
        
        assert isinstance(report, dict)
        assert 'accuracy' in report
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = y_true.copy()
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, average='binary')
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_all_wrong_predictions(self):
        """Test metrics with all incorrect predictions"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, average='binary')
        
        assert metrics['accuracy'] == 0.0
    
    def test_edge_case_all_same_class(self):
        """Test handling of edge case with single class"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1, 1])
        
        # Should not raise error
        metrics = evaluator.calculate_metrics(y_true, y_pred, average='binary')
        assert 'accuracy' in metrics
    
    @pytest.mark.parametrize("average_method", ['binary', 'micro', 'macro', 'weighted'])
    def test_different_averaging_methods(self, average_method):
        """Test different averaging methods for multi-class"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 2, 1, 2, 0, 1, 2, 0, 1] * 3)
        y_pred = np.array([0, 1, 2, 1, 1, 0, 2, 2, 0, 1] * 3)
        
        if average_method == 'binary':
            # Binary requires 2 classes
            y_true = (y_true > 0).astype(int)
            y_pred = (y_pred > 0).astype(int)
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, average=average_method)
        
        assert all(0 <= v <= 1 for v in metrics.values())


class TestMetricVisualization:
    """Test visualization functions"""
    
    def test_plot_confusion_matrix(self, tmp_path):
        """Test confusion matrix plotting"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0] * 5)
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0] * 5)
        
        output_path = tmp_path / "confusion_matrix.png"
        
        # Should not raise error
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            labels=['Negative', 'Positive'],
            save_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_plot_roc_curve(self, tmp_path):
        """Test ROC curve plotting"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.85, 0.95, 0.15, 0.25])
        
        output_path = tmp_path / "roc_curve.png"
        
        evaluator.plot_roc_curve(y_true, y_scores, save_path=str(output_path))
        
        assert output_path.exists()


class TestCostSensitiveMetrics:
    """Test cost-sensitive evaluation metrics"""
    
    def test_cost_sensitive_accuracy(self):
        """Test custom cost-sensitive accuracy calculation"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        # False negatives cost more than false positives (e.g., fraud detection)
        cost_matrix = {
            'tn': 0, 'fp': 1,  # FP costs 1
            'fn': 5, 'tp': 0   # FN costs 5
        }
        
        total_cost = evaluator.calculate_cost(y_true, y_pred, cost_matrix)
        
        assert isinstance(total_cost, (int, float))
        assert total_cost >= 0
