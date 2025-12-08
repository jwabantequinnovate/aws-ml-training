"""
Model evaluation utilities

Provides common evaluation metrics and visualization functions for all modules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        Calculate common classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            average: Averaging method for multi-class
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        if y_pred_proba is not None and average == 'binary':
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='.2f' if normalize else 'd',
            cmap='Blues', xticklabels=labels, yticklabels=labels
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = 'ROC Curve',
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            figsize: Figure size
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = 'Precision-Recall Curve',
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot precision-recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            figsize: Figure size
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_models(
        models: Dict[str, Dict],
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        figsize: Tuple[int, int] = (12, 6)
    ) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model results
            metrics: Metrics to compare
            figsize: Figure size
            
        Returns:
            Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in models.items():
            row = {'Model': model_name}
            for metric in metrics:
                if metric in results:
                    row[metric] = results[metric]
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Plot comparison
        df_plot = df_comparison.set_index('Model')
        df_plot.plot(kind='bar', figsize=figsize)
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
        
        return df_comparison
    
    @staticmethod
    def print_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> None:
        """
        Print detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
        """
        print("\nClassification Report:")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=labels))


class BusinessMetrics:
    """
    Calculate business-relevant metrics
    """
    
    @staticmethod
    def calculate_cost_sensitive_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_fp: float,
        cost_fn: float,
        cost_tp: float = 0,
        cost_tn: float = 0
    ) -> Dict[str, float]:
        """
        Calculate cost-sensitive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
            cost_tp: Cost of true positive
            cost_tn: Cost of true negative
            
        Returns:
            Dictionary with cost metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = (tp * cost_tp + tn * cost_tn + 
                     fp * cost_fp + fn * cost_fn)
        
        return {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'cost_tp': tp * cost_tp,
            'cost_tn': tn * cost_tn,
            'cost_fp': fp * cost_fp,
            'cost_fn': fn * cost_fn,
            'total_cost': total_cost
        }
    
    @staticmethod
    def calculate_lift(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_quantiles: int = 10
    ) -> pd.DataFrame:
        """
        Calculate lift chart
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_quantiles: Number of quantiles
            
        Returns:
            Lift chart dataframe
        """
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred_proba': y_pred_proba
        })
        
        df['quantile'] = pd.qcut(df['y_pred_proba'], n_quantiles, 
                                 labels=False, duplicates='drop')
        
        lift_data = df.groupby('quantile').agg({
            'y_true': ['sum', 'count']
        }).reset_index()
        
        lift_data.columns = ['quantile', 'positives', 'total']
        lift_data['positive_rate'] = lift_data['positives'] / lift_data['total']
        
        baseline_rate = y_true.sum() / len(y_true)
        lift_data['lift'] = lift_data['positive_rate'] / baseline_rate
        
        return lift_data


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot feature importance
    
    Args:
        feature_names: Names of features
        importances: Feature importance values
        top_n: Number of top features to show
        figsize: Figure size
    """
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'f1'
) -> Dict[str, float]:
    """
    Perform cross-validation
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target
        cv: Number of folds
        scoring: Scoring metric
        
    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'scores': scores.tolist()
    }
