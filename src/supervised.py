"""
Supervised Learning Module for Titanic Dataset
Implements multiple classification models and evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')


class SupervisedModels:
    """Class to handle multiple supervised learning models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize multiple classification models"""
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        print(f"✓ Initialized {len(self.models)} models")
        return self.models
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found")
            return None
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        print(f"✓ {model_name} trained successfully")
        return model
    
    def train_all_models(self, X_train, y_train):
        """
        Train all initialized models
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        for name in self.models.keys():
            self.train_model(name, X_train, y_train)
        
        print(f"\n✓ All {len(self.models)} models trained")
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to evaluate
        X_test : pd.DataFrame
            Testing features
        y_test : pd.Series
            Testing target
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found")
            return None
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Testing features
        y_test : pd.Series
            Testing target
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all model results
        """
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        
        results_list = []
        
        for name in self.models.keys():
            metrics = self.evaluate_model(name, X_test, y_test)
            results_list.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics.get('roc_auc', np.nan)
            })
        
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "-"*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("-"*60)
        print(results_df.to_string(index=False))
        
        # Identify best model
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n✓ Best Model: {self.best_model_name}")
        print(f"  Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
        
        return results_df
    
    def cross_validate(self, model_name, X, y, cv=5):
        """
        Perform cross-validation on a model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        cv : int
            Number of folds
            
        Returns:
        --------
        dict
            Cross-validation scores
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found")
            return None
        
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        print(f"\n{model_name} - Cross-Validation Results ({cv}-fold):")
        print(f"  Scores: {scores}")
        print(f"  Mean Accuracy: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
        
        return cv_results
    
    def get_feature_importance(self, model_name, feature_names):
        """
        Get feature importance for tree-based models
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_names : list
            List of feature names
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature importance
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        else:
            print(f"✗ {model_name} does not support feature importance")
            return None
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
        filepath : str
            Path to save the model
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found")
            return
        
        joblib.dump(self.models[model_name], filepath)
        print(f"✓ {model_name} saved to {filepath}")
    
    def load_model(self, filepath, model_name):
        """
        Load a trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        model_name : str
            Name to assign to the loaded model
        """
        try:
            self.models[model_name] = joblib.load(filepath)
            print(f"✓ {model_name} loaded from {filepath}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : array
        Confusion matrix
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, model_name, save_path=None):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_test : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Plot comparison of model performances
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model results
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        results_sorted = results_df.sort_values(metric, ascending=True)
        
        ax.barh(results_sorted['Model'], results_sorted[metric], color='steelblue')
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(results_sorted[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Supervised Learning module loaded successfully!")
    print("Available class: SupervisedModels")
    print("Available functions:")
    print("  - plot_confusion_matrix()")
    print("  - plot_roc_curve()")
    print("  - plot_model_comparison()")
