"""
Visualization Module for Titanic Dataset
Provides comprehensive visualization utilities for EDA and model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_missing_values(df, save_path=None):
    """
    Visualize missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    save_path : str, optional
        Path to save the plot
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("✓ No missing values in the dataset")
        return
    
    missing_pct = (missing / len(df) * 100).round(2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(missing)), missing_pct.values, color='coral')
    ax.set_yticks(range(len(missing)))
    ax.set_yticklabels(missing.index)
    ax.set_xlabel('Percentage of Missing Values (%)', fontsize=12)
    ax.set_title('Missing Values Analysis', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (count, pct) in enumerate(zip(missing.values, missing_pct.values)):
        ax.text(pct + 1, i, f'{count} ({pct}%)', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Missing values plot saved to {save_path}")
    
    plt.show()


def plot_target_distribution(df, target_col='Survived', save_path=None):
    """
    Plot distribution of target variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_col : str
        Name of target column
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    target_counts = df[target_col].value_counts()
    axes[0].bar(target_counts.index, target_counts.values, color=['#e74c3c', '#2ecc71'], edgecolor='black')
    axes[0].set_xlabel(target_col, fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'{target_col} Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xticks(target_counts.index)
    axes[0].set_xticklabels(['Not Survived', 'Survived'])
    
    # Add value labels
    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    colors = ['#e74c3c', '#2ecc71']
    axes[1].pie(target_counts.values, labels=['Not Survived', 'Survived'], autopct='%1.1f%%',
                colors=colors, startangle=90, explode=(0.05, 0.05))
    axes[1].set_title(f'{target_col} Percentage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Target distribution plot saved to {save_path}")
    
    plt.show()


def plot_numerical_distributions(df, numerical_cols, save_path=None):
    """
    Plot distributions of numerical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    numerical_cols : list
        List of numerical column names
    save_path : str, optional
        Path to save the plot
    """
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend(fontsize=9)
    
    # Hide extra subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Numerical distributions plot saved to {save_path}")
    
    plt.show()


def plot_categorical_distributions(df, categorical_cols, save_path=None):
    """
    Plot distributions of categorical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    categorical_cols : list
        List of categorical column names
    save_path : str, optional
        Path to save the plot
    """
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 4))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        value_counts = df[col].value_counts()
        ax.bar(range(len(value_counts)), value_counts.values, color='teal', edgecolor='black')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + max(value_counts.values) * 0.02, str(v), ha='center', fontsize=9)
    
    # Hide extra subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Categorical distributions plot saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(df, save_path=None):
    """
    Plot correlation matrix heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    save_path : str, optional
        Path to save the plot
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.shape[1] < 2:
        print("✗ Not enough numerical columns for correlation matrix")
        return
    
    corr = numerical_df.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation matrix saved to {save_path}")
    
    plt.show()


def plot_feature_vs_target(df, feature_col, target_col='Survived', save_path=None):
    """
    Plot relationship between a feature and target variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    feature_col : str
        Name of feature column
    target_col : str
        Name of target column
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    if df[feature_col].dtype in ['object', 'category'] or df[feature_col].nunique() < 10:
        # Categorical or discrete feature
        pd.crosstab(df[feature_col], df[target_col]).plot(kind='bar', ax=axes[0], 
                                                           color=['#e74c3c', '#2ecc71'],
                                                           edgecolor='black')
        axes[0].set_xlabel(feature_col, fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title(f'{feature_col} vs {target_col} (Count)', fontsize=12, fontweight='bold')
        axes[0].legend(['Not Survived', 'Survived'])
        axes[0].tick_params(axis='x', rotation=45)
        
        # Survival rate
        survival_rate = df.groupby(feature_col)[target_col].mean()
        axes[1].bar(range(len(survival_rate)), survival_rate.values, color='steelblue', edgecolor='black')
        axes[1].set_xticks(range(len(survival_rate)))
        axes[1].set_xticklabels(survival_rate.index, rotation=45)
        axes[1].set_xlabel(feature_col, fontsize=12)
        axes[1].set_ylabel('Survival Rate', fontsize=12)
        axes[1].set_title(f'Survival Rate by {feature_col}', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(survival_rate.values):
            axes[1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=9)
    else:
        # Numerical feature
        for survived in [0, 1]:
            data = df[df[target_col] == survived][feature_col].dropna()
            axes[0].hist(data, bins=20, alpha=0.6, label=f'Survived={survived}', edgecolor='black')
        axes[0].set_xlabel(feature_col, fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'{feature_col} Distribution by {target_col}', fontsize=12, fontweight='bold')
        axes[0].legend(['Not Survived', 'Survived'])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Box plot
        df.boxplot(column=feature_col, by=target_col, ax=axes[1])
        axes[1].set_xlabel(target_col, fontsize=12)
        axes[1].set_ylabel(feature_col, fontsize=12)
        axes[1].set_title(f'{feature_col} by {target_col}', fontsize=12, fontweight='bold')
        axes[1].set_xticklabels(['Not Survived', 'Survived'])
        plt.suptitle('')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature vs target plot saved to {save_path}")
    
    plt.show()


def plot_boxplots(df, numerical_cols, save_path=None):
    """
    Plot boxplots for outlier detection
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    numerical_cols : list
        List of numerical column names
    save_path : str, optional
        Path to save the plot
    """
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        df.boxplot(column=col, ax=ax, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='black'),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'))
        ax.set_ylabel(col, fontsize=11)
        ax.set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Boxplots saved to {save_path}")
    
    plt.show()


def plot_pairplot(df, target_col='Survived', save_path=None):
    """
    Create pairplot for numerical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_col : str
        Name of target column for color coding
    save_path : str, optional
        Path to save the plot
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit to reasonable number of features
    if len(numerical_cols) > 6:
        print(f"⚠ Too many numerical features ({len(numerical_cols)}). Selecting top 5 + target.")
        # Select features with highest correlation to target
        if target_col in numerical_cols:
            corr_with_target = df[numerical_cols].corr()[target_col].abs().sort_values(ascending=False)
            selected_cols = corr_with_target.head(6).index.tolist()
        else:
            selected_cols = numerical_cols[:5] + [target_col]
    else:
        selected_cols = numerical_cols
    
    if target_col not in selected_cols:
        selected_cols.append(target_col)
    
    pairplot_df = df[selected_cols].copy()
    
    g = sns.pairplot(pairplot_df, hue=target_col, palette={0: '#e74c3c', 1: '#2ecc71'},
                     diag_kind='hist', plot_kws={'alpha': 0.6, 'edgecolor': 'black'},
                     height=2.5)
    g.fig.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Pairplot saved to {save_path}")
    
    plt.show()


def create_eda_summary(df, target_col='Survived'):
    """
    Create comprehensive EDA summary report
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_col : str
        Name of target column
        
    Returns:
    --------
    dict
        Dictionary containing EDA summary
    """
    summary = {
        'dataset_shape': df.shape,
        'numerical_features': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'target_distribution': df[target_col].value_counts().to_dict() if target_col in df.columns else None
    }
    
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nDataset Shape: {summary['dataset_shape'][0]} rows × {summary['dataset_shape'][1]} columns")
    print(f"\nNumerical Features ({len(summary['numerical_features'])}): {', '.join(summary['numerical_features'])}")
    print(f"\nCategorical Features ({len(summary['categorical_features'])}): {', '.join(summary['categorical_features'])}")
    print(f"\nDuplicate Rows: {summary['duplicates']}")
    
    if summary['target_distribution']:
        print(f"\nTarget Distribution ({target_col}):")
        for key, value in summary['target_distribution'].items():
            print(f"  {key}: {value} ({value/sum(summary['target_distribution'].values())*100:.1f}%)")
    
    return summary


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("Available functions:")
    print("  - plot_missing_values()")
    print("  - plot_target_distribution()")
    print("  - plot_numerical_distributions()")
    print("  - plot_categorical_distributions()")
    print("  - plot_correlation_matrix()")
    print("  - plot_feature_vs_target()")
    print("  - plot_boxplots()")
    print("  - plot_pairplot()")
    print("  - create_eda_summary()")
