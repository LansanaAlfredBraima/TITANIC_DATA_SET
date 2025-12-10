"""
Data Preprocessing Module for Titanic Dataset
Handles data loading, cleaning, transformation, and exploratory data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Load the Titanic dataset from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None


def explore_data(df):
    """
    Perform initial data exploration
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    dict
        Dictionary containing exploration results
    """
    exploration = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes,
        'missing_values': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'duplicates': df.duplicated().sum(),
        'numerical_summary': df.describe(),
        'categorical_summary': df.describe(include=['object'])
    }
    
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    print(f"\nShape: {exploration['shape'][0]} rows × {exploration['shape'][1]} columns")
    print(f"\nDuplicate rows: {exploration['duplicates']}")
    
    print("\n" + "-"*60)
    print("MISSING VALUES")
    print("-"*60)
    missing_df = pd.DataFrame({
        'Column': exploration['missing_values'].index,
        'Missing Count': exploration['missing_values'].values,
        'Percentage': exploration['missing_percentage'].values
    })
    print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))
    
    return exploration


def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with missing values handled
    """
    df_clean = df.copy()
    
    # Age: Fill with median grouped by Pclass and Sex
    df_clean['Age'] = df_clean.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Embarked: Fill with mode
    if df_clean['Embarked'].isnull().sum() > 0:
        df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    
    # Cabin: Create binary feature for cabin known/unknown
    df_clean['Cabin_Known'] = df_clean['Cabin'].notna().astype(int)
    
    # Fare: Fill with median
    if df_clean['Fare'].isnull().sum() > 0:
        df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
    
    print("\n✓ Missing values handled successfully")
    return df_clean


def feature_engineering(df):
    """
    Create new features from existing ones
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with engineered features
    """
    df_eng = df.copy()
    
    # Family size
    df_eng['FamilySize'] = df_eng['SibSp'] + df_eng['Parch'] + 1
    
    # Is alone
    df_eng['IsAlone'] = (df_eng['FamilySize'] == 1).astype(int)
    
    # Title extraction from name
    df_eng['Title'] = df_eng['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
        'Capt': 'Rare', 'Ms': 'Miss'
    }
    df_eng['Title'] = df_eng['Title'].map(title_mapping)
    df_eng['Title'].fillna('Rare', inplace=True)
    
    # Age groups
    df_eng['AgeGroup'] = pd.cut(df_eng['Age'], bins=[0, 12, 18, 35, 60, 100],
                                 labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Fare per person
    df_eng['FarePerPerson'] = df_eng['Fare'] / df_eng['FamilySize']
    
    print("\n✓ Feature engineering completed")
    print(f"  New features: FamilySize, IsAlone, Title, AgeGroup, FarePerPerson, Cabin_Known")
    
    return df_eng


def detect_outliers(df, columns):
    """
    Detect outliers using IQR method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    columns : list
        List of column names to check for outliers
        
    Returns:
    --------
    dict
        Dictionary with outlier information
    """
    outliers_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_info[col] = {
            'count': len(outliers),
            'percentage': round(len(outliers) / len(df) * 100, 2),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    print("\n" + "-"*60)
    print("OUTLIER DETECTION (IQR Method)")
    print("-"*60)
    for col, info in outliers_info.items():
        print(f"{col}: {info['count']} outliers ({info['percentage']}%)")
    
    return outliers_info


def encode_categorical(df, target_col='Survived'):
    """
    Encode categorical variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_col : str
        Name of target column to exclude from encoding
        
    Returns:
    --------
    pd.DataFrame
        Dataset with encoded categorical variables
    """
    df_encoded = df.copy()
    
    # Binary encoding for Sex
    df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
    
    # One-hot encoding for Embarked
    df_encoded = pd.get_dummies(df_encoded, columns=['Embarked'], prefix='Embarked', drop_first=True)
    
    # One-hot encoding for Title
    df_encoded = pd.get_dummies(df_encoded, columns=['Title'], prefix='Title', drop_first=True)
    
    # One-hot encoding for AgeGroup
    df_encoded = pd.get_dummies(df_encoded, columns=['AgeGroup'], prefix='AgeGroup', drop_first=True)
    
    print("\n✓ Categorical variables encoded")
    
    return df_encoded


def prepare_features(df, target_col='Survived'):
    """
    Prepare final feature set for modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_col : str
        Name of target column
        
    Returns:
    --------
    tuple
        (X, y) features and target
    """
    # Drop unnecessary columns
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    
    # Keep only columns that exist
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    df_model = df.drop(columns=drop_cols)
    
    # Separate features and target
    if target_col in df_model.columns:
        X = df_model.drop(columns=[target_col])
        y = df_model[target_col]
    else:
        X = df_model
        y = None
    
    print(f"\n✓ Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n✓ Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    # Identify numerical columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fit on training data and transform both sets
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"\n✓ Features scaled: {len(numerical_cols)} numerical columns")
    
    return X_train_scaled, X_test_scaled, scaler


def save_processed_data(df, filepath):
    """
    Save processed dataset to CSV
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed dataset
    filepath : str
        Output file path
    """
    df.to_csv(filepath, index=False)
    print(f"\n✓ Processed data saved to: {filepath}")


if __name__ == "__main__":
    print("Preprocessing module loaded successfully!")
    print("Available functions:")
    print("  - load_data()")
    print("  - explore_data()")
    print("  - handle_missing_values()")
    print("  - feature_engineering()")
    print("  - detect_outliers()")
    print("  - encode_categorical()")
    print("  - prepare_features()")
    print("  - split_data()")
    print("  - scale_features()")
