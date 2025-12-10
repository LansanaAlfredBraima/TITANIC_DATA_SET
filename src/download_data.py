"""
Dataset downloader for Titanic dataset
Downloads the dataset from a public source if not already present
"""

import pandas as pd
import os


def download_titanic_dataset(output_dir='data/raw'):
    """
    Download Titanic dataset from seaborn or create sample data
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'titanic.csv')
    
    if os.path.exists(filepath):
        print(f"✓ Dataset already exists at {filepath}")
        return filepath
    
    try:
        # Try to download from GitHub (original Kaggle format)
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        df.to_csv(filepath, index=False)
        print(f"✓ Titanic dataset downloaded successfully to {filepath}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return filepath
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("  Please download manually from:")
        print("  https://www.kaggle.com/c/titanic/data")
        print("  or")
        print("  https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        return None


if __name__ == "__main__":
    download_titanic_dataset()
