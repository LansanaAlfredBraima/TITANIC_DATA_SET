# Machine Learning Group Project: Titanic Survival Prediction

## Overview

This project applies supervised and unsupervised machine learning techniques to the Titanic dataset to predict passenger survival and discover patterns in the data.

## Dataset

**Source**: [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

**Description**: The dataset contains information about passengers aboard the Titanic, including demographics, ticket class, and survival status.

**Features**:
- PassengerId: Unique identifier
- Survived: Survival status (0 = No, 1 = Yes)
- Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Name: Passenger name
- Sex: Gender
- Age: Age in years
- SibSp: Number of siblings/spouses aboard
- Parch: Number of parents/children aboard
- Ticket: Ticket number
- Fare: Passenger fare
- Cabin: Cabin number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Structure

```
PROJECT/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned and processed data
├── notebooks/
│   └── ml_analysis.ipynb # Main analysis notebook
├── src/
│   ├── preprocessing.py  # Data preprocessing functions
│   ├── supervised.py     # Supervised learning models
│   ├── unsupervised.py   # Unsupervised learning models
│   └── visualization.py  # Visualization utilities
├── outputs/
│   ├── figures/          # Generated plots and charts
│   └── models/           # Saved model files
├── docs/
│   ├── poster.html       # Project poster
│   └── report.md         # Final report
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   - Visit [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
   - Download `train.csv` and `test.csv`
   - Place them in the `data/raw/` directory

## Usage

### Running the Analysis

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open and run** `notebooks/ml_analysis.ipynb`

### Using Individual Modules

```python
# Import preprocessing functions
from src.preprocessing import load_data, clean_data, perform_eda

# Import supervised learning models
from src.supervised import train_models, evaluate_models

# Import unsupervised learning functions
from src.unsupervised import perform_clustering, apply_pca
```

## Project Objectives

1. **Data Preprocessing**: Handle missing values, outliers, and perform feature engineering
2. **Exploratory Data Analysis**: Understand patterns and relationships in the data
3. **Supervised Learning**: Build classification models to predict survival
4. **Unsupervised Learning**: Discover passenger segments through clustering
5. **Insights**: Extract meaningful conclusions and recommendations

## Assessment Criteria

- Data Preprocessing and EDA: 20 marks
- Supervised Learning Models: 30 marks
- Unsupervised Learning Models: 20 marks
- Insights and Conclusions: 15 marks
- Poster: 10 marks
- Printed Documentation: 5 marks

**Total**: 100 marks

## Team Information

**Course**: Data Science Fundamentals  
**Institution**: LUCT  
**Academic Year**: 2024/2025  
**Semester**: Year 3, Semester 1

## License

This project is for educational purposes as part of a university assignment.

## Acknowledgments

- Dataset provided by Kaggle
- Titanic data originally from Encyclopedia Titanica
