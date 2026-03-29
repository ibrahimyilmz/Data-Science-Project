"""
Script to translate Turkish comments to English in all Python files
"""

import os
import re

# Translation dictionary for common Turkish phrases and words
translations = {
    # Module/class descriptions
    "Data Preparation and Balancing": "Data Preparation and Balancing",
    "Person 1 (Data Architect)": "Person 1 (Data Architect)",
    "Person 2 (Model Architect)": "Person 2 (Model Architect)",
    "Person 3 (Visualization Architect)": "Person 3 (Visualization Architect)",
    "Person 2's classification and forecasting models": "Person 2's classification and forecasting models",
    "labeled data": "labeled data",
    "prepares": "prepares",
    "Functions": "Functions",
    "Features": "Features",
    
    # Common phrases
    "Data preprocessing and preparation class": "Data preprocessing and preparation class",
    "Dataset balancing (equal RS/RP distribution)": "Dataset balancing (equal RS/RP distribution)",
    "Train/test split preserving time series structure": "Train/test split preserving time series structure",
    "Missing value handling": "Missing value handling",
    "Feature normalization": "Feature normalization",
    
    # Method descriptions
    "Initialize DataPreprocessor": "Initialize DataPreprocessor",
    "Test set ratio (0-1)": "Test set ratio (0-1)",
    "Random seed for reproducibility": "Random seed for reproducibility",
    "Balances dataset to ensure equal number of RS and RP examples in test set": "Balances dataset to ensure equal number of RS and RP examples in test set",
    "Method: Minority class oversampling, majority class undersampling": "Method: Minority class oversampling, majority class undersampling",
    "Original dataset": "Original dataset",
    "Label column name (contains 'RS' or 'RP')": "Label column name (contains 'RS' or 'RP')",
    "Balanced dataset": "Balanced dataset",
    "Check class distribution": "Check class distribution",
    "Original class distribution": "Original class distribution",
    "Take equal number of samples from each class": "Take equal number of samples from each class",
    "Balanced class distribution": "Balanced class distribution",
    "Dataset balancing error": "Dataset balancing error",
    
    # Zaman serisi ile ilgili
    "Train/test split preserving time series structure yapar": "Performs train/test split while preserving time series structure",
    "Uses TimeSeriesSplit to prevent data leakage": "Uses TimeSeriesSplit to prevent data leakage",
    "Dataset": "Dataset",
    "Time column name": "Time column name",
    "List of feature columns": "List of feature columns",
    "Sort by time column": "Sort by time column",
    "If features_cols not specified, use all numeric columns": "If features_cols not specified, use all numeric columns",
    "Time series split": "Time series split",
    "Training set size": "Training set size",
    "Test set size": "Test set size",
    "Number of features": "Number of features",
    "Train/Test split error": "Train/Test split error",
    
    # Feature normalization
    "Feature normalization uygular": "Applies feature normalization",
    "Training features": "Training features",
    "Test features": "Test features",
    "Normalization method ('standard' or 'minmax')": "Normalization method ('standard' or 'minmax')",
    "Unknown normalization method": "Unknown normalization method",
    "STANDARD normalization applied": "STANDARD normalization applied",
    "MINMAX normalization applied": "MINMAX normalization applied",
    "normalization applied": "normalization applied",
    "Normalization error": "Normalization error",
    
    # Missing value handling
    "Handles missing values": "Handles missing values",
    "Processing method ('interpolate', 'forward_fill', 'backward_fill')": "Processing method ('interpolate', 'forward_fill', 'backward_fill')",
    "Dataset with missing values handled": "Dataset with missing values handled",
    "Unknown missing value handling method": "Unknown missing value handling method",
    "Missing values handled using": "Missing values handled using",
    "method": "method",
    "Missing value handling errorsı": "Missing value handling error",
    
    # Aykırı değer
    "Detects and removes outliers": "Detects and removes outliers",
    "Data (n_samples x n_features)": "Data (n_samples x n_features)",
    "Method ('iqr' or 'zscore')": "Method ('iqr' or 'zscore')",
    "Threshold value": "Threshold value",
    "Data with outliers removed": "Data with outliers removed",
    "Unknown outlier removal method": "Unknown outlier removal method",
    "Outliers removed:": "Outliers removed:",
    "samples": "samples",
    "Outlier removal error": "Outlier removal error",
    
    # Lag features
    "Creates lagged (lag) features for forecasting model": "Creates lagged (lag) features for forecasting model",
    "Yesterday's consumption (lag=24), last week's (lag=7), etc": "Yesterday's consumption (lag=24), last week's (lag=7), etc",
    "Time series dataset": "Time series dataset",
    "Value column name": "Value column name",
    "Lag values to create (in hours)": "Lag values to create (in hours)",
    "Dataset with lag features added": "Dataset with lag features added",
    "Remove NaN values from first rows": "Remove NaN values from first rows",
    "Lag features created:": "Lag features created:",
    "Lag features creation error": "Lag features creation error",
    
    # Test code
    "Test code": "Test code",
    "Model Preparation Module - Test": "Model Preparation Module - Test",
    "Create sample dataset": "Create sample dataset",
    "Generate fake data": "Generate fake data",
    "Original dataset şekli": "Original dataset shape",
    "Class distribution": "Class distribution",
    "Test DataPreprocessor": "Test DataPreprocessor",
    "Dataset balancing": "Dataset balancing",
    "Balanced dataset şekli": "Balanced dataset shape",
    "X_train shape": "X_train shape",
    "X_test shape": "X_test shape",
    "Normalization": "Normalization",
    "X_train (normalized) mean:": "X_train (normalized) mean:",
    "X_train (normalized) std:": "X_train (normalized) std:",
    
    # Logger
    "Logger configuration": "Logger configuration",
    
    # Classification module
    "Classification Models - RS/RP Classification": "Classification Models - RS/RP Classification",
    "Classifies data as RS or RP": "Classifies data as RS or RP",
    "Binary classification problem": "Binary classification problem",
    "Base Classifier abstract class": "Base Classifier abstract class",
    "Logistic Regression Classifier": "Logistic Regression Classifier",
    "Neural Network Classifier (PyTorch based)": "Neural Network Classifier (PyTorch based)",
    "3-layer architecture [128, 64, 32]": "3-layer architecture [128, 64, 32]",
    "Initialize model": "Initialize model",
    "Train model": "Train model",
    "Learning rate": "Learning rate",
    "Number of epochs": "Number of epochs",
    "Batch size": "Batch size",
    "Model training error": "Model training error",
    "Training": "Training",
    "Test": "Test",
    
    # Forecasting module
    "Forecasting Models": "Forecasting Models",
    "Forecasts energy consumption 24 hours ahead": "Forecasts energy consumption 24 hours ahead",
    "Time series forecasting models": "Time series forecasting models",
    "Linear Forecasting Model": "Linear Forecasting Model",
    "ARIMA Forecasting Model": "ARIMA Forecasting Model",
    "Prophet Forecasting Model": "Prophet Forecasting Model",
    "Train model ve tahmin yap": "Train model and make predictions",
    "Forecasting error": "Forecasting error",
    
    # Evaluator module
    "Model Evaluation and Comparison": "Model Evaluation and Comparison",
    "Classification and Forecasting Evaluation": "Classification and Forecasting Evaluation",
    "Model Comparison": "Model Comparison",
    "Classification metrics (Accuracy, Precision, Recall, F1)": "Classification metrics (Accuracy, Precision, Recall, F1)",
    "Forecasting metrics (MAE, RMSE, MAPE)": "Forecasting metrics (MAE, RMSE, MAPE)",
    "Calculate metrics": "Calculate metrics",
    "Evaluate": "Evaluate",
    
    # Integration module
    "Model Integration and Data Pipeline": "Model Integration and Data Pipeline",
    "Integration layer between Person 1's data and Person 3's dashboard": "Integration layer between Person 1's data and Person 3's dashboard",
    "Load models and make predictions": "Load models and make predictions",
    "Format results": "Format results",
    
    # Common log messages
    "saved successfully": "saved successfully",
    "loaded successfully": "loaded successfully",
    "error": "error",
    "warning": "warning",
    "info": "info",
    
    # Başlangıç/Validation
    "Initialization": "Initialization",
    "Validation": "Validation",
    "Error": "Error",
    "Warning": "Warning",
}

# Function to translate text
def translate_text(text):
    """Translate Turkish text to English using dictionary"""
    result = text
    for turkish, english in translations.items():
        result = result.replace(turkish, english)
    return result

# Get all Python files
python_files = []
project_root = r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project"

for root, dirs, files in os.walk(project_root):
    # Skip __pycache__ and .git directories
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'models', 'data', 'logs']]
    
    for file in files:
        if file.endswith('.py'):
            python_files.append(os.path.join(root, file))

print(f"Found {len(python_files)} Python files to translate:")
for f in python_files:
    print(f"  - {f}")

# Process each file
for filepath in python_files:
    print(f"\nProcessing: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Translate content
        translated_content = translate_text(content)
        
        # Check if there were any changes
        if content != translated_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            print(f"  ✓ Translated successfully")
        else:
            print(f"  - No Turkish content found")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("Translation complete!")
