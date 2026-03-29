"""
Model Preparation Module - Data Preparation and Balancing
========================================================
Person 1 (Data Architect) provided labeled data,
Person 2's classification and forecasting models prepares.

Functions:
    - balance_dataset(): Ensures equal RS/RP distribution in test set
    - train_test_split_timeseries(): Preserves time series structure
    - prepare_features(): Prepares feature vector for model input
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from typing import Tuple, Dict, Optional, List

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing and preparation class.
    
    Features:
    - Dataset balancing (equal RS/RP distribution)
    - Train/test split preserving time series structure
    - Missing value handling
    - Feature normalization
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataPreprocessor.
        
        Args:
            test_size (float): Test set ratio (0-1)
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.logger = logger
        
    def balance_dataset(self, df: pd.DataFrame, label_col: str = 'label') -> pd.DataFrame:
        """
        Balances dataset to ensure equal number of RS and RP examples in test set.
        
        Method: Minority class oversampling, majority class undersampling
        
        Args:
            df (pd.DataFrame): Original dataset
            label_col (str): Label column name (contains 'RS' or 'RP')
            
        Returns:
            pd.DataFrame: Balanced dataset
        """
        try:
            # Check class distribution
            class_distribution = df[label_col].value_counts()
            self.logger.info(f"Original class distribution:\n{class_distribution}")
            
            # Take equal number of samples from each class
            min_samples = class_distribution.min()
            
            balanced_dfs = []
            for class_label in df[label_col].unique():
                class_df = df[df[label_col] == class_label]
                balanced_dfs.append(class_df.sample(n=min_samples, 
                                                     random_state=self.random_state))
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            balanced_df = balanced_df.sample(frac=1, 
                                            random_state=self.random_state).reset_index(drop=True)
            
            self.logger.info(f"Balanced class distribution:\n{balanced_df[label_col].value_counts()}")
            return balanced_df
            
        except Exception as e:
            self.logger.error(f"Dataset balancing error: {str(e)}")
            raise
    
    def train_test_split_timeseries(self, 
                                    df: pd.DataFrame, 
                                    time_col: Optional[str] = None,
                                    features_cols: Optional[List[str]] = None,
                                    label_col: str = 'label') -> Tuple:
        """
        Performs train/test split while preserving time series structure.
        
        Method: Uses TimeSeriesSplit to prevent data leakage
        
        Args:
            df (pd.DataFrame): Dataset
            time_col (str, optional): Time column name
            features_cols (list, optional): List of feature columns
            label_col (str): Label column name
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            if time_col:
                # Sort by time column
                df = df.sort_values(by=time_col).reset_index(drop=True)
                
            # If features_cols not specified, use all numeric columns
            if features_cols is None:
                features_cols = [col for col in df.columns 
                               if col not in [label_col, time_col] and df[col].dtype in [np.float64, np.int64]]
            
            X = df[features_cols].values
            y = df[label_col].values
            
            # Time series split
            split_point = int(len(df) * (1 - self.test_size))
            
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            self.logger.info(f"Training set size: {X_train.shape[0]}")
            self.logger.info(f"Test set size: {X_test.shape[0]}")
            self.logger.info(f"Number of features: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test, features_cols
            
        except Exception as e:
            self.logger.error(f"Train/Test split error: {str(e)}")
            raise
    
    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray, 
                          method: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies feature normalization.
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            method (str): Normalization method ('standard' or 'minmax')
            
        Returns:
            Tuple: (X_train_normalized, X_test_normalized)
        """
        try:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            X_train_normalized = self.scaler.fit_transform(X_train)
            X_test_normalized = self.scaler.transform(X_test)
            
            self.logger.info(f"{method.upper()} normalization applied")
            return X_train_normalized, X_test_normalized
            
        except Exception as e:
            self.logger.error(f"Normalization error: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handles missing values.
        
        Args:
            df (pd.DataFrame): Dataset
            method (str): Processing method ('interpolate', 'forward_fill', 'backward_fill')
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        try:
            if method == 'interpolate':
                df_filled = df.interpolate(method='linear', limit_direction='both')
            elif method == 'forward_fill':
                df_filled = df.fillna(method='ffill').fillna(method='bfill')
            elif method == 'backward_fill':
                df_filled = df.fillna(method='bfill').fillna(method='ffill')
            else:
                raise ValueError(f"Unknown missing value handling method: {method}")
            
            self.logger.info(f"Missing values handled using {method} method")
            return df_filled
            
        except Exception as e:
            self.logger.error(f"Missing value handling error: {str(e)}")
            raise
    
    def remove_outliers(self, X: np.ndarray, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
        """
        Detects and removes outliers.
        
        Args:
            X (np.ndarray): Data (n_samples x n_features)
            method (str): Method ('iqr' or 'zscore')
            threshold (float): Threshold value
            
        Returns:
            np.ndarray: Data with outliers removed
        """
        try:
            if method == 'iqr':
                Q1 = np.percentile(X, 25, axis=0)
                Q3 = np.percentile(X, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
                
            elif method == 'zscore':
                z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
                mask = np.all(z_scores < threshold, axis=1)
            else:
                raise ValueError(f"Unknown outlier removal method: {method}")
            
            removed_count = X.shape[0] - mask.sum()
            self.logger.info(f"Outliers removed: {removed_count} samples")
            
            return X[mask]
            
        except Exception as e:
            self.logger.error(f"Outlier removal error: {str(e)}")
            raise


def create_lag_features(data: pd.DataFrame, 
                        value_col: str, 
                        lags: List[int] = [1, 7, 24]) -> pd.DataFrame:
    """
    Creates lagged (lag) features for forecasting model.
    
    Example: Yesterday's consumption (lag=24), last week's (lag=7), etc.
    
    Args:
        data (pd.DataFrame): Time series dataset
        value_col (str): Value column name
        lags (list): Lag values to create (in hours)
        
    Returns:
        pd.DataFrame: Dataset with lag features added
    """
    try:
        df = data.copy()
        
        for lag in lags:
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
        
        # Remove NaN values from first rows
        df = df.dropna()
        
        logger.info(f"Lag features created: {lags}")
        return df
        
    except Exception as e:
        logger.error(f"Lag features creation error: {str(e)}")
        raise


# Test code
if __name__ == "__main__":
    print("Model Preparation Module - Test")
    print("=" * 50)
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate fake data
    X_data = np.random.randn(n_samples, 10)
    y_data = np.random.choice(['RS', 'RP'], n_samples)
    
    df_test = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(10)])
    df_test['label'] = y_data
    df_test['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='h')
    
    print(f"\nOriginal dataset shape: {df_test.shape}")
    print(f"Class distribution:\n{df_test['label'].value_counts()}")
    
    # Test DataPreprocessor
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    
    # Dataset balancing
    df_balanced = preprocessor.balance_dataset(df_test)
    print(f"\nBalanced dataset shape: {df_balanced.shape}")
    
    # Train/Test split
    X_train, X_test, y_train, y_test, feature_names = preprocessor.train_test_split_timeseries(
        df_balanced, 
        time_col='timestamp',
        label_col='label'
    )
    
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Normalization
    X_train_norm, X_test_norm = preprocessor.normalize_features(X_train, X_test, method='standard')
    print(f"\nX_train (normalized) mean: {X_train_norm.mean():.4f}")
    print(f"X_train (normalized) std: {X_train_norm.std():.4f}")
