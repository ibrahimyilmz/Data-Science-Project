"""Classification and forecasting ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ClassificationModels:
    """Classification models for RS/RP prediction."""

    @staticmethod
    def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Train Logistic Regression.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data
        X_test, y_test : np.ndarray
            Test data

        Returns
        -------
        dict
            Model and metrics
        """
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return {
            "model": model,
            "predictions": y_pred,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    @staticmethod
    def train_neural_network(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 100) -> dict:
        """
        Train Neural Network.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data
        X_test, y_test : np.ndarray
            Test data
        epochs : int
            Number of training epochs

        Returns
        -------
        dict
            Model and metrics
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)

        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.numpy()

        return {
            "model": model,
            "predictions": y_pred,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }


class ForecastingModels:
    """Forecasting models for consumption prediction."""

    @staticmethod
    def train_arima(timeseries: np.ndarray, order: tuple = (1, 1, 1), forecast_steps: int = 24) -> dict:
        """
        Train ARIMA model.

        Parameters
        ----------
        timeseries : np.ndarray
            Historical timeseries data
        order : tuple
            ARIMA order (p, d, q)
        forecast_steps : int
            Number of steps to forecast

        Returns
        -------
        dict
            Forecast and metrics
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels not available")

        model = ARIMA(timeseries, order=order)
        results = model.fit()
        forecast = results.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean.values

        return {
            "forecast": forecast_values,
            "mae": float(np.mean(np.abs(np.diff(timeseries)[:forecast_steps]))),
        }

    @staticmethod
    def create_sequences(data: np.ndarray, seq_length: int = 24) -> tuple:
        """
        Create sequences for LSTM.

        Parameters
        ----------
        data : np.ndarray
            Input data
        seq_length : int
            Sequence length

        Returns
        -------
        tuple
            (X, y) sequences
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    @staticmethod
    def train_lstm(timeseries: np.ndarray, forecast_steps: int = 24, seq_length: int = 24, epochs: int = 50) -> dict:
        """
        Train LSTM model.

        Parameters
        ----------
        timeseries : np.ndarray
            Historical timeseries data
        forecast_steps : int
            Number of steps to forecast
        seq_length : int
            Sequence length
        epochs : int
            Training epochs

        Returns
        -------
        dict
            Forecast and metrics
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        # Normalize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(timeseries.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = ForecastingModels.create_sequences(data_scaled, seq_length)

        if len(X) == 0:
            raise ValueError(f"Not enough data for sequences of length {seq_length}")

        # Train/test split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train_t = torch.FloatTensor(X_train).unsqueeze(2)
        y_train_t = torch.FloatTensor(y_train)

        # LSTM model
        model = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        linear = nn.Linear(64, 1)

        optimizer = optim.Adam(list(model.parameters()) + list(linear.parameters()), lr=0.01)
        criterion = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            lstm_out, _ = model(X_train_t)
            outputs = linear(lstm_out[:, -1, :])
            loss = criterion(outputs, y_train_t.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Forecast
        forecast = []
        current = data_scaled[-seq_length:]

        for _ in range(forecast_steps):
            current_t = torch.FloatTensor(current).unsqueeze(0).unsqueeze(2)
            with torch.no_grad():
                lstm_out, _ = model(current_t)
                next_val = linear(lstm_out[:, -1, :]).item()
            forecast.append(next_val)
            current = np.append(current[1:], next_val)

        # Inverse transform
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

        return {
            "forecast": forecast,
            "mae": float(np.mean(np.abs(np.diff(timeseries)[:forecast_steps]))),
        }
