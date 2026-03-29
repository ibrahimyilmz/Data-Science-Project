"""Clustering and feature extraction functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fourier_features(values: np.ndarray, n_components: int = 5) -> np.ndarray:
    """
    Extract Fourier features using FFT.

    Parameters
    ----------
    values : np.ndarray
        Input timeseries
    n_components : int
        Number of frequency components to extract

    Returns
    -------
    np.ndarray
        Magnitude of top frequency components
    """
    fft_values = np.fft.rfft(values - np.mean(values))
    magnitudes = np.abs(fft_values)
    top_indices = np.argsort(magnitudes)[-n_components:]
    return magnitudes[top_indices[::-1]]


def extract_consumption_features(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    time_col: str = "timestamp",
    power_col: str = "power_kw",
) -> pd.DataFrame:
    """
    Extract behavioral features from consumption data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw consumption data
    customer_col : str
        Customer ID column name
    time_col : str
        Timestamp column name
    power_col : str
        Power consumption column name

    Returns
    -------
    pd.DataFrame
        Customer-level features
    """
    features = []

    for customer_id, group in df.groupby(customer_col):
        if len(group) < 24:
            continue

        power_values = group[power_col].values
        df_group = group.copy()
        df_group[time_col] = pd.to_datetime(df_group[time_col])
        df_group["hour"] = df_group[time_col].dt.hour

        features_dict = {"customer_id": customer_id}

        # Consumption statistics
        features_dict["mean_power"] = float(np.mean(power_values))
        features_dict["std_power"] = float(np.std(power_values))
        features_dict["min_power"] = float(np.min(power_values))
        features_dict["max_power"] = float(np.max(power_values))
        features_dict["occupancy_rate"] = float(
            np.sum(power_values > np.percentile(power_values, 25)) / len(power_values)
        )

        # Peak hours
        morning_peak = df_group[(df_group["hour"] >= 6) & (df_group["hour"] < 9)][
            power_col
        ].mean()
        evening_peak = df_group[(df_group["hour"] >= 18) & (df_group["hour"] < 21)][
            power_col
        ].mean()

        features_dict["morning_peak"] = float(morning_peak) if not np.isnan(
            morning_peak
        ) else 0.0
        features_dict["evening_peak"] = float(evening_peak) if not np.isnan(
            evening_peak
        ) else 0.0

        # Fourier components
        try:
            fourier = fourier_features(power_values, n_components=5)
            for i, comp in enumerate(fourier):
                features_dict[f"fourier_{i}"] = float(comp)
        except Exception:
            for i in range(5):
                features_dict[f"fourier_{i}"] = 0.0

        features.append(features_dict)

    return pd.DataFrame(features)


def perform_clustering(
    features_df: pd.DataFrame, n_clusters: int = 3, use_pca: bool = True
) -> tuple:
    """
    Perform K-Means clustering.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with customer_id in first column
    n_clusters : int
        Number of clusters
    use_pca : bool
        Whether to apply PCA first

    Returns
    -------
    tuple
        (cluster_labels, scaler, kmeans_model)
    """
    # Extract features (exclude customer_id)
    X = features_df.iloc[:, 1:].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA
    if use_pca and X_scaled.shape[1] > 2:
        pca = PCA(n_components=min(10, X_scaled.shape[1]))
        X_scaled = pca.fit_transform(X_scaled)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    return labels, scaler, kmeans


def apply_pca_2d(
    features_df: pd.DataFrame,
) -> tuple:
    """
    Apply PCA for 2D visualization.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix

    Returns
    -------
    tuple
        (pca_data, pca_model)
    """
    X = features_df.iloc[:, 1:].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca
