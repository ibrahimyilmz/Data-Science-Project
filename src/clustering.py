from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .clustering_engine import DEFAULT_FEATURE_COLUMNS, _run_cluster_pipeline


DEFAULT_CLUSTER_FEATURES: list[str] = [
	*DEFAULT_FEATURE_COLUMNS,
]


def fourier_features(values: np.ndarray, n_components: int = 5) -> np.ndarray:
	"""Extract Fourier features using FFT."""
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
	"""Extract behavioral features from consumption data."""
	# Convert to proper types at the beginning
	df = df.copy()
	
	# Convert power to numeric
	try:
		df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
	except Exception as e:
		raise ValueError(f"Failed to convert '{power_col}' to numeric. Error: {str(e)}")
	
	# Convert timestamp to datetime with multiple format attempts
	try:
		# Try multiple date formats
		formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d']
		datetime_converted = False
		
		for fmt in formats:
			try:
				df[time_col] = pd.to_datetime(df[time_col], format=fmt, errors='coerce')
				non_null_count = df[time_col].notna().sum()
				if non_null_count > len(df) * 0.8:  # At least 80% converted successfully
					datetime_converted = True
					break
			except:
				continue
		
		# If none of the formats worked, try pandas auto-detection
		if not datetime_converted:
			df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True, errors='coerce')
			non_null_count = df[time_col].notna().sum()
			if non_null_count <= len(df) * 0.5:
				raise ValueError(f"Could not convert '{time_col}' to datetime")
	
	except Exception as e:
		raise ValueError(f"Failed to convert '{time_col}' to datetime. Tried formats: YYYY-MM-DD HH:MM:SS, YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY. Error: {str(e)}")
	
	# Check for invalid values
	if df[time_col].isna().any():
		invalid_count = df[time_col].isna().sum()
		raise ValueError(f"Found {invalid_count} invalid datetime values in '{time_col}'. Ensure dates are in a standard format (e.g., 2023-11-01 or 01/11/2023)")
	
	if df[power_col].isna().any():
		invalid_count = df[power_col].isna().sum()
		raise ValueError(f"Found {invalid_count} invalid numeric values in '{power_col}'. All values must be numbers.")
	
	# Now safely use .dt accessor
	df["hour"] = df[time_col].dt.hour
	
	features = []

	for customer_id, group in df.groupby(customer_col):
		if len(group) < 24:
			continue

		power_values = group[power_col].values
		df_group = group.copy()

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
	"""Perform K-Means clustering."""
	X = features_df.iloc[:, 1:].values
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	if use_pca and X_scaled.shape[1] > 2:
		pca = PCA(n_components=min(10, X_scaled.shape[1]))
		X_scaled = pca.fit_transform(X_scaled)

	kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	labels = kmeans.fit_predict(X_scaled)

	return labels, scaler, kmeans


def apply_pca_2d(
	features_df: pd.DataFrame,
) -> tuple:
	"""Apply PCA for 2D visualization."""
	X = features_df.iloc[:, 1:].values
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(X_scaled)

	return X_pca, pca


def assign_residence_labels(
	features_df: pd.DataFrame,
	*,
	customer_col: str = "customer_id",
	feature_cols: Sequence[str] | None = None,
	n_clusters: int = 2,
	n_components: int = 2,
	random_state: int = 42,
) -> pd.DataFrame:
	"""Apply StandardScaler + PCA + KMeans, then map clusters to RS/RP labels."""
	selected_cols = list(feature_cols or DEFAULT_CLUSTER_FEATURES)
	return _run_cluster_pipeline(
		features_df,
		feature_cols=selected_cols,
		customer_col=customer_col,
		n_components=n_components,
		n_clusters=n_clusters,
		random_state=random_state,
	)

