from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_CLUSTER_FEATURES: list[str] = [
	"occupancy_rate",
	"weekend_weekday_ratio",
	"fft_daily_amp",
	"fft_weekly_amp",
]


def assign_residence_labels(
	features_df: pd.DataFrame,
	*,
	customer_col: str = "customer_id",
	feature_cols: Sequence[str] | None = None,
	n_clusters: int = 2,
	n_components: int = 2,
	random_state: int = 42,
) -> pd.DataFrame:
	"""
	Apply StandardScaler + PCA + KMeans, then map clusters to RS/RP labels.

	Mapping rule:
	- cluster with lower mean occupancy_rate -> RS
	- other cluster -> RP
	"""
	if features_df.empty:
		raise ValueError("features_df is empty.")

	selected_cols = list(feature_cols or DEFAULT_CLUSTER_FEATURES)
	missing = [col for col in selected_cols if col not in features_df.columns]
	if missing:
		raise ValueError(f"Missing feature columns for clustering: {missing}")
	if "occupancy_rate" not in features_df.columns:
		raise ValueError("Column 'occupancy_rate' is required for RS/RP mapping.")

	x = features_df[selected_cols].copy()
	for col in selected_cols:
		x[col] = pd.to_numeric(x[col], errors="coerce")
	if x.isna().any().any():
		bad = x.columns[x.isna().any()].tolist()
		raise ValueError(f"Non-numeric or missing values found in clustering columns: {bad}")

	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)

	if len(selected_cols) < n_components:
		raise ValueError(
			f"n_components={n_components} cannot exceed number of features={len(selected_cols)}"
		)
	if len(features_df) < n_clusters:
		raise ValueError(
			f"n_clusters={n_clusters} cannot exceed number of rows={len(features_df)}"
		)

	pca = PCA(n_components=n_components, random_state=random_state)
	x_pca = pca.fit_transform(x_scaled)

	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
	clusters = kmeans.fit_predict(x_pca)

	result_df = features_df.copy()
	result_df["cluster"] = clusters

	occupancy_by_cluster = result_df.groupby("cluster", as_index=False)["occupancy_rate"].mean()
	rs_cluster = int(occupancy_by_cluster.sort_values("occupancy_rate", ascending=True).iloc[0]["cluster"])
	result_df["label"] = result_df["cluster"].apply(lambda c: "RS" if c == rs_cluster else "RP")

	# Keep PCA coordinates for optional visualization in dashboard.
	if n_components >= 1:
		result_df["pca_1"] = x_pca[:, 0]
	if n_components >= 2:
		result_df["pca_2"] = x_pca[:, 1]

	if customer_col in result_df.columns:
		result_df = result_df.sort_values(customer_col).reset_index(drop=True)

	return result_df

