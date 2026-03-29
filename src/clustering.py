from __future__ import annotations

from typing import Sequence

import pandas as pd
from .clustering_engine import DEFAULT_FEATURE_COLUMNS, _run_cluster_pipeline


DEFAULT_CLUSTER_FEATURES: list[str] = [
	*DEFAULT_FEATURE_COLUMNS,
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
	selected_cols = list(feature_cols or DEFAULT_CLUSTER_FEATURES)
	return _run_cluster_pipeline(
		features_df,
		feature_cols=selected_cols,
		customer_col=customer_col,
		n_components=n_components,
		n_clusters=n_clusters,
		random_state=random_state,
	)

