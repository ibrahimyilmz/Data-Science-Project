from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLUMNS: list[str] = [
    "occupancy_rate",
    "weekend_weekday_ratio",
    "fft_daily_amp",
    "fft_weekly_amp",
]


def reduce_and_cluster(
    features_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str] | None = None,
    customer_col: str = "customer_id",
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reduce feature space with PCA, run K-Means (k=2), and assign RS/RP labels.

    Label interpretation rule:
    - lower mean occupancy_rate cluster -> RS
    - higher mean occupancy_rate cluster -> RP
    """
    if features_df.empty:
        raise ValueError("features_df is empty.")

    selected_cols = list(feature_cols or DEFAULT_FEATURE_COLUMNS)
    missing = [col for col in selected_cols if col not in features_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "occupancy_rate" not in features_df.columns:
        raise ValueError("Column 'occupancy_rate' is required for RS/RP interpretation.")
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for this project step.")
    if len(selected_cols) < n_components:
        raise ValueError(
            f"n_components={n_components} cannot exceed number of selected features={len(selected_cols)}"
        )
    if len(features_df) < 2:
        raise ValueError("At least 2 rows are required for K-Means with k=2.")

    x = features_df[selected_cols].copy()
    for col in selected_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if x.isna().any().any():
        bad_cols = x.columns[x.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values detected in: {bad_cols}")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=n_components, random_state=random_state)
    x_pca = pca.fit_transform(x_scaled)

    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=20)
    cluster_ids = kmeans.fit_predict(x_pca)

    result = features_df.copy()
    result["cluster"] = cluster_ids
    result["pca_1"] = x_pca[:, 0]
    result["pca_2"] = x_pca[:, 1]
    if n_components == 3:
        result["pca_3"] = x_pca[:, 2]

    occupancy_by_cluster = result.groupby("cluster", as_index=False)["occupancy_rate"].mean()
    rs_cluster = int(occupancy_by_cluster.sort_values("occupancy_rate", ascending=True).iloc[0]["cluster"])
    result["label"] = result["cluster"].apply(lambda c: "RS" if c == rs_cluster else "RP")

    if customer_col in result.columns:
        result = result.sort_values(customer_col).reset_index(drop=True)

    return result
