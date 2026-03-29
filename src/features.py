from __future__ import annotations

import numpy as np
import pandas as pd


def _amplitude_for_period(values: np.ndarray, period_in_samples: int) -> float:
    if len(values) < 4 or period_in_samples <= 0:
        return 0.0

    centered = values - np.mean(values)
    fft_values = np.fft.rfft(centered)
    n = len(values)

    target_bin = int(round(n / period_in_samples))
    target_bin = max(1, min(target_bin, len(fft_values) - 1))

    # Scale amplitude to be comparable across different series lengths.
    return float((2.0 / n) * np.abs(fft_values[target_bin]))


def build_behavioral_features(
    df: pd.DataFrame,
    *,
    customer_col: str = "customer_id",
    time_col: str = "timestamp",
    energy_col: str = "energy_kwh",
    empty_day_threshold_kwh: float = 0.5,
    sampling_hours: float = 0.5,
) -> pd.DataFrame:
    """
    Build customer-level behavioral features for clustering/classification.

    Features:
    - occupancy_rate: share of days with daily energy > empty_day_threshold_kwh
    - low_consumption_day_ratio: share of days <= empty_day_threshold_kwh
    - weekday/weekend balance from daily consumption
    - Fourier amplitudes for daily (24h) and weekly (7d) rhythms
    """
    required = {customer_col, time_col, energy_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df[[customer_col, time_col, energy_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    if work[time_col].isna().any():
        raise ValueError(f"Column '{time_col}' contains invalid datetime values.")

    work[energy_col] = pd.to_numeric(work[energy_col], errors="coerce")
    if work[energy_col].isna().any():
        raise ValueError(f"Column '{energy_col}' contains non-numeric values.")

    work["date"] = work[time_col].dt.normalize()
    daily = (
        work.groupby([customer_col, "date"], as_index=False)[energy_col]
        .sum()
        .rename(columns={energy_col: "daily_energy_kwh"})
    )

    daily["is_weekend"] = daily["date"].dt.weekday >= 5
    daily["is_low_day"] = daily["daily_energy_kwh"] <= empty_day_threshold_kwh

    customer_features = (
        daily.groupby(customer_col, as_index=False)
        .agg(
            total_days=("daily_energy_kwh", "size"),
            mean_daily_kwh=("daily_energy_kwh", "mean"),
            std_daily_kwh=("daily_energy_kwh", "std"),
            low_consumption_day_ratio=("is_low_day", "mean"),
        )
        .fillna({"std_daily_kwh": 0.0})
    )
    customer_features["occupancy_rate"] = 1.0 - customer_features["low_consumption_day_ratio"]

    weekday_weekend = (
        daily.groupby([customer_col, "is_weekend"], as_index=False)["daily_energy_kwh"].mean()
        .pivot(index=customer_col, columns="is_weekend", values="daily_energy_kwh")
        .rename(columns={False: "weekday_mean_kwh", True: "weekend_mean_kwh"})
        .reset_index()
    )
    if "weekday_mean_kwh" not in weekday_weekend.columns:
        weekday_weekend["weekday_mean_kwh"] = 0.0
    if "weekend_mean_kwh" not in weekday_weekend.columns:
        weekday_weekend["weekend_mean_kwh"] = 0.0
    weekday_weekend["weekday_mean_kwh"] = weekday_weekend["weekday_mean_kwh"].fillna(0.0)
    weekday_weekend["weekend_mean_kwh"] = weekday_weekend["weekend_mean_kwh"].fillna(0.0)
    weekday_weekend["weekend_weekday_ratio"] = (
        weekday_weekend["weekend_mean_kwh"] / (weekday_weekend["weekday_mean_kwh"] + 1e-9)
    )

    # Fourier features are computed on the original high-frequency series.
    samples_per_day = int(round(24.0 / sampling_hours))
    samples_per_week = samples_per_day * 7

    fft_rows = []
    for customer_id, part in work.sort_values(time_col).groupby(customer_col):
        values = part[energy_col].to_numpy(dtype=float)
        fft_rows.append(
            {
                customer_col: customer_id,
                "fft_daily_amp": _amplitude_for_period(values, samples_per_day),
                "fft_weekly_amp": _amplitude_for_period(values, samples_per_week),
            }
        )
    fft_df = pd.DataFrame(fft_rows)

    features = customer_features.merge(weekday_weekend, on=customer_col, how="left")
    features = features.merge(fft_df, on=customer_col, how="left")

    for col in [
        "weekday_mean_kwh",
        "weekend_mean_kwh",
        "weekend_weekday_ratio",
        "fft_daily_amp",
        "fft_weekly_amp",
    ]:
        features[col] = features[col].fillna(0.0)

    return features
