from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_TIME_CANDIDATES = ("timestamp", "datetime", "date", "time", "horodate")
DEFAULT_POWER_CANDIDATES = ("power_kw", "kw", "puissance_kw", "load_kw", "power")
DEFAULT_ID_CANDIDATES = ("customer_id", "id_pdl", "pdl", "id")


def _find_column(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> str | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    if required:
        raise ValueError(f"Column not found. Tried: {list(candidates)}")
    return None


def load_consumption_data(
    csv_path: str | Path,
    *,
    time_col: str | None = None,
    power_col: str | None = None,
    customer_col: str | None = None,
    interval_hours: float = 0.5,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load Enedis-style consumption CSV data and convert power (kW) to energy (kWh).

    Conversion formula:
        E_kWh = P_kW * interval_hours
    For 30-minute measurements, interval_hours=0.5.
    """
    df = pd.read_csv(csv_path, **read_csv_kwargs)

    selected_time_col = time_col or _find_column(df.columns, DEFAULT_TIME_CANDIDATES, required=True)
    selected_power_col = power_col or _find_column(df.columns, DEFAULT_POWER_CANDIDATES, required=True)
    selected_customer_col = customer_col or _find_column(df.columns, DEFAULT_ID_CANDIDATES, required=False)

    parsed_time = pd.to_datetime(df[selected_time_col], errors="coerce")
    if parsed_time.isna().any():
        invalid_count = int(parsed_time.isna().sum())
        raise ValueError(
            f"{invalid_count} rows have non-parseable datetime values in column '{selected_time_col}'."
        )

    power_kw = pd.to_numeric(df[selected_power_col], errors="coerce")
    if power_kw.isna().any():
        invalid_count = int(power_kw.isna().sum())
        raise ValueError(
            f"{invalid_count} rows have non-numeric power values in column '{selected_power_col}'."
        )

    result = df.copy()
    result["timestamp"] = parsed_time
    result["power_kw"] = power_kw
    result["energy_kwh"] = result["power_kw"] * interval_hours

    if selected_customer_col is not None and selected_customer_col != "customer_id":
        result["customer_id"] = result[selected_customer_col]

    # Keep a stable ordering for downstream time-series processing.
    sort_cols = ["timestamp"]
    if "customer_id" in result.columns:
        sort_cols = ["customer_id", "timestamp"]
    result = result.sort_values(sort_cols).reset_index(drop=True)

    return result
