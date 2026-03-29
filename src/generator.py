"""Synthetic energy consumption profile generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class SyntheticProfileGenerator:
    """Generate realistic synthetic RS/RP energy consumption profiles."""

    RS_PROFILE = {
        "base_load_mean": 0.25,
        "base_load_std": 0.05,
        "morning_peak_intensity": 1.0,
        "morning_peak_width": 2.0,
        "evening_peak_intensity": 0.9,
        "evening_peak_width": 3.0,
        "night_reduction_min": 0.3,
        "night_reduction_max": 0.4,
        "noise_std": 0.1,
    }

    RP_PROFILE = {
        "base_load_mean": 0.50,
        "base_load_std": 0.10,
        "morning_peak_intensity": 1.8,
        "morning_peak_width": 2.5,
        "evening_peak_intensity": 1.5,
        "evening_peak_width": 3.5,
        "night_reduction_min": 0.3,
        "night_reduction_max": 0.4,
        "noise_std": 0.12,
    }

    def __init__(self, profile_class: str = "RS", seed: int | None = None):
        """
        Initialize generator.

        Parameters
        ----------
        profile_class : str
            "RS" for standard or "RP" for premium
        seed : int, optional
            Random seed for reproducibility
        """
        self.profile_class = profile_class
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.profile = self.RS_PROFILE if profile_class == "RS" else self.RP_PROFILE

    def _gaussian_peak(
        self, hour: float, center: float, intensity: float, width: float
    ) -> float:
        """Generate Gaussian-shaped peak."""
        sigma = width / 2.355
        return intensity * np.exp(-((hour - center) ** 2) / (2 * sigma**2))

    def generate_24h_profile(self, seed: int | None = None) -> np.ndarray:
        """
        Generate 24-hour consumption profile.

        Parameters
        ----------
        seed : int, optional
            Override random seed for this generation

        Returns
        -------
        np.ndarray
            24-hour profile (1 value per hour)
        """
        if seed is not None:
            np.random.seed(seed)

        profile_24h = np.zeros(24)
        base_load = np.random.normal(
            self.profile["base_load_mean"], self.profile["base_load_std"]
        )

        for hour in range(24):
            # Base load with night reduction
            if 22 <= hour or hour < 6:
                night_factor = np.random.uniform(
                    self.profile["night_reduction_min"],
                    self.profile["night_reduction_max"],
                )
                load = base_load * night_factor
            else:
                load = base_load

            # Morning peak (6-9h)
            morning_peak = self._gaussian_peak(
                hour,
                center=7.5,
                intensity=self.profile["morning_peak_intensity"],
                width=self.profile["morning_peak_width"],
            )

            # Evening peak (18-21h)
            evening_peak = self._gaussian_peak(
                hour,
                center=19.5,
                intensity=self.profile["evening_peak_intensity"],
                width=self.profile["evening_peak_width"],
            )

            # Combine and add noise
            profile_24h[hour] = load + morning_peak + evening_peak

        # Add Gaussian noise
        noise = np.random.normal(0, self.profile["noise_std"], 24)
        profile_24h = np.maximum(profile_24h + noise, 0)

        return profile_24h

    def generate_multiple_profiles(
        self, n_profiles: int = 100, rs_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate multiple profiles.

        Parameters
        ----------
        n_profiles : int
            Number of profiles to generate
        rs_ratio : float
            Ratio of RS profiles (0-1)

        Returns
        -------
        pd.DataFrame
            Profiles with customer_id, hour, power_kw
        """
        profiles = []

        for i in range(n_profiles):
            profile_24h = self.generate_24h_profile()
            customer_type = "RS" if np.random.random() < rs_ratio else "RP"

            for hour, power in enumerate(profile_24h):
                profiles.append(
                    {
                        "customer_id": f"CUST_{i+1:05d}",
                        "customer_type": customer_type,
                        "hour": hour,
                        "power_kw": power,
                    }
                )

        return pd.DataFrame(profiles)

    def calculate_similarity_metrics(
        self, synthetic: np.ndarray, real: np.ndarray
    ) -> dict:
        """
        Compare synthetic vs real profiles.

        Parameters
        ----------
        synthetic : np.ndarray
            Synthetic consumption values
        real : np.ndarray
            Real consumption values

        Returns
        -------
        dict
            Similarity metrics
        """
        ks_stat, ks_pvalue = stats.ks_2samp(synthetic, real)
        wasserstein = stats.wasserstein_distance(synthetic, real)

        return {
            "mean_diff": float(np.abs(np.mean(synthetic) - np.mean(real))),
            "std_diff": float(np.abs(np.std(synthetic) - np.std(real))),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "wasserstein_distance": float(wasserstein),
            "synthetic_quantiles": {
                "q25": float(np.quantile(synthetic, 0.25)),
                "q50": float(np.quantile(synthetic, 0.50)),
                "q75": float(np.quantile(synthetic, 0.75)),
            },
            "real_quantiles": {
                "q25": float(np.quantile(real, 0.25)),
                "q50": float(np.quantile(real, 0.50)),
                "q75": float(np.quantile(real, 0.75)),
            },
        }
