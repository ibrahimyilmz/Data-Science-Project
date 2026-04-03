"""Synthetic energy consumption profile generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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

    def __init__(self, profile_class: str = "RS", seed: int | None = None, learned_profile: dict | None = None):
        """
        Initialize generator.

        Parameters
        ----------
        profile_class : str
            "RS" for standard or "RP" for premium
        seed : int, optional
            Random seed for reproducibility
        learned_profile : dict, optional
            Profile learned from real data. If provided, overrides default profiles.
        """
        self.profile_class = profile_class
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        if learned_profile is not None:
            self.profile = learned_profile
        else:
            self.profile = self.RS_PROFILE if profile_class == "RS" else self.RP_PROFILE

    def _gaussian_peak(
        self, hour: float, center: float, intensity: float, width: float
    ) -> float:
        """Generate Gaussian-shaped peak."""
        sigma = width / 2.355
        return intensity * np.exp(-((hour - center) ** 2) / (2 * sigma**2))

    @classmethod
    def learn_from_data(
        cls, df: pd.DataFrame, profile_class: str = "RS", seed: int | None = None
    ) -> "SyntheticProfileGenerator":
        """
        Learn profile characteristics from real consumption data.

        Parameters
        ----------
        df : pd.DataFrame
            Real consumption data with columns: customer_id, power_kw, hour (or timestamp)
        profile_class : str
            "RS" for standard or "RP" for premium
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        SyntheticProfileGenerator
            Generator with learned profile
        """
        # If timestamp is provided, extract hour
        if "timestamp" in df.columns and "hour" not in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"] = df["timestamp"].dt.hour

        # Create hourly profiles per customer
        hourly_profiles = []
        for customer_id, group in df.groupby("customer_id"):
            if "hour" in group.columns:
                hourly_avg = group.groupby("hour")["power_kw"].mean()
                if len(hourly_avg) >= 12:  # At least 12 hours of data
                    hourly_profiles.append(hourly_avg.values)

        if not hourly_profiles:
            # Fallback to default profiles if no valid data
            return cls(profile_class=profile_class, seed=seed)

        # Learn characteristics from real data
        morning_mean = df[df["hour"].isin([6, 7, 8, 9])]["power_kw"].mean() if len(df[df["hour"].isin([6, 7, 8, 9])]) > 0 else np.mean(df["power_kw"])
        evening_mean = df[df["hour"].isin([18, 19, 20, 21])]["power_kw"].mean() if len(df[df["hour"].isin([18, 19, 20, 21])]) > 0 else np.mean(df["power_kw"])
        overall_mean = np.mean(df["power_kw"])
        
        learned_profile = {
            "base_load_mean": float(np.percentile(df["power_kw"], 10)),
            "base_load_std": float(np.std(df["power_kw"]) * 0.3),
            "morning_peak_intensity": float(morning_mean / overall_mean) if overall_mean > 0 else 1.0,
            "morning_peak_width": 2.0,
            "evening_peak_intensity": float(evening_mean / overall_mean) if overall_mean > 0 else 0.9,
            "evening_peak_width": 3.0,
            "night_reduction_min": 0.3,
            "night_reduction_max": 0.4,
            "noise_std": float(np.std(df["power_kw"]) * 0.15),
            "mean_power": float(np.mean(df["power_kw"])),
            "std_power": float(np.std(df["power_kw"])),
            "min_power": float(np.min(df["power_kw"])),
            "max_power": float(np.max(df["power_kw"])),
        }

        return cls(profile_class=profile_class, seed=seed, learned_profile=learned_profile)

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
        self, n_profiles: int = 100, rs_ratio: float = 0.5, base_date: str = "2024-01-01"
    ) -> pd.DataFrame:
        """
        Generate multiple profiles.

        Parameters
        ----------
        n_profiles : int
            Number of profiles to generate
        rs_ratio : float
            Ratio of RS profiles (0-1)
        base_date : str
            Base date for timestamps (YYYY-MM-DD format)

        Returns
        -------
        pd.DataFrame
            Profiles with customer_id, timestamp, power_kw (clustering-compatible format)
        """
        profiles = []
        base_timestamp = pd.to_datetime(base_date)

        for i in range(n_profiles):
            profile_24h = self.generate_24h_profile()
            customer_type = "RS" if np.random.random() < rs_ratio else "RP"

            for hour, power in enumerate(profile_24h):
                timestamp = base_timestamp + pd.Timedelta(hours=int(hour))
                profiles.append(
                    {
                        "customer_id": i + 1,
                        "timestamp": timestamp,
                        "power_kw": power,
                    }
                )

        return pd.DataFrame(profiles)

    def generate_from_real_data(
        self, n_profiles: int = 100, real_df: pd.DataFrame | None = None, base_date: str = "2024-01-01"
    ) -> pd.DataFrame:
        """
        Generate synthetic profiles similar to real data patterns.

        Parameters
        ----------
        n_profiles : int
            Number of profiles to generate
        real_df : pd.DataFrame, optional
            Real data to extract patterns from. If provided, overrides learned profiles.
        base_date : str
            Base date for timestamps (YYYY-MM-DD format)

        Returns
        -------
        pd.DataFrame
            Synthetic profiles with customer_id, timestamp, power_kw (clustering-compatible format)
        """
        if real_df is not None:
            # Learn patterns from real data
            gen = self.__class__.learn_from_data(real_df, self.profile_class, self.seed)
            self.profile = gen.profile

        profiles = []
        mean_power = self.profile.get("mean_power", 1.0)
        std_power = self.profile.get("std_power", 0.3)
        base_timestamp = pd.to_datetime(base_date)

        for i in range(n_profiles):
            profile_24h = self.generate_24h_profile()
            
            # Scale to learned mean/std
            profile_24h = (profile_24h - np.mean(profile_24h)) / (np.std(profile_24h) + 1e-6)
            profile_24h = profile_24h * std_power + mean_power
            profile_24h = np.maximum(profile_24h, 0)  # Ensure non-negative

            for hour, power in enumerate(profile_24h):
                timestamp = base_timestamp + pd.Timedelta(hours=int(hour))
                profiles.append(
                    {
                        "customer_id": i + 1,
                        "timestamp": timestamp,
                        "power_kw": float(power),
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


class VAE(nn.Module):
    """Variational Autoencoder for energy consumption profiles."""

    def __init__(self, input_dim=24, hidden_dim=64, latent_dim=8):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus(),  # Ensure positive outputs
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x, x_recon, mu, logvar):
    """VAE loss: reconstruction + KL divergence."""
    mse = nn.MSELoss(reduction="mean")(x_recon, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


class VAEGenerator:
    """Generate synthetic profiles using Variational Autoencoder."""

    def __init__(self, seed: int | None = None, device="cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for VAE. Install with: pip install torch")
        
        self.seed = seed
        self.device = device
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def train_on_data(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        latent_dim: int = 8,
    ):
        """Train VAE on real consumption data."""
        # Extract hourly profiles
        profiles = []
        for customer_id, group in df.groupby("customer_id"):
            if "hour" in group.columns:
                hourly = group.groupby("hour")["power_kw"].first()
            else:
                # If no hour, create hourly values from timestamp
                group_copy = group.copy()
                group_copy["timestamp"] = pd.to_datetime(group_copy["timestamp"])
                group_copy["hour"] = group_copy["timestamp"].dt.hour
                hourly = group_copy.groupby("hour")["power_kw"].mean()

            if len(hourly) >= 12:
                profiles.append(hourly.values)

        if not profiles:
            raise ValueError("No valid profiles extracted from data")

        X = np.array(profiles).astype(np.float32)

        # Pad/truncate to 24 hours
        if X.shape[1] < 24:
            X = np.pad(X, ((0, 0), (0, 24 - X.shape[1])), mode="mean")
        elif X.shape[1] > 24:
            X = X[:, :24]

        # Normalize
        self.scaler_mean = X.mean(axis=0, keepdims=True)
        self.scaler_std = X.std(axis=0, keepdims=True) + 1e-6
        X_scaled = (X - self.scaler_mean) / self.scaler_std

        # Create DataLoader
        dataset = TensorDataset(torch.FloatTensor(X_scaled))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        self.model = VAE(input_dim=24, hidden_dim=64, latent_dim=latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Train
        for epoch in range(epochs):
            total_loss = 0
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                x_recon, mu, logvar = self.model(x_batch)
                loss = vae_loss(x_batch, x_recon, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return self

    def generate(self, n_profiles: int = 100, base_date: str = "2024-01-01") -> pd.DataFrame:
        """Generate synthetic profiles."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_on_data() first.")

        self.model.eval()
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(n_profiles, self.model.latent_dim).to(self.device)
            samples = self.model.decode(z).cpu().numpy()

        # Denormalize
        samples = samples * self.scaler_std + self.scaler_mean
        samples = np.maximum(samples, 0)  # Ensure non-negative

        # Create DataFrame with timestamps (clustering-compatible format)
        profiles = []
        base_timestamp = pd.to_datetime(base_date)
        
        for i, profile_24h in enumerate(samples):
            for hour, power in enumerate(profile_24h):
                timestamp = base_timestamp + pd.Timedelta(hours=int(hour))
                profiles.append(
                    {
                        "customer_id": i + 1,
                        "timestamp": timestamp,
                        "power_kw": float(power),
                    }
                )

        return pd.DataFrame(profiles)
