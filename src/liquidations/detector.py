"""
Liquidation Cluster Detector

Finds concentration zones of liquidations that act as
price magnets and potential cascade triggers.
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import zscore

from src.logger import logger


class LiquidationClusterDetector:
    """
    Detect clusters of liquidations in price space.

    Algorithm:
    1. Build histogram of liquidation values in price bins
    2. Compute z-score for each bin
    3. Find bins with z-score > adaptive threshold
    4. Merge adjacent significant bins into clusters
    """

    def __init__(
        self,
        price_range_pct: float = 0.05,
        bin_size_pct: float = 0.001,
        z_score_percentile: int = 90,
    ):
        """
        Args:
            price_range_pct: Range around current price (±5%)
            bin_size_pct: Size of each price bin (0.1%)
            z_score_percentile: Percentile for z-score threshold
        """
        self.price_range_pct = price_range_pct
        self.bin_size_pct = bin_size_pct
        self.z_score_percentile = z_score_percentile

    def detect_clusters(
        self,
        liquidations_df: pd.DataFrame,
        current_price: float,
    ) -> list[dict]:
        """
        Find liquidation clusters around current price.

        Args:
            liquidations_df: DataFrame with columns: price, value_usd
            current_price: Current market price

        Returns:
            List of cluster dictionaries
        """
        if liquidations_df.empty or "price" not in liquidations_df.columns:
            logger.warning("No liquidation data for cluster detection")
            return []

        df = liquidations_df.copy()

        # Filter to price range
        price_min = current_price * (1 - self.price_range_pct)
        price_max = current_price * (1 + self.price_range_pct)
        df = df[(df["price"] >= price_min) & (df["price"] <= price_max)]

        if df.empty:
            return []

        # Ensure value_usd column
        if "value_usd" not in df.columns:
            if "quantity" in df.columns and "price" in df.columns:
                df["value_usd"] = df["quantity"] * df["price"]
            else:
                df["value_usd"] = 1.0  # Count each as 1

        # Build histogram
        bin_size = current_price * self.bin_size_pct
        bins = np.arange(price_min, price_max + bin_size, bin_size)

        histogram, bin_edges = np.histogram(
            df["price"],
            bins=bins,
            weights=df["value_usd"]
        )

        if histogram.sum() == 0:
            return []

        # Compute z-scores
        mean_val = np.mean(histogram)
        std_val = np.std(histogram)

        if std_val == 0:
            logger.warning("Zero std in liquidation histogram")
            return []

        z_scores = (histogram - mean_val) / std_val

        # Adaptive threshold
        z_threshold = np.percentile(
            np.abs(z_scores), self.z_score_percentile
        )
        z_threshold = max(z_threshold, 1.5)  # Minimum 1.5 sigma

        # Find significant bins
        significant = []
        for i, z in enumerate(z_scores):
            if abs(z) > z_threshold:
                significant.append({
                    "bin_center": float(
                        (bin_edges[i] + bin_edges[i + 1]) / 2
                    ),
                    "total_value": float(histogram[i]),
                    "z_score": float(z),
                    "bin_index": i,
                })

        if not significant:
            return []

        # Merge adjacent bins
        clusters = self._merge_bins(significant, bin_size)

        # Classify relative to current price
        for cluster in clusters:
            cluster["distance_pct"] = (
                (cluster["price_center"] - current_price) / current_price
            )
            cluster["position"] = (
                "above" if cluster["distance_pct"] > 0 else "below"
            )

        logger.info(
            f"Found {len(clusters)} liquidation clusters "
            f"(threshold z={z_threshold:.2f})"
        )
        return clusters

    def build_heatmap_data(
        self,
        liquidations_df: pd.DataFrame,
        current_price: float,
        time_windows: list[str] = None,
    ) -> pd.DataFrame:
        """
        Build heatmap data: time windows × price bins.

        Args:
            liquidations_df: Must have 'timestamp', 'price', 'value_usd'
            current_price: Current price
            time_windows: List like ['4h', '12h', '24h']

        Returns:
            DataFrame with heatmap values
        """
        if time_windows is None:
            time_windows = ["4h", "12h", "24h", "48h"]

        if liquidations_df.empty or "timestamp" not in liquidations_df.columns:
            return pd.DataFrame()

        df = liquidations_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        now = df["timestamp"].max()

        price_min = current_price * (1 - self.price_range_pct)
        price_max = current_price * (1 + self.price_range_pct)
        n_bins = 100
        bins = np.linspace(price_min, price_max, n_bins)

        rows = []
        for window in time_windows:
            hours = int(window.replace("h", ""))
            start_time = now - pd.Timedelta(hours=hours)

            window_data = df[df["timestamp"] >= start_time]

            if "value_usd" in window_data.columns:
                weights = window_data["value_usd"]
            else:
                weights = np.ones(len(window_data))

            hist, _ = np.histogram(
                window_data["price"],
                bins=bins,
                weights=weights
            )
            rows.append(hist)

        return pd.DataFrame(
            rows,
            index=time_windows,
            columns=[f"{p:.0f}" for p in bins[:-1]]
        )

    def get_nearest_cluster(
        self,
        clusters: list[dict],
        current_price: float,
        direction: str = "any",
    ) -> Optional[dict]:
        """
        Find nearest liquidation cluster.

        Args:
            clusters: List of cluster dicts
            current_price: Current price
            direction: 'above', 'below', or 'any'

        Returns:
            Nearest cluster dict or None
        """
        if not clusters:
            return None

        filtered = clusters
        if direction == "above":
            filtered = [c for c in clusters if c["position"] == "above"]
        elif direction == "below":
            filtered = [c for c in clusters if c["position"] == "below"]

        if not filtered:
            return None

        return min(
            filtered,
            key=lambda c: abs(c["price_center"] - current_price)
        )

    def _merge_bins(
        self,
        significant_bins: list[dict],
        bin_size: float,
    ) -> list[dict]:
        """Merge adjacent significant bins into clusters."""
        if not significant_bins:
            return []

        significant_bins.sort(key=lambda b: b["bin_index"])

        clusters = []
        current_group = [significant_bins[0]]

        for i in range(1, len(significant_bins)):
            prev_idx = significant_bins[i - 1]["bin_index"]
            curr_idx = significant_bins[i]["bin_index"]

            if curr_idx - prev_idx <= 2:  # Adjacent or one apart
                current_group.append(significant_bins[i])
            else:
                clusters.append(self._aggregate_cluster(current_group))
                current_group = [significant_bins[i]]

        # Don't forget last group
        clusters.append(self._aggregate_cluster(current_group))

        return clusters

    def _aggregate_cluster(self, bins: list[dict]) -> dict:
        """Aggregate bins into a single cluster."""
        total_value = sum(b["total_value"] for b in bins)

        # Weighted center
        if total_value > 0:
            price_center = sum(
                b["bin_center"] * b["total_value"] for b in bins
            ) / total_value
        else:
            price_center = np.mean([b["bin_center"] for b in bins])

        return {
            "price_center": float(price_center),
            "total_value": float(total_value),
            "radius": float(
                (bins[-1]["bin_center"] - bins[0]["bin_center"]) / 2
            ),
            "max_z_score": float(max(abs(b["z_score"]) for b in bins)),
            "bin_count": len(bins),
        }
