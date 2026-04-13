"""
EWMA (Exponentially Weighted Moving Average) Calculator

Computes:
- EWMA volatility for fast adaptation to market shocks
- Adaptive lambda optimization
"""

import numpy as np
import pandas as pd
from typing import Optional


class EWMACalculator:
    """
    EWMA volatility calculator.

    Uses exponential weighting to give more importance
    to recent returns, adapting faster to market changes.
    """

    def __init__(self, lambda_decay: float = 0.94):
        """
        Args:
            lambda_decay: Decay factor (0.94 = 6% weight on newest return)
        """
        self.lambda_decay = lambda_decay

    def compute_volatility(
        self,
        returns: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """
        Compute EWMA volatility.

        Args:
            returns: Array of log returns
            annualize: Convert to annualized vol

        Returns:
            EWMA volatility
        """
        if len(returns) < 10:
            return np.std(returns) * (np.sqrt(252) if annualize else 1.0)

        # EWMA variance
        variance = self._ewma_variance(returns)
        vol = np.sqrt(variance)

        if annualize:
            vol *= np.sqrt(252)

        return vol

    def compute_volatility_series(
        self,
        prices: pd.Series,
        returns_period: int = 1,
    ) -> pd.Series:
        """
        Compute rolling EWMA volatility series.

        Args:
            prices: Price series
            returns_period: Period for returns (1 = single period)

        Returns:
            Series of EWMA volatilities
        """
        # Calculate log returns
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()

        # EWMA variance
        variance = self._ewma_variance_series(returns)
        vol = np.sqrt(variance) * np.sqrt(252)

        return vol

    def optimize_lambda(
        self,
        returns: np.ndarray,
        lambda_range: tuple = (0.85, 0.99),
        steps: int = 50,
    ) -> float:
        """
        Optimize lambda by minimizing forecast error.

        Args:
            returns: Historical returns
            lambda_range: (min, max) lambda to search
            steps: Number of grid points

        Returns:
            Optimal lambda
        """
        lambdas = np.linspace(lambda_range[0], lambda_range[1], steps)
        errors = []

        for lam in lambdas:
            calc = EWMACalculator(lam)

            # Use last 80% for training, first 20% for validation
            split = int(len(returns) * 0.8)
            if split < 20:
                continue

            train_returns = returns[:split]
            test_returns = returns[split:]

            # Forecast variance
            train_var = calc._ewma_variance(train_returns)
            forecast_vol = np.sqrt(train_var) * np.sqrt(252)

            # Actual variance in test set
            actual_vol = np.std(test_returns) * np.sqrt(252)

            # Forecast error
            error = abs(forecast_vol - actual_vol)
            errors.append(error)

        if not errors:
            return self.lambda_decay

        best_idx = np.argmin(errors)
        return lambdas[best_idx]

    def _ewma_variance(self, returns: np.ndarray) -> float:
        """Compute single EWMA variance value."""
        n = len(returns)
        if n == 0:
            return 0.0

        # Weights: exponential decay
        weights = np.array([
            (1 - self.lambda_decay) * self.lambda_decay ** i
            for i in range(n - 1, -1, -1)
        ])
        weights = weights / weights.sum()  # Normalize

        # Squared returns
        squared_returns = returns ** 2

        return np.sum(weights * squared_returns)

    def _ewma_variance_series(
        self, returns: pd.Series
    ) -> pd.Series:
        """Compute EWMA variance for each point in time."""
        n = len(returns)
        variance = pd.Series(index=returns.index, dtype=float)

        # Initialize with sample variance
        if n < 2:
            return variance

        variance.iloc[0] = returns.iloc[0] ** 2

        for i in range(1, n):
            variance.iloc[i] = (
                self.lambda_decay * variance.iloc[i - 1]
                + (1 - self.lambda_decay) * returns.iloc[i] ** 2
            )

        return variance

    def half_life(self) -> float:
        """
        Calculate half-life of EWMA.

        Number of periods for weight to decay to 50%.
        """
        if self.lambda_decay >= 1.0:
            return float("inf")
        return -np.log(2) / np.log(self.lambda_decay)
