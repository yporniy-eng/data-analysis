"""
Black-Scholes-Merton Option Pricing

Implements:
- BSM formula for call and put options
- Greeks: delta, gamma, vega, theta, rho
- Kurtosis/skewness correction (Edgeworth expansion)
"""

import numpy as np
from scipy.stats import norm
from typing import Optional


class BlackScholesCalculator:
    """
    Black-Scholes-Merton option pricing with corrections.

    Based on:
    - Black & Scholes (1973) - original formula
    - Merton (1973) - extensions
    - Edgeworth expansion for non-normal distributions
    """

    def __init__(
        self,
        risk_free_rate: float = 0.053,
        use_correction: bool = True,
    ):
        """
        Args:
            risk_free_rate: Annual risk-free rate (SOFR)
            use_correction: Apply kurtosis/skewness correction
        """
        self.r = risk_free_rate
        self.use_correction = use_correction

    def price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = "call",
        kurtosis: Optional[float] = None,
        skewness: Optional[float] = None,
    ) -> float:
        """
        Calculate option price.

        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
            kurtosis: Historical kurtosis (for correction)
            skewness: Historical skewness (for correction)

        Returns:
            Option price
        """
        if T <= 0:
            # Expired option - intrinsic value
            if option_type == "call":
                return max(0, S - K)
            else:
                return max(0, K - S)

        # Standard BSM price
        bs_price = self._bsm_price(S, K, T, sigma, option_type)

        # Apply correction for non-normal distribution
        if self.use_correction and kurtosis is not None:
            correction = self._edgeworth_correction(
                S, K, T, sigma, option_type, kurtosis, skewness
            )
            bs_price += correction

        return max(0, bs_price)  # Option price can't be negative

    def _bsm_price(
        self, S: float, K: float, T: float, sigma: float, option_type: str
    ) -> float:
        """Standard Black-Scholes price."""
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def delta(
        self, S: float, K: float, T: float, sigma: float,
        option_type: str = "call"
    ) -> float:
        """Option delta."""
        d1, _ = self._calculate_d1_d2(S, K, T, sigma)
        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """Option gamma (same for call and put)."""
        d1, _ = self._calculate_d1_d2(S, K, T, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """Option vega (same for call and put)."""
        d1, _ = self._calculate_d1_d2(S, K, T, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)

    def theta(
        self, S: float, K: float, T: float, sigma: float,
        option_type: str = "call"
    ) -> float:
        """Option theta (time decay per day)."""
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma)

        common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type == "call":
            theta = common - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            theta = common + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)

        return theta / 365  # Per day

    def rho(
        self, S: float, K: float, T: float, sigma: float,
        option_type: str = "call"
    ) -> float:
        """Option rho (interest rate sensitivity)."""
        _, d2 = self._calculate_d1_d2(S, K, T, sigma)

        if option_type == "call":
            return K * T * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return -K * T * np.exp(-self.r * T) * norm.cdf(-d2)

    def all_greeks(
        self, S: float, K: float, T: float, sigma: float,
        option_type: str = "call"
    ) -> dict:
        """Calculate all Greeks at once."""
        return {
            "delta": self.delta(S, K, T, sigma, option_type),
            "gamma": self.gamma(S, K, T, sigma),
            "vega": self.vega(S, K, T, sigma),
            "theta": self.theta(S, K, T, sigma, option_type),
            "rho": self.rho(S, K, T, sigma, option_type),
        }

    def _edgeworth_correction(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,
        kurtosis: float,
        skewness: Optional[float] = None,
    ) -> float:
        """
        Edgeworth expansion correction for non-normal distributions.

        Adjusts BSM price for:
        - Excess kurtosis (fat tails): κ > 3
        - Skewness: ν ≠ 0
        """
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma)

        # Kurtosis coefficient
        kappa_4 = (kurtosis - 3) / 24  # Excess kurtosis normalized

        # Base correction term
        correction = kappa_4 * S * sigma * np.sqrt(T) * (
            d1 * d2 - 1
        ) * norm.pdf(d1)

        # Add skewness correction if available
        if skewness is not None:
            kappa_3 = skewness / 6
            correction += kappa_3 * S * sigma * np.sqrt(T) * (
                d1 + d2
            ) * norm.pdf(d1)

        return correction

    def _calculate_d1_d2(
        self, S: float, K: float, T: float, sigma: float
    ) -> tuple:
        """Calculate d1 and d2 parameters."""
        sigma_sqrt_T = sigma * np.sqrt(T)
        d1 = (
            np.log(S / K) + (self.r + sigma ** 2 / 2) * T
        ) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        return d1, d2

    def implied_volatility(
        self,
        S: float,
        K: float,
        T: float,
        market_price: float,
        option_type: str = "call",
        max_iter: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry
            market_price: Observed market price
            option_type: 'call' or 'put'

        Returns:
            Implied volatility
        """
        sigma = 0.3  # Initial guess

        for _ in range(max_iter):
            price = self._bsm_price(S, K, T, sigma, option_type)
            vega = self.vega(S, K, T, sigma)

            diff = price - market_price

            if abs(diff) < tolerance:
                return sigma

            if vega == 0:
                return sigma

            sigma = sigma - diff / vega

            # Bounds check
            sigma = np.clip(sigma, 0.01, 5.0)

        return sigma
