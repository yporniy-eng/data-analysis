"""
Signal Generator

Combines:
- Liquidation cluster analysis
- BSM option pricing
- Market state clustering
- Position sizing (Kelly)

To generate actionable option signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.math_core.bsm import BlackScholesCalculator
from src.math_core.ewma import EWMACalculator
from src.liquidations.detector import LiquidationClusterDetector
from src.logger import logger


class SignalGenerator:
    """
    Generate option trading signals based on
    liquidation clusters and option mispricing.
    """

    def __init__(
        self,
        bsm_calculator: BlackScholesCalculator,
        ewma_calculator: EWMACalculator,
        liq_detector: LiquidationClusterDetector,
        min_profit_ratio: float = 1.5,
        min_composite_score: float = 0.6,
        kelly_fraction: float = 0.25,
        max_risk_pct: float = 0.02,
    ):
        """
        Args:
            bsm_calculator: BSM pricing engine
            ewma_calculator: EWMA volatility engine
            liq_detector: Liquidation cluster detector
            min_profit_ratio: Minimum expected_move / option_price
            min_composite_score: Minimum signal quality score
            kelly_fraction: Fraction of Kelly to use
            max_risk_pct: Max % of account per trade
        """
        self.bsm = bsm_calculator
        self.ewma = ewma_calculator
        self.liq = liq_detector
        self.min_profit_ratio = min_profit_ratio
        self.min_composite_score = min_composite_score
        self.kelly_fraction = kelly_fraction
        self.max_risk_pct = max_risk_pct

    def generate_signals(
        self,
        current_price: float,
        liquidations_df: pd.DataFrame,
        historical_prices: pd.Series,
        account_value: float = 100000,
    ) -> list[dict]:
        """
        Generate trading signals.

        Args:
            current_price: Current market price
            liquidations_df: Recent liquidation data
            historical_prices: Price history for volatility
            account_value: Account size for position sizing

        Returns:
            List of signal dicts
        """
        signals = []

        # 1. Detect liquidation clusters
        clusters = self.liq.detect_clusters(
            liquidations_df, current_price
        )

        if not clusters:
            logger.info("No liquidation clusters found")
            return []

        # 2. Compute EWMA volatility
        if len(historical_prices) > 10:
            log_returns = np.log(historical_prices).diff().dropna()
            sigma = self.ewma.compute_volatility(
                log_returns.values, annualize=True
            )
        else:
            sigma = 0.65  # Default for crypto

        # 3. Compute market stats for correction
        if len(historical_prices) > 50:
            log_returns = np.log(historical_prices).diff().dropna()
            kurtosis = float(
                pd.Series(log_returns).kurtosis() + 3
            )  # Excess → actual
            skewness = float(pd.Series(log_returns).skew())
        else:
            kurtosis = 3.0
            skewness = 0.0

        # 4. Generate signals for each cluster
        for cluster in clusters:
            signal = self._evaluate_cluster(
                cluster=cluster,
                current_price=current_price,
                sigma=sigma,
                kurtosis=kurtosis,
                skewness=skewness,
                account_value=account_value,
            )
            if signal:
                signals.append(signal)

        # Sort by composite score
        signals.sort(
            key=lambda s: s["composite_score"], reverse=True
        )

        logger.info(
            f"Generated {len(signals)} signals "
            f"(price: {current_price:.0f}, vol: {sigma:.1%})"
        )
        return signals

    def _evaluate_cluster(
        self,
        cluster: dict,
        current_price: float,
        sigma: float,
        kurtosis: float,
        skewness: float,
        account_value: float,
    ) -> Optional[dict]:
        """Evaluate a single cluster as a potential signal."""

        target_price = cluster["price_center"]
        distance_pct = abs(cluster["distance_pct"])

        # Determine option type
        if cluster["position"] == "above":
            option_type = "call"
        else:
            option_type = "put"

        # Strike = cluster center (rounded)
        strike = round(target_price, -2)  # Round to nearest 100

        # Time to expiry: estimate based on distance
        # Farther targets need more time
        days_to_expiry = self._estimate_expiry(distance_pct)
        T = days_to_expiry / 365.0

        # BSM fair value (corrected)
        fair_value = self.bsm.price(
            S=current_price,
            K=strike,
            T=T,
            sigma=sigma,
            option_type=option_type,
            kurtosis=kurtosis,
            skewness=skewness,
        )

        if fair_value <= 0:
            return None

        # Market price (simulated as BSM + small spread)
        # In production, this would come from actual option order book
        spread_pct = max(0.05, 0.15 - distance_pct * 2)
        market_price = fair_value * (1 + spread_pct)

        # Expected move (based on cluster strength)
        expected_move = self._estimate_move(
            cluster, current_price, sigma, days_to_expiry
        )

        # Profit ratio
        if market_price <= 0:
            return None

        profit_ratio = expected_move / market_price

        if profit_ratio < self.min_profit_ratio:
            return None

        # Greeks
        greeks = self.bsm.all_greeks(
            S=current_price, K=strike, T=T, sigma=sigma,
            option_type=option_type
        )

        # Composite score
        composite_score = self._compute_score(
            fair_value=fair_value,
            market_price=market_price,
            profit_ratio=profit_ratio,
            cluster_z_score=cluster["max_z_score"],
            distance_pct=distance_pct,
            greeks=greeks,
        )

        if composite_score < self.min_composite_score:
            return None

        # Position size (Kelly)
        position_size = self._kelly_position(
            win_probability=self._estimate_win_prob(distance_pct, sigma),
            avg_win=expected_move,
            avg_loss=market_price * 0.5,  # Stop at -50% of premium
            account_value=account_value,
        )

        if position_size <= 0:
            return None

        expiry_date = datetime.utcnow() + timedelta(days=days_to_expiry)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "signal_type": f"BUY_{option_type.upper()}",
            "symbol": "BTC",
            "strike": strike,
            "expiry": expiry_date.strftime("%Y-%m-%d"),
            "option_type": option_type.upper(),
            "market_price": round(market_price, 2),
            "fair_value": round(fair_value, 2),
            "mispricing_pct": round(
                (fair_value - market_price) / market_price, 4
            ),
            "expected_move": round(expected_move, 2),
            "profit_ratio": round(profit_ratio, 2),
            "composite_score": round(composite_score, 4),
            "probability_of_success": round(
                self._estimate_win_prob(distance_pct, sigma), 4
            ),
            "target_price": round(target_price, 2),
            "target_distance_pct": round(
                cluster["distance_pct"], 4
            ),
            "position_size": int(position_size),
            "risk_usd": round(position_size * market_price * 0.5, 2),
            "reward_usd": round(position_size * expected_move, 2),
            "greeks": {k: round(v, 4) for k, v in greeks.items()},
            "cluster_info": {
                "total_value": cluster["total_value"],
                "max_z_score": cluster["max_z_score"],
                "radius": cluster["radius"],
            },
        }

    def _estimate_expiry(self, distance_pct: float) -> int:
        """Estimate days to expiry based on target distance."""
        # Closer targets → shorter expiry
        # Further targets → longer expiry
        days = max(7, min(60, int(distance_pct * 500)))
        return days

    def _estimate_move(
        self,
        cluster: dict,
        current_price: float,
        sigma: float,
        days: int,
    ) -> float:
        """Estimate expected price move."""
        # Base move from volatility
        vol_move = current_price * sigma * np.sqrt(days / 365)

        # Cluster strength multiplier
        cluster_strength = min(
            2.0, cluster["max_z_score"] / 2.0
        )

        return vol_move * cluster_strength * 0.5

    def _compute_score(
        self,
        fair_value: float,
        market_price: float,
        profit_ratio: float,
        cluster_z_score: float,
        distance_pct: float,
        greeks: dict,
    ) -> float:
        """
        Compute composite signal score (0-1).

        Factors:
        - Mispricing alpha
        - Profit ratio
        - Cluster strength
        - Delta (directional conviction)
        """
        # Normalize each factor to 0-1 range
        mispricing = max(0, (fair_value - market_price) / market_price)
        mispricing = min(1, mispricing * 5)  # Scale

        profit = min(1, profit_ratio / 5)  # Cap at 5x

        cluster = min(1, cluster_z_score / 5)  # Z-score normalized

        # Closer clusters are more actionable
        proximity = max(0, 1 - distance_pct * 10)

        # Equal weights (can be made adaptive)
        w1, w2, w3, w4 = 0.3, 0.3, 0.2, 0.2

        score = (
            w1 * mispricing
            + w2 * profit
            + w3 * cluster
            + w4 * proximity
        )

        return np.clip(score, 0, 1)

    def _estimate_win_prob(
        self, distance_pct: float, sigma: float
    ) -> float:
        """Estimate probability of reaching target."""
        # Simple Brownian motion probability
        # P(reach target) ≈ N(distance / (sigma * sqrt(T)))
        T = self._estimate_expiry(distance_pct) / 365
        z = distance_pct / (sigma * np.sqrt(T))

        from scipy.stats import norm
        return float(1 - norm.cdf(z))

    def _kelly_position(
        self,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        account_value: float,
    ) -> float:
        """
        Calculate position size using fractional Kelly criterion.

        f* = (b * p - q) / b
        where b = avg_win/avg_loss, p = win_prob, q = 1-p
        """
        if avg_loss <= 0:
            return 0

        b = avg_win / avg_loss  # Payoff ratio
        p = win_probability
        q = 1 - p

        kelly_f = (b * p - q) / b

        if kelly_f <= 0:
            return 0

        # Fractional Kelly
        adjusted_kelly = kelly_f * self.kelly_fraction

        # Risk limit
        max_risk = account_value * self.max_risk_pct

        # Position in USD
        position_usd = min(
            account_value * adjusted_kelly,
            max_risk,
        )

        # Convert to contracts (rough estimate)
        return max(0, position_usd / 100)
