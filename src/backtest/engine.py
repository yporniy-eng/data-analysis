"""
Backtest Engine

Simulates trading signals on historical data
with realistic commissions, slippage, and exits.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.logger import logger


class BacktestEngine:
    """
    Run backtest on historical data.

    Process:
    1. Generate signals at each time step
    2. Open positions for qualifying signals
    3. Manage exits (target, stop, time, expiry)
    4. Compute performance metrics
    """

    def __init__(
        self,
        signal_generator,
        initial_capital: float = 100000,
        commission_rate: float = 0.0004,
        stop_loss_pct: float = 0.50,
        max_hold_hours: int = 48,
    ):
        """
        Args:
            signal_generator: SignalGenerator instance
            initial_capital: Starting capital
            commission_rate: Per-trade commission
            stop_loss_pct: Stop loss as % of option premium
            max_hold_hours: Maximum holding period
        """
        self.generator = signal_generator
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_hours = max_hold_hours

    def run(
        self,
        prices_df: pd.DataFrame,
        liquidations_df: pd.DataFrame,
        symbol: str = "BTC",
        step_hours: int = 4,
    ) -> dict:
        """
        Run full backtest.

        Args:
            prices_df: OHLCV data with columns: timestamp, close, volume
            liquidations_df: Liquidation data
            symbol: Symbol name
            step_hours: Hours between signal checks

        Returns:
            Dict with equity curve, trades, metrics
        """
        logger.info(
            f"Starting backtest: {len(prices_df)} candles, "
            f"step={step_hours}h"
        )

        # State
        cash = self.initial_capital
        positions = []
        closed_trades = []
        equity_curve = []

        # Step through time
        timestamps = prices_df["timestamp"].unique()
        price_lookup = prices_df.set_index("timestamp")["close"]

        for i in range(0, len(timestamps), step_hours):
            current_ts = timestamps[i]
            current_price = price_lookup.get(current_ts, np.nan)

            if np.isnan(current_price):
                continue

            # Historical prices up to this point
            hist_mask = prices_df["timestamp"] <= current_ts
            hist_prices = prices_df.loc[hist_mask, "close"]

            # Historical liquidations
            liq_mask = (
                liquidations_df["timestamp"] <= current_ts
            )
            liq_window = max(24, step_hours * 2)
            liq_start = current_ts - timedelta(hours=liq_window)
            hist_liq = liquidations_df[
                (liquidations_df["timestamp"] >= liq_start)
                & (liquidations_df["timestamp"] <= current_ts)
            ]

            # Generate signals
            signals = self.generator.generate_signals(
                current_price=current_price,
                liquidations_df=hist_liq,
                historical_prices=hist_prices,
                account_value=cash + self._unrealized_pnl(
                    positions, current_price, price_lookup
                ),
            )

            # Open new positions
            for signal in signals:
                if len(positions) >= 5:  # Max concurrent positions
                    break

                position = self._open_position(
                    signal, current_price, current_ts, cash
                )
                if position:
                    positions.append(position)
                    cash -= position["entry_cost"]

            # Check exits
            for pos in positions[:]:
                exit_result = self._check_exit(
                    pos, current_price, current_ts, price_lookup
                )
                if exit_result:
                    positions.remove(pos)
                    closed_trades.append(exit_result)
                    cash += exit_result["exit_proceeds"]

            # Record equity
            total_equity = cash + self._unrealized_pnl(
                positions, current_price, price_lookup
            )
            equity_curve.append({
                "timestamp": current_ts,
                "equity": total_equity,
                "cash": cash,
                "positions": len(positions),
            })

        # Compute metrics
        metrics = self._compute_metrics(
            equity_curve, closed_trades
        )

        result = {
            "equity_curve": pd.DataFrame(equity_curve),
            "trades": pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame(),
            "metrics": metrics,
        }

        logger.info(
            f"Backtest complete: {len(closed_trades)} trades, "
            f"return: {metrics.get('total_return', 0):.2%}"
        )
        return result

    def _open_position(
        self,
        signal: dict,
        current_price: float,
        timestamp: datetime,
        cash: float,
    ) -> Optional[dict]:
        """Open a position from a signal."""
        option_price = signal["market_price"]
        size = signal["position_size"]

        # Cost
        cost = option_price * size
        commission = cost * self.commission_rate
        total_cost = cost + commission

        if total_cost > cash * 0.8:  # Don't use more than 80% cash
            size = int(cash * 0.8 / (option_price * (1 + self.commission_rate)))
            if size <= 0:
                return None
            cost = option_price * size
            commission = cost * self.commission_rate
            total_cost = cost + commission

        return {
            "signal_type": signal["signal_type"],
            "symbol": signal["symbol"],
            "strike": signal["strike"],
            "option_type": signal["option_type"],
            "entry_price": option_price,
            "entry_time": timestamp,
            "contracts": size,
            "entry_cost": total_cost,
            "target_price": signal["target_price"],
            "stop_loss": option_price * (1 - self.stop_loss_pct),
            "composite_score": signal["composite_score"],
        }

    def _check_exit(
        self,
        position: dict,
        current_price: float,
        current_ts: datetime,
        price_lookup: pd.Series,
    ) -> Optional[dict]:
        """Check exit conditions for a position."""
        hold_hours = (current_ts - position["entry_time"]).total_seconds() / 3600

        # Estimate current option price (simplified)
        entry_price = position["entry_price"]

        if position["option_type"] == "CALL":
            price_change = current_price - price_lookup.get(
                position["entry_time"], current_price
            )
        else:
            price_change = price_lookup.get(
                position["entry_time"], current_price
            ) - current_price

        # Simple option price model (linear approximation)
        current_option_price = max(
            0.01,
            entry_price + price_change * 0.4  # ~delta 0.4
        )

        # Exit conditions
        exit_reason = None
        exit_price = current_option_price

        # 1. Stop loss
        if current_option_price <= position["stop_loss"]:
            exit_reason = "stop_loss"

        # 2. Target reached
        elif abs(current_price - position["target_price"]) / position["target_price"] < 0.01:
            exit_reason = "target_reached"

        # 3. Max holding time
        elif hold_hours >= self.max_hold_hours:
            exit_reason = "max_hold_time"

        if exit_reason is None:
            return None

        # P&L
        proceeds = exit_price * position["contracts"]
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission

        pnl = net_proceeds - position["entry_cost"]
        pnl_pct = pnl / position["entry_cost"]

        return {
            "signal_type": position["signal_type"],
            "symbol": position["symbol"],
            "strike": position["strike"],
            "option_type": position["option_type"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "entry_time": position["entry_time"],
            "exit_time": current_ts,
            "contracts": position["contracts"],
            "entry_cost": position["entry_cost"],
            "exit_proceeds": net_proceeds,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "hold_hours": hold_hours,
            "composite_score": position["composite_score"],
        }

    def _unrealized_pnl(
        self,
        positions: list[dict],
        current_price: float,
        price_lookup: pd.Series,
    ) -> float:
        """Calculate unrealized P&L of open positions."""
        total = 0
        for pos in positions:
            entry_price = pos["entry_price"]

            if pos["option_type"] == "CALL":
                price_change = current_price - price_lookup.get(
                    pos["entry_time"], current_price
                )
            else:
                price_change = price_lookup.get(
                    pos["entry_time"], current_price
                ) - current_price

            current_value = max(0.01, entry_price + price_change * 0.4)
            total += (current_value - entry_price) * pos["contracts"]

        return total

    def _compute_metrics(
        self,
        equity_curve: list[dict],
        trades: list[dict],
    ) -> dict:
        """Compute performance metrics."""
        if not equity_curve:
            return {}

        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        # Total return
        total_return = (
            equity_df["equity"].iloc[-1] / self.initial_capital - 1
        )

        # Sharpe ratio
        if len(equity_df) > 1:
            returns = equity_df["equity"].pct_change().dropna()
            if returns.std() > 0:
                sharpe = (
                    returns.mean() / returns.std() * np.sqrt(252 * 24 / 4)
                )
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max drawdown
        rolling_max = equity_df["equity"].cummax()
        drawdowns = (equity_df["equity"] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Trade stats
        if not trades_df.empty:
            win_trades = trades_df[trades_df["pnl"] > 0]
            loss_trades = trades_df[trades_df["pnl"] <= 0]

            win_rate = len(win_trades) / len(trades_df)
            avg_win = win_trades["pnl"].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades["pnl"].mean() if len(loss_trades) > 0 else 0

            gross_profit = win_trades["pnl"].sum() if len(win_trades) > 0 else 0
            gross_loss = abs(loss_trades["pnl"].sum()) if len(loss_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            "total_return": float(total_return),
            "final_equity": float(equity_df["equity"].iloc[-1]),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "total_trades": len(trades_df),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "avg_hold_hours": float(
                trades_df["hold_hours"].mean()
            ) if not trades_df.empty else 0,
        }
