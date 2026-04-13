#!/usr/bin/env python
"""
Backtest script

Usage:
    python scripts/run_backtest.py --symbol BTC/USDT --days 180
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import DataCollector
from src.math_core.bsm import BlackScholesCalculator
from src.math_core.ewma import EWMACalculator
from src.liquidations.detector import LiquidationClusterDetector
from src.signals.generator import SignalGenerator
from src.backtest.engine import BacktestEngine
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    logger.info(f"Starting backtest: {args.symbol} ({args.days} days)")

    # Load data
    collector = DataCollector(exchange_name="binance")
    ohlcv = collector.fetch_ohlcv_history(args.symbol, "1h", args.days)

    if ohlcv.empty:
        logger.error("No OHLCV data")
        return

    sym_raw = args.symbol.replace("/", "")
    liqs = collector.fetch_liquidations(sym_raw, limit=2000)

    logger.info(f"Loaded {len(ohlcv)} candles, {len(liqs)} liquidations")

    # Initialize components
    bsm = BlackScholesCalculator(risk_free_rate=0.053)
    ewma = EWMACalculator(lambda_decay=0.94)
    liq_detector = LiquidationClusterDetector()

    generator = SignalGenerator(
        bsm_calculator=bsm,
        ewma_calculator=ewma,
        liq_detector=liq_detector,
        min_profit_ratio=1.5,
        min_composite_score=0.5,
        kelly_fraction=0.25,
        max_risk_pct=0.02,
    )

    # Run backtest
    engine = BacktestEngine(
        signal_generator=generator,
        initial_capital=args.capital,
        commission_rate=0.0004,
        stop_loss_pct=0.50,
        max_hold_hours=48,
    )

    result = engine.run(ohlcv, liqs, symbol=args.symbol, step_hours=4)

    # Print results
    metrics = result["metrics"]
    logger.info("=" * 50)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
    logger.info(f"Final Equity: ${metrics.get('final_equity', 0):,.2f}")
    logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
    logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    logger.info(f"Avg Hold: {metrics.get('avg_hold_hours', 0):.1f}h")
    logger.info("=" * 50)

    # Save to file
    if args.output:
        output = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "symbol": args.symbol,
            "days": args.days,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
