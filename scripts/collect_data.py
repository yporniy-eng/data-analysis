#!/usr/bin/env python
"""
Data collection script

Usage:
    python scripts/collect_data.py --symbol BTC/USDT --days 90
    python scripts/collect_data.py --all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import DataCollector
from src.data.storage import ClickHouseStorage
from src.logger import logger
import time


def main():
    parser = argparse.ArgumentParser(description="Collect market data")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading pair (e.g., BTC/USDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of history to fetch",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect for all configured symbols",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database storage (CSV fallback)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        choices=["binance", "bybit"],
        help="Exchange to use",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1h,4h",
        help="Comma-separated timeframes",
    )

    args = parser.parse_args()

    symbols = ["BTC/USDT", "ETH/USDT"] if args.all else [args.symbol]
    timeframes = args.timeframes.split(",")

    collector = DataCollector(exchange_name=args.exchange)

    # Try to connect to database
    storage = None
    if not args.no_db:
        try:
            storage = ClickHouseStorage()
            logger.info("Database connection OK")
        except Exception as e:
            logger.warning(f"Database not available, skipping: {e}")
            storage = None

    for symbol in symbols:
        for tf in timeframes:
            logger.info(f"Collecting {symbol} {tf} ({args.days} days)")

            # Fetch OHLCV
            df = collector.fetch_ohlcv_history(symbol, tf, args.days)

            if df.empty:
                logger.warning(f"No data for {symbol} {tf}")
                continue

            # Save to database
            if storage:
                try:
                    storage.insert_ohlcv(df)
                except Exception as e:
                    logger.error(f"Error saving OHLCV: {e}")

            # Fetch liquidations
            sym_raw = symbol.replace("/", "")
            liqs = collector.fetch_liquidations(sym_raw, limit=1000)

            if not liqs.empty and storage:
                try:
                    storage.insert_liquidations(liqs)
                except Exception as e:
                    logger.error(f"Error saving liquidations: {e}")

            # Fetch funding rates
            funding = collector.fetch_funding_rates(symbol, limit=100)

            if not funding.empty and storage:
                try:
                    storage.insert_funding_rates(funding)
                except Exception as e:
                    logger.error(f"Error saving funding rates: {e}")

            time.sleep(1)  # Rate limit

    logger.info("Data collection complete")


if __name__ == "__main__":
    main()
