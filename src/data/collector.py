"""
Data Collector - Fetch market data from exchanges

Supports:
- Binance (public data, no API key required)
- Bybit (public data, no API key required)

Data types:
- OHLCV (candlestick data)
- Liquidations (force orders)
- Funding rates
- Open interest
"""

import ccxt
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
from typing import Optional

from src.logger import logger


class DataCollector:
    """Collect market data from cryptocurrency exchanges."""

    def __init__(
        self,
        exchange_name: str = "binance",
        api_key: str = None,
        api_secret: str = None,
    ):
        """
        Initialize exchange connection.

        Args:
            exchange_name: 'binance' or 'bybit'
            api_key: Optional API key (not needed for public data)
            api_secret: Optional API secret
        """
        exchange_class = getattr(ccxt, exchange_name)
        exchange_config = {}

        if api_key and api_secret:
            exchange_config = {
                "apiKey": api_key,
                "secret": api_secret,
            }

        self.exchange = exchange_class(exchange_config)
        self.exchange_name = exchange_name
        logger.info(f"Connected to {exchange_name}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
        since: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            limit: Number of candles (max 1000 per request)
            since: Start time (optional)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(
            f"Fetching OHLCV: {symbol} {timeframe} limit={limit}"
        )

        since_ms = None
        if since:
            since_ms = int(since.timestamp() * 1000)

        ohlcv = self.exchange.fetch_ohlcv(
            symbol, timeframe, since=since_ms, limit=limit
        )

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol
        df["exchange"] = self.exchange_name
        df["timeframe"] = timeframe

        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df

    def fetch_ohlcv_history(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data (handles pagination).

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days to fetch

        Returns:
            DataFrame with all candles
        """
        logger.info(f"Fetching {days} days of {symbol} {timeframe} history")

        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)
        current_dt = start_dt

        all_data = []
        max_per_request = 1000

        while current_dt < end_dt:
            df = self.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=max_per_request,
                since=current_dt,
            )

            if df.empty:
                break

            all_data.append(df)

            # Move forward
            current_dt = df["timestamp"].max() + timedelta(
                minutes=self._timeframe_to_minutes(timeframe)
            )

            # Rate limit
            time.sleep(0.2)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values(
            "timestamp"
        ).reset_index(drop=True)

        logger.info(f"Total: {len(result)} candles for {symbol}")
        return result

    def fetch_liquidations(
        self,
        symbol: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch recent liquidation data.

        Note: Binance force orders are public but limited.
        Bybit has better liquidation data via API.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT' for Binance)
            limit: Number of records

        Returns:
            DataFrame with liquidation data
        """
        # Normalize symbol format for REST API
        if self.exchange_name == "binance":
            return self._fetch_binance_liquidations(symbol, limit)
        elif self.exchange_name == "bybit":
            return self._fetch_bybit_liquidations(symbol, limit)

    def _fetch_binance_liquidations(
        self,
        symbol: str,
        limit: int,
    ) -> pd.DataFrame:
        """Fetch Binance force orders (public endpoint)."""
        # Binance symbol format: BTCUSDT
        sym = symbol.replace("/", "").replace("USDT", "USDT")

        url = "https://fapi.binance.com/fapi/v1/allForceOrders"
        params = {"symbol": sym, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error fetching Binance liquidations: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["symbol"] = symbol
        df["exchange"] = "binance"

        df = df.rename(
            columns={
                "time": "timestamp",
                "price": "price",
                "qty": "quantity",
                "side": "side",
            }
        )

        if "quantity" in df.columns and "price" in df.columns:
            df["value_usd"] = df["quantity"] * df["price"]

        logger.info(f"Fetched {len(df)} Binance liquidations")
        return df

    def _fetch_bybit_liquidations(
        self,
        symbol: str,
        limit: int,
    ) -> pd.DataFrame:
        """Fetch Bybit liquidations."""
        try:
            # Bybit v5 API
            liqs = self.exchange.fetch_liquidations(symbol, limit=limit)
            df = pd.DataFrame(liqs)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["symbol"] = symbol
                df["exchange"] = "bybit"
                if "quantity" in df.columns and "price" in df.columns:
                    df["value_usd"] = df["quantity"] * df["price"]
            logger.info(f"Fetched {len(df)} Bybit liquidations")
            return df
        except Exception as e:
            logger.error(f"Error fetching Bybit liquidations: {e}")
            return pd.DataFrame()

    def fetch_funding_rates(
        self,
        symbol: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch funding rate history.

        Args:
            symbol: Trading pair
            limit: Number of records

        Returns:
            DataFrame with funding rates
        """
        logger.info(f"Fetching funding rates for {symbol}")

        try:
            rates = self.exchange.fetch_funding_rate_history(
                symbol, limit=limit
            )
            df = pd.DataFrame(rates)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["symbol"] = symbol
                df["exchange"] = self.exchange_name
            logger.info(f"Fetched {len(df)} funding rate records")
            return df
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker["last"]

    def get_ticker(self, symbol: str) -> dict:
        """Get full ticker data."""
        return self.exchange.fetch_ticker(symbol)

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        mapping = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440,
        }
        return mapping.get(timeframe, 60)
