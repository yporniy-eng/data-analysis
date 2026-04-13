"""
Data Storage - ClickHouse integration

Handles:
- Writing data to ClickHouse
- Reading data for analysis
- Schema management
"""

import clickhouse_connect
import pandas as pd
from datetime import datetime
from typing import Optional, List

from src.config import config
from src.logger import logger


class ClickHouseStorage:
    """ClickHouse database interface."""

    def __init__(self):
        """Initialize ClickHouse connection."""
        db_config = config["database"]["clickhouse"]
        self.client = clickhouse_connect.get_client(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
        )
        logger.info(
            f"Connected to ClickHouse at "
            f"{db_config['host']}:{db_config['port']}"
        )

    def insert_ohlcv(self, df: pd.DataFrame) -> int:
        """
        Insert OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0

        columns = [
            "timestamp", "symbol", "exchange", "timeframe",
            "open", "high", "low", "close", "volume", "trades_count"
        ]

        # Ensure all columns exist
        for col in columns:
            if col not in df.columns:
                df[col] = 0 if col == "trades_count" else 0.0

        data = df[columns].to_dict("records")
        self.client.insert("ohlcv", data, column_names=columns)

        logger.info(f"Inserted {len(data)} OHLCV rows")
        return len(data)

    def insert_liquidations(self, df: pd.DataFrame) -> int:
        """Insert liquidation data."""
        if df.empty:
            return 0

        columns = ["timestamp", "symbol", "exchange", "side", "price",
                    "quantity", "value_usd"]

        # Ensure columns
        for col in columns:
            if col not in df.columns:
                df[col] = "" if col == "side" else 0.0

        data = df[columns].to_dict("records")
        self.client.insert("liquidations", data, column_names=columns)

        logger.info(f"Inserted {len(data)} liquidation rows")
        return len(data)

    def insert_funding_rates(self, df: pd.DataFrame) -> int:
        """Insert funding rate data."""
        if df.empty:
            return 0

        columns = [
            "timestamp", "symbol", "exchange",
            "funding_rate", "next_funding_time"
        ]

        for col in columns:
            if col not in df.columns:
                if col == "next_funding_time":
                    df[col] = datetime.utcnow()
                else:
                    df[col] = 0.0

        data = df[columns].to_dict("records")
        self.client.insert("funding_rates", data, column_names=columns)

        logger.info(f"Inserted {len(data)} funding rate rows")
        return len(data)

    def insert_signals(self, signals: list[dict]) -> int:
        """Insert signal data."""
        if not signals:
            return 0

        columns = [
            "timestamp", "symbol", "signal_type", "strike", "expiry",
            "option_type", "market_price", "fair_value", "expected_move",
            "profit_ratio", "composite_score", "probability_of_success",
            "position_size", "risk_usd", "reward_usd",
            "exit_price", "pnl", "exit_reason"
        ]

        # Ensure all signals have all columns
        for signal in signals:
            for col in columns:
                if col not in signal:
                    signal[col] = None

        self.client.insert("signals_history", signals, column_names=columns)
        logger.info(f"Inserted {len(signals)} signal rows")
        return len(signals)

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from database.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_dt: Start time filter
            end_dt: End time filter
            limit: Max rows

        Returns:
            DataFrame with OHLCV data
        """
        query = f"""
            SELECT timestamp, symbol, exchange, timeframe,
                   open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = '{symbol}'
              AND timeframe = '{timeframe}'
        """

        if start_dt:
            query += f" AND timestamp >= '{start_dt}'"
        if end_dt:
            query += f" AND timestamp <= '{end_dt}'"

        query += f" ORDER BY timestamp ASC LIMIT {limit}"

        result = self.client.query(query)
        df = pd.DataFrame(result.result_rows, columns=result.column_names)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        logger.info(f"Fetched {len(df)} OHLCV rows for {symbol}")
        return df

    def get_liquidations(
        self,
        symbol: str,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """Fetch liquidation data from database."""
        query = f"""
            SELECT timestamp, symbol, exchange, side, price, quantity, value_usd
            FROM liquidations
            WHERE symbol = '{symbol}'
        """

        if start_dt:
            query += f" AND timestamp >= '{start_dt}'"
        if end_dt:
            query += f" AND timestamp <= '{end_dt}'"

        query += f" ORDER BY timestamp ASC LIMIT {limit}"

        result = self.client.query(query)
        df = pd.DataFrame(result.result_rows, columns=result.column_names)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        logger.info(f"Fetched {len(df)} liquidation rows for {symbol}")
        return df

    def get_funding_rates(
        self,
        symbol: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch funding rates from database."""
        query = f"""
            SELECT timestamp, symbol, exchange, funding_rate
            FROM funding_rates
            WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC
            LIMIT {limit}
        """

        result = self.client.query(query)
        df = pd.DataFrame(result.result_rows, columns=result.column_names)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def execute(self, query: str) -> pd.DataFrame:
        """Execute custom query."""
        result = self.client.query(query)
        return pd.DataFrame(result.result_rows, columns=result.column_names)

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        query = f"""
            SELECT count() FROM system.tables
            WHERE database = 'liquidity_detector' AND name = '{table_name}'
        """
        result = self.client.query(query)
        return result.result_rows[0][0] > 0

    def close(self):
        """Close connection."""
        self.client.close()
