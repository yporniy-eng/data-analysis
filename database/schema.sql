-- Liquidity Impulse Detector - ClickHouse Schema

CREATE DATABASE IF NOT EXISTS liquidity_detector;

USE liquidity_detector;

-- OHLCV Data
CREATE TABLE IF NOT EXISTS ohlcv (
    timestamp DateTime64(3),
    symbol String,
    exchange String,
    timeframe String,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    trades_count UInt64 DEFAULT 0
) ENGINE = MergeTree()
ORDER BY (symbol, exchange, timeframe, timestamp);

-- Liquidations Data
CREATE TABLE IF NOT EXISTS liquidations (
    timestamp DateTime64(3),
    symbol String,
    exchange String,
    side String,
    price Float64,
    quantity Float64,
    value_usd Float64
) ENGINE = MergeTree()
ORDER BY (symbol, exchange, timestamp);

-- Funding Rates
CREATE TABLE IF NOT EXISTS funding_rates (
    timestamp DateTime64(3),
    symbol String,
    exchange String,
    funding_rate Float64,
    next_funding_time DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (symbol, exchange, timestamp);

-- Open Interest
CREATE TABLE IF NOT EXISTS open_interest (
    timestamp DateTime64(3),
    symbol String,
    exchange String,
    open_interest Float64,
    open_interest_value Float64
) ENGINE = MergeTree()
ORDER BY (symbol, exchange, timestamp);

-- Signals History
CREATE TABLE IF NOT EXISTS signals_history (
    timestamp DateTime64(3),
    symbol String,
    signal_type String,
    strike Float64,
    expiry DateTime64(3),
    option_type String,
    market_price Float64,
    fair_value Float64,
    expected_move Float64,
    profit_ratio Float64,
    composite_score Float64,
    probability_of_success Float64,
    position_size Int32,
    risk_usd Float64,
    reward_usd Float64,
    exit_price Nullable(Float64),
    pnl Nullable(Float64),
    exit_reason Nullable(String)
) ENGINE = MergeTree()
ORDER BY (timestamp, symbol);

-- Model Parameters (for adaptive retraining)
CREATE TABLE IF NOT EXISTS model_params (
    timestamp DateTime64(3),
    symbol String,
    param_name String,
    param_value Float64
) ENGINE = MergeTree()
ORDER BY (timestamp, symbol, param_name);
