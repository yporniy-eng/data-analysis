"""
Streamlit Dashboard - Liquidity Impulse Detector

Displays:
- Price chart with liquidation clusters
- Current signals table
- P&L history
- Key metrics
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.collector import DataCollector
from src.math_core.bsm import BlackScholesCalculator
from src.math_core.ewma import EWMACalculator
from src.liquidations.detector import LiquidationClusterDetector
from src.signals.generator import SignalGenerator
from src.backtest.engine import BacktestEngine
from src.logger import logger

st.set_page_config(
    page_title="Liquidity Impulse Detector",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Liquidity Impulse Detector")
st.caption("Опционный аналитик на основе кластеров ликвидности")


# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Параметры")

    symbol = st.selectbox(
        "Инструмент",
        ["BTC/USDT", "ETH/USDT"],
        index=0,
    )

    timeframe = st.selectbox(
        "Таймфрейм",
        ["1h", "4h", "1d"],
        index=0,
    )

    days_back = st.slider("Дней истории", 30, 365, 180, 30)

    run_analysis = st.button("🚀 Запустить анализ", type="primary")

    st.divider()
    st.info(
        "💡 **Совет:** Для работы нужен Docker с ClickHouse "
        "(опционально). Без БД данные загружаются напрямую с биржи."
    )


# ---- Main logic ----
@st.cache_data(ttl=300)
def load_data(sym: str, tf: str, days: int) -> tuple:
    """Load OHLCV and liquidation data."""
    collector = DataCollector(exchange_name="binance")

    # OHLCV
    ohlcv = collector.fetch_ohlcv_history(sym, tf, days)

    # Liquidations
    sym_raw = sym.replace("/", "")
    liqs = collector.fetch_liquidations(sym_raw, limit=1000)

    return ohlcv, liqs


def build_price_chart(ohlcv: pd.DataFrame, clusters: list, current_price: float):
    """Build candlestick chart with liquidation clusters."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=ohlcv["timestamp"],
            open=ohlcv["open"],
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            name="Price",
        ),
        row=1, col=1,
    )

    # Liquidation cluster lines
    colors = {"above": "red", "below": "green"}
    for cluster in clusters:
        fig.add_hline(
            y=cluster["price_center"],
            line_dash="dash",
            line_color=colors.get(cluster["position"], "gray"),
            opacity=0.7,
            row=1, col=1,
            annotation_text=(
                f"{'🔴' if cluster['position'] == 'above' else '🟢'} "
                f"{cluster['distance_pct']:+.1%} | "
                f"z={cluster['max_z_score']:.1f}"
            ),
            annotation_position="top left" if cluster["position"] == "above" else "bottom left",
        )

    # Volume bars
    colors_vol = [
        "green" if c >= o else "red"
        for o, c in zip(ohlcv["open"], ohlcv["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=ohlcv["timestamp"],
            y=ohlcv["volume"],
            marker_color=colors_vol,
            name="Volume",
            opacity=0.6,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        xaxis_title="",
    )

    return fig


def build_signal_table(signals: list) -> pd.DataFrame:
    """Format signals for display."""
    if not signals:
        return pd.DataFrame()

    df = pd.DataFrame(signals)
    display_cols = [
        "signal_type", "strike", "option_type",
        "market_price", "fair_value",
        "profit_ratio", "composite_score",
        "probability_of_success", "position_size",
        "risk_usd", "reward_usd",
    ]
    available = [c for c in display_cols if c in df.columns]
    return df[available]


# ---- Run analysis ----
if run_analysis or True:  # Auto-run on load
    with st.spinner("Загрузка данных..."):
        ohlcv, liqs = load_data(symbol, timeframe, days_back)

    if ohlcv.empty:
        st.error("Не удалось загрузить данные OHLCV")
        st.stop()

    current_price = ohlcv["close"].iloc[-1]

    # Display current price
    st.metric(
        f"{symbol} — Текущая цена",
        f"${current_price:,.2f}",
        delta=f"{ohlcv['close'].iloc[-1] - ohlcv['close'].iloc[-2]:+.2f}"
    )

    st.divider()

    # ---- Liquidation clusters ----
    st.subheader("🎯 Кластеры ликвидности")

    liq_detector = LiquidationClusterDetector()
    clusters = liq_detector.detect_clusters(liqs, current_price)

    if clusters:
        col1, col2, col3 = st.columns(3)
        for i, cluster in enumerate(clusters[:6]):
            with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
                icon = "🔴" if cluster["position"] == "above" else "🟢"
                st.metric(
                    f"{icon} Кластер {i + 1}",
                    f"${cluster['price_center']:,.0f}",
                    delta=f"{cluster['distance_pct']:+.2%}"
                )
                st.caption(
                    f"z-score: {cluster['max_z_score']:.1f} | "
                    f"Объём: ${cluster['total_value']:,.0f}"
                )
    else:
        st.info("Кластеры ликвидности не обнаружены в диапазоне ±5%")

    # ---- Price chart ----
    st.subheader("📈 График цены")
    fig = build_price_chart(ohlcv, clusters, current_price)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Signal generation ----
    st.subheader("💡 Торговые сигналы")

    bsm = BlackScholesCalculator(risk_free_rate=0.053)
    ewma = EWMACalculator(lambda_decay=0.94)

    generator = SignalGenerator(
        bsm_calculator=bsm,
        ewma_calculator=ewma,
        liq_detector=liq_detector,
        min_profit_ratio=1.5,
        min_composite_score=0.5,
    )

    hist_prices = ohlcv["close"]
    signals = generator.generate_signals(
        current_price=current_price,
        liquidations_df=liqs,
        historical_prices=hist_prices,
        account_value=100000,
    )

    if signals:
        sig_df = build_signal_table(signals)
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

        # Show best signal details
        best = signals[0]
        st.success(
            f"✅ Лучший сигнал: **{best['signal_type']}** "
            f"страйк ${best['strike']:,.0f} | "
            f"Потенциал: {best['profit_ratio']:.1f}x | "
            f"Вероятность: {best['probability_of_success']:.1%}"
        )
    else:
        st.info("Нет сигналов, соответствующих критериям")

    # ---- Backtest ----
    st.divider()
    st.subheader("📊 Бэктест")

    if st.button("Запустить бэктест"):
        with st.spinner("Бэктест запущен..."):
            engine = BacktestEngine(
                signal_generator=generator,
                initial_capital=100000,
                commission_rate=0.0004,
                stop_loss_pct=0.50,
                max_hold_hours=48,
            )

            result = engine.run(ohlcv, liqs, symbol=symbol, step_hours=4)

            metrics = result["metrics"]

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
            col2.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
            col3.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
            col4.metric("Max DD", f"{metrics.get('max_drawdown', 0):.2%}")
            col5.metric("Trades", f"{metrics.get('total_trades', 0)}")

            # Equity curve
            if not result["equity_curve"].empty:
                eq_df = result["equity_curve"]
                fig_eq = go.Figure()
                fig_eq.add_trace(
                    go.Scatter(
                        x=eq_df["timestamp"],
                        y=eq_df["equity"],
                        mode="lines",
                        fill="tozeroy",
                        name="Equity",
                    )
                )
                fig_eq.update_layout(
                    title="Кривая доходности",
                    xaxis_title="Дата",
                    yaxis_title="Equity ($)",
                    height=400,
                )
                st.plotly_chart(fig_eq, use_container_width=True)

            # Trades table
            if not result["trades"].empty:
                st.dataframe(result["trades"], use_container_width=True, hide_index=True)

    # ---- Volatility analysis ----
    st.divider()
    st.subheader("📉 Анализ волатильности")

    log_returns = np.log(ohlcv["close"]).diff().dropna()
    ewma_calc = EWMACalculator(0.94)
    vol_series = ewma_calc.compute_volatility_series(ohlcv["close"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Текущая EWMA волатильность",
            f"{vol_series.iloc[-1]:.1%}",
        )
    with col2:
        st.metric(
            "Период полураспада",
            f"{ewma_calc.half_life():.0f} периодов",
        )

    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Scatter(
            x=vol_series.dropna().index,
            y=vol_series.dropna().values,
            mode="lines",
            name="EWMA Volatility",
            line=dict(color="orange"),
        )
    )
    fig_vol.update_layout(
        title="EWMA Волатильность (annualized)",
        xaxis_title="Дата",
        yaxis_title="Volatility",
        height=350,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # ---- Distribution analysis ----
    st.subheader("📊 Распределение доходностей")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Средняя", f"{log_returns.mean():.4%}")
    with col2:
        kurt = pd.Series(log_returns).kurtosis()
        st.metric("Эксцесс (избыт.)", f"{kurt:.2f}")
    with col3:
        skew = pd.Series(log_returns).skew()
        st.metric("Скошенность", f"{skew:.2f}")

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=log_returns,
            nbinsx=50,
            name="Returns",
            marker_color="steelblue",
        )
    )
    fig_hist.update_layout(
        title="Распределение лог-доходностей",
        xaxis_title="Return",
        yaxis_title="Count",
        height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
