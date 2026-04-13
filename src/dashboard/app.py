"""
Liquidity Impulse Detector - Simple Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.collector import DataCollector
from src.math_core.ewma import EWMACalculator
from src.liquidations.detector import LiquidationClusterDetector
from src.signals.generator import SignalGenerator
from src.math_core.bsm import BlackScholesCalculator

st.set_page_config(page_title="Liquidity Impulse Detector", page_icon="📊", layout="wide")

st.title("📊 Liquidity Impulse Detector")

# ---- Sidebar ----
with st.sidebar:
    st.header("Параметры")
    symbol = st.selectbox("Инструмент", ["BTC/USDT", "ETH/USDT"], index=0)
    timeframe = st.selectbox("Таймфрейм", ["1h", "4h", "1d"], index=0)
    st.divider()
    st.caption("Данные с Binance (бесплатно, без ключей)")

# ---- Main ----
if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
    with st.spinner("Загрузка данных..."):
        collector = DataCollector("binance")
        ohlcv = collector.fetch_ohlcv(symbol, timeframe, limit=1000)
        liqs = collector.fetch_liquidations(symbol.replace("/", ""), limit=500)

    if ohlcv.empty:
        st.error("Не удалось загрузить данные")
        st.stop()

    current_price = ohlcv["close"].iloc[-1]
    st.metric(f"{symbol}", f"${current_price:,.2f}")

    # ---- CLUSTERS ----
    st.subheader("🎯 Кластеры ликвидности")
    detector = LiquidationClusterDetector()
    clusters = detector.detect_clusters(liqs, current_price)

    if clusters:
        cols = st.columns(min(3, len(clusters)))
        for i, c in enumerate(clusters[:6]):
            icon = "🔴" if c["position"] == "above" else "🟢"
            cols[i % 3].metric(f"{icon}", f"${c['price_center']:,.0f}", f"{c['distance_pct']:+.2%}")
    else:
        st.info("Кластеры не найдены в диапазоне ±5%")

    # ---- CHART ----
    st.subheader("📈 График цены")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=ohlcv["timestamp"],
        open=ohlcv["open"],
        high=ohlcv["high"],
        low=ohlcv["low"],
        close=ohlcv["close"],
        name="Price",
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
    ))

    # Cluster lines
    for c in clusters:
        fig.add_hline(
            y=c["price_center"],
            line_dash="dash",
            line_color="red" if c["position"] == "above" else "green",
            opacity=0.6,
            annotation_text=f"Cluster {c['distance_pct']:+.1%}",
        )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="pan",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
        },
    )

    # ---- SIGNALS ----
    st.subheader("💡 Торговые сигналы")
    bsm = BlackScholesCalculator(risk_free_rate=0.053)
    ewma = EWMACalculator(0.94)
    generator = SignalGenerator(bsm, ewma, detector, min_profit_ratio=1.0, min_composite_score=0.3)
    signals = generator.generate_signals(
        current_price=current_price,
        liquidations_df=liqs,
        historical_prices=ohlcv["close"],
        account_value=100000,
    )

    if signals:
        for sig in signals[:5]:
            st.success(
                f"**{sig['signal_type']}** | K=${sig['strike']:,.0f} | "
                f"Потенциал: {sig['profit_ratio']:.1f}x | "
                f"Вероятность: {sig['probability_of_success']:.1%}"
            )
    else:
        st.info("Нет сигналов")

    # ---- VOLATILITY ----
    st.subheader("📉 Волатильность")
    ewma_calc = EWMACalculator(0.94)
    vol_series = ewma_calc.compute_volatility_series(ohlcv["close"])
    st.metric("EWMA волатильность", f"{vol_series.iloc[-1]:.1%}")

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=vol_series.dropna().index,
        y=vol_series.dropna().values,
        mode="lines",
        line=dict(color="orange"),
    ))
    fig_vol.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_vol, use_container_width=True, config={"scrollZoom": True})

    # ---- DISTRIBUTION ----
    st.subheader("📊 Распределение доходностей")
    log_returns = np.log(ohlcv["close"]).diff().dropna()
    col1, col2, col3 = st.columns(3)
    col1.metric("Средняя", f"{log_returns.mean():.4%}")
    col2.metric("Эксцесс", f"{pd.Series(log_returns).kurtosis():.2f}")
    col3.metric("Скошенность", f"{pd.Series(log_returns).skew():.2f}")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=log_returns, nbinsx=50, marker_color="steelblue"))
    fig_hist.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_hist, use_container_width=True, config={"scrollZoom": True})

else:
    st.markdown("### Нажмите **🚀 Запустить анализ** для загрузки данных")
    st.info("Загрузка ~5 секунд: 1000 свечей + ликвидации с Binance")
