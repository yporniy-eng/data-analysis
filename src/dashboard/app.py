"""
Liquidity Impulse Detector — Real-time Binance-like Dashboard

Features:
- Real-time candle updates via efficient polling (last 5 candles only)
- Zoom/pan state PRESERVED across updates (uirevision)
- Live price line, liquidation clusters, signals
- Auto-refresh every N seconds
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.collector import DataCollector
from src.math_core.ewma import EWMACalculator
from src.liquidations.detector import LiquidationClusterDetector
from src.signals.generator import SignalGenerator
from src.math_core.bsm import BlackScholesCalculator

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="LID Realtime", page_icon="📊", layout="wide")

# ---- Session state ----
if "running" not in st.session_state:
    st.session_state.running = False
if "ohlcv" not in st.session_state:
    st.session_state.ohlcv = None
if "update_count" not in st.session_state:
    st.session_state.update_count = 0
if "initial_loaded" not in st.session_state:
    st.session_state.initial_loaded = False

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Настройки")

    symbol = st.selectbox(
        "Инструмент",
        ["BTC/USDT", "ETH/USDT"],
        index=0,
        disabled=st.session_state.running,
    )

    timeframe = st.selectbox(
        "Таймфрейм",
        ["1m", "3m", "5m", "15m", "1h"],
        index=2,
        disabled=st.session_state.running,
    )

    if HAS_AUTOREFRESH:
        refresh_sec = st.slider("Интервал (сек)", 2, 60, 5, 1)
    else:
        refresh_sec = 10

    st.divider()

    if st.button("▶️ Запустить", type="primary", use_container_width=True):
        st.session_state.running = True
        st.session_state.initial_loaded = False
        st.session_state.ohlcv = None
        st.rerun()

    if st.button("⏸️ Остановить", use_container_width=True):
        st.session_state.running = False
        st.rerun()

    st.divider()

    if st.session_state.running:
        st.success("🟢 Работает")
        st.caption(f"Обновление каждые {refresh_sec}с")
        st.caption(f"Обновлений: {st.session_state.update_count}")
    else:
        st.info("⏸️ Остановлено")

    st.divider()
    now = datetime.now().strftime("%H:%M:%S")
    st.metric("🕐 Время", now)


# ---- Title ----
st.title("📊 Liquidity Impulse Detector — Realtime")

if not st.session_state.running:
    st.info("Нажмите **▶️ Запустить** в sidebar для старта")
    st.stop()

# ---- Auto-refresh ----
if HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_sec * 1000, key="lid_autorefresh")

# ---- Fetch/update data ----
collector = DataCollector("binance")

if not st.session_state.initial_loaded:
    # Initial load: get 200 candles
    with st.spinner(f"Загрузка {symbol} {timeframe}..."):
        df = collector.fetch_ohlcv(symbol, timeframe, limit=200)
    if df.empty:
        st.error("Не удалось загрузить данные")
        st.stop()
    st.session_state.ohlcv = df
    st.session_state.initial_loaded = True
else:
    # Incremental update: only last 5 candles
    new_candles = collector.fetch_ohlcv(symbol, timeframe, limit=5)
    if not new_candles.empty and st.session_state.ohlcv is not None:
        # Append new candles, remove duplicates
        combined = pd.concat([st.session_state.ohlcv, new_candles], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        st.session_state.ohlcv = combined.tail(300)  # Keep last 300 candles

ohlcv = st.session_state.ohlcv
if ohlcv is None or ohlcv.empty:
    st.stop()

current_price = ohlcv["close"].iloc[-1]
prev_price = ohlcv["close"].iloc[-2] if len(ohlcv) > 1 else current_price
st.session_state.update_count += 1

# ---- Top metrics ----
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    symbol,
    f"${current_price:,.2f}",
    delta=f"{current_price - prev_price:+.2f} ({(current_price/prev_price-1)*100:+.2f}%)"
)

ewma_calc = EWMACalculator(0.94)
vol_series = ewma_calc.compute_volatility_series(ohlcv["close"])
col2.metric("Волатильность", f"{vol_series.iloc[-1]:.1%}")

liqs = collector.fetch_liquidations(symbol.replace("/", ""), limit=200)
detector = LiquidationClusterDetector()
clusters = detector.detect_clusters(liqs, current_price)
col3.metric("Кластеры", str(len(clusters)))
col4.metric("Свечей", str(len(ohlcv)))

# ---- REAL-TIME CHART (preserves zoom/pan) ----
st.subheader("📈 Цена")

fig = go.Figure()

# Candlestick
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

# Current price line
fig.add_hline(
    y=current_price,
    line_dash="dot",
    line_color="yellow",
    line_width=1,
    opacity=0.7,
)

# Cluster lines
for c in clusters[:5]:
    fig.add_hline(
        y=c["price_center"],
        line_dash="dash",
        line_color="red" if c["position"] == "above" else "green",
        opacity=0.4,
    )

# ═══════════════════════════════════════════════════════
# uirevision = KEEPS zoom/pan when data changes!
# dragmode = "pan" so user can drag to move around
# ═══════════════════════════════════════════════════════
fig.update_layout(
    height=450,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    dragmode="pan",
    margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False,
    uirevision="lid_chart_stable",  # ← PRESERVES zoom/pan across rerenders
    xaxis=dict(
        fixedrange=False,  # Allow zoom/pan
    ),
    yaxis=dict(
        fixedrange=False,  # Allow zoom/pan
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
        "scrollZoom": True,
    },
    key=f"main_chart",  # Same key every time → preserves state
)

# ---- Signals ----
st.subheader("💡 Сигналы")
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
    cols = st.columns(min(3, len(signals)))
    for i, sig in enumerate(signals[:3]):
        with cols[i % 3]:
            icon = "🟢" if "CALL" in sig["signal_type"] else "🔴"
            st.metric(f"{icon} {sig['signal_type']}", f"K=${sig['strike']:,.0f}", f"×{sig['profit_ratio']:.1f}")
else:
    st.caption("Нет сигналов")

# ---- Volatility ----
st.subheader("📉 Волатильность")
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=vol_series.dropna().index,
    y=vol_series.dropna().values,
    mode="lines",
    line=dict(color="orange", width=2),
    fill="tozeroy",
))
fig_vol.update_layout(
    height=200, margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False, uirevision="vol_stable",
)
st.plotly_chart(fig_vol, use_container_width=True, config={"scrollZoom": True}, key="vol_chart")

# ---- Distribution ----
log_returns = np.log(ohlcv["close"]).diff().dropna()
c1, c2, c3 = st.columns(3)
c1.metric("Средняя", f"{log_returns.mean():.4%}")
c2.metric("Эксцесс", f"{pd.Series(log_returns).kurtosis():.2f}")
c3.metric("Скошенность", f"{pd.Series(log_returns).skew():.2f}")

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=log_returns, nbinsx=40, marker_color="steelblue"))
fig_hist.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, uirevision="hist_stable")
st.plotly_chart(fig_hist, use_container_width=True, config={"scrollZoom": True}, key="hist_chart")
