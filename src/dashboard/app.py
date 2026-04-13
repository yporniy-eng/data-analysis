"""
Liquidity Impulse Detector - Real-time Dashboard
Auto-refreshes chart every N seconds with live market data.
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

# Try to import auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="LID — Realtime", page_icon="📊", layout="wide")

# ---- Session state ----
if "running" not in st.session_state:
    st.session_state.running = False
if "data" not in st.session_state:
    st.session_state.data = None
if "last_update" not in st.session_state:
    st.session_state.last_update = None

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Настройки")
    symbol = st.selectbox("Инструмент", ["BTC/USDT", "ETH/USDT"], index=0)
    timeframe = st.selectbox("Таймфрейм", ["1m", "5m", "15m", "1h"], index=0)

    if HAS_AUTOREFRESH:
        refresh_sec = st.slider("Интервал обновления (сек)", 5, 120, 30, 5)
    else:
        refresh_sec = 30

    st.divider()

    if st.button("▶️ Запустить", type="primary", use_container_width=True):
        st.session_state.running = True
        st.rerun()

    if st.button("⏸️ Остановить", use_container_width=True):
        st.session_state.running = False
        st.rerun()

    st.divider()
    st.caption(
        f"Автообновление: {'✅ ' + str(refresh_sec) + 'с' if st.session_state.running else '⏸️'}"
    )

    # Live clock
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

# ---- Fetch data ----
with st.status("📡 Загрузка данных...", expanded=False) as status:
    collector = DataCollector("binance")
    ohlcv = collector.fetch_ohlcv(symbol, timeframe, limit=200)

    if not ohlcv.empty:
        liqs = collector.fetch_liquidations(symbol.replace("/", ""), limit=500)
        st.session_state.data = {"ohlcv": ohlcv, "liqs": liqs}
        st.session_state.last_update = datetime.now()
        status.update(label=f"✅ Данные обновлены ({datetime.now().strftime('%H:%M:%S')})", state="complete")
    else:
        status.update(label="❌ Ошибка загрузки данных", state="error")
        st.stop()

data = st.session_state.data
ohlcv = data["ohlcv"]
liqs = data["liqs"]
current_price = ohlcv["close"].iloc[-1]
prev_price = ohlcv["close"].iloc[-2] if len(ohlcv) > 1 else current_price

# ---- Top metrics ----
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    symbol,
    f"${current_price:,.2f}",
    delta=f"{current_price - prev_price:+.2f} ({(current_price/prev_price-1)*100:+.2f}%)"
)

# EWMA vol
ewma_calc = EWMACalculator(0.94)
vol_series = ewma_calc.compute_volatility_series(ohlcv["close"])
col2.metric("Волатильность (24h)", f"{vol_series.iloc[-1]:.1%}")

# Clusters count
detector = LiquidationClusterDetector()
clusters = detector.detect_clusters(liqs, current_price)
col3.metric("Кластеры ликвидности", str(len(clusters)))

# Last update
if st.session_state.last_update:
    col4.metric("Обновлено", st.session_state.last_update.strftime("%H:%M:%S"))

# ---- REAL-TIME CHART ----
st.subheader("📈 График цены (realtime)")

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

# Current price line
fig.add_hline(
    y=current_price,
    line_dash="dot",
    line_color="yellow",
    line_width=2,
    opacity=0.8,
    annotation_text=f"● ${current_price:,.0f}",
    annotation_position="right",
)

# Cluster lines
for c in clusters:
    fig.add_hline(
        y=c["price_center"],
        line_dash="dash",
        line_color="red" if c["position"] == "above" else "green",
        opacity=0.5,
        annotation_text=f"{'🔴' if c['position']=='above' else '🟢'} {c['distance_pct']:+.1%}",
    )

fig.update_layout(
    height=500,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    dragmode="pan",
    margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
    },
    key=f"chart_{st.session_state.last_update}",  # Force re-render on update
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
    cols = st.columns(min(3, len(signals)))
    for i, sig in enumerate(signals[:3]):
        with cols[i % 3]:
            icon = "🟢" if "CALL" in sig["signal_type"] else "🔴"
            st.metric(
                f"{icon} {sig['signal_type']}",
                f"K=${sig['strike']:,.0f}",
                f"Потенциал: {sig['profit_ratio']:.1f}x"
            )
            st.caption(f"Вероятность: {sig['probability_of_success']:.1%}")
else:
    st.caption("Нет сигналов, соответствующих критериям")

# ---- VOLATILITY CHART ----
st.subheader("📉 EWMA Волатильность")
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=vol_series.dropna().index,
    y=vol_series.dropna().values,
    mode="lines",
    line=dict(color="orange", width=2),
    fill="tozeroy",
    name="Volatility",
))
fig_vol.add_hline(
    y=vol_series.median(),
    line_dash="dash",
    line_color="gray",
    opacity=0.5,
    annotation_text=f"Median: {vol_series.median():.1%}",
)
fig_vol.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
st.plotly_chart(fig_vol, use_container_width=True, config={"scrollZoom": True}, key=f"vol_{st.session_state.last_update}")

# ---- DISTRIBUTION ----
st.subheader("📊 Распределение доходностей")
log_returns = np.log(ohlcv["close"]).diff().dropna()
c1, c2, c3 = st.columns(3)
c1.metric("Средняя", f"{log_returns.mean():.4%}")
c2.metric("Эксцесс", f"{pd.Series(log_returns).kurtosis():.2f}")
c3.metric("Скошенность", f"{pd.Series(log_returns).skew():.2f}")

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=log_returns, nbinsx=40, marker_color="steelblue"))
fig_hist.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
st.plotly_chart(fig_hist, use_container_width=True, config={"scrollZoom": True}, key=f"hist_{st.session_state.last_update}")
