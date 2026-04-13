"""
Streamlit Dashboard - Liquidity Impulse Detector

Displays:
- Price chart with liquidation clusters
- Current signals table
- P&L history
- Key metrics

Data is loaded ONLY when user clicks the button (no auto-load).
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

    days_back = st.slider("Дней истории", 30, 365, 90, 30)

    st.divider()

    # Show current BTC price (fast, single API call)
    if st.button("📡 Проверить цену BTC", use_container_width=True):
        with st.spinner("Запрос цены..."):
            try:
                c = DataCollector("binance")
                price = c.get_current_price("BTC/USDT")
                st.success(f"BTC = **${price:,.2f}**")
            except Exception as e:
                st.error(f"Ошибка: {e}")

    st.divider()

    # Chart size controls
    st.subheader("📐 Размер графика")
    chart_height = st.slider(
        "Высота (px)",
        min_value=400,
        max_value=1200,
        value=700,
        step=50,
        key="chart_height_slider",
    )
    st.session_state.chart_height = chart_height

    st.divider()

    st.info(
        "💡 Нажмите **«Запустить анализ»** "
        "для загрузки данных и генерации сигналов.\n\n"
        "Данные загружаются напрямую с Binance (API ключ не нужен)."
    )


# ---- Data loading (cached) ----
@st.cache_data(ttl=600, show_spinner=False)
def load_ohlcv_data(sym: str, tf: str, days: int) -> pd.DataFrame:
    """Load OHLCV candlestick data."""
    collector = DataCollector(exchange_name="binance")
    df = collector.fetch_ohlcv_history(sym, tf, days)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_liquidations(sym: str, limit: int = 1000) -> pd.DataFrame:
    """Load liquidation data."""
    collector = DataCollector(exchange_name="binance")
    sym_raw = sym.replace("/", "")
    df = collector.fetch_liquidations(sym_raw, limit=limit)
    return df


# ---- Welcome screen (shown until analysis is started) ----
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


def _run_analysis(symbol, timeframe, days_back):
    """Store params and rerun to trigger analysis."""
    st.session_state.selected_symbol = symbol
    st.session_state.selected_timeframe = timeframe
    st.session_state.selected_days = days_back
    st.session_state.data_loaded = True
    st.rerun()


if not st.session_state.data_loaded:
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🚀 Быстрый старт")
        st.markdown("""
1. Выберите инструмент в sidebar
2. Выберите таймфрейм и глубину истории
3. Нажмите **«Запустить анализ»**

Данные загружаются с Binance (публичные API, без ключей).
        """)

        if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
            _run_analysis(symbol, timeframe, days_back)

    with col2:
        st.subheader("📋 Что показывает приложение")
        st.markdown("""
- **Кластеры ликвидности** — зоны концентрации ликвидаций (ценовые магниты)
- **График цены** — свечи + объём + линии кластеров
- **Торговые сигналы** — CALL/PUT опционы с оценкой потенциала
- **Анализ волатильности** — EWMA, эксцесс, скошенность
- **Распределение доходностей** — проверка на нормальность
- **Бэктест** — валидация стратегии на исторических данных
        """)

    st.divider()
    st.caption(
        "Научная основа: Bachelier · Black-Scholes-Merton · "
        "Simons (Renaissance) · Andrew Lo · Market Microstructure"
    )
    st.stop()


# ---- Analysis results (shown after data is loaded) ----
def run_full_analysis():
    """Main analysis flow."""
    sym = st.session_state.get("selected_symbol", "BTC/USDT")
    tf = st.session_state.get("selected_timeframe", "1h")
    days = st.session_state.get("selected_days", 90)

    with st.spinner(f"Загрузка данных {sym} ({tf}, {days} дней)..."):
        ohlcv = load_ohlcv_data(sym, tf, days)
        liqs = load_liquidations(sym, limit=1000)

    if ohlcv.empty:
        st.error("Не удалось загрузить данные OHLCV")
        st.session_state.data_loaded = False
        return

    current_price = ohlcv["close"].iloc[-1]
    prev_price = ohlcv["close"].iloc[-2] if len(ohlcv) > 1 else current_price

    # Store in session state
    st.session_state.ohlcv = ohlcv
    st.session_state.liquidations = liqs
    st.session_state.current_price = current_price
    st.session_state.prev_price = prev_price

    # Show header
    st.metric(
        f"{sym} — Текущая цена",
        f"${current_price:,.2f}",
        delta=f"{current_price - prev_price:+.2f}"
    )

    st.divider()

    # ---- Liquidation clusters ----
    st.subheader("🎯 Кластеры ликвидности")

    liq_detector = LiquidationClusterDetector()
    clusters = liq_detector.detect_clusters(liqs, current_price)

    st.session_state.clusters = clusters

    if clusters:
        cols = st.columns(3)
        for i, cluster in enumerate(clusters[:6]):
            with cols[i % 3]:
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
    chart_h = st.session_state.get("chart_height", 700)
    fig = build_price_chart(ohlcv, clusters, current_price, chart_height=chart_h)
    st.plotly_chart(fig, use_container_width=True, key="price_chart")

    # ---- Signal generation ----
    st.subheader("💡 Торговые сигналы")

    bsm = BlackScholesCalculator(risk_free_rate=0.053)
    ewma = EWMACalculator(lambda_decay=0.94)

    generator = SignalGenerator(
        bsm_calculator=bsm,
        ewma_calculator=ewma,
        liq_detector=liq_detector,
        min_profit_ratio=1.0,
        min_composite_score=0.3,
    )

    hist_prices = ohlcv["close"]
    signals = generator.generate_signals(
        current_price=current_price,
        liquidations_df=liqs,
        historical_prices=hist_prices,
        account_value=100000,
    )

    st.session_state.signals = signals
    st.session_state.generator = generator

    if signals:
        sig_df = build_signal_table(signals)
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

        best = signals[0]
        st.success(
            f"✅ Лучший сигнал: **{best['signal_type']}** "
            f"страйк ${best['strike']:,.0f} | "
            f"Потенциал: {best['profit_ratio']:.1f}x | "
            f"Вероятность: {best['probability_of_success']:.1%}"
        )
    else:
        st.info("Нет сигналов, соответствующих критериям")

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

    # ---- Backtest ----
    st.divider()
    st.subheader("📊 Бэктест")
    st.caption("Симуляция торговли на исторических данных (комиссия 0.04%, стоп-лосс 50%)")

    if st.button("▶️ Запустить бэктест"):
        run_backtest(ohlcv, liqs, sym, generator)

    # ---- Re-run button ----
    st.divider()
    if st.button("🔄 Обновить данные", use_container_width=True):
        st.session_state.data_loaded = False
        st.rerun()


def run_backtest(ohlcv, liqs, symbol, generator):
    """Run backtest and display results."""
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

    if not result["trades"].empty:
        st.dataframe(result["trades"], use_container_width=True, hide_index=True)


def build_price_chart(
    ohlcv: pd.DataFrame,
    clusters: list,
    current_price: float,
    chart_height: int = 700,
):
    """Build interactive candlestick chart with liquidation clusters.

    Interactions:
    - Scroll / trackpad pinch → zoom in/out
    - Click + drag → pan (move around)
    - Box select → zoom to selection
    - Modebar buttons → reset, zoom, pan, screenshot
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Price", "Volume"),
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
            increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
        ),
        row=1, col=1,
    )

    # Liquidation cluster lines
    colors_map = {"above": "rgba(255,0,0,0.7)", "below": "rgba(0,200,0,0.7)"}
    for cluster in clusters:
        fig.add_hline(
            y=cluster["price_center"],
            line_dash="dash",
            line_color=colors_map.get(cluster["position"], "gray"),
            line_width=2,
            opacity=0.8,
            row=1, col=1,
            annotation_text=(
                f"{'🔴' if cluster['position'] == 'above' else '🟢'} "
                f"{cluster['distance_pct']:+.1%} | "
                f"z={cluster['max_z_score']:.1f} | "
                f"${cluster['total_value']:,.0f}"
            ),
            annotation_font=dict(size=11),
            annotation_position=(
                "top left" if cluster["position"] == "above" else "bottom left"
            ),
        )

    # Volume bars
    colors_vol = [
        "rgba(38,166,154,0.6)" if c >= o else "rgba(239,83,80,0.6)"
        for o, c in zip(ohlcv["open"], ohlcv["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=ohlcv["timestamp"],
            y=ohlcv["volume"],
            marker_color=colors_vol,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Interactive modebar configuration
    fig.update_layout(
        height=chart_height,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        xaxis_title="",
        dragmode="zoom",  # Default: scroll to zoom, drag to pan
        hovermode="x unified",
        xaxis=dict(
            type="date",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
            domain=[0.25, 1.0],
        ),
        yaxis2=dict(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            domain=[0.0, 0.22],
        ),
        modebar=dict(
            add=[
                "drawline",       # Draw trend lines
                "eraseshape",     # Erase drawings
            ],
            remove=[
                "lasso2d",
                "select2d",
                "autoScale2d",
            ],
        ),
        config=dict(
            scrollZoom=True,          # Trackpad scroll to zoom
            displayModeBar=True,       # Show modebar
            modeBarButtonsToAdd=[
                "drawline",
                "drawopenpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ],
        ),
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


# ---- Main entry point ----
if st.session_state.data_loaded:
    run_full_analysis()
