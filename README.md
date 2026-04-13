# Liquidity Impulse Detector

Опционный аналитик криптовалют, вычисляющий закономерности импульсного движения цены
и определяющий точки входа для покупки опционов на основе кластеров ликвидности.

## Научная основа

- **Louis Bachelier** — случайные процессы, броуновское движение
- **Black-Scholes-Merton** — ценообразование опционов, греки
- **Jim Simons** — кластеризация состояний рынка, PCA
- **Andrew Lo** — адаптивная гипотеза рынков, переобучение
- **Market Makers** — кластеры ликвидаций, ценовые магниты

## Быстрый старт

### 1. Установка

```bash
# Python 3.10+
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Запуск баз данных (опционально)

```bash
docker-compose up -d
```

Без Docker приложение работает напрямую с биржей (данные не сохраняются).

### 3. Запуск Dashboard

```bash
python scripts/run_dashboard.py
```

Откройте http://localhost:8501

### 4. Сбор данных

```bash
# Загрузить 180 дней BTC/USDT
python scripts/collect_data.py --symbol BTC/USDT --days 180

# Загрузить всё (BTC + ETH)
python scripts/collect_data.py --all --days 365
```

### 5. Бэктест

```bash
python scripts/run_backtest.py --symbol BTC/USDT --days 180
```

## Архитектура

```
liquidity_detector/
├── src/
│   ├── data/           # Сбор данных (Binance, Bybit)
│   ├── math_core/      # BSM, EWMA, коррекции
│   ├── liquidations/   # Детектор кластеров ликвидности
│   ├── signals/        # Генератор торговых сигналов
│   ├── backtest/       # Бэктест-движок
│   └── dashboard/      # Streamlit интерфейс
├── scripts/            # Скрипты запуска
├── database/           # ClickHouse схема
└── config.yaml         # Конфигурация
```

## Как работает

1. **Сбор данных** — OHLCV, ликвидации, funding rates с Binance/Bybit
2. **Детекция кластеров** — поиск зон концентрации ликвидаций (z-score анализ)
3. **Оценка опционов** — BSM + поправка на эксцесс/скошенность + EWMA волатильность
4. **Генерация сигналов** — комбинация кластеров, mispricing, Kelly sizing
5. **Бэктест** — валидация на исторических данных с комиссиями и проскальзыванием

## Конфигурация

Все параметры в `config.yaml`. Ничего магического — пороги вычисляются адаптивно:

| Параметр | Описание | Значение |
|---|---|---|
| `ewma.lambda_decay` | Скорость адаптации волатильности | 0.94 |
| `liquidations.z_score_percentile` | Порог обнаружения кластеров | 90-й перцентиль |
| `signals.min_profit_ratio` | Мин. соотношение прибыль/цена | 1.5 |
| `signals.min_composite_score` | Мин. скор сигнала | 0.5 |
| `position_sizing.kelly_fraction` | Доля Келли | 0.25 |

## Технологии

| Компонент | Технология |
|---|---|
| Данные | ccxt (Binance, Bybit) |
| Хранение | ClickHouse (опционально) |
| Математика | scipy, numpy, py_vollib |
| ML | scikit-learn |
| Dashboard | Streamlit + Plotly |

## API ключи

Для публичных данных (OHLCV, ликвидации) ключи **не нужны**.

Для приватных данных создайте `.env`:
```bash
cp .env.example .env
# Редактируйте .env
```

## Лицензия

MIT
