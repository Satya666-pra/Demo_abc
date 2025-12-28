
import io
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import requests

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# =============================
# Streamlit Page Config
# =============================
st.set_page_config(
    page_title="AI-Powered Stock Price & Investment Analysis",
    page_icon="📈",
    layout="wide"
)

# =============================
# Data Loading Helpers
# =============================
@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch single-ticker OHLCV from yfinance.
    Ensures a flat column index and numeric Close/Volume.
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    # yfinance can sometimes return MultiIndex columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.copy()
    df.dropna(how="all", inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_ticker_info(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    try:
        info = tk.info if hasattr(tk, "info") else {}
        return info or {}
    except Exception:
        return {}


# =============================
# NSE Index → Ticker List (NIFTY 50/100/500)
# =============================
@st.cache_data(show_spinner=True)
def get_nse_index_tickers(index_source: str):
    """
    Returns a list of Yahoo Finance tickers (e.g. TCS.NS) for a selected NSE index.

    Primary: niftyindices.com CSV
    Fallback: GitHub raw static CSV mirrors (keeps app working when niftyindices times out)
    """
    primary = {
        "NIFTY 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
        "NIFTY 100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
        "NIFTY 500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    }

    fallback = {
        # If you want, you can replace these with your own hosted lists.
        "NIFTY 50": "https://raw.githubusercontent.com/tyrone-fonseca/nifty50/master/nifty50.csv",
        "NIFTY 100": "https://raw.githubusercontent.com/ityouknow/nifty100/master/nifty100.csv",
        "NIFTY 500": "https://raw.githubusercontent.com/VarunVats9/NIFTY-500-Stock-List/master/ind_nifty500list.csv",
    }

    url = primary.get(index_source)
    if url is None:
        return []

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    def _read_csv_from_url(u: str, timeout: int = 15) -> pd.DataFrame:
        resp = requests.get(u, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))

    # Retry primary
    df = None
    last_err = None
    for attempt in range(3):
        try:
            df = _read_csv_from_url(url, timeout=15 + 5 * attempt)
            break
        except Exception as e:
            last_err = e
            time.sleep(0.5 + attempt * 0.5)

    # Fallback
    if df is None:
        fb = fallback.get(index_source)
        if fb:
            try:
                df = _read_csv_from_url(fb, timeout=20)
                st.warning(f"Primary NSE CSV timed out; using fallback list for {index_source}.")
            except Exception as e:
                st.error(f"Failed to fetch {index_source} list. Please use Manual mode. Error: {e}")
                return []
        else:
            st.error(f"Failed to fetch {index_source} list. Please use Manual mode. Error: {last_err}")
            return []

    # Find symbol column
    symbol_col = None
    for col in df.columns:
        if str(col).strip().lower() in {"symbol", "ticker", "stock", "security"}:
            symbol_col = col
            break
    if symbol_col is None:
        # common in some mirrors
        for col in df.columns:
            if "symbol" in str(col).lower():
                symbol_col = col
                break

    if symbol_col is None:
        st.error(f"Could not find a Symbol/Ticker column in {index_source} CSV.")
        return []

    symbols = df[symbol_col].astype(str).str.strip()
    symbols = symbols[symbols.str.len() > 0].unique().tolist()

    # Convert NSE symbol to Yahoo Finance: TCS -> TCS.NS
    yahoo_tickers = []
    for s in symbols:
        s2 = s.replace(".NS", "").replace(".BO", "")
        if s2:
            yahoo_tickers.append(f"{s2}.NS")

    # Sometimes lists contain duplicates
    yahoo_tickers = list(dict.fromkeys(yahoo_tickers))
    return yahoo_tickers


# =============================
# Technical Indicators (pandas only)
# =============================
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators and returns a clean df with required columns present.
    If OHLCV missing, returns empty.
    """
    df = df.copy()
    required_price_cols = ["Close", "High", "Low", "Volume"]
    for c in required_price_cols:
        if c not in df.columns:
            return pd.DataFrame()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["RSI_14"] = calc_rsi(close, 14)

    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    df["SMA_200"] = close.rolling(window=200, min_periods=200).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd.ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    ma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    df["BB_HIGH"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(window=14, min_periods=14).mean()

    direction = np.sign(close.diff().fillna(0))
    df["OBV"] = (direction * volume).cumsum()

    lowest_low = low.rolling(window=14, min_periods=14).min()
    highest_high = high.rolling(window=14, min_periods=14).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    df["STOCH_K"] = stoch_k
    df["STOCH_D"] = stoch_k.rolling(window=3, min_periods=3).mean()

    # Ensure required indicator cols exist even if NaN
    required_ind_cols = [
        "RSI_14", "EMA_20", "EMA_50", "MACD", "MACD_SIGNAL", "MACD_HIST",
        "BB_HIGH", "BB_LOW", "ATR_14", "OBV", "STOCH_K", "STOCH_D", "SMA_200"
    ]
    for c in required_ind_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df.dropna(subset=["Close"])  # keep as much as possible
    return df


# =============================
# LSTM
# =============================
def create_lstm_dataset(series, lookback=60):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i, 0])
        y.append(series[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.expand_dims(X, axis=2)
    return X, y


def train_lstm_model(df: pd.DataFrame, lookback=60, epochs=10, batch_size=32):
    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    X, y = create_lstm_dataset(scaled, lookback=lookback)
    if len(X) < 20:
        raise ValueError("Not enough history for LSTM. Try a longer period or smaller lookback.")

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    test_pred = model.predict(X_test, verbose=0)
    test_pred_rescaled = scaler.inverse_transform(test_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    return model, scaler, history, (y_test_rescaled, test_pred_rescaled)


def forecast_future_prices(model, scaler, df: pd.DataFrame, lookback=60, horizon=15):
    close_prices = df["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(close_prices)

    if len(scaled) < lookback:
        raise ValueError("Not enough history to forecast. Increase period or reduce lookback.")

    last_seq = scaled[-lookback:]
    preds_scaled = []

    current_seq = last_seq.copy()
    for _ in range(horizon):
        pred = model.predict(current_seq.reshape(1, lookback, 1), verbose=0)
        preds_scaled.append(pred[0, 0])
        current_seq = np.vstack([current_seq[1:], pred])

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]
    return pd.DataFrame({"Predicted_Close": preds}, index=future_dates)


# =============================
# Investment Scoring
# =============================
def _to_scalar(x):
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return np.nan
        return float(x.iloc[-1])
    if isinstance(x, np.ndarray):
        x = x.flatten()
        if len(x) == 0:
            return np.nan
        return float(x[-1])
    try:
        return float(x)
    except Exception:
        return np.nan


def z_score(value, mean, std, reverse: bool = False) -> float:
    value = _to_scalar(value)
    mean = _to_scalar(mean)
    std = _to_scalar(std)
    if np.isnan(value) or np.isnan(std) or std == 0:
        score = 0.5
    else:
        score = 0.5 + 0.1 * (value - mean) / std
    score = max(0.0, min(1.0, score))
    return 1.0 - score if reverse else score


def compute_investment_score(df: pd.DataFrame, info: dict) -> dict:
    if df is None or df.empty:
        raise ValueError("Not enough clean data for scoring.")

    latest_close = float(df["Close"].iloc[-1])
    returns_1m = float(df["Close"].pct_change(21).iloc[-1]) if len(df) > 30 else np.nan
    returns_3m = float(df["Close"].pct_change(63).iloc[-1]) if len(df) > 90 else np.nan
    volatility_3m = float(df["Close"].pct_change().iloc[-63:].std()) if len(df) > 90 else np.nan
    avg_volume = float(df["Volume"].iloc[-60:].mean()) if "Volume" in df.columns and len(df) > 60 else np.nan

    pe = info.get("trailingPE", np.nan) if isinstance(info, dict) else np.nan
    pb = info.get("priceToBook", np.nan) if isinstance(info, dict) else np.nan
    profit_margin = info.get("profitMargins", np.nan) if isinstance(info, dict) else np.nan
    roe = info.get("returnOnEquity", np.nan) if isinstance(info, dict) else np.nan
    revenue_growth = info.get("revenueGrowth", np.nan) if isinstance(info, dict) else np.nan
    beta = info.get("beta", np.nan) if isinstance(info, dict) else np.nan

    score_valuation = z_score(pe, mean=15, std=10, reverse=True)
    score_growth = z_score(revenue_growth, mean=0.10, std=0.10)
    score_profitability = z_score(profit_margin, mean=0.10, std=0.10)
    score_risk = z_score(volatility_3m, mean=0.02, std=0.01, reverse=True)
    score_momentum = z_score(returns_3m, mean=0.05, std=0.10)
    score_liquidity = z_score(avg_volume, mean=1_000_000, std=2_000_000)
    score_management = z_score(roe, mean=0.12, std=0.08)
    score_industry = z_score(beta, mean=1.0, std=0.3)

    weights = {
        "Financial Performance": 0.18,
        "Valuation": 0.14,
        "Growth Potential": 0.14,
        "Risk": 0.14,
        "Momentum": 0.14,
        "Liquidity": 0.10,
        "Management Quality": 0.08,
        "Industry Strength": 0.08,
    }

    factor_scores = {
        "Financial Performance": score_profitability * 100,
        "Valuation": score_valuation * 100,
        "Growth Potential": score_growth * 100,
        "Risk": score_risk * 100,
        "Momentum": score_momentum * 100,
        "Liquidity": score_liquidity * 100,
        "Management Quality": score_management * 100,
        "Industry Strength": score_industry * 100,
    }

    overall_score = float(sum(factor_scores[k] * weights[k] for k in weights))

    if overall_score >= 80:
        rating = "Strong Buy"
    elif overall_score >= 65:
        rating = "Buy"
    elif overall_score >= 50:
        rating = "Hold"
    elif overall_score >= 35:
        rating = "Sell"
    else:
        rating = "Strong Sell"

    return {
        "factor_scores": factor_scores,
        "overall_score": overall_score,
        "rating": rating,
        "raw_metrics": {
            "latest_close": latest_close,
            "returns_1m": returns_1m,
            "returns_3m": returns_3m,
            "volatility_3m": volatility_3m,
            "avg_volume": avg_volume,
            "pe": pe,
            "pb": pb,
            "profit_margin": profit_margin,
            "roe": roe,
            "revenue_growth": revenue_growth,
            "beta": beta,
        },
    }


# =============================
# Extra: Factor exposures (A) + Sentiment (B) + Regime (E) + Monte Carlo (C)
# =============================
def factor_investing_exposures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple 5-factor proxy from price/volatility trends (educational).
    """
    close = df["Close"]
    ret_252 = close.pct_change(252)
    ret_63 = close.pct_change(63)
    vol_63 = close.pct_change().rolling(63).std()
    sma_200 = df.get("SMA_200", close.rolling(200).mean())
    ema_20 = df.get("EMA_20", close.ewm(span=20, adjust=False).mean())
    ema_50 = df.get("EMA_50", close.ewm(span=50, adjust=False).mean())

    latest = df.iloc[-1]
    # Value proxy: cheaper pullbacks vs long trend
    value = 100 * max(0, min(1, (float(sma_200.iloc[-1]) / float(close.iloc[-1])) if close.iloc[-1] else 0))
    # Quality proxy: stable trend (ema20>ema50 and price>sma200)
    quality = 100 * (1.0 if (float(ema_20.iloc[-1]) > float(ema_50.iloc[-1]) and float(close.iloc[-1]) > float(sma_200.iloc[-1])) else 0.3)
    # Momentum proxy: 3M return
    mom = 50 if np.isnan(ret_63.iloc[-1]) else max(0, min(100, 50 + 500 * float(ret_63.iloc[-1])))
    # Low vol proxy: inverse of 3M vol
    lv = 50 if np.isnan(vol_63.iloc[-1]) else max(0, min(100, 100 - 3000 * float(vol_63.iloc[-1])))
    # Size proxy: unknown without market cap; keep neutral
    size = 50.0

    return pd.DataFrame({
        "Factor": ["Value", "Quality", "Momentum", "Low Volatility", "Size (Neutral)"],
        "Exposure Score (0-100)": [value, quality, mom, lv, size]
    })


def analyst_sentiment_proxy(info: dict, score_summary: dict) -> dict:
    """
    Uses yfinance info (if available) and investment score as a proxy.
    """
    reco = None
    if isinstance(info, dict):
        reco = info.get("recommendationKey") or info.get("recommendationMean")
    base = float(score_summary["overall_score"]) if score_summary else 50.0

    if isinstance(reco, str):
        # Map yfinance keys to rough number
        m = {
            "strong_buy": 80, "buy": 70, "hold": 50, "underperform": 35, "sell": 25
        }
        reco_score = m.get(reco.lower(), 50)
    elif isinstance(reco, (int, float)) and not np.isnan(reco):
        # recommendationMean: 1=strong buy, 5=sell
        reco_score = max(0, min(100, 110 - 20 * float(reco)))
    else:
        reco_score = 50

    sentiment = 0.6 * base + 0.4 * reco_score
    if sentiment >= 75:
        label = "Bullish"
    elif sentiment >= 55:
        label = "Positive"
    elif sentiment >= 45:
        label = "Neutral"
    elif sentiment >= 30:
        label = "Negative"
    else:
        label = "Bearish"

    return {"sentiment_score": float(sentiment), "label": label, "raw_reco": reco}


def market_regime_detection(df: pd.DataFrame) -> dict:
    close = df["Close"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    vol20 = close.pct_change().rolling(20).std()

    if len(df) < 210:
        return {"regime": "Unknown (Need 200+ trading days)", "volatility": float(vol20.iloc[-1]) if len(vol20.dropna()) else np.nan}

    trend = "Bull" if ma50.iloc[-1] > ma200.iloc[-1] else "Bear"
    vol = float(vol20.iloc[-1]) if not np.isnan(vol20.iloc[-1]) else np.nan

    if np.isnan(vol):
        vol_label = "Unknown"
    elif vol < 0.012:
        vol_label = "Low Vol"
    elif vol < 0.022:
        vol_label = "Normal Vol"
    else:
        vol_label = "High Vol"

    return {"regime": f"{trend} / {vol_label}", "volatility": vol}


def monte_carlo_simulation(df: pd.DataFrame, horizon_days: int | None = None, n_sims: int | None = None):
    """
    Monte Carlo using log returns; auto-picks horizon & simulations if None.
    Keeps it 'dynamic' without user sliders.
    """
    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < 60:
        raise ValueError("Not enough data for Monte Carlo (need ~60+ points).")

    mu = float(log_ret.mean())
    sigma = float(log_ret.std())

    # Dynamic defaults
    if horizon_days is None:
        horizon_days = int(np.clip(len(df) // 10, 15, 60))
    if n_sims is None:
        n_sims = int(np.clip(len(df) * 2, 300, 1200))

    s0 = float(close.iloc[-1])
    drift = mu - 0.5 * sigma * sigma
    shocks = sigma * np.random.normal(size=(horizon_days, n_sims))
    increments = drift + shocks
    paths = s0 * np.exp(np.cumsum(increments, axis=0))

    end_prices = paths[-1, :]
    expected = float(end_prices.mean())
    prob_up = float((end_prices > s0).mean())
    prob_dd20 = float((end_prices < s0 * 0.8).mean())
    return {
        "paths": paths,
        "horizon_days": horizon_days,
        "n_sims": n_sims,
        "current_price": s0,
        "expected_price": expected,
        "prob_up": prob_up,
        "prob_dd20": prob_dd20,
    }


# =============================
# Indicator Health Meter (dot status)
# =============================
def health_dot(color: str) -> str:
    return f"<span style='font-size:1.2rem; color:{color};'>●</span>"


def build_indicator_health_meter(df: pd.DataFrame) -> list[dict]:
    """
    Creates status lines similar to your screenshot.
    Returns list of dicts: {dot_html, title, detail}
    """
    req = ["RSI_14", "STOCH_K", "STOCH_D", "MACD", "MACD_SIGNAL", "MACD_HIST", "SMA_200", "EMA_20", "EMA_50", "ATR_14", "BB_LOW", "BB_HIGH", "OBV"]
    for c in req:
        if c not in df.columns:
            return [{"dot_html": health_dot("red"), "title": "Indicators", "detail": f"Missing column: {c}"}]

    last = df.iloc[-1]
    close = float(last["Close"])

    items = []

    # RSI
    rsi = float(last["RSI_14"])
    if 30 <= rsi <= 70:
        items.append({"dot_html": health_dot("limegreen"), "title": f"RSI (14) ({rsi:.1f})", "detail": "Green – Healthy momentum (30–70 zone)."})
    elif rsi < 30:
        items.append({"dot_html": health_dot("orange"), "title": f"RSI (14) ({rsi:.1f})", "detail": "Yellow – Oversold (<30)."})
    else:
        items.append({"dot_html": health_dot("orange"), "title": f"RSI (14) ({rsi:.1f})", "detail": "Yellow – Overbought (>70)."})

    # Stochastic
    k = float(last["STOCH_K"])
    d = float(last["STOCH_D"])
    if 20 <= k <= 80 and 20 <= d <= 80:
        dot = "limegreen"; txt = "Green – Within normal 20–80 band."
    else:
        dot = "orange"; txt = "Yellow – Outside normal 20–80 band."
    items.append({"dot_html": health_dot(dot), "title": f"Stochastic %K/%D (K={k:.1f}, D={d:.1f})", "detail": txt})

    # MACD level
    macd = float(last["MACD"])
    dot = "limegreen" if macd > 0 else "red"
    txt = "Green – MACD above 0 (bullish bias)." if macd > 0 else "Red – MACD below 0 (bearish bias)."
    items.append({"dot_html": health_dot(dot), "title": f"MACD ({macd:.3f})", "detail": txt})

    # MACD signal cross
    sig = float(last["MACD_SIGNAL"])
    if macd >= sig:
        dot = "limegreen"; txt = "Green – MACD ≥ Signal (bullish crossover)."
    else:
        dot = "red"; txt = "Red – MACD < Signal (bearish crossover)."
    items.append({"dot_html": health_dot(dot), "title": f"MACD Signal Cross (MACD={macd:.3f}, Signal={sig:.3f})", "detail": txt})

    # MACD histogram
    hist = float(last["MACD_HIST"])
    if hist >= 0:
        dot = "limegreen"; txt = "Green – Positive histogram (bullish momentum)."
    else:
        dot = "red"; txt = "Red – Negative histogram (bearish momentum)."
    items.append({"dot_html": health_dot(dot), "title": f"MACD Histogram ({hist:.3f})", "detail": txt})

    # SMA 200 trend
    sma200 = float(last["SMA_200"]) if not np.isnan(last["SMA_200"]) else np.nan
    if np.isnan(sma200):
        items.append({"dot_html": health_dot("gray"), "title": "SMA 200 Trend", "detail": "Gray – Need 200 trading days."})
    else:
        if close > sma200:
            dot="limegreen"; txt="Green – Price above SMA 200 (long-term uptrend)."
        else:
            dot="red"; txt="Red – Price below SMA 200 (long-term downtrend)."
        items.append({"dot_html": health_dot(dot), "title": f"SMA 200 Trend (Close={close:.2f}, SMA200={sma200:.2f})", "detail": txt})

    # EMA 20 vs EMA 50
    ema20 = float(last["EMA_20"])
    ema50 = float(last["EMA_50"])
    if ema20 > ema50:
        dot="limegreen"; txt="Green – EMA20 > EMA50 (short-term uptrend)."
    else:
        dot="red"; txt="Red – EMA20 < EMA50 (short-term downtrend)."
    items.append({"dot_html": health_dot(dot), "title": f"EMA 20 vs EMA 50 (EMA20={ema20:.2f}, EMA50={ema50:.2f})", "detail": txt})

    # ATR %
    atr = float(last["ATR_14"]) if not np.isnan(last["ATR_14"]) else np.nan
    if np.isnan(atr) or close == 0:
        items.append({"dot_html": health_dot("gray"), "title": "ATR (14)", "detail": "Gray – Not enough data."})
    else:
        atr_pct = 100 * atr / close
        if atr_pct < 1.5:
            dot="limegreen"; txt="Green – Low/normal volatility."
        elif atr_pct < 3.5:
            dot="gold"; txt="Yellow – Normal volatility."
        else:
            dot="red"; txt="Red – High volatility."
        items.append({"dot_html": health_dot(dot), "title": f"ATR (14) ({atr:.2f} ({atr_pct:.1f}%))", "detail": txt})

    # Bollinger Bands
    bb_low = float(last["BB_LOW"]) if not np.isnan(last["BB_LOW"]) else np.nan
    bb_high = float(last["BB_HIGH"]) if not np.isnan(last["BB_HIGH"]) else np.nan
    if np.isnan(bb_low) or np.isnan(bb_high):
        items.append({"dot_html": health_dot("gray"), "title": "Bollinger Bands", "detail": "Gray – Need 20 days."})
    else:
        if bb_low <= close <= bb_high:
            dot="limegreen"; txt="Green – Price comfortably inside bands."
        else:
            dot="orange"; txt="Yellow – Price outside bands (breakout/extreme)."
        items.append({"dot_html": health_dot(dot), "title": f"Bollinger Bands (Close={close:.2f}, Low={bb_low:.2f}, High={bb_high:.2f})", "detail": txt})

    # OBV trend (10d)
    if "OBV" in df.columns and len(df) >= 12:
        obv = df["OBV"].astype(float)
        obv_chg = float((obv.iloc[-1] - obv.iloc[-11]) / (abs(obv.iloc[-11]) + 1e-9)) * 100
        if obv_chg > 2:
            dot="limegreen"; txt="Green – OBV rising (volume confirms trend)."
        elif obv_chg < -2:
            dot="red"; txt="Red – OBV falling (distribution)."
        else:
            dot="gold"; txt="Yellow – OBV flat (no strong volume trend)."
        items.append({"dot_html": health_dot(dot), "title": f"OBV Trend (10-day) ({obv_chg:.1f}%)", "detail": txt})

    return items


# =============================
# Rating badge
# =============================
def rating_badge(rating: str) -> str:
    color = "#6b7280"
    if rating == "Strong Buy":
        color = "#15803d"
    elif rating == "Buy":
        color = "#22c55e"
    elif rating == "Hold":
        color = "#eab308"
    elif rating == "Sell":
        color = "#f97316"
    elif rating == "Strong Sell":
        color = "#dc2626"

    return f"""
    <span style="
        background-color:{color};
        color:white;
        padding:2px 8px;
        border-radius:12px;
        font-size:0.8rem;
        font-weight:600;
        margin-left:4px;
    ">{rating}</span>
    """


# =============================
# Portfolio helper (sector table row)
# =============================
def analyse_single_ticker_for_sector(ticker: str, period: str = "1y", interval: str = "1d"):
    price_df = load_price_data(ticker, period=period, interval=interval)
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return None

    info = load_ticker_info(ticker)
    tech_df = add_technical_indicators(price_df)
    if tech_df.empty:
        return None

    score_summary = compute_investment_score(tech_df, info)
    factors = score_summary["factor_scores"]

    current_price = float(price_df["Close"].iloc[-1])
    row = {
        "Ticker": ticker,
        "Current_Price": round(current_price, 2),
        "Sector": info.get("sector", "Unknown") if isinstance(info, dict) else "Unknown",
        "Industry": info.get("industry", "Unknown") if isinstance(info, dict) else "Unknown",
        "Overall_Score": round(float(score_summary["overall_score"]), 1),
        "Rating": score_summary["rating"],
        "PE": score_summary["raw_metrics"]["pe"],
        "PB": score_summary["raw_metrics"]["pb"],
        "Profit_Margin": score_summary["raw_metrics"]["profit_margin"],
        "ROE": score_summary["raw_metrics"]["roe"],
        "Revenue_Growth": score_summary["raw_metrics"]["revenue_growth"],
        "Beta": score_summary["raw_metrics"]["beta"],
        "F_Financial_Performance": factors["Financial Performance"],
        "F_Valuation": factors["Valuation"],
        "F_Growth_Potential": factors["Growth Potential"],
        "F_Risk": factors["Risk"],
        "F_Momentum": factors["Momentum"],
        "F_Liquidity": factors["Liquidity"],
        "F_Management_Quality": factors["Management Quality"],
        "F_Industry_Strength": factors["Industry Strength"],
    }
    return row


# =============================
# UI
# =============================
st.title("🚀 AI-Powered Stock Price Prediction & Investment Analysis App 📈")
st.markdown(
    """
This app blends **LSTM Neural Networks**, **Technical Analysis**,  
an **8-Factor Investment Scoring Model**, and additional quant views  
(**Factor Investing, Sentiment, Monte Carlo, Market Regime, Indicator Health Meter**)  
for learning & research (**not** financial advice).
"""
)

with st.sidebar:
    st.header("⚙️ Single-Stock Controls")
    ticker = st.text_input("Stock Ticker", value="AAPL")
    period = st.selectbox("History Period", ["1y", "2y", "5y", "10y", "max"], index=2)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    lookback = st.slider("LSTM Lookback Window (days)", 30, 120, 60, step=5)
    forecast_horizon = st.slider("Forecast Horizon (days)", 5, 60, 15, step=5)
    epochs = st.slider("LSTM Training Epochs", 5, 50, 10, step=5)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    run_single = st.button("🚀 Run Single-Stock Analysis")

    st.markdown("---")
    st.header("📂 Portfolio / Sector View")

    index_source = st.selectbox(
        "Index Source",
        ["Manual (comma-separated below)", "NIFTY 50", "NIFTY 100", "NIFTY 500"],
        index=1
    )

    if index_source == "Manual (comma-separated below)":
        portfolio_tickers_text = st.text_area(
            "Enter tickers (comma-separated)",
            value="TCS.NS, INFY.NS, HDFCBANK.NS, ITC.NS"
        )
    else:
        portfolio_tickers_text = ""
        st.caption("✔ Tickers will be loaded automatically from the selected NSE index.")

    min_score = st.slider("Minimum Overall Score", 0, 100, 50, step=5)

    portfolio_button = st.button("📊 Analyse Portfolio by Sector")
    sector_button = st.button("📂 Quick Sector, Score & Rating Lookup")


def get_portfolio_tickers():
    if index_source == "Manual (comma-separated below)":
        return [t.strip().upper() for t in portfolio_tickers_text.split(",") if t.strip()]
    else:
        tickers = get_nse_index_tickers(index_source)
        if tickers:
            st.success(f"Loaded {len(tickers)} tickers from {index_source}.")
        return tickers


# =============================
# Single stock output (tabs)
# =============================
if run_single:
    try:
        with st.spinner("Fetching market data..."):
            price_df = load_price_data(ticker, period=period, interval=interval)
            info = load_ticker_info(ticker)

        if price_df is None or price_df.empty:
            st.error("No data received. Check ticker or internet connection.")
        else:
            current_price = float(price_df["Close"].iloc[-1])
            st.success(f"Loaded {len(price_df)} rows for {ticker}")

            # Build technicals once
            tech_df = add_technical_indicators(price_df)
            if tech_df.empty or len(tech_df) < 60:
                st.warning("Not enough clean indicator history for all sections. Try longer period.")
            # Investment scoring (needs technicals, but we allow even if some NaN)
            score_summary = compute_investment_score(tech_df if not tech_df.empty else price_df, info)

            tabs = st.tabs([
                "Overview",
                "Price & LSTM",
                "Factor Investing (A)",
                "Sentiment (B)",
                "Monte Carlo (C)",
                "Market Regime (E)",
            ])

            # --- Overview
            with tabs[0]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Current Price", f"{current_price:.2f}")
                with c2:
                    st.metric("Overall Score", f"{score_summary['overall_score']:.1f} / 100")
                with c3:
                    st.markdown(f"**Rating:** {rating_badge(score_summary['rating'])}", unsafe_allow_html=True)

                st.subheader("📈 Price History")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(price_df.index, price_df["Close"], label="Close")
                ax.set_title(f"{ticker} - Close Price")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                st.pyplot(fig)

                st.subheader("🩺 Indicator Health Meter")
                if tech_df is None or tech_df.empty:
                    st.info("Not enough data for indicator health meter.")
                else:
                    items = build_indicator_health_meter(tech_df)
                    st.markdown("### 🧪 Indicator Health Meter", unsafe_allow_html=True)
                    for it in items:
                        st.markdown(
                            f"{it['dot_html']} <b>{it['title']}</b> — {it['detail']}",
                            unsafe_allow_html=True
                        )

                st.subheader("📌 8-Factor Scoring Breakdown")
                factor_df = pd.DataFrame(
                    {"Factor": list(score_summary["factor_scores"].keys()),
                     "Score": list(score_summary["factor_scores"].values())}
                )
                st.dataframe(factor_df)
                st.bar_chart(factor_df.set_index("Factor"))

            # --- Price & LSTM
            with tabs[1]:
                st.subheader("📊 Technical Indicators (tail)")
                if tech_df is None or tech_df.empty:
                    st.warning("No technical dataframe. Try a longer period.")
                else:
                    st.dataframe(tech_df.tail(10))

                st.subheader("🤖 LSTM Price Prediction")
                if tech_df is None or tech_df.empty:
                    st.warning("Not enough data for LSTM.")
                else:
                    with st.spinner("Training LSTM model..."):
                        model, scaler, history, (y_test, y_pred) = train_lstm_model(
                            tech_df, lookback=lookback, epochs=epochs, batch_size=batch_size
                        )

                    col1, col2 = st.columns(2)
                    with col1:
                        fig_loss, ax_loss = plt.subplots(figsize=(5, 3))
                        ax_loss.plot(history.history["loss"], label="Train Loss")
                        ax_loss.plot(history.history["val_loss"], label="Val Loss")
                        ax_loss.legend()
                        st.pyplot(fig_loss)
                    with col2:
                        fig_test, ax_test = plt.subplots(figsize=(5, 3))
                        ax_test.plot(y_test, label="Actual")
                        ax_test.plot(y_pred, label="Predicted")
                        ax_test.legend()
                        st.pyplot(fig_test)

                    with st.spinner("Forecasting future prices..."):
                        future_df = forecast_future_prices(
                            model, scaler, tech_df, lookback=lookback, horizon=forecast_horizon
                        )

                    st.subheader(f"🔮 {forecast_horizon}-Day Forecast")
                    fig_future, ax_future = plt.subplots(figsize=(10, 4))
                    ax_future.plot(tech_df.index[-120:], tech_df["Close"].iloc[-120:], label="Historical Close")
                    ax_future.plot(future_df.index, future_df["Predicted_Close"], label="Forecast", linestyle="--")
                    ax_future.set_xlabel("Date")
                    ax_future.set_ylabel("Price")
                    ax_future.legend()
                    st.pyplot(fig_future)
                    st.dataframe(future_df)

            # --- Factor Investing (A)
            with tabs[2]:
                st.subheader("A) Factor Investing Exposures (5-factor proxy)")
                if tech_df is None or tech_df.empty:
                    st.warning("Not enough data for factor exposures.")
                else:
                    fdf = factor_investing_exposures(tech_df)
                    st.dataframe(fdf)
                    st.bar_chart(fdf.set_index("Factor"))

            # --- Sentiment (B)
            with tabs[3]:
                st.subheader("B) Sentiment Score (Analyst/Score Proxy)")
                sent = analyst_sentiment_proxy(info, score_summary)
                st.metric("Sentiment Score", f"{sent['sentiment_score']:.1f} / 100")
                st.write(f"Label: **{sent['label']}**")
                st.write(f"yfinance recommendation raw: `{sent['raw_reco']}`")

            # --- Monte Carlo (C)
            with tabs[4]:
                st.subheader("C) Monte Carlo Simulation (Dynamic)")
                if tech_df is None or tech_df.empty:
                    st.warning("Not enough data for Monte Carlo.")
                else:
                    mc = monte_carlo_simulation(tech_df)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Price", f"{mc['current_price']:.2f}")
                    c2.metric("Expected Price (mean)", f"{mc['expected_price']:.2f}")
                    c3.metric("Prob(Price > Current)", f"{mc['prob_up']*100:.1f}%")
                    c4.metric("Prob(>20% Drawdown)", f"{mc['prob_dd20']*100:.1f}%")

                    st.caption(f"Horizon: {mc['horizon_days']} days | Simulations: {mc['n_sims']}")
                    fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
                    # plot a subset of paths for readability
                    paths = mc["paths"]
                    for j in range(min(30, paths.shape[1])):
                        ax_mc.plot(paths[:, j], alpha=0.35)
                    ax_mc.set_title(f"{ticker} Monte Carlo Simulated Paths ({mc['horizon_days']} days)")
                    ax_mc.set_xlabel("Days")
                    ax_mc.set_ylabel("Price")
                    st.pyplot(fig_mc)

            # --- Market Regime (E)
            with tabs[5]:
                st.subheader("E) Market Regime Detection")
                if tech_df is None or tech_df.empty:
                    st.warning("Not enough data for regime detection.")
                else:
                    reg = market_regime_detection(tech_df)
                    st.metric("Regime", reg["regime"])
                    if not np.isnan(reg["volatility"]):
                        st.metric("20D Volatility (std of returns)", f"{reg['volatility']:.4f}")

    except Exception as e:
        st.error(f"Single-stock analysis failed: {e}")
else:
    st.info("Set your parameters in the sidebar, then click **🚀 Run Single-Stock Analysis** to start.")


# =============================
# Portfolio / Sector-wise View
# =============================
if portfolio_button:
    tickers = get_portfolio_tickers()
    if not tickers:
        st.warning("No tickers available. Check index source or manual list.")
    else:
        st.subheader(f"📂 Portfolio Overview (Sector-wise, Score ≥ {min_score})")

        rows = []
        with st.spinner("Fetching data & computing scores for portfolio..."):
            for t in tickers:
                try:
                    row = analyse_single_ticker_for_sector(t, period=period, interval=interval)
                    if row is not None:
                        rows.append(row)
                except Exception as ex:
                    st.error(f"Error for {t}: {ex}")

        if not rows:
            st.error("No valid tickers processed.")
        else:
            dfp = pd.DataFrame(rows)
            dfp = dfp[dfp["Overall_Score"] >= min_score].copy()
            if dfp.empty:
                st.warning(f"No stocks with Overall Score ≥ {min_score}.")
            else:
                dfp.sort_values(["Sector", "Overall_Score"], ascending=[True, False], inplace=True)
                dfp.reset_index(drop=True, inplace=True)

                sector_options = sorted(dfp["Sector"].fillna("Unknown").unique().tolist())
                selected_sectors = st.multiselect("Filter by Sector", sector_options, default=sector_options)
                filtered_df = dfp[dfp["Sector"].isin(selected_sectors)].reset_index(drop=True)

                st.write("### 🧾 All Stocks (Filtered)")
                st.dataframe(filtered_df)

                st.write(f"### 🏭 Sector Ranking (Avg Score, Score ≥ {min_score})")
                sector_summary = (
                    filtered_df.groupby("Sector")["Overall_Score"]
                    .agg(["count", "mean"])
                    .rename(columns={"count": "Num_Stocks", "mean": "Avg_Score"})
                    .sort_values("Avg_Score", ascending=False)
                )
                st.dataframe(sector_summary)

                st.write("### 🌡 Sector Heatmap (Average Scores)")
                st.dataframe(sector_summary.style.background_gradient(subset=["Avg_Score"]))

                # Radar: top 3 sectors
                st.write("### 🕸 Sector Factor Radar (Top Sectors)")
                factor_cols = [
                    "F_Financial_Performance",
                    "F_Valuation",
                    "F_Growth_Potential",
                    "F_Risk",
                    "F_Momentum",
                    "F_Liquidity",
                    "F_Management_Quality",
                    "F_Industry_Strength",
                ]
                labels = ["Fin Perf", "Valuation", "Growth", "Risk", "Momentum", "Liquidity", "Mgmt", "Industry"]

                sector_factors = filtered_df.groupby("Sector")[factor_cols].mean()
                top_sectors = sector_summary.head(3).index.tolist()

                if len(top_sectors) > 0:
                    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                    angles += angles[:1]

                    fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                    for sec in top_sectors:
                        vals = sector_factors.loc[sec, factor_cols].tolist()
                        vals += vals[:1]
                        ax_r.plot(angles, vals, label=sec)
                        ax_r.fill(angles, vals, alpha=0.1)
                    ax_r.set_xticks(angles[:-1])
                    ax_r.set_xticklabels(labels)
                    ax_r.set_title("Top Sector Factor Profiles")
                    ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
                    st.pyplot(fig_r)

                st.write("### 👑 Top 10 Stocks per Sector – Watchlist")
                watchlist_df = (
                    filtered_df
                    .sort_values(["Sector", "Overall_Score"], ascending=[True, False])
                    .groupby("Sector", group_keys=False)
                    .head(10)
                    .reset_index(drop=True)
                )

                watchlist_view = watchlist_df[[
                    "Sector", "Ticker", "Current_Price", "Overall_Score", "Rating", "Industry"
                ]]
                st.dataframe(watchlist_view)

                st.download_button(
                    "⬇️ Download Watchlist (Top 10 per Sector)",
                    data=watchlist_view.to_csv(index=False).encode("utf-8"),
                    file_name="sector_watchlist_top10.csv",
                    mime="text/csv",
                )

                st.write("### 📋 Watchlist Cards (Top 10 by Sector)")
                for sec, grp in watchlist_df.groupby("Sector"):
                    st.markdown(f"#### Sector: **{sec}**")
                    for _, r in grp.sort_values("Overall_Score", ascending=False).iterrows():
                        st.markdown(
                            f"""
                            <div style="margin-bottom:6px;">
                                <strong>{r['Ticker']}</strong>
                                &nbsp;| Price: {float(r['Current_Price']):.2f}
                                &nbsp;| Score: {float(r['Overall_Score']):.1f}
                                {rating_badge(r['Rating'])}
                                <br/>
                                <span style="font-size:0.8rem; color:#6b7280;">
                                    Industry: {r['Industry']}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown("---")


# =============================
# Quick Sector + Score + Rating Lookup
# =============================
if sector_button:
    tickers = get_portfolio_tickers()
    if not tickers:
        st.warning("No tickers available. Check index source or manual list.")
    else:
        st.subheader("📂 All Stocks – Sector, Industry, Price, Score & Rating")

        rows = []
        with st.spinner("Fetching sectors, prices, scores & ratings..."):
            for t in tickers:
                try:
                    row = analyse_single_ticker_for_sector(t, period=period, interval=interval)
                    if row is not None:
                        rows.append(row)
                except Exception as ex:
                    st.error(f"Error for {t}: {ex}")

        if not rows:
            st.error("No valid tickers processed for lookup.")
        else:
            df = pd.DataFrame(rows)
            df_simple = df[[
                "Ticker", "Sector", "Industry", "Current_Price", "Overall_Score", "Rating"
            ]].sort_values(["Sector", "Overall_Score"], ascending=[True, False]).reset_index(drop=True)

            st.write("### 🧾 Table – All Stocks by Sector")
            st.dataframe(df_simple)

            st.write("### 🎯 Cards – Rating Badges")
            for _, r in df_simple.iterrows():
                st.markdown(
                    f"""
                    <div style="margin-bottom:6px;">
                        <strong>{r['Ticker']}</strong>
                        &nbsp;| Price: {float(r['Current_Price']):.2f}
                        &nbsp;| Score: {float(r['Overall_Score']):.1f}
                        {rating_badge(r['Rating'])}
                        <br/>
                        <span style="font-size:0.8rem; color:#6b7280;">
                            Sector: {r['Sector']} | Industry: {r['Industry']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
