import io
from datetime import timedelta
import os
import random

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# -----------------------------
# Global Random Seed (Deterministic Behaviour)
# -----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="AI-Powered Stock Price & Investment Analysis",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Data Loading Helpers
# -----------------------------
@st.cache_data(show_spinner=True)
def load_price_data(ticker: str, period: str = "5y", interval: str = "1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)
    return df


@st.cache_data(show_spinner=True)
def load_ticker_info(ticker: str):
    tk = yf.Ticker(ticker)
    info = tk.info if hasattr(tk, "info") else {}
    return info

# -----------------------------
# Simple Technical Indicators (pandas only)
# -----------------------------
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
    Add a set of technical indicators using ONLY df['Close']/['High']/['Low']/['Volume'].
    """
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 1) RSI
    df["RSI_14"] = calc_rsi(close, 14)

    # 2) EMAs & SMA
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    df["SMA_200"] = close.rolling(window=200, min_periods=200).mean()

    # 3) MACD & Signal
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd.ewm(span=9, adjust=False).mean()

    # 4) Bollinger Bands (20, 2)
    ma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    df["BB_HIGH"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    # 5) ATR(14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(window=14, min_periods=14).mean()

    # 6) OBV
    direction = np.sign(close.diff().fillna(0))
    df["OBV"] = (direction * volume).cumsum()

    # 7) Stochastic %K and %D (14, 3)
    lowest_low = low.rolling(window=14, min_periods=14).min()
    highest_high = high.rolling(window=14, min_periods=14).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    df["STOCH_K"] = stoch_k
    df["STOCH_D"] = stoch_k.rolling(window=3, min_periods=3).mean()

    df = df.dropna()
    return df

# -----------------------------
# Indicator Health Meter Logic
# -----------------------------
def _status_color(status: str) -> str:
    if status == "Green":
        return "🟢"
    if status == "Yellow":
        return "🟡"
    return "🔴"


def evaluate_indicator_health(tech_df: pd.DataFrame):
    """
    Returns a list of dicts:
    {name, value_str, status (Green/Yellow/Red), message}
    based on commonly used ideal ranges for each indicator.
    """
    latest = tech_df.iloc[-1]
    last_close = float(latest["Close"])
    results = []

    def get_val(col):
        val = latest.get(col, np.nan)
        try:
            return float(val)
        except Exception:
            return np.nan

    # 1) RSI 14
    rsi = get_val("RSI_14")
    if np.isnan(rsi):
        status, msg = "Yellow", "Not enough data for RSI."
    else:
        if 30 <= rsi <= 70:
            status, msg = "Green", "Healthy momentum (30–70 zone)."
        elif 20 <= rsi < 30 or 70 < rsi <= 80:
            status, msg = "Yellow", "Near overbought/oversold levels."
        else:
            status, msg = "Red", "Extreme overbought/oversold zone."
    results.append(
        {
            "name": "RSI (14)",
            "value_str": f"{rsi:.1f}" if not np.isnan(rsi) else "NA",
            "status": status,
            "message": msg,
        }
    )

    # 2) Stochastic K/D
    stoch_k = get_val("STOCH_K")
    stoch_d = get_val("STOCH_D")
    if np.isnan(stoch_k) or np.isnan(stoch_d):
        status, msg = "Yellow", "Not enough data for Stochastic."
    else:
        if 20 <= stoch_k <= 80 and 20 <= stoch_d <= 80:
            status, msg = "Green", "Within normal 20–80 band."
        elif (stoch_k < 20 or stoch_k > 80) or (stoch_d < 20 or stoch_d > 80):
            status, msg = "Red", "Overbought/oversold Stochastic region."
        else:
            status, msg = "Yellow", "Borderline Stochastic levels."
    results.append(
        {
            "name": "Stochastic %K/%D",
            "value_str": f"K={stoch_k:.1f}, D={stoch_d:.1f}"
            if not (np.isnan(stoch_k) or np.isnan(stoch_d))
            else "NA",
            "status": status,
            "message": msg,
        }
    )

    # 3) MACD & Histogram
    macd = get_val("MACD")
    macd_signal = get_val("MACD_SIGNAL")
    if np.isnan(macd) or np.isnan(macd_signal):
        status1, msg1 = "Yellow", "Not enough data for MACD."
        status2, msg2 = "Yellow", "Not enough data for MACD Signal."
        hist = np.nan
    else:
        hist = macd - macd_signal
        if macd > 0:
            status1, msg1 = "Green", "MACD above 0 (bullish bias)."
        elif -0.2 <= macd <= 0.2:
            status1, msg1 = "Yellow", "MACD near zero (sideways trend)."
        else:
            status1, msg1 = "Red", "MACD below 0 (bearish bias)."

        if macd > macd_signal:
            status2, msg2 = "Green", "MACD > Signal (bullish crossover)."
        elif abs(macd - macd_signal) < 0.05:
            status2, msg2 = "Yellow", "MACD close to Signal (unclear)."
        else:
            status2, msg2 = "Red", "MACD < Signal (bearish crossover)."

    results.append(
        {
            "name": "MACD",
            "value_str": f"{macd:.3f}" if not np.isnan(macd) else "NA",
            "status": status1,
            "message": msg1,
        }
    )
    results.append(
        {
            "name": "MACD Signal Cross",
            "value_str": f"MACD={macd:.3f}, Signal={macd_signal:.3f}"
            if not (np.isnan(macd) or np.isnan(macd_signal))
            else "NA",
            "status": status2,
            "message": msg2,
        }
    )
    results.append(
        {
            "name": "MACD Histogram",
            "value_str": f"{hist:.3f}" if not np.isnan(hist) else "NA",
            "status": "Green" if not np.isnan(hist) and hist > 0 else
            ("Yellow" if not np.isnan(hist) and abs(hist) <= 0.05 else "Red"),
            "message": "Positive histogram (bullish momentum)"
            if not np.isnan(hist) and hist > 0
            else ("Flat histogram (weak momentum)"
                  if not np.isnan(hist) and abs(hist) <= 0.05
                  else "Negative histogram (bearish momentum)"),
        }
    )

    # 4) SMA 200 (Price vs long-term trend)
    sma200 = get_val("SMA_200")
    if np.isnan(sma200):
        status, msg = "Yellow", "Not enough data for 200-day SMA."
    else:
        if last_close > sma200:
            status, msg = "Green", "Price above SMA 200 (long-term uptrend)."
        elif abs(last_close - sma200) / sma200 < 0.02:
            status, msg = "Yellow", "Price near SMA 200 (transition zone)."
        else:
            status, msg = "Red", "Price below SMA 200 (long-term downtrend)."
    results.append(
        {
            "name": "SMA 200 Trend",
            "value_str": f"Close={last_close:.2f}, SMA200={sma200:.2f}"
            if not np.isnan(sma200)
            else "NA",
            "status": status,
            "message": msg,
        }
    )

    # 5) EMA20 vs EMA50
    ema20 = get_val("EMA_20")
    ema50 = get_val("EMA_50")
    if np.isnan(ema20) or np.isnan(ema50):
        status, msg = "Yellow", "Not enough data for EMAs."
    else:
        if ema20 > ema50:
            status, msg = "Green", "EMA20 > EMA50 (short-term uptrend)."
        elif abs(ema20 - ema50) / ema50 < 0.01:
            status, msg = "Yellow", "EMAs close (sideways market)."
        else:
            status, msg = "Red", "EMA20 < EMA50 (short-term downtrend)."
    results.append(
        {
            "name": "EMA 20 vs EMA 50",
            "value_str": f"EMA20={ema20:.2f}, EMA50={ema50:.2f}"
            if not (np.isnan(ema20) or np.isnan(ema50))
            else "NA",
            "status": status,
            "message": msg,
        }
    )

    # 6) ATR volatility (as % of price)
    atr = get_val("ATR_14")
    if np.isnan(atr):
        status, msg = "Yellow", "Not enough data for ATR."
        atr_pct = np.nan
    else:
        atr_pct = atr / last_close
        if atr_pct <= 0.015:
            status, msg = "Green", "Low volatility (calm market)."
        elif atr_pct <= 0.03:
            status, msg = "Yellow", "Normal volatility."
        else:
            status, msg = "Red", "High volatility (risk zone)."
    results.append(
        {
            "name": "ATR (14)",
            "value_str": f"{atr:.2f} ({atr_pct*100:.1f}%)"
            if not np.isnan(atr)
            else "NA",
            "status": status,
            "message": msg,
        }
    )

    # 7) Bollinger Bands (Price inside / outside)
    bb_high = get_val("BB_HIGH")
    bb_low = get_val("BB_LOW")
    if np.isnan(bb_high) or np.isnan(bb_low):
        status, msg = "Yellow", "Not enough data for Bollinger Bands."
    else:
        if bb_low <= last_close <= bb_high:
            margin = min(
                (last_close - bb_low) / last_close,
                (bb_high - last_close) / last_close,
            )
            if margin < 0.02:
                status, msg = "Yellow", "Price close to band edge."
            else:
                status, msg = "Green", "Price comfortably inside bands."
        else:
            status, msg = "Red", "Price outside bands (extreme move)."
    results.append(
        {
            "name": "Bollinger Bands",
            "value_str": (
                f"Close={last_close:.2f}, Low={bb_low:.2f}, High={bb_high:.2f}"
                if not (np.isnan(bb_high) or np.isnan(bb_low))
                else "NA"
            ),
            "status": status,
            "message": msg,
        }
    )

    # 8) OBV trend (rising/flat/falling)
    obv_latest = get_val("OBV")
    if len(tech_df) > 10:
        obv_prev = float(tech_df["OBV"].iloc[-11])
    else:
        obv_prev = float(tech_df["OBV"].iloc[0])
    if np.isnan(obv_latest) or np.isnan(obv_prev) or obv_prev == 0:
        status, msg = "Yellow", "Not enough data for OBV trend."
        obv_chg_pct = np.nan
    else:
        obv_chg_pct = (obv_latest - obv_prev) / abs(obv_prev)
        if obv_chg_pct > 0.05:
            status, msg = "Green", "OBV rising (buying pressure)."
        elif obv_chg_pct < -0.05:
            status, msg = "Red", "OBV falling (selling pressure)."
        else:
            status, msg = "Yellow", "OBV flat (no strong volume trend)."
    results.append(
        {
            "name": "OBV Trend (10-day)",
            "value_str": f"{obv_chg_pct*100:.1f}%"
            if not np.isnan(obv_chg_pct)
            else "NA",
            "status": status,
            "message": msg,
        }
    )

    return results

# -----------------------------
# LSTM Data Prep & Training
# -----------------------------
def create_lstm_dataset(series, lookback=60):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i, 0])
        y.append(series[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.expand_dims(X, axis=2)  # [samples, timesteps, features]
    return X, y


def train_lstm_model(df: pd.DataFrame, lookback=60, epochs=10, batch_size=32):
    """
    Train LSTM on Close price.
    Scaler is fit only on training portion to avoid leakage.
    """
    close_prices = df["Close"].values.reshape(-1, 1)

    train_size_data = int(len(close_prices) * 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_prices[:train_size_data])

    scaled = scaler.transform(close_prices)

    X, y = create_lstm_dataset(scaled, lookback=lookback)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        shuffle=False,      # IMPORTANT: deterministic & time-series friendly
        verbose=0,
    )

    test_pred = model.predict(X_test, verbose=0)
    test_pred_rescaled = scaler.inverse_transform(test_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    return model, scaler, history, (y_test_rescaled, test_pred_rescaled)


def forecast_future_prices(model, scaler, df: pd.DataFrame, lookback=60, horizon=15):
    close_prices = df["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(close_prices)

    last_seq = scaled[-lookback:]
    preds_scaled = []

    current_seq = last_seq.copy()
    for _ in range(horizon):
        pred = model.predict(current_seq.reshape(1, lookback, 1), verbose=0)
        preds_scaled.append(pred[0, 0])
        current_seq = np.vstack([current_seq[1:], pred])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()

    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]
    future_df = pd.DataFrame({"Predicted_Close": preds}, index=future_dates)
    return future_df

# -----------------------------
# 8-Factor Investment Scoring
# -----------------------------
def _to_scalar(x):
    """Convert Series/ndarray to a single float for scoring."""
    if isinstance(x, pd.Series):
        return float(x.iloc[-1])
    if isinstance(x, np.ndarray):
        x = x.flatten()
        return float(x[-1])
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def z_score(value, mean, std, reverse: bool = False) -> float:
    """
    Safe z-score style normalization in [0,1].
    Handles Series/ndarray/None without raising 'truth value of Series' errors.
    """
    value = _to_scalar(value)
    mean = _to_scalar(mean)
    std = _to_scalar(std)

    if np.isnan(value) or np.isnan(std) or std == 0:
        score = 0.5  # neutral
    else:
        score = 0.5 + 0.1 * (value - mean) / std

    score = max(0.0, min(1.0, score))
    return 1.0 - score if reverse else score


def compute_investment_score(df: pd.DataFrame, info: dict) -> dict:
    latest_close = df["Close"].iloc[-1]
    returns_1m = df["Close"].pct_change(21).iloc[-1]
    returns_3m = df["Close"].pct_change(63).iloc[-1]
    volatility_3m = df["Close"].pct_change().iloc[-63:].std()
    avg_volume = df["Volume"].iloc[-60:].mean()

    pe = info.get("trailingPE", np.nan)
    pb = info.get("priceToBook", np.nan)
    profit_margin = info.get("profitMargins", np.nan)
    roe = info.get("returnOnEquity", np.nan)
    revenue_growth = info.get("revenueGrowth", np.nan)
    beta = info.get("beta", np.nan)

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

    overall_score = sum(factor_scores[k] * weights[k] for k in weights)

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

# -----------------------------
# Excel Export Helper
# -----------------------------
def to_excel(price_df, tech_df, future_df, score_summary):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        price_df.to_excel(writer, sheet_name="Prices")
        tech_df.to_excel(writer, sheet_name="Technicals")
        future_df.to_excel(writer, sheet_name="Forecast")

        score_df = pd.DataFrame(
            {
                "Factor": list(score_summary["factor_scores"].keys()),
                "Score": list(score_summary["factor_scores"].values()),
            }
        )
        score_df.to_excel(writer, sheet_name="Investment_Score", index=False)

        meta_df = pd.DataFrame(
            {
                "Metric": list(score_summary["raw_metrics"].keys()),
                "Value": list(score_summary["raw_metrics"].values()),
            }
        )
        meta_df.to_excel(writer, sheet_name="Raw_Metrics", index=False)

    return output.getvalue()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚀 AI-Powered Stock Price Prediction & Investment Analysis App 📈")

st.markdown(
    """
This app blends **LSTM Neural Networks**, **Technical Analysis**,  
an **Indicator Health Meter**, and an **8-Factor Investment Scoring Model**  
for learning and research (**not** financial advice).
"""
)

with st.sidebar:
    st.header("⚙️ Controls")

    ticker = st.text_input("Stock Ticker", value="AAPL")
    period = st.selectbox("History Period", ["1y", "2y", "5y", "10y", "max"], index=2)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    lookback = st.slider("LSTM Lookback Window (days)", 30, 120, 60, step=5)
    forecast_horizon = st.slider("Forecast Horizon (days)", 5, 60, 15, step=5)
    epochs = st.slider("LSTM Training Epochs", 5, 50, 10, step=5)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    train_button = st.button("🚀 Run Analysis (Fetch + Train + Forecast)")

# -----------------------------
# Main App Logic
# -----------------------------
if train_button:
    try:
        with st.spinner("Fetching market data..."):
            price_df = load_price_data(ticker, period=period, interval=interval)
            info = load_ticker_info(ticker)

        if price_df.empty:
            st.error("No data received. Check the ticker or internet connection.")
        else:
            st.success(f"Loaded {len(price_df)} rows of data for {ticker}")

            # Price history chart
            st.subheader("📈 Price History")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(price_df.index, price_df["Close"], label="Close")
            ax.set_title(f"{ticker} - Close Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # Technical indicators
            st.subheader("📊 Technical Indicators")
            tech_df = add_technical_indicators(price_df)
            st.dataframe(tech_df.tail(10))

            # Indicator Health Meter
            st.subheader("🩺 Indicator Health Meter")
            health = evaluate_indicator_health(tech_df)
            for h in health:
                emoji = _status_color(h["status"])
                st.markdown(
                    f"{emoji} **{h['name']}** "
                    f"({h['value_str']}) → **{h['status']}** – {h['message']}"
                )

            # LSTM model
            st.subheader("🤖 LSTM Price Prediction")
            with st.spinner("Training LSTM model..."):
                model, scaler, history, (y_test, y_pred) = train_lstm_model(
                    tech_df, lookback=lookback, epochs=epochs, batch_size=batch_size
                )

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Training Loss Curve")
                fig_loss, ax_loss = plt.subplots(figsize=(5, 3))
                ax_loss.plot(history.history["loss"], label="Train Loss")
                ax_loss.plot(history.history["val_loss"], label="Val Loss")
                ax_loss.legend()
                st.pyplot(fig_loss)

            with col2:
                st.caption("Test Predictions (LSTM vs Actual)")
                fig_test, ax_test = plt.subplots(figsize=(5, 3))
                ax_test.plot(y_test, label="Actual")
                ax_test.plot(y_pred, label="Predicted")
                ax_test.legend()
                st.pyplot(fig_test)

            # Future forecast
            with st.spinner("Forecasting future prices..."):
                future_df = forecast_future_prices(
                    model, scaler, tech_df, lookback=lookback, horizon=forecast_horizon
                )

            st.subheader(f"🔮 {forecast_horizon}-Day Price Forecast")
            fig_future, ax_future = plt.subplots(figsize=(10, 4))
            ax_future.plot(
                tech_df.index[-120:], tech_df["Close"].iloc[-120:], label="Historical Close"
            )
            ax_future.plot(
                future_df.index, future_df["Predicted_Close"],
                label="Forecast", linestyle="--"
            )
            ax_future.set_xlabel("Date")
            ax_future.set_ylabel("Price")
            ax_future.legend()
            st.pyplot(fig_future)

            st.dataframe(future_df)

            # 8-factor scoring
            st.subheader("📌 8-Factor Investment Scoring Model (0–100)")
            score_summary = compute_investment_score(tech_df, info)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric(
                    label="Overall Investment Score",
                    value=f"{score_summary['overall_score']:.1f} / 100",
                )
                st.markdown(f"**Rating:** `{score_summary['rating']}`")

            with c2:
                factor_df = pd.DataFrame(
                    {
                        "Factor": list(score_summary["factor_scores"].keys()),
                        "Score": list(score_summary["factor_scores"].values()),
                    }
                )
                st.bar_chart(factor_df.set_index("Factor"))

            with st.expander("View Raw Metrics Used in Scoring"):
                st.json(score_summary["raw_metrics"])

            # Excel export
            st.subheader("📤 Export to Excel")
            excel_bytes = to_excel(price_df, tech_df, future_df, score_summary)
            st.download_button(
                label="⬇️ Download Full Analysis as Excel",
                data=excel_bytes,
                file_name=f"{ticker}_analysis.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Set your parameters in the sidebar, then click **🚀 Run Analysis** to start.")
