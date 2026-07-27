import os
import csv
import math
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone, time as dt_time

import requests
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
JAKARTA_TZ = timezone(timedelta(hours=7))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TICKER_FILE = os.getenv("TICKER_FILE", os.path.join(BASE_DIR, "stocks.txt"))
SECTOR_FILE = os.getenv("SECTOR_FILE", os.path.join(BASE_DIR, "sectors.csv"))
OUTPUT_FILE = os.getenv("SCANNER_OUTPUT_FILE", os.path.join(BASE_DIR, "alert.csv"))
LOG_FILE = os.getenv("SCANNER_LOG_FILE", os.path.join(BASE_DIR, "scanner.log"))

YF_PERIOD = os.getenv("YF_PERIOD", "6mo")
YF_INTERVAL = os.getenv("YF_INTERVAL", "1d")
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "0"))  # 0 = all

RUN_ONLY_OPEN_SESSION = os.getenv("RUN_ONLY_OPEN_SESSION", "false").lower() == "true"
OPEN_START = dt_time(9, 0)
OPEN_END = dt_time(10, 30)

MIN_RVOL = float(os.getenv("MIN_RVOL", "1.8"))
MIN_AVG_VOLUME = float(os.getenv("MIN_AVG_VOLUME", "100000"))
MAX_PRICE = float(os.getenv("MAX_PRICE", "50000"))

# Flow thresholds for IDX in IDR
STRONG_ACCUM = 1_000_000_000
MED_ACCUM = 250_000_000
MED_DIST = -250_000_000
STRONG_DIST = -1_000_000_000

# Breakout / signal filters
BREAKOUT_LOOKBACK = 20
BREAKOUT_BUFFER_PCT = 0.002
PULLBACK_WINDOW = 5

# Rate limiting
SLEEP_EVERY = 3
SLEEP_SECONDS = 2

# =========================
# LOGGING
# =========================
logger = logging.getLogger("scanner")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
stream = logging.StreamHandler()
stream.setFormatter(fmt)
logger.addHandler(stream)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

# =========================
# HELPERS
# =========================
def now_jakarta():
    return datetime.now(JAKARTA_TZ)

def in_open_session():
    t = now_jakarta().time()
    return OPEN_START <= t <= OPEN_END

def load_tickers(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ticker file not found: {path}")
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if not t or t.startswith("#"):
                continue
            tickers.append(t)
    if MAX_TICKERS and MAX_TICKERS > 0:
        tickers = tickers[:MAX_TICKERS]
    return tickers

def load_sectors(path):
    sectors = {}
    if not os.path.exists(path):
        return sectors
    try:
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        if "ticker" in cols and "sector" in cols:
            ticker_col = df.columns[cols.index("ticker")]
            sector_col = df.columns[cols.index("sector")]
            for _, row in df.iterrows():
                t = str(row[ticker_col]).strip().upper()
                s = str(row[sector_col]).strip()
                if t:
                    sectors[t] = s
    except Exception as e:
        logger.warning(f"Failed to read sector file: {e}")
    return sectors

def clean_yf_df(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip() for col in df.columns.values]
    df = df.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Adj Close" in df.columns and "Adjclose" not in df.columns:
        df["Adjclose"] = df["Adj Close"]
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Adjclose"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in ["Close", "Volume"] if c in df.columns])
    return df

def fetch_history(ticker):
    import yfinance as yf
    try:
        df = yf.download(
            ticker,
            period=YF_PERIOD,
            interval=YF_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return clean_yf_df(df)
    except Exception as e:
        logger.warning(f"{ticker} download failed: {e}")
        return pd.DataFrame()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def macd_hist(series):
    macd_line = ema(series, 12) - ema(series, 26)
    signal = ema(macd_line, 9)
    return macd_line - signal

def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def safe_last(series):
    if series is None or len(series) == 0:
        return np.nan
    return series.iloc[-1]

def flow_label(flow_m):
    if flow_m >= STRONG_ACCUM:
        return "STRONG ACCUMULATION"
    if flow_m >= MED_ACCUM:
        return "MEDIUM ACCUMULATION"
    if flow_m <= STRONG_DIST:
        return "STRONG DISTRIBUTION"
    if flow_m <= MED_DIST:
        return "MEDIUM DISTRIBUTION"
    return "NEUTRAL"

def flow_score(flow_m):
    if flow_m >= STRONG_ACCUM:
        return 4
    if flow_m >= MED_ACCUM:
        return 2
    if flow_m <= STRONG_DIST:
        return -4
    if flow_m <= MED_DIST:
        return -2
    return 0

def breakout_level(df):
    if len(df) < BREAKOUT_LOOKBACK + 2:
        return np.nan
    prior_high = df["High"].shift(1).rolling(BREAKOUT_LOOKBACK).max().iloc[-1]
    return prior_high

def support_level(df):
    if len(df) < PULLBACK_WINDOW + 2:
        return np.nan
    return df["Low"].shift(1).rolling(PULLBACK_WINDOW).min().iloc[-1]

def price_snapshot(df):
    close = safe_last(df["Close"])
    vol = safe_last(df["Volume"])
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else np.nan
    return close, prev_close, vol

def score_ticker(df, ticker, sector=""):
    if df.empty or len(df) < 35:
        return None

    close, prev_close, vol = price_snapshot(df)
    if not np.isfinite(close) or close <= 0:
        return None
    if close > MAX_PRICE:
        return None

    avg_vol20 = df["Volume"].rolling(20).mean().iloc[-1]
    if not np.isfinite(avg_vol20) or avg_vol20 < MIN_AVG_VOLUME:
        return None

    rsi_val = safe_last(rsi(df["Close"]))
    macd_h = safe_last(macd_hist(df["Close"]))
    atr_val = safe_last(atr(df))
    if not np.isfinite(rsi_val) or not np.isfinite(macd_h) or not np.isfinite(atr_val):
        return None

    breakout = breakout_level(df)
    support = support_level(df)
    last_high = safe_last(df["High"])
    last_low = safe_last(df["Low"])
    last_open = safe_last(df["Open"])

    rv_ratio = vol / avg_vol20 if avg_vol20 > 0 else np.nan

    # Flow proxy from price * volume change
    dollar_flow = (df["Close"] * df["Volume"]).rolling(5).sum().iloc[-1] - (df["Close"] * df["Volume"]).rolling(5).sum().iloc[-6] if len(df) > 6 else np.nan
    if not np.isfinite(dollar_flow):
        dollar_flow = 0.0

    # Use recent signed pressure
    recent_ret = df["Close"].pct_change().tail(5)
    signed_flow = float((recent_ret.fillna(0).values * df["Volume"].tail(5).fillna(0).values).sum() * close)

    flow_m = signed_flow
    fscore = flow_score(flow_m)
    flabel = flow_label(flow_m)

    # Breakout / pullback logic
    breakout_confirmed = np.isfinite(breakout) and close > breakout * (1 + BREAKOUT_BUFFER_PCT)
    pullback_ok = np.isfinite(support) and close <= support * 1.03

    # Momentum score
    score = 0
    score += fscore
    score += 2 if breakout_confirmed else 0
    score += 1 if pullback_ok else 0
    score += 1 if rv_ratio >= MIN_RVOL else 0
    score += 1 if rsi_val >= 55 else 0
    score += 1 if macd_h > 0 else 0
    score += 1 if close > df["Close"].rolling(20).mean().iloc[-1] else 0
    score -= 1 if rsi_val > 78 else 0
    score -= 1 if atr_val / close > 0.12 else 0

    direction = None
    setup_type = None
    entry = None
    sl = None
    tp1 = None
    tp2 = None

    # Long breakout
    if breakout_confirmed and score >= 5 and rv_ratio >= MIN_RVOL:
        direction = "long"
        setup_type = "BREAKOUT"
        entry = round(float(breakout), 2)
        sl = round(float(max(support if np.isfinite(support) else entry * 0.95, entry - 1.5 * atr_val)), 2)
        tp1 = round(float(entry + 2 * (entry - sl)), 2)
        tp2 = round(float(entry + 3 * (entry - sl)), 2)

    # Long pullback
    elif pullback_ok and score >= 4 and rsi_val >= 45 and macd_h > -0.5:
        direction = "long"
        setup_type = "PULLBACK"
        entry = round(float(support), 2)
        sl = round(float(entry - 1.4 * atr_val), 2)
        tp1 = round(float(entry + 2 * (entry - sl)), 2)
        tp2 = round(float(entry + 3 * (entry - sl)), 2)

    # Short distribution
    elif flow_m <= MED_DIST and rsi_val < 45 and macd_h < 0 and close < df["Close"].rolling(20).mean().iloc[-1]:
        direction = "short"
        setup_type = "DISTRIBUTION"
        entry = round(float(close), 2)
        sl = round(float(entry + 1.4 * atr_val), 2)
        tp1 = round(float(entry - 2 * (sl - entry)), 2)
        tp2 = round(float(entry - 3 * (sl - entry)), 2)

    if direction is None:
        return None

    confidence = max(1, min(10, int(round(3 + score))))
    reason = (
        f"{ticker} menunjukkan {flabel.lower()} dengan RVOL {rv_ratio:.2f}x, RSI {rsi_val:.1f}, dan MACD histogram {macd_h:.2f}. "
        f"Strukturnya mendukung {setup_type.lower()} karena {('breakout atas resistance' if setup_type == 'BREAKOUT' else 'pantulan dari support' if setup_type == 'PULLBACK' else 'tekanan jual masih dominan')}. "
        f"Flow dan volume mendukung arah ini, jadi setup ini layak dipantau."
    )

    return {
        "ticker": ticker,
        "sector": sector,
        "direction": direction,
        "setup_type": setup_type,
        "entry": entry,
        "stop_loss": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rsi": round(float(rsi_val), 2),
        "macd_hist": round(float(macd_h), 4),
        "atr": round(float(atr_val), 2),
        "rvol": round(float(rv_ratio), 2),
        "flow_m": round(float(flow_m), 2),
        "flow_label": flabel,
        "score": score,
        "confidence": confidence,
        "reason": reason,
        "last_close": round(float(close), 2),
        "last_high": round(float(last_high), 2) if np.isfinite(last_high) else "",
        "last_low": round(float(last_low), 2) if np.isfinite(last_low) else "",
        "last_open": round(float(last_open), 2) if np.isfinite(last_open) else "",
        "avg_vol20": int(avg_vol20) if np.isfinite(avg_vol20) else "",
    }

def save_results(rows, path):
    fieldnames = [
        "timestamp", "ticker", "sector", "direction", "setup_type",
        "entry", "stop_loss", "tp1", "tp2", "rsi", "macd_hist", "atr",
        "rvol", "flow_m", "flow_label", "score", "confidence", "reason",
        "last_close", "last_open", "last_high", "last_low", "avg_vol20"
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main():
    if RUN_ONLY_OPEN_SESSION and not in_open_session():
        logger.info("OUTSIDE 09:00-10:30 WIB - scan skipped")
        return

    tickers = load_tickers(TICKER_FILE)
    sectors = load_sectors(SECTOR_FILE)

    logger.info(f"Loaded {len(tickers)} tickers")
    results = []
    scanned = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            df = fetch_history(ticker)
            if df.empty:
                logger.info(f"{ticker} - no data")
                continue

            row = score_ticker(df, ticker, sectors.get(ticker, ""))
            scanned += 1

            if row:
                row["timestamp"] = now_jakarta().strftime("%Y-%m-%d %H:%M:%S")
                results.append(row)
                logger.info(f"{ticker} - {row['direction'].upper()} {row['setup_type']} score={row['score']} flow={row['flow_label']}")

            if i % SLEEP_EVERY == 0:
                time.sleep(SLEEP_SECONDS)

        except Exception as e:
            logger.exception(f"{ticker} failed: {e}")

    results = sorted(results, key=lambda x: (x["score"], x["confidence"]), reverse=True)
    save_results(results, OUTPUT_FILE)

    logger.info(f"SCAN DONE | scanned={scanned} setups={len(results)} output={OUTPUT_FILE}")

if __name__ == "__main__":
    main()
