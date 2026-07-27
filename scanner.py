#!/usr/bin/env python3
import os
import csv
import time
import random
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime

TICKER_FILE = os.getenv("SCANNER_TICKERS_FILE", "stocks.txt")
OUTPUT_FILE = os.getenv("SCANNER_OUTPUT_FILE", "setups.csv")
REQUEST_DELAY = 0.10
MIN_DAILY_VALUE = 3_000_000_000
MIN_BARS = 60

BREAKOUT_MIN_SCORE = 40
BREAKOUT_GAP_MIN = 1.5
BREAKOUT_GAP_MAX = 6.5

REVERSAL_MIN_SCORE = 20
REVERSAL_RSI_MAX = 40
REVERSAL_RVOL_MIN = 1.2
REVERSAL_DROP_MAX = 0.5

OPEN_SESSION_START = "09:00"
OPEN_SESSION_END = "10:30"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def in_open_session():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    hm = now.strftime("%H:%M")
    return OPEN_SESSION_START <= hm <= OPEN_SESSION_END

def sleep_jitter(base=0.10):
    time.sleep(round(random.uniform(base * 0.5, base * 1.5), 3))

def load_ticker_list(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip()]

def fetch_data(ticker, interval="1d", range_="6mo", retries=3):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": range_, "interval": interval, "includePrePost": "false"}
    headers = {"User-Agent": os.getenv("YAHOO_USER_AGENT", "Mozilla/5.0")}
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logging.warning(f"fetch_data error {ticker}: {e}")
            sleep_jitter(0.2)
    return None

def extract_series(data):
    try:
        if not data:
            return None
        chart = data.get("chart") or {}
        result_list = chart.get("result") or []
        if not result_list:
            return None
        result = result_list[0] or {}
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quotes_list = indicators.get("quote") or []
        if not timestamps or not quotes_list:
            return None
        quotes = quotes_list[0] or {}
        df = pd.DataFrame({
            "open": quotes.get("open") or [],
            "high": quotes.get("high") or [],
            "low": quotes.get("low") or [],
            "close": quotes.get("close") or [],
            "volume": quotes.get("volume") or [],
        }, index=pd.to_datetime(timestamps, unit="s"))
        df = df.dropna(subset=["close", "volume"])
        df = df[df["volume"] > 0]
        if len(df) < MIN_BARS:
            return None
        return df
    except Exception as e:
        logging.warning(f"extract_series error: {e}")
        return None

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_support_resistance(df, lookback=20):
    recent = df.tail(lookback)
    return float(recent["low"].min()), float(recent["high"].max())

def td_sequential(close, period=4):
    if len(close) < period + 5:
        return 0
    score = 0
    for i in range(1, period + 1):
        if close.iloc[-i] > close.iloc[-i - period]:
            score += 1
        elif close.iloc[-i] < close.iloc[-i - period]:
            score -= 1
    return score

def get_flow_proxy(df):
    if df is None or len(df) < 10:
        return 0.0
    close = df["close"].values
    volume = df["volume"].values
    high = df["high"].values
    low = df["low"].values
    net = 0.0
    for i in range(1, min(len(df), 30)):
        tp = (high[i] + low[i] + close[i]) / 3.0
        if close[i] > close[i - 1]:
            net += tp * volume[i]
        elif close[i] < close[i - 1]:
            net -= tp * volume[i]
    return float(net)

def flow_score(net_flow_idr, avg_daily_value):
    score = 0
    label = "NEUTRAL"
    if avg_daily_value <= 0:
        return score, label
    flow_pct = (net_flow_idr / avg_daily_value) * 100
    if flow_pct <= -15:
        score += 25
        label = "ACCUMULATION STRONG"
    elif flow_pct <= -5:
        score += 12
        label = "ACCUMULATION"
    elif flow_pct >= 20:
        score -= 20
        label = "DISTRIBUTION STRONG"
    elif flow_pct >= 8:
        score -= 10
        label = "DISTRIBUTION"
    return score, label

def level_entry(signal_type, price, support, atr_val):
    if signal_type == "breakout":
        entry = max(support, price * 0.995) if support else price * 0.995
        stop = max(1, entry - max(price * 0.01, atr_val * 0.7))
        tp1 = entry + min(atr_val * 1.0, price * 0.03)
        tp2 = entry + min(atr_val * 1.8, price * 0.06)
        tp3 = entry + min(atr_val * 2.6, price * 0.09)
    else:
        entry = price * 0.995
        stop = max(1, price - max(price * 0.02, atr_val * 0.8))
        tp1 = price + min(atr_val * 0.9, price * 0.025)
        tp2 = price + min(atr_val * 1.5, price * 0.05)
        tp3 = ""
    return round(entry, 0), round(stop, 0), round(tp1, 0), round(tp2, 0), (round(tp3, 0) if tp3 else "")

def scan_market(tickers):
    results = []
    scanned = 0

    for ticker in tickers:
        scanned += 1
        sleep_jitter(REQUEST_DELAY)

        data = fetch_data(ticker, interval="1d", range_="6mo")
        df = extract_series(data)
        if df is None:
            continue

        close = df["close"]
        volume = df["volume"]
        price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        gap_pct = ((price / prev_price) - 1) * 100 if prev_price > 0 else 0
        daily_value = float(price * volume.iloc[-1])
        if daily_value < MIN_DAILY_VALUE:
            continue

        atr_series = calc_atr(df)
        atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else price * 0.02
        rsi_series = calc_rsi(close)
        rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0
        avg_vol_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
        rvol = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0

        sma20 = float(close.tail(20).mean()) if len(close) >= 20 else price
        sma50 = float(close.tail(50).mean()) if len(close) >= 50 else price
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else price

        td_setup = td_sequential(close)
        support, resistance = calc_support_resistance(df)
        proxy_flow_idr = get_flow_proxy(df)
        flow_bonus, flow_label = flow_score(proxy_flow_idr, daily_value)
        sector = ticker.replace(".JK", "")

        score_m = 0
        score_m += (atr_val / price) * 220
        score_m += rvol * 12
        score_m += flow_bonus
        score_m += min(max(gap_pct, -2), 12) * 2.0
        if price > sma20:
            score_m += 8
        if price > sma50:
            score_m += 6
        if price > sma50 > sma200:
            score_m += 8
        if td_setup >= 3:
            score_m += 6
        if 55 <= rsi_val <= 75:
            score_m += 8
        if rvol < 1.2:
            score_m -= 12
        elif rvol < 1.5:
            score_m -= 6
        else:
            score_m += 8

        score_r = 0
        score_r += rvol * 6
        if rsi_val < REVERSAL_RSI_MAX:
            score_r += (REVERSAL_RSI_MAX - rsi_val) * 2
        score_r += abs(flow_bonus) * 0.4
        score_r += (atr_val / price) * 150
        if td_setup >= 5:
            score_r += 10

        if score_m >= BREAKOUT_MIN_SCORE and BREAKOUT_GAP_MIN <= gap_pct <= BREAKOUT_GAP_MAX and rvol >= 1.5 and 55 <= rsi_val <= 75 and "DISTRIBUTION" not in flow_label:
            entry, stop, tp1, tp2, tp3 = level_entry("breakout", price, support, atr_val)
            results.append({
                "timestamp_scan": datetime.now().isoformat(),
                "ticker": ticker,
                "type": "breakout",
                "sector": sector,
                "price": round(price, 0),
                "prev_price": round(prev_price, 0),
                "gap_pct": round(gap_pct, 2),
                "volume": int(volume.iloc[-1]),
                "daily_value": int(daily_value),
                "rvol": round(rvol, 2),
                "atr_pct": round((atr_val / price) * 100, 2),
                "rsi": round(rsi_val, 1),
                "flow": flow_label,
                "score": round(score_m, 1),
                "entry": entry,
                "stop": stop,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "support": round(support, 0),
                "resistance": round(resistance, 0),
                "status": "NEW"
            })

        if score_r >= REVERSAL_MIN_SCORE and rsi_val < REVERSAL_RSI_MAX and rvol > REVERSAL_RVOL_MIN and gap_pct <= REVERSAL_DROP_MAX:
            entry, stop, tp1, tp2, tp3 = level_entry("reversal", price, support, atr_val)
            results.append({
                "timestamp_scan": datetime.now().isoformat(),
                "ticker": ticker,
                "type": "reversal",
                "sector": sector,
                "price": round(price, 0),
                "prev_price": round(prev_price, 0),
                "gap_pct": round(gap_pct, 2),
                "volume": int(volume.iloc[-1]),
                "daily_value": int(daily_value),
                "rvol": round(rvol, 2),
                "atr_pct": round((atr_val / price) * 100, 2),
                "rsi": round(rsi_val, 1),
                "flow": flow_label,
                "score": round(score_r, 1),
                "entry": entry,
                "stop": stop,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "support": round(support, 0),
                "resistance": round(resistance, 0),
                "status": "NEW"
            })

    results.sort(key=lambda x: float(x["score"]), reverse=True)
    return scanned, results[:20]

def save_setups(results):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp_scan","ticker","type","sector","price","prev_price","gap_pct","volume","daily_value","rvol","atr_pct","rsi","flow","score","entry","stop","tp1","tp2","tp3","support","resistance","status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

def main():
    if not in_open_session():
        print("OUTSIDE 09:00-10:30 WIB - scan skipped")
        return
    tickers = load_ticker_list(TICKER_FILE)
    scanned, results = scan_market(tickers)
    save_setups(results)
    print(f"SCAN DONE - scanned={scanned} setups={len(results)}")

if __name__ == "__main__":
    main()
