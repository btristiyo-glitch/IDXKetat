#!/usr/bin/env python3
import os
import csv
import logging
import requests
import pandas as pd
from datetime import datetime

ACTIVE_FILE = os.getenv("ACTIVE_OUTPUT_FILE", "active_setups.csv")
RESULT_FILE = "backtest_results.csv"
CANDLE_INTERVAL = "5m"
LOOKAHEAD_BARS = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(str(x).replace(",", ""))
    except:
        return default

def fetch_hist(ticker, range_="10d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": range_, "interval": CANDLE_INTERVAL, "includePrePost": "false"}
    headers = {"User-Agent": os.getenv("YAHOO_USER_AGENT", "Mozilla/5.0")}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    result = (data.get("chart", {}).get("result") or [None])[0]
    if not result:
        return None
    ts = result.get("timestamp") or []
    q = (result.get("indicators", {}).get("quote") or [{}])[0]
    df = pd.DataFrame({
        "ts": pd.to_datetime(ts, unit="s"),
        "open": q.get("open") or [],
        "high": q.get("high") or [],
        "low": q.get("low") or [],
        "close": q.get("close") or [],
        "volume": q.get("volume") or [],
    }).dropna(subset=["open", "high", "low", "close"])
    return df.reset_index(drop=True)

def simulate_trade(df, entry_idx, entry, stop, tp1, signal_type):
    for i in range(entry_idx + 1, min(len(df), entry_idx + 1 + LOOKAHEAD_BARS)):
        bar = df.iloc[i]
        o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
        if signal_type == "breakout":
            stop_hit = l <= stop
            tp_hit = h >= tp1
        else:
            stop_hit = h >= stop
            tp_hit = l <= tp1

        if stop_hit and tp_hit:
            if abs(o - stop) < abs(o - tp1):
                return "LOSS", stop
            return "WIN", tp1
        if stop_hit:
            return "LOSS", stop
        if tp_hit:
            return "WIN", tp1
    return "OPEN", float(df.iloc[min(len(df) - 1, entry_idx + LOOKAHEAD_BARS)]["close"])

def evaluate_row(row):
    ticker = row.get("ticker")
    signal_type = (row.get("type") or "").lower()
    entry = safe_float(row.get("entry"))
    stop = safe_float(row.get("stop"))
    tp1 = safe_float(row.get("tp1"))
    if not ticker or None in [entry, stop, tp1]:
        return {**row, "result": "UNKNOWN", "r_multiple": "", "note": "missing fields"}

    df = fetch_hist(ticker, range_="10d")
    if df is None or df.empty:
        return {**row, "result": "UNKNOWN", "r_multiple": "", "note": "no history"}

    entry_idx = None
    for i in range(len(df) - 1):
        next_open = float(df.iloc[i + 1]["open"])
        if abs(next_open - entry) / entry <= 0.02:
            entry_idx = i + 1
            break
    if entry_idx is None:
        entry_idx = max(0, len(df) - LOOKAHEAD_BARS - 1)

    outcome, exit_price = simulate_trade(df, entry_idx, entry, stop, tp1, signal_type)
    risk = abs(entry - stop)
    if risk <= 0:
        r_mult = ""
    else:
        r_mult = round((exit_price - entry) / risk, 2) if signal_type == "breakout" else round((entry - exit_price) / risk, 2)

    return {
        **row,
        "result": outcome,
        "exit_price": round(exit_price, 0),
        "r_multiple": r_mult,
        "note": "tp1 before sl" if outcome == "WIN" else ("sl before tp1" if outcome == "LOSS" else "expired")
    }

def main():
    rows = load_csv(ACTIVE_FILE)
    if not rows:
        print("No active setups found.")
        return

    results = [evaluate_row(r) for r in rows]
    df = pd.DataFrame(results)
    df.to_csv(RESULT_FILE, index=False)

    total = len(df)
    wins = len(df[df["result"] == "WIN"])
    losses = len(df[df["result"] == "LOSS"])
    open_trades = len(df[df["result"] == "OPEN"])
    unknown = len(df[df["result"] == "UNKNOWN"])
    closed = wins + losses
    win_rate = (wins / closed * 100) if closed > 0 else 0

    print(f"Total: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Open: {open_trades}")
    print(f"Unknown: {unknown}")
    print(f"Win rate: {win_rate:.2f}%")

if __name__ == "__main__":
    main()
