#!/usr/bin/env python3
import os
import csv
import time
import logging
import requests
from datetime import datetime, timedelta

SETUP_FILE = os.getenv("SCANNER_OUTPUT_FILE", "setups.csv")
ACTIVE_FILE = os.getenv("ACTIVE_OUTPUT_FILE", "active_setups.csv")
STATE_FILE = os.getenv("STATE_FILE", "position_state.csv")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

CHECK_INTERVAL_SECONDS = 900
COOLDOWN_MINUTES = 30
NEAR_ENTRY_PCT = 0.8
MAX_BREAKOUT_CHASE_PCT = 1.5
MAX_SETUP_AGE_HOURS = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LAST_ALERT_TIME = {}

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def telegram_send(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram env missing")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        return r.status_code == 200
    except Exception as e:
        logging.warning(f"Telegram error: {e}")
        return False

def fetch_live_price(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "1d", "interval": "5m", "includePrePost": "false"}
    headers = {"User-Agent": os.getenv("YAHOO_USER_AGENT", "Mozilla/5.0")}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        result = (data.get("chart", {}).get("result") or [None])[0]
        if not result:
            return None
        closes = result.get("indicators", {}).get("quote", [{}])[0].get("close") or []
        closes = [x for x in closes if x is not None]
        return float(closes[-1]) if closes else None
    except Exception as e:
        logging.warning(f"fetch_live_price error {ticker}: {e}")
        return None

def cooldown_pass(ticker):
    now = datetime.now()
    last = LAST_ALERT_TIME.get(ticker)
    return not (last and (now - last) < timedelta(minutes=COOLDOWN_MINUTES))

def mark_alert_sent(ticker):
    LAST_ALERT_TIME[ticker] = datetime.now()

def is_setup_fresh(ts):
    try:
        dt = datetime.fromisoformat(ts)
        return (datetime.now() - dt) <= timedelta(hours=MAX_SETUP_AGE_HOURS)
    except:
        return False

def near_entry_pass(price, entry, signal_type):
    dist_pct = abs(price - entry) / entry * 100
    if signal_type == "breakout":
        return price <= entry * (1 + NEAR_ENTRY_PCT / 100)
    return dist_pct <= NEAR_ENTRY_PCT

def breakout_chase_skip(price, entry):
    return price > entry * (1 + MAX_BREAKOUT_CHASE_PCT / 100)

def load_state():
    rows = load_csv(STATE_FILE)
    return {r["ticker"]: r for r in rows if r.get("ticker")}

def save_state(state_dict):
    rows = list(state_dict.values())
    fieldnames = ["ticker", "last_status", "last_alert_time", "last_price", "updated_at"]
    save_csv(STATE_FILE, rows, fieldnames)

def format_alert(item, live_price):
    tp3 = item.get("tp3") or "-"
    return (
        f"{'🟢' if item['type']=='breakout' else '🟣'} {item['ticker']} - {item['type'].upper()}\n"
        f"Live: {live_price:,.0f}\n"
        f"Entry: {float(item['entry']):,.0f}\n"
        f"SL: {float(item['stop']):,.0f}\n"
        f"TP1: {float(item['tp1']):,.0f} | TP2: {float(item['tp2']):,.0f} | TP3: {tp3}\n"
        f"Score: {float(item['score']):.1f} | RSI: {float(item['rsi']):.1f} | RVOL: {float(item['rvol']):.2f}x\n"
        f"Gap: {float(item['gap_pct']):.2f}% | Flow: {item['flow']}\n"
    )

def write_active_rows(active_rows):
    if not active_rows:
        return
    fieldnames = list(active_rows[0].keys())
    with open(ACTIVE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in active_rows:
            writer.writerow(row)

def process_setups():
    setups = load_csv(SETUP_FILE)
    state = load_state()
    active_rows = []

    for item in setups:
        if not item.get("ticker") or not item.get("timestamp_scan"):
            continue
        if not is_setup_fresh(item["timestamp_scan"]):
            continue

        ticker = item["ticker"]
        live_price = fetch_live_price(ticker)
        if live_price is None:
            continue

        entry = float(item["entry"])
        if not cooldown_pass(ticker):
            continue
        if item["type"] == "breakout" and breakout_chase_skip(live_price, entry):
            continue
        if not near_entry_pass(live_price, entry, item["type"]):
            continue

        msg = format_alert(item, live_price)
        sent = telegram_send(msg)

        if sent:
            mark_alert_sent(ticker)

        state[ticker] = {
            "ticker": ticker,
            "last_status": "ALERT_SENT" if sent else "ALERT_FAILED",
            "last_alert_time": datetime.now().isoformat(),
            "last_price": f"{live_price:.0f}",
            "updated_at": datetime.now().isoformat()
        }

        item["status"] = "ALERT_SENT" if sent else "ALERT_FAILED"
        item["live_price"] = f"{live_price:.0f}"
        item["timestamp_alert"] = datetime.now().isoformat()
        item["alert_status"] = "SENT" if sent else "FAILED"
        active_rows.append(item)

    write_active_rows(active_rows)
    save_state(state)
    logging.info(f"checked={len(setups)} active={len(active_rows)}")

def main():
    while True:
        try:
            process_setups()
        except Exception as e:
            logging.exception(f"alert checker fatal: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
