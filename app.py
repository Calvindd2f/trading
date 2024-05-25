import requests
import websocket
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from retrying import retry
import time

# Constants
API_BASE_URL = "https://api.exchange.com"
WEBSOCKET_URL = "wss://ws.exchange.com/realtime"
SYMBOL = "BTCUSD"
BUY_THRESHOLD = 0.05  # 5% price increase for pump detection
SELL_THRESHOLD = -0.05  # 5% price decrease for dump detection
TRADE_AMOUNT = 0.01  # Amount of BTC to trade
INITIAL_BACKOFF = 1  # Initial backoff duration in seconds

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Fetch historical data
@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_historical_data(symbol):
    response = requests.get(f"{API_BASE_URL}/historical/{symbol}")
    response.raise_for_status()  # Raise an error for bad status
    data = response.json()
    return pd.DataFrame(data)

# WebSocket callback for real-time data
def on_message(ws, message):
    data = json.loads(message)
    process_real_time_data(data)

# WebSocket error handler
def on_error(ws, error):
    logging.error(f"Error: {error}")

# WebSocket close handler
def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket closed")

# WebSocket open handler
def on_open(ws):
    subscribe_message = json.dumps({"type": "subscribe", "channels": [{"name": "ticker", "product_ids": [SYMBOL]}]})
    ws.send(subscribe_message)
    logging.info("WebSocket connection opened and subscription message sent")

# Process real-time data
def process_real_time_data(data):
    global historical_data

    # Append new data to historical DataFrame
    timestamp = datetime.fromisoformat(data['time'].replace("Z", ""))
    price = float(data['price'])
    volume = float(data['volume'])
    new_row = pd.DataFrame([[timestamp, price, volume]], columns=['time', 'price', 'volume'])
    historical_data = pd.concat([historical_data, new_row]).reset_index(drop=True)

    # Log received data
    logging.info(f"Received data - Time: {timestamp}, Price: {price}, Volume: {volume}")

    # Check for anomalies
    detect_anomalies()

# Detect pumps and dumps
def detect_anomalies():
    global historical_data

    # Calculate moving averages and price changes
    historical_data['ma'] = historical_data['price'].rolling(window=10).mean()
    historical_data['price_change'] = historical_data['price'].pct_change()

    latest_change = historical_data['price_change'].iloc[-1]

    if latest_change >= BUY_THRESHOLD:
        logging.info(f"Pump detected with change: {latest_change}")
        execute_trade("buy", TRADE_AMOUNT)
    elif latest_change <= SELL_THRESHOLD:
        logging.info(f"Dump detected with change: {latest_change}")
        execute_trade("sell", TRADE_AMOUNT)

# Execute trade with dynamic backoff
@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def execute_trade(side, amount):
    global backoff_duration

    trade_data = {
        "symbol": SYMBOL,
        "side": side,
        "type": "market",
        "quantity": amount
    }
    response = requests.post(f"{API_BASE_URL}/order", json=trade_data)
    response.raise_for_status()  # Raise an error for bad status

    logging.info(f"Executed {side} trade for {amount} {SYMBOL}. Response: {response.json()}")

    # Adjust backoff duration based on response time
    response_time = response.elapsed.total_seconds()
    backoff_duration = min(backoff_duration * (1 + response_time), 60)
    logging.info(f"Adjusted backoff duration to: {backoff_duration} seconds")

# Main function
if __name__ == "__main__":
    backoff_duration = INITIAL_BACKOFF

    historical_data = fetch_historical_data(SYMBOL)
    logging.info("Fetched historical data")

    ws = websocket.WebSocketApp(WEBSOCKET_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    logging.info("Starting WebSocket connection")
    ws.run_forever()
