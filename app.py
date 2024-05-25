import requests
import websocket
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Constants
API_BASE_URL = "https://api.exchange.com"
WEBSOCKET_URL = "wss://ws.exchange.com/realtime"
SYMBOL = "BTCUSD"
BUY_THRESHOLD = 0.05  # 5% price increase for pump detection
SELL_THRESHOLD = -0.05  # 5% price decrease for dump detection
TRADE_AMOUNT = 0.01  # Amount of BTC to trade

# Fetch historical data
def fetch_historical_data(symbol):
    response = requests.get(f"{API_BASE_URL}/historical/{symbol}")
    data = response.json()
    return pd.DataFrame(data)

# WebSocket callback for real-time data
def on_message(ws, message):
    data = json.loads(message)
    process_real_time_data(data)

# WebSocket error handler
def on_error(ws, error):
    print(f"Error: {error}")

# WebSocket close handler
def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

# WebSocket open handler
def on_open(ws):
    subscribe_message = json.dumps({"type": "subscribe", "channels": [{"name": "ticker", "product_ids": [SYMBOL]}]})
    ws.send(subscribe_message)

# Process real-time data
def process_real_time_data(data):
    global historical_data

    # Append new data to historical DataFrame
    timestamp = datetime.fromisoformat(data['time'].replace("Z", ""))
    price = float(data['price'])
    volume = float(data['volume'])
    new_row = pd.DataFrame([[timestamp, price, volume]], columns=['time', 'price', 'volume'])
    historical_data = pd.concat([historical_data, new_row]).reset_index(drop=True)

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
        execute_trade("buy", TRADE_AMOUNT)
    elif latest_change <= SELL_THRESHOLD:
        execute_trade("sell", TRADE_AMOUNT)

# Execute trade
def execute_trade(side, amount):
    trade_data = {
        "symbol": SYMBOL,
        "side": side,
        "type": "market",
        "quantity": amount
    }
    response = requests.post(f"{API_BASE_URL}/order", json=trade_data)
    print(response.json())

# Main function
if __name__ == "__main__":
    historical_data = fetch_historical_data(SYMBOL)

    ws = websocket.WebSocketApp(WEBSOCKET_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
