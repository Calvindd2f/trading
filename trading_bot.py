import joblib
import pandas as pd
from datetime import datetime
import requests
import websocket
import json
import logging
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = joblib.load('pump_dump_model.pkl')

# Constants
API_BASE_URL = "https://api.exchange.com"
WEBSOCKET_URL = "wss://ws.exchange.com/realtime"
SYMBOL = "BTCUSD"
TRADE_AMOUNT = 0.01

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Fetch historical data for initial processing
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

    # Predict anomaly
    predict_anomaly()

# Predict anomaly using the trained model
def predict_anomaly():
    global historical_data

    # Prepare the latest data for prediction
    latest_data = historical_data.iloc[-1:]
    latest_data['price_change'] = latest_data['price'].pct_change()
    latest_data['volume_change'] = latest_data['volume'].pct_change()
    latest_data['ma_10'] = latest_data['price'].rolling(window=10).mean()
    latest_data['ma_50'] = latest_data['price'].rolling(window=50).mean()
    latest_data['ma_200'] = latest_data['price'].rolling(window=200).mean()
    latest_data['ma_diff'] = latest_data['ma_10'] - latest_data['ma_50']
    latest_data.dropna(inplace=True)

    if latest_data.empty:
        return

    # Extract features
    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff']
    X_latest = latest_data[features]

    # Predict using the model
    prediction = model.predict(X_latest)[0]

    if prediction == 1:
        logging.info("Pump detected")
        execute_trade("buy", TRADE_AMOUNT)
    elif prediction == -1:
        logging.info("Dump detected")
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
    response.raise_for_status()  # Raise an error for bad status

    trade_result = response.json()
    logging.info(f"Executed {side} trade for {amount} {SYMBOL}. Response: {trade_result}")

# Main function
if __name__ == "__main__":
    historical_data = fetch_historical_data(SYMBOL)
    logging.info("Fetched historical data")

    ws = websocket.WebSocketApp(WEBSOCKET_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    logging.info("Starting WebSocket connection")
    ws.run_forever()
