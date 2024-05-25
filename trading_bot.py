import sqlite3
import logging
from datetime import datetime
import joblib
import pandas as pd
import requests
import websocket
import json
import logging
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the trained model
model = joblib.load('pump_dump_model.pkl')

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize global variables for performance metrics
total_loss = 0
trade_count = 0
backoff_duration = INITIAL_BACKOFF
equity_curve = []

# Constants
API_BASE_URL = "https://api.exchange.com"
WEBSOCKET_URL = "wss://ws.exchange.com/realtime"
SYMBOL = "BTCUSD"
TRADE_AMOUNT = 0.01

# Calculate Sharpe Ratio
def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.01):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate assuming 252 trading days in a year
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio * np.sqrt(252)  # Annualize the Sharpe ratio

# Log performance metrics with Sharpe Ratio
def log_performance_metrics(total_trades, total_profit_loss, max_drawdown, sharpe_ratio):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO performance_metrics (timestamp, total_trades, total_profit_loss, max_drawdown, sharpe_ratio) VALUES (?, ?, ?, ?, ?)''',
                   (datetime.now().isoformat(), total_trades, total_profit_loss, max_drawdown, sharpe_ratio))
    conn.commit()
    conn.close()



def fetch_historical_data_from_db():
    conn = sqlite3.connect('trading_bot.db')
    df = pd.read_sql_query("SELECT * FROM historical_data", conn)
    conn.close()
    return df

def log_performance_metrics(total_trades, total_profit_loss, max_drawdown):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO performance_metrics (timestamp, total_trades, total_profit_loss, max_drawdown) VALUES (?, ?, ?, ?)''',
                   (datetime.now().isoformat(), total_trades, total_profit_loss, max_drawdown))
    conn.commit()
    conn.close()

# Calculate maximum drawdown
def calculate_max_drawdown(equity_curve):
    peak = equity_curve[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

# Predict anomaly using the trained model
def predict_anomaly():
    global historical_data

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

    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff']
    X_latest = latest_data[features]

    prediction = model.predict(X_latest)[0]

    if prediction == 1:
        simulate_trade("buy", TRADE_AMOUNT)
    elif prediction == -1:
        simulate_trade("sell", TRADE_AMOUNT)

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

def store_historical_data(df):
    conn = sqlite3.connect('trading_bot.db')
    df.to_sql('historical_data', conn, if_exists='append', index=False)
    conn.close()

# Fetch and preprocess data
symbols = ["BTCUSD", "ETHUSD", "LTCUSD"]  # Add more symbols as needed
all_data = pd.concat([preprocess_data(fetch_historical_data(symbol)) for symbol in symbols])

# Store preprocessed data in the database
store_historical_data(all_data)

# Define the execute_trade function
def log_trade(trade_data, response):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO trade_logs (timestamp, symbol, side, quantity, price, response) VALUES (?, ?, ?, ?, ?, ?)''',
                   (datetime.now().isoformat(), trade_data['symbol'], trade_data['side'], trade_data['quantity'], trade_data['price'], str(response)))
    conn.commit()
    conn.close()

def log_performance_metrics(total_trades, total_profit_loss, max_drawdown):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO performance_metrics (timestamp, total_trades, total_profit_loss, max_drawdown) VALUES (?, ?, ?, ?)''',
                   (datetime.now().isoformat(), total_trades, total_profit_loss, max_drawdown))
    conn.commit()
    conn.close()

# Update the execute_trade function to log trades
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

    # Log the trade
    trade_data['price'] = trade_result.get('price', 0)  # Assuming the response contains the trade price
    log_trade(trade_data, trade_result)

    # Update performance metrics (simplified example)
    global trade_count, total_loss
    trade_count += 1
    total_loss += trade_result.get('loss', 0)  # Assuming the response contains the trade loss
    log_performance_metrics(trade_count, total_loss, max_drawdown=0)  # max_drawdown calculation can be added

def fetch_historical_data_from_db():
    conn = sqlite3.connect('trading_bot.db')
    df = pd.read_sql_query("SELECT * FROM historical_data", conn)
    conn.close()
    return df

def fetch_trade_logs():
    conn = sqlite3.connect('trading_bot.db')
    df = pd.read_sql_query("SELECT * FROM trade_logs", conn)
    conn.close()
    return df

def fetch_performance_metrics():
    conn = sqlite3.connect('trading_bot.db')
    df = pd.read_sql_query("SELECT * FROM performance_metrics", conn)
    conn.close()
    return df

# Example usage of analysis methods
if __name__ == "__main__":
    # Fetch and analyze historical data
    historical_data_df = fetch_historical_data_from_db()
    print(historical_data_df.describe())

    # Fetch and analyze trade logs
    trade_logs_df = fetch_trade_logs()
    print(trade_logs_df)

    # Fetch and analyze performance metrics
    performance_metrics_df = fetch_performance_metrics()
    print(performance_metrics_df)


# Main function
if __name__ == "__main__":
    historical_data = fetch_historical_data(SYMBOL)
    store_historical_data(historical_data)
    logging.info("Fetched and stored historical data")

    ws = websocket.WebSocketApp(WEBSOCKET_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    logging.info("Starting WebSocket connection")
    ws.run_forever()

# Simulate trade execution for paper trading
def simulate_trade(side, amount):
    global trade_count, total_loss, equity_curve

    trade_price = historical_data.iloc[-1]['price']
    trade_loss = 0  # Simplified example without calculating actual profit/loss

    trade_count += 1
    total_loss += trade_loss

    # Update equity curve
    if equity_curve:
        equity_curve.append(equity_curve[-1] - trade_loss)
    else:
        equity_curve = [10000 - trade_loss]  # Starting equity of 10000

    logging.info(f"Simulated {side} trade for {amount} at price {trade_price}")

    # Calculate and log performance metrics
    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe_ratio = calculate_sharpe_ratio(equity_curve)
    log_performance_metrics(trade_count, total_loss, max_drawdown, sharpe_ratio)

# Execute trade and simulate performance
def execute_trade(side, amount):
    global trade_count, total_loss, equity_curve

    trade_price = historical_data.iloc[-1]['price']
    trade_loss = 0  # Simplified example without calculating actual profit/loss

    trade_count += 1
    total_loss += trade_loss

    # Update equity curve
    if equity_curve:
        equity_curve.append(equity_curve[-1] - trade_loss)
    else:
        equity_curve = [10000 - trade_loss]  # Starting equity of 10000

    logging.info(f"Executed {side} trade for {amount} at price {trade_price}")

    # Calculate and log performance metrics
    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe_ratio = calculate_sharpe_ratio(equity_curve)
    log_performance_metrics(trade_count, total_loss, max_drawdown, sharpe_ratio)

# Main function for live testing (paper trading)
if __name__ == "__main__":
    historical_data = fetch_historical_data_from_db()
    logging.info("Fetched historical data")

    ws = websocket.WebSocketApp(WEBSOCKET_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    logging.info("Starting WebSocket connection")
    ws.run_forever()

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

    timestamp = datetime.fromisoformat(data['time'].replace("Z", ""))
    price = float(data['price'])
    volume = float(data['volume'])
    new_row = pd.DataFrame([[timestamp, price, volume]], columns=['time', 'price', 'volume'])
    historical_data = pd.concat([historical_data, new_row]).reset_index(drop=True)

    predict_anomaly()

# Main function for live testing (paper trading)
if __name__ == "__main__":
    historical_data = fetch_historical_data_from_db()
    logging.info("Fetched historical data")

    ws = websocket.WebSocketApp(WEBSOCKET_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    logging.info("Starting WebSocket connection")
    ws.run_forever()