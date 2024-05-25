import sqlite3
import logging
from datetime import datetime
import pandas as pd
import requests
import websocket
import json
import logging
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import schedule # Schedule periodic review
import time

import numpy as np

# Load the optimized model
model = joblib.load('optimized_pump_dump_model.pkl')

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize global variables for performance metrics
total_loss = 0
trade_count = 0
backoff_duration = INITIAL_BACKOFF
equity_curve = []

# Define risk management parameters
MAX_POSITION_SIZE = 10000  # Maximum position size in USD
STOP_LOSS_PERCENT = 0.05  # 5% stop-loss


# Constants
API_BASE_URL = "https://api.exchange.com"
WEBSOCKET_URL = "wss://ws.exchange.com/realtime"
SYMBOL = "BTCUSD"
TRADE_AMOUNT = 0.01

# Calculate position size based on risk management
def calculate_position_size(account_balance, risk_per_trade):
    position_size = account_balance * risk_per_trade
    return min(position_size, MAX_POSITION_SIZE)

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

# Calculate Sharpe Ratio
def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.01):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate assuming 252 trading days in a year
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio * np.sqrt(252)  # Annualize the Sharpe ratio


# Fetch historical data for initial processing
def fetch_historical_data(symbol):
    response = requests.get(f"{API_BASE_URL}/historical/{symbol}")
    data = response.json()
    return pd.DataFrame(data)

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

def store_historical_data(df):
    conn = sqlite3.connect('trading_bot.db')
    df.to_sql('historical_data', conn, if_exists='append', index=False)
    conn.close()

# Fetch and preprocess data
symbols = ["BTCUSD", "ETHUSD", "LTCUSD"]  # Add more symbols as needed
all_data = pd.concat([preprocess_data(fetch_historical_data(symbol)) for symbol in symbols])

# Store preprocessed data in the database
store_historical_data(all_data)

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {np.mean(cv_scores)}")

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

# Additional backtesting
def additional_backtesting():
    global historical_data

    for symbol in ["BTCUSD", "ETHUSD", "LTCUSD"]:
        historical_data = preprocess_data(fetch_historical_data(symbol))
        backtest_trading_bot(historical_data)

if __name__ == "__main__":
    additional_backtesting()


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

# Execute trade with stop-loss
def execute_trade(side, amount):
    global trade_count, total_loss, equity_curve

    trade_price = historical_data.iloc[-1]['price']
    position_size = calculate_position_size(10000, 0.01)  # Example: 1% risk per trade
    trade_amount = position_size / trade_price

    if side == 'buy':
        stop_loss_price = trade_price * (1 - STOP_LOSS_PERCENT)
    else:
        stop_loss_price = trade_price * (1 + STOP_LOSS_PERCENT)

    # Simulate trade execution (use a dummy price for simplicity)
    trade_loss = 0  # Simplified example without calculating actual profit/loss

    trade_count += 1
    total_loss += trade_loss

    if equity_curve:
        equity_curve.append(equity_curve[-1] - trade_loss)
    else:
        equity_curve = [10000 - trade_loss]  # Starting equity of 10000

    logging.info(f"Executed {side} trade for {trade_amount} at price {trade_price}, stop loss at {stop_loss_price}")

    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe_ratio = calculate_sharpe_ratio(equity_curve)
    log_performance_metrics(trade_count, total_loss, max_drawdown, sharpe_ratio)

    # Adjust backoff logic based on strategy optimization
    global backoff_duration
    backoff_duration = max(1, backoff_duration * 0.9 if side == 'buy' else backoff_duration * 1.1)

# Use the best model for prediction
model = best_model

# Predict anomaly using the optimized model
def predict_anomaly():
    global historical_data

    latest_data = historical_data.iloc[-1:]
    latest_data['price_change'] = latest_data['price'].pct_change()
    latest_data['volume_change'] = latest_data['volume'].pct_change()
    latest_data['ma_10'] = latest_data['price'].rolling(window=10).mean()
    latest_data['ma_50'] = latest_data['price'].rolling(window=50).mean()
    latest_data['ma_200'] = latest_data['price'].rolling(window=200).mean()
    latest_data['ma_diff'] = latest_data['ma_10'] - latest_data['ma_50']
    latest_data['std_10'] = latest_data['price'].rolling(window=10).std()
    latest_data['std_50'] = latest_data['price'].rolling(window=50).std()
    latest_data['momentum'] = latest_data['price'] - latest_data['price'].shift(4)
    latest_data['volatility'] = latest_data['price'].rolling(window=20).std() / latest_data['price'].rolling(window=20).mean()
    latest_data.dropna(inplace=True)

    if latest_data.empty:
        return

    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility']
    X_latest = latest_data[features]

    prediction = model.predict(X_latest)[0]

    if prediction == 1:
        execute_trade("buy", TRADE_AMOUNT)
    elif prediction == -1:
        execute_trade("sell", TRADE_AMOUNT)

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

# Periodic review function to analyze performance and adjust strategy
# Fetch performance metrics from the database
def fetch_performance_metrics():
    conn = sqlite3.connect('trading_bot.db')
    df = pd.read_sql_query("SELECT * FROM performance_metrics", conn)
    conn.close()
    return df

# Periodic review function to analyze performance and adjust strategy
def periodic_review():
    metrics_df = fetch_performance_metrics()
    
    # Analyze recent performance
    recent_metrics = metrics_df.tail(50)  # Analyze the last 50 records
    avg_sharpe_ratio = recent_metrics['sharpe_ratio'].mean()
    avg_max_drawdown = recent_metrics['max_drawdown'].mean()

    print(f"Average Sharpe Ratio (last 50 trades): {avg_sharpe_ratio}")
    print(f"Average Max Drawdown (last 50 trades): {avg_max_drawdown}")

    # Adjust strategy based on performance
    if avg_sharpe_ratio < 0.5:  # Example threshold for Sharpe Ratio
        print("Sharpe Ratio below threshold, adjusting strategy...")
        # Adjust parameters, risk management, or trading logic here
        adjust_strategy()

    if avg_max_drawdown > 0.2:  # Example threshold for Max Drawdown
        print("Max Drawdown above threshold, increasing risk management...")
        # Increase stop-loss percentage or reduce position size
        increase_risk_management()

# Adjust strategy based on performance insights
def adjust_strategy():
    global BUY_THRESHOLD, SELL_THRESHOLD
    BUY_THRESHOLD += 0.01  # Example adjustment
    SELL_THRESHOLD -= 0.01
    print(f"Adjusted BUY_THRESHOLD to {BUY_THRESHOLD} and SELL_THRESHOLD to {SELL_THRESHOLD}")

# Increase risk management measures
def increase_risk_management():
    global STOP_LOSS_PERCENT, MAX_POSITION_SIZE
    STOP_LOSS_PERCENT -= 0.01  # Example adjustment
    MAX_POSITION_SIZE *= 0.9  # Reduce maximum position size by 10%
    print(f"Adjusted STOP_LOSS_PERCENT to {STOP_LOSS_PERCENT} and MAX_POSITION_SIZE to {MAX_POSITION_SIZE}")

def schedule_periodic_review():
    schedule.every().day.at("00:00").do(periodic_review)  # Schedule the review to run daily at midnight

    while True:
        schedule.run_pending()
        time.sleep(1)

# Main function to initialize and start the trading bot
if __name__ == "__main__":
    historical_data = fetch_historical_data_from_db()
    logging.info("Fetched historical data")

    # Start the WebSocket connection for real-time data
    ws = websocket.WebSocketApp(WEBSOCKET_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    logging.info("Starting WebSocket connection")
    
    # Schedule periodic performance reviews
    schedule_periodic_review()
    
    # Run the WebSocket connection in a separate thread to allow scheduling
    import threading
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.start()