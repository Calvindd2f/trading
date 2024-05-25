import sqlite3
import websocket
import json
import logging
from datetime import datetime
import joblib
import pandas as pd
import sqlite3
import logging
from datetime import datetime
import joblib
import pandas as pd
import websocket
import json
import logging

# Load the trained model
model = joblib.load('pump_dump_model.pkl')

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

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

def log_performance_metrics(total_trades, total_profit_loss, max_drawdown):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO performance_metrics (timestamp, total_trades, total_profit_loss, max_drawdown) VALUES (?, ?, ?, ?)''',
                   (datetime.now().isoformat(), total_trades, total_profit_loss, max_drawdown))
    conn.commit()
    conn.close()

    logging.info(f"Performance metrics logged: total_trades={total_trades}, total_profit_loss={total_profit_loss}, max_drawdown={max_drawdown}")

# Simulate trade execution for paper trading (updated)
def simulate_trade(side, amount):
    global trade_count, total_loss

    trade_price = historical_data.iloc[-1]['price']
    trade_loss = 0  # Simplified example without calculating actual profit/loss

    trade_count += 1
    total_loss += trade_loss

    logging.info(f"Simulated {side} trade for {amount} at price {trade_price}")

    # Log performance metrics
    log_performance_metrics(trade_count, total_loss, max_drawdown=0)  # max_drawdown calculation can be added

# Execute trade and simulate performance (updated)
def execute_trade(side, amount):
    global trade_count, total_loss

    # Simulate trade execution (use a dummy price for simplicity)
    trade_price = historical_data.iloc[-1]['price']
    trade_loss = 0  # Simplified example without calculating actual profit/loss

    trade_count += 1
    total_loss += trade_loss

    # Log the trade (in real application, store it in the database)
    print(f"Executed {side} trade for {amount} at price {trade_price}")

    # Log performance metrics
    log_performance_metrics(trade_count, total_loss, max_drawdown=0)  # max_draw






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
