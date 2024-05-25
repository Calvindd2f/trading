import logging
import pandas as pd
import websocket
import json
from datetime import datetime
from model import load_model, predict_anomaly
from data_processing import fetch_historical_data_from_db, process_real_time_data
from utils import log_performance_metrics, execute_trade

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize global variables for performance metrics
total_loss = 0
trade_count = 0
backoff_duration = 1  # Example initial backoff duration
equity_curve = []

# WebSocket callback for real-time data
def on_message(ws, message):
    data = json.loads(message)
    process_real_time_data(data, predict_anomaly)

# WebSocket error handler
def on_error(ws, error):
    logging.error(f"Error: {error}")

# WebSocket close handler
def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket closed")

# WebSocket open handler
def on_open(ws):
    subscribe_message = json.dumps({"type": "subscribe", "channels": [{"name": "ticker", "product_ids": ["BTCUSD"]}]})
    ws.send(subscribe_message)
    logging.info("WebSocket connection opened and subscription message sent")

# Main function for live testing (paper trading)
if __name__ == "__main__":
    historical_data = fetch_historical_data_from_db()
    logging.info("Fetched historical data")

    ws = websocket.WebSocketApp("wss://example.com/realtime", on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    logging.info("Starting WebSocket connection")

    # Run the WebSocket connection in a separate thread to allow scheduling
    import threading
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.start()
