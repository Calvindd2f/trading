import os
from flask import Flask, render_template, jsonify, request, Response
from data_processing import fetch_historical_data_from_db
from retraining.training import three_pass_training, train_model, save_model
from model import load_model, preprocess_data
import logging
import pandas as pd
import random
import asyncio
import threading
from ta import add_all_ta_features

app = Flask(__name__)

# Initialize global variables for metrics
metrics = {'total_loss': 0, 'trade_count': 0, 'equity_curve': []}
loss_threshold = -1000  # Example loss threshold

# Path to the model file
model_path = 'src/optimized_pump_dump_model.pkl'

# Load initial model if it exists, otherwise log an error
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    logging.error(f"Model file not found at {model_path}. Please generate the model first.")

# Fetch historical data
historical_data = fetch_historical_data_from_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    data = fetch_historical_data_from_db()
    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']
    data = add_all_ta_features(data, open='open', high='high', low='low', close='close', volume='volume')
    final_result = three_pass_training(data, features)
    logging.info(f"Final result after three passes: {final_result}")
    if final_result > 0:
        best_model = train_model(data, features)['GradientBoosting']
        save_model(best_model, model_path)
        global model
        model = load_model(model_path)
        logging.info("Retraining completed and model updated.")
        return jsonify({'status': 'success', 'message': 'Retraining completed successfully.'})
    else:
        logging.warning("Training failed to achieve positive gain/loss. Model not updated.")
        return jsonify({'status': 'failure', 'message': 'Retraining failed to achieve positive gain/loss.'})

@app.route('/get_metrics')
def get_metrics():
    return jsonify(metrics)

@app.route('/get_trades', methods=['GET'])
def get_trades():
    # Placeholder for real-time trades
    trades = [
        {'time': '2023-05-01T12:34:56', 'type': 'buy', 'amount': 1.2, 'price': 50000},
        {'time': '2023-05-01T12:35:56', 'type': 'sell', 'amount': 0.5, 'price': 50500}
    ]
    return jsonify(trades)

def websocket_thread():
    asyncio.run(websocket_handler())

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    threading.Thread(target=websocket_thread).start()
    app.run(debug=True)

