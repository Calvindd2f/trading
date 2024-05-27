import logging
import pandas as pd
import json
from datetime import datetime
import asyncio
import aiohttp
from aiologger import Logger
from data_processing import fetch_historical_data_from_db, process_real_time_data
from model import load_model, predict_anomaly
from utils import log_performance_metrics, execute_trade
from retraining.training import train_model, save_model, three_pass_training, preprocess_data
import model_tuning

# Configure asynchronous logger
logger = Logger.with_default_handlers(name='trading_bot_logger')

# Initialize global variables for performance metrics
total_loss = 0
trade_count = 0
backoff_duration = 1  # Example initial backoff duration
equity_curve = []
loss_threshold = -1000  # Example loss threshold

# Load model
model = load_model()

# Fetch historical data
historical_data = fetch_historical_data_from_db()

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def process_real_time_data_async(data):
    global historical_data, total_loss, trade_count

    timestamp = datetime.fromisoformat(data['time'].replace("Z", ""))
    price = float(data['price'])
    volume = float(data['volume'])
    new_row = pd.DataFrame([[timestamp, price, volume]], columns=['time', 'price', 'volume'])
    historical_data = pd.concat([historical_data, new_row]).reset_index(drop=True)

    predictions = predict_anomaly(model, historical_data, TRADE_AMOUNT=1000)

    # Log the predictions for debugging purposes
    logger.info(f"Predictions: {predictions}")

    # Execute trade if an anomaly is detected
    if predictions is not None and predictions[-1] == 1:
        execute_trade('buy', 1000)  # Example trade side 'buy', amount 1000

    # Check if losses exceed threshold
    if total_loss <= loss_threshold:
        await cease_live_executions()

async def on_message(message):
    data = json.loads(message)
    await process_real_time_data_async(data)

async def websocket_handler():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect('wss://example.com/realtime') as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await on_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

async def cease_live_executions():
    logger.warning("Ceasing live executions due to excessive losses.")
    # Implement logic to cease live executions, such as closing positions and stopping the bot
    await retrain_model()

async def retrain_model():
    logger.info("Initiating retraining session...")
    data = fetch_historical_data_from_db()
    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']
    final_result = three_pass_training(data, features)
    logger.info(f"Final result after three passes: {final_result}")
    if final_result > 0:
        best_model = train_model(data, features)['GradientBoosting']
        save_model(best_model, 'src/optimized_pump_dump_model.pkl')
        global model
        model = load_model('src/optimized_pump_dump_model.pkl')
        logger.info("Retraining completed and model updated.")
    else:
        logger.warning("Training failed to achieve positive gain/loss. Model not updated.")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(websocket_handler())
