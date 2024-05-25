import joblib
import pandas as pd
from utils import execute_trade

model = joblib.load('optimized_pump_dump_model.pkl')

def load_model(model_path='optimized_pump_dump_model.pkl'):
    return joblib.load(model_path)

def predict_anomaly():
    global historical_data

    latest_data = historical_data.iloc[-1:]
    latest_data['price_change'] = latest_data['price'].pct_change().values
    latest_data['volume_change'] = latest_data['volume'].pct_change().values
    latest_data['ma_10'] = latest_data['price'].rolling(window=10).mean().values
    latest_data['ma_50'] = latest_data['price'].rolling(window=50).mean().values
    latest_data['ma_200'] = latest_data['price'].rolling(window=200).mean().values
    latest_data['ma_diff'] = latest_data['ma_10'] - latest_data['ma_50']
    latest_data['std_10'] = latest_data['price'].rolling(window=10).std().values
    latest_data['std_50'] = latest_data['price'].rolling(window=50).std().values
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
