import logging
import sqlite3
import pandas as pd
import requests
import websocket
import json
import argparse
import numpy as np
from numba import jit
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sqlalchemy import create_engine

logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

API_BASE_URL = "https://api.exchange.com"
WEBSOCKET_URL = "wss://ws.exchange.com/realtime"
SYMBOL = "BTCUSD"
TRADE_AMOUNT = 0.01
MAX_POSITION_SIZE = 10000  # Maximum position size in USD
STOP_LOSS_PERCENT = 0.05  # 5% stop-loss
TRAINING_SPLIT_RATIO = 0.8
TRAINING_CV_FOLDS = 5
DB_URI = "sqlite:///trading_bot.db"

scaler = MinMaxScaler()

def calculate_position_size(account_balance, risk_per_trade):
    position_size = account_balance * risk_per_trade
    return min(position_size, MAX_POSITION_SIZE)

def fetch_data(symbol):
    response = requests.get(f"{API_BASE_URL}/historical/{symbol}")
    data = response.json()
    return pd.DataFrame(data)

def preprocess_data(data):
    data = calculate_features(data)
    data = scaler.fit_transform(data)
    return data

def calculate_features(data):
    data['price_change'] = data['price'].pct_change().values
    data['volume_change'] = data['volume'].pct_change().values
    data['ma_10'] = data['price'].rolling(window=10).mean().values
    data['ma_50'] = data['price'].rolling(window=50).mean().values
    data['ma_200'] = data['price'].rolling(window=200).mean().values
    data['ma_diff'] = data['ma_10'] - data['ma_50']
    data['std_10'] = data['price'].rolling(window=10).std().values
    data['std_50'] = data['price'].rolling(window=50).std().values
    data['momentum'] = data['price'] - data['price'].shift(4)
    data['volatility'] = data['price'].rolling(window=20).std() / data['price'].rolling(window=20).mean()
    data.dropna(inplace=True, how='any')
    return data

def train_model(X, y):
    models = [
        ('RandomForest', RandomForestClassifier()),
        ('GradientBoosting', GradientBoostingClassifier()),
        ('SVC', SVC())
    ]

    best_model = None
    best_score = -1

    for name, model in models:
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(model, param_grid, cv=TRAINING_CV_FOLDS, scoring='accuracy')
        grid_search.fit(X, y)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_

    return best_model

def store_model(model, filename):
    dump(model, filename)

def load_model(filename):
    return load(filename)

def store_data(data, table_name):
    engine = create_engine(DB_URI)
    data.to_sql(table_name, engine, if_exists='replace')

def fetch_data_from_db(table_name):
    engine = create_engine(DB_URI)
    data = pd.read_sql_table(table_name, engine)
    return data

def main(args):
    data = fetch_data(SYMBOL)
    historical_data = preprocess_data(data)

    if args.train:
        split_index = int(len(historical_data) * TRAINING_SPLIT_RATIO)
        X_train = historical_data[:split_index]
        y_train = np.sign(X_train.pop('price_change'))

        model = train_model(X_train, y_train)
        store_model(model, 'optimized_model.pkl')

    elif args.predict:
        model = load_model('optimized_model.pkl')
        X_test = preprocess_data(fetch_data(SYMBOL))
        y_pred = model.predict(X_test)
        print(classification_report(np.sign(X_test.pop('price_change')), y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Make predictions")

    args = parser.parse_args()

    main(args)
