import sqlite3
import pandas as pd
import numpy as np
from numba import jit
from datetime import datetime
from multiprocessing import Pool

def preprocess_chunk(chunk):
    return preprocess_data(chunk)

def preprocess_data_parallel(df, num_chunks=4):
    chunks = np.array_split(df, num_chunks)
    with Pool(num_chunks) as pool:
        df = pd.concat(pool.map(preprocess_chunk, chunks))
    return df

def preprocess_data(data):
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
    data['rsi'] = calculate_rsi(data['price'].values)
    data['macd'] = calculate_macd(data['price'].values)
    data.dropna(inplace=True)
    return data

def fetch_historical_data_from_db():
    conn = sqlite3.connect('trading_bot.db')
    df = pd.read_sql_query("SELECT * FROM historical_data", conn)
    conn.close()
    return df


def process_real_time_data(data, predict_anomaly):
    global historical_data

    timestamp = datetime.fromisoformat(data['time'].replace("Z", ""))
    price = float(data['price'])
    volume = float(data['volume'])
    new_row = pd.DataFrame([[timestamp, price, volume]], columns=['time', 'price', 'volume'])
    historical_data = pd.concat([historical_data, new_row]).reset_index(drop=True)

    predict_anomaly()

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('data/historical_data.csv')
    processed_data = preprocess_data_parallel(data, num_chunks=4)
    print(processed_data.head())