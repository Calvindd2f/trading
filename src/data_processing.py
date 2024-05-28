import sqlite3
import pandas as pd
import numpy as np
from numba import jit
from datetime import datetime
from multiprocessing import Pool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def preprocess_chunk(chunk: pd.DataFrame):
    preprocess_data(chunk)

def preprocess_data_parallel(df: pd.DataFrame, num_chunks: int = 4) -> pd.DataFrame:
    if not df.empty:
        df.sort_values('time', inplace=True)
        chunks = np.array_split(df, num_chunks)
        with Pool(num_chunks) as pool:
            df_list = pool.map(preprocess_chunk, chunks)
        df = pd.concat(df_list).reset_index(drop=True)
    return df

def preprocess_data(data: pd.DataFrame):
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
    data['macd'], _, _ = calculate_macd(data['price'].values)
    data.dropna(inplace=True)
    return data

@jit
def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    deltas = np.diff(prices)
    seed = deltas[:window + 1]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100. / (1. + rs)

    for i in range(window, len(prices)):
        delta = deltas[i - 1]  # the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

@jit
def calculate_macd(prices: np.ndarray, slow: int = 26, fast: int = 12, signal: int = 9):
    exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd.values, signal_line.values, macd_hist.values

def fetch_historical_data_from_db(db_path='trading_bot.db', table_name='historical_data') -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def process_real_time_data(data: dict, historical_data: pd.DataFrame, predict_anomaly: callable):
    try:
        timestamp = datetime.fromisoformat(data['time'].replace("Z", ""))
        price = float(data['price'])
        volume = float(data['volume'])
        new_row = pd.DataFrame([[timestamp, price, volume]], columns=['time', 'price', 'volume'])
        historical_data = pd.concat([historical_data, new_row]).reset_index(drop=True)

        predict_anomaly(historical_data)
    except Exception as e:
        logging.error(f"Error processing real-time data: {e}")

if __name__ == "__main__":
    historical_data = fetch_historical_data_from_db()
    data = pd.read_csv('data/historical_data.csv')
    processed_data = preprocess_data_parallel(data, num_chunks=4)
    logging.info(f"Processed data: \n{processed_data.head()}")
