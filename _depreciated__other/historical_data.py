import pandas as pd
import requests
import logging
from joblib import dump

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set API base URL, buy, and sell thresholds as global constants
API_BASE_URL = "https://api.example.com"
BUY_THRESHOLD = 0.05
SELL_THRESHOLD = -0.05

def fetch_historical_data(symbol):
    """
    Fetch historical data for a given symbol.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/historical/{symbol}")
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch historical data for {symbol}: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess data to create features and labels.
    """
    try:
        df['price_change'] = df['price'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['ma_10'] = df['price'].rolling(window=10).mean()
        df['ma_50'] = df['price'].rolling(window=50).mean()
        df['ma_200'] = df['price'].rolling(window=200).mean()
        df['ma_diff'] = df['ma_10'] - df['ma_50']
        df['label'] = 0  # Default label for normal behavior
        df.loc[df['price_change'] >= BUY_THRESHOLD, 'label'] = 1  # Label for pump
        df.loc[df['price_change'] <= SELL_THRESHOLD, 'label'] = -1  # Label for dump
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Failed to preprocess data: {e}")
        return None

def save_preprocessed_data(df, filename):
    """
    Save preprocessed data to a CSV file.
    """
    try:
        df.to_csv(filename, index=False)
        logging.info(f"Preprocessed data saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save preprocessed data: {e}")

def main():
    """
    Fetch and preprocess historical data for multiple symbols.
    """
    symbols = ["BTCUSD", "ETHUSD", "LTCUSD"]  # Add more symbols as needed
    all_data = pd.concat([preprocess_data(fetch_historical_data(symbol)) for symbol in symbols if fetch_historical_data(symbol) is not None])
    save_preprocessed_data(all_data, 'historical_data.csv')

if __name__ == "__main__":
    main()
