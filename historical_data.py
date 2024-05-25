import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import requests

# Fetch historical data for multiple symbols
def fetch_historical_data(symbol):
    response = requests.get(f"{API_BASE_URL}/historical/{symbol}")
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    return df

# Preprocess data to create features and labels
def preprocess_data(df):
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

# Fetch and preprocess data
symbols = ["BTCUSD", "ETHUSD", "LTCUSD"]  # Add more symbols as needed
all_data = pd.concat([preprocess_data(fetch_historical_data(symbol)) for symbol in symbols])

# Save preprocessed data
all_data.to_csv('historical_data.csv', index=False)
