import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

def preprocess_data(data):
    # Feature engineering
    data['price_change'] = data['price'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['ma_10'] = data['price'].rolling(window=10).mean()
    data['ma_50'] = data['price'].rolling(window=50).mean()
    data['ma_200'] = data['price'].rolling(window=200).mean()
    data['ma_diff'] = data['ma_10'] - data['ma_50']
    data['std_10'] = data['price'].rolling(window=10).std()
    data['std_50'] = data['price'].rolling(window=50).std()
    data['momentum'] = data['price'] - data['price'].shift(4)
    data['volatility'] = data['price'].rolling(window=10).std()

    rsi = RSIIndicator(close=data['price'])
    data['rsi'] = rsi.rsi()

    macd = MACD(close=data['price'])
    data['macd'] = macd.macd()

    bollinger = BollingerBands(close=data['price'])
    data['bb_high'] = bollinger.bollinger_hband()
    data['bb_low'] = bollinger.bollinger_lband()

    # Drop NaN values after adding features
    data = data.dropna()
    return data
