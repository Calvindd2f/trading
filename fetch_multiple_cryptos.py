import aiohttp
import asyncio
import pandas as pd
import numpy as np
import t

async def fetch_historical_data(crypto_id, vs_currency, days):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        #'interval': 'hourly'
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            return data

def process_data(data, crypto_id):
    if 'prices' not in data or 'total_volumes' not in data:
        raise KeyError("Expected keys not found in the API response")

    prices = data['prices']
    volumes = data['total_volumes']

    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['volume'] = [v[1] for v in volumes]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['crypto'] = crypto_id
    return df

def calculate_indicators(df):
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['ma_10'] = talib.SMA(df['price'], timeperiod=10)
    df['ma_50'] = talib.SMA(df['price'], timeperiod=50)
    df['ma_200'] = talib.SMA(df['price'], timeperiod=200)
    df['ma_diff'] = df['ma_10'] - df['ma_50']
    df['std_10'] = df['price'].rolling(window=10).std()
    df['std_50'] = df['price'].rolling(window=50).std()
    df['momentum'] = talib.MOM(df['price'], timeperiod=4)
    df['volatility'] = df['price'].rolling(window=20).std() / df['price'].rolling(window=20).mean()
    df['rsi'] = talib.RSI(df['price'], timeperiod=14)
    df['macd'], macdsignal, macdhist = talib.MACD(df['price'], fastperiod=12, slowperiod=26, signalperiod=9)
    return df

def save_data(df, filename):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

async def main():
    crypto_ids = ['bitcoin', 'ethereum', 'litecoin']  # Example cryptocurrencies
    vs_currency = 'usd'
    days = '30'  # Fetch data for the past 30 days

    combined_df = pd.DataFrame()

    for crypto_id in crypto_ids:
        print(f"Fetching data for {crypto_id} from CoinGecko API")
        data = await fetch_historical_data(crypto_id, vs_currency, days)
        df = process_data(data, crypto_id)
        df = calculate_indicators(df)
        combined_df = pd.concat([combined_df, df])

    save_data(combined_df, 'data/historical_data.csv')

if __name__ == "__main__":
    asyncio.run(main())
