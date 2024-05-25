import asyncio
import aiohttp
from anyio import Path
#import requests
import pandas as pd




async def fetch_historical_data(crypto_id: str, vs_currency: str, days: int) -> dict:
    if crypto_id is None or vs_currency is None or days is None:
        raise ValueError("Arguments cannot be None")
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        #'interval': 'hourly'
    }
    print(f"Fetching data for {crypto_id} from Coingecko API")
    print(f"URL: {url}")
    print(f"Parameters: {params}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            return data

    #response = requests.get(url, params=params)
    #print(f"Status code: {response.status_code}")
    #if response.status_code != 200:
    #    raise Exception(f"Failed to fetch data for {crypto_id}. Status code: {response.status_code}")
    #data = response.json()
    #if data is None:
    #    raise Exception(f"Failed to fetch data for {crypto_id}. No response found.")
    #print(f"Data: {data}")
    #return data


def remove_bom(data):
    if data is None:
        raise ValueError("Cannot remove BOM from None")
    bom = b'\xef\xbb\xbf'
    if data.startswith(bom):
        print(f"Remove BOM from data: {data[:10]}")
        return data[len(bom):]
    print(f"No BOM found in data: {data[:10]}")
    return data


def process_data(data: dict, crypto_id: str) -> pd.DataFrame:
    """
    Process the data fetched from the Coingecko API.

    Args:
        data (dict): The response from the Coingecko API.
        crypto_id (str): The id of the cryptocurrency that the data is for.

    Returns:
        pd.DataFrame: The processed DataFrame.

    Raises:
        ValueError: If the data is empty or no response is found.
        KeyError: If the required keys are not found in the response.
    """
    print(f"Processing data for {crypto_id}")
    if data is None:
        raise ValueError(f"Failed to fetch data for {crypto_id}. No response found.")
    if 'prices' not in data or 'total_volumes' not in data:
        raise KeyError("Expected keys not found in the API response")

    prices = data['prices']
    volumes = data['total_volumes']

    # Create a DataFrame
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['volume'] = [v[1] for v in volumes]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['crypto'] = crypto_id  # Add the crypto column

    print(f"DataFrame: {df}")
    return df


def save_data(df, filename):
    if df is None:
        raise ValueError(f"Failed to save data to {filename}. No DataFrame provided.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame but got {type(df).__name__}")
    if not filename:
        raise ValueError(f"Failed to save data to file. No filename provided.")
    filepath = Path(filename)
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    try:
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    except Exception as e:
        raise ValueError(f"Failed to save data to {filename}. Error: {e}") from e

async def main():
    crypto_ids = ['bitcoin', 'ethereum', 'litecoin']  # Example cryptocurrencies
    vs_currency = 'usd'
    days = '30'  # Fetch data for the past 30 days

    combined_df = pd.DataFrame()

    for crypto_id in crypto_ids:
        print(f"Fetching data for {crypto_id} from Coingecko API")
        data = await fetch_historical_data(crypto_id, vs_currency, days)
        df = process_data(data, crypto_id)
        combined_df = pd.concat([combined_df, df])

    save_data(combined_df, 'data/historical_data.csv')

if __name__ == "__main__":
    asyncio.run(main())