import aiohttp
import asyncio
import pandas as pd
import numpy as np
import talib
from datetime import datetime
from aiohttp import ClientTimeout

async def fetch_historical_data(crypto_id, vs_currency, days):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
    }
    timeout = ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                print(f"Error fetching data for {crypto_id}: {response.status}")
                return None
            data = await response.json()
            return data

